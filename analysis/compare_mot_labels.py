#!/usr/bin/env python3
"""
Compare MOT labels across three sources at the scene-group level.

Sources
-------
  human        — dataset/MOT_main/clips_human
  sam3_native  — dataset/MOT_main/labels_sam3_native
  sam3_botsort — dataset/MOT_main/labels_sam3_botsort

Scene groups:  acs_s1 | acs_s2 | ie | rectorat

── Table 1 — basic track statistics ──────────────────────────────────────────
  clips, frames, unique_ids, total_tracks,
  avg/med/min/max track length

── Table 2 — correction delta (SAM3 → human) ─────────────────────────────────
  Δ unique IDs           (ghost IDs removed)
  Δ total tracks         (fragmentation reduction)
  Δ avg track length     (continuity improvement, should be > 0)
  ghost tracks deleted   (SAM3 tracks with no matching human track)
  tracks merged          (multiple SAM3 IDs merged into one human track)
  ID switches corrected  (= tracks merged; each merge fixes one switch)
  detections interpolated (human annotations in gaps of matched SAM3 tracks)

── Figure — centroid heatmaps ────────────────────────────────────────────────
  analysis/figures/centroids_{scene_group}.png
  3 subplots: human | sam3_native | sam3_botsort

Output
------
  analysis/results/compare_mot_labels.csv       (table 1)
  analysis/results/compare_mot_corrections.csv  (table 2)
  analysis/results/compare_mot_labels.json      (both tables)
  analysis/figures/centroids_*.png

Usage
-----
  uv run python analysis/compare_mot_labels.py
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MOT_DIR   = REPO_ROOT / "dataset" / "MOT_main"

SOURCES: dict[str, Path] = {
    "human":        MOT_DIR / "clips_human",
    "sam3_native":  MOT_DIR / "labels_sam3_native",
    "sam3_botsort": MOT_DIR / "labels_sam3_botsort",
}
SAM3_METHODS = ["sam3_native", "sam3_botsort"]

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

SCENE_GROUPS = ["acs_s1", "acs_s2", "ie", "rectorat"]
PAPER_NAMES  = {
    "acs_s1":   "acs_ec",
    "acs_s2":   "acs_eg",
    "ie":       "ie_central",
    "rectorat": "r_central",
}
IOU_THRESH   = 0.5


def scene_group(scene_name: str) -> str:
    return scene_name.split("_recording_")[0]


# ── Parsing ────────────────────────────────────────────────────────────────────


def parse_gt(path: Path) -> dict[int, list[int]]:
    """{tid: [frame, ...]} — frame list per track."""
    tracks: dict[int, list[int]] = defaultdict(list)
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        tracks[int(p[1])].append(int(p[0]))
    return dict(tracks)


def parse_gt_full(path: Path) -> dict[int, dict[int, list[float]]]:
    """{frame: {tid: [x, y, w, h]}}"""
    data: dict[int, dict[int, list[float]]] = defaultdict(dict)
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        frame, tid = int(p[0]), int(p[1])
        data[frame][tid] = [float(p[2]), float(p[3]), float(p[4]), float(p[5])]
    return dict(data)


def clip_total_frames(clip_dir: Path) -> int:
    d = clip_dir / "frames"
    return sum(1 for f in d.iterdir() if f.suffix.lower() == ".jpg") if d.exists() else 0


def clip_image_size(clip_dir: Path) -> tuple[int, int]:
    """Return (width, height) from the first frame."""
    d = clip_dir / "frames"
    if not d.exists():
        return 1920, 1080
    first = next(iter(sorted(d.glob("*.jpg"))), None)
    if first is None:
        return 1920, 1080
    import cv2
    img = cv2.imread(str(first))
    if img is None:
        return 1920, 1080
    return img.shape[1], img.shape[0]


# ── Clip discovery ─────────────────────────────────────────────────────────────


def discover_clips() -> list[dict]:
    ref_root = SOURCES["human"]
    clips = []
    for split_dir in sorted(ref_root.iterdir()):
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            for clip_dir in sorted(scene_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                rel = Path(split_dir.name) / scene_dir.name / clip_dir.name
                gt_paths = {
                    src: root / rel / "gt" / "gt.txt"
                    for src, root in SOURCES.items()
                }
                clips.append({
                    "split":       split_dir.name,
                    "scene":       scene_dir.name,
                    "scene_group": scene_group(scene_dir.name),
                    "clip":        clip_dir.name,
                    "clip_dir":    clip_dir,
                    "gt_paths":    gt_paths,
                })
    return clips


# ── IoU matching & correction analysis ────────────────────────────────────────


def _iou(b1: list[float], b2: list[float]) -> float:
    """IoU of two TLWH boxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-6)


def corrections_for_clip(human_path: Path, sam3_path: Path) -> dict:
    """
    Compare SAM3 labels to human GT for one clip.
    Returns correction counts.
    """
    h_data = parse_gt_full(human_path)   # {frame: {tid: box}}
    s_data = parse_gt_full(sam3_path)

    if not h_data and not s_data:
        return dict(ghost_deleted=0, merged=0, id_switches=0, interpolated=0)

    h_tids = {tid for fd in h_data.values() for tid in fd}
    s_tids = {tid for fd in s_data.values() for tid in fd}

    # Co-occurrence: (s_tid, h_tid) → frames matched
    cooc: dict[tuple[int, int], int] = defaultdict(int)
    for frame, h_frame in h_data.items():
        s_frame = s_data.get(frame, {})
        if not s_frame:
            continue
        for h_tid, hb in h_frame.items():
            for s_tid, sb in s_frame.items():
                if _iou(hb, sb) >= IOU_THRESH:
                    cooc[(s_tid, h_tid)] += 1

    # Best human match for each SAM3 track
    sam3_to_human: dict[int, int] = {}
    for s_tid in s_tids:
        best = max(
            ((h_tid, cnt) for (s, h_tid), cnt in cooc.items() if s == s_tid),
            key=lambda x: x[1], default=None
        )
        if best:
            sam3_to_human[s_tid] = best[0]

    # human → [sam3 tracks matched]
    human_to_sam3s: dict[int, list[int]] = defaultdict(list)
    for s_tid, h_tid in sam3_to_human.items():
        human_to_sam3s[h_tid].append(s_tid)

    # Ghost tracks: SAM3 tracks with no human match
    ghost_deleted = sum(1 for s in s_tids if s not in sam3_to_human)

    # Tracks merged: for each human track with >1 SAM3 tracks, count the merges
    merged = sum(max(0, len(ss) - 1) for ss in human_to_sam3s.values())

    # ID switches corrected = tracks merged (each merge removes one ID switch)
    id_switches = merged

    # Detections interpolated: human frames not covered by any matched SAM3 track
    interpolated = 0
    for h_tid in h_tids:
        h_frames = {f for f, fd in h_data.items() if h_tid in fd}
        s_tids_matched = human_to_sam3s.get(h_tid, [])
        s_frames = {f for f, fd in s_data.items()
                    for st in s_tids_matched if st in fd}
        interpolated += len(h_frames - s_frames)

    return dict(
        ghost_deleted=ghost_deleted,
        merged=merged,
        id_switches=id_switches,
        interpolated=interpolated,
    )


# ── Aggregation ────────────────────────────────────────────────────────────────


def aggregate(clips: list[dict]) -> tuple[dict, dict]:
    """
    Returns:
      stats[sg][src]  = {n_clips, frames, track_lengths}
      corr[sg][src]   = {ghost_deleted, merged, id_switches, interpolated}
    """
    stats = {
        sg: {src: {"n_clips": 0, "frames": 0, "track_lengths": []}
             for src in SOURCES}
        for sg in SCENE_GROUPS
    }
    corr = {
        sg: {m: {"ghost_deleted": 0, "merged": 0,
                 "id_switches": 0, "interpolated": 0}
             for m in SAM3_METHODS}
        for sg in SCENE_GROUPS
    }

    for clip in clips:
        sg = clip["scene_group"]
        if sg not in stats:
            continue

        n_frames = clip_total_frames(clip["clip_dir"])
        for src, gt_path in clip["gt_paths"].items():
            tracks  = parse_gt(gt_path)
            lengths = [len(fs) for fs in tracks.values()]
            d = stats[sg][src]
            d["n_clips"] += 1
            d["frames"]  += n_frames
            d["track_lengths"].extend(lengths)

        h_path = clip["gt_paths"]["human"]
        for m in SAM3_METHODS:
            c = corrections_for_clip(h_path, clip["gt_paths"][m])
            for k in corr[sg][m]:
                corr[sg][m][k] += c[k]

    return stats, corr


def summarize_stats(stats: dict) -> list[dict]:
    rows = []
    for sg in SCENE_GROUPS:
        for src in SOURCES:
            d = stats[sg][src]
            L = d["track_lengths"]
            rows.append({
                "scene_group":  PAPER_NAMES.get(sg, sg),
                "method":       src,
                "clips":        d["n_clips"],
                "frames":       d["frames"],
                "unique_ids":   len(L),
                "total_tracks": len(L),
                "avg_len":      round(statistics.mean(L),   2) if L else 0,
                "med_len":      round(statistics.median(L), 2) if L else 0,
                "min_len":      min(L) if L else 0,
                "max_len":      max(L) if L else 0,
            })
    return rows


def summarize_corrections(stats: dict, corr: dict) -> list[dict]:
    rows = []
    for sg in SCENE_GROUPS:
        h = stats[sg]["human"]
        h_L = h["track_lengths"]
        h_avg = statistics.mean(h_L) if h_L else 0
        h_ids = len(h_L)

        for m in SAM3_METHODS:
            s = stats[sg][m]
            s_L = s["track_lengths"]
            s_avg = statistics.mean(s_L) if s_L else 0
            s_ids = len(s_L)
            c = corr[sg][m]
            rows.append({
                "scene_group":       PAPER_NAMES.get(sg, sg),
                "sam3_method":       m,
                "delta_unique_ids":  s_ids - h_ids,
                "delta_tracks":      s_ids - h_ids,
                "delta_avg_len":     round(h_avg - s_avg, 2),
                "ghost_deleted":     c["ghost_deleted"],
                "tracks_merged":     c["merged"],
                "id_switches_fixed": c["id_switches"],
                "interpolated":      c["interpolated"],
            })
    return rows


# ── Centroid heatmaps ──────────────────────────────────────────────────────────


def collect_centroids(clips: list[dict]) -> dict[str, dict[str, list]]:
    """
    Returns {scene_group: {method: [(cx_norm, cy_norm), ...]}}
    Coordinates normalised to [0, 1] using the clip's image dimensions.
    """
    centroids: dict[str, dict[str, list]] = {
        sg: {src: [] for src in SOURCES} for sg in SCENE_GROUPS
    }
    img_sizes: dict[str, tuple[int, int]] = {}   # clip key → (w, h)

    for clip in clips:
        sg = clip["scene_group"]
        if sg not in centroids:
            continue
        w, h = clip_image_size(clip["clip_dir"])
        for src, gt_path in clip["gt_paths"].items():
            data = parse_gt_full(gt_path)
            for frame_data in data.values():
                for bx, by, bw, bh in frame_data.values():
                    cx = (bx + bw / 2) / w
                    cy = (by + bh / 2) / h
                    centroids[sg][src].append((cx, cy))

    return centroids


def plot_heatmaps(centroids: dict) -> None:
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    BINS   = 150
    SIGMA  = 2.0          # Gaussian smoothing σ in bins
    CMAP   = "YlOrRd"     # white → yellow → orange → red; print-safe
    METHODS = ["human", "sam3_native", "sam3_botsort"]
    METHOD_TITLES = {
        "human":        "Human GT",
        "sam3_native":  "SAM3 native",
        "sam3_botsort": "SAM3 + BoT-SORT",
    }

    # ── Pass 1: build all smoothed histograms, find global max ────────────────
    hists: dict[str, dict[str, np.ndarray]] = {}
    for sg in SCENE_GROUPS:
        hists[sg] = {}
        for src in METHODS:
            pts = centroids[sg][src]
            if pts:
                xs = np.clip([p[0] for p in pts], 0.0, 1.0)
                ys = np.clip([p[1] for p in pts], 0.0, 1.0)
                h2d, _, _ = np.histogram2d(
                    xs, ys, bins=BINS, range=[[0, 1], [0, 1]]
                )
                h2d = gaussian_filter(h2d.T, sigma=SIGMA)  # smooth; .T → (y, x)
            else:
                h2d = np.zeros((BINS, BINS))
            hists[sg][src] = h2d

    global_max = max(
        h.max() for sg_h in hists.values() for h in sg_h.values()
    )
    if global_max <= 0:
        global_max = 1.0

    # ── Pass 2: render one figure per scene group ─────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for sg in SCENE_GROUPS:
        paper = PAPER_NAMES.get(sg, sg)

        fig, axes = plt.subplots(
            1, 3,
            figsize=(10, 3.6),
            gridspec_kw={"wspace": 0.06},
        )
        fig.patch.set_facecolor("white")

        img_obj = None
        for ax, src in zip(axes, METHODS):
            h_norm = hists[sg][src] / global_max   # normalise to [0, 1] globally

            img_obj = ax.imshow(
                h_norm,
                origin="upper",
                extent=[0, 1, 1, 0],               # x: 0→1 left-right,
                                                    # y: 0(top)→1(bottom) image convention
                aspect="auto",
                cmap=CMAP,
                vmin=0.0,
                vmax=1.0,
                interpolation="bicubic",
            )
            ax.set_facecolor("white")
            ax.set_title(METHOD_TITLES[src], fontsize=9, pad=4)
            ax.set_xlabel("$x$", fontsize=8)
            if ax is axes[0]:
                ax.set_ylabel("$y$", fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

        # Shared colorbar spanning all three axes
        fig.subplots_adjust(right=0.87)
        cbar_ax = fig.add_axes([0.89, 0.13, 0.018, 0.74])
        cb = fig.colorbar(img_obj, cax=cbar_ax)
        cb.set_label("Relative density", fontsize=8, labelpad=5)
        cb.ax.tick_params(labelsize=7)

        out = FIGURES_DIR / f"centroids_{paper}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  figure → {out}")


# ── Printing ───────────────────────────────────────────────────────────────────


def print_stats_table(rows: list[dict]) -> None:
    W = 100
    header = (f"{'Scene':<10} {'Method':<14} {'Clips':>6} {'Frames':>7}"
              f" {'UniqIDs':>8} {'AvgLen':>8} {'MedLen':>8}"
              f" {'MinLen':>7} {'MaxLen':>7}")
    cur = None
    for r in rows:
        if r["scene_group"] != cur:
            cur = r["scene_group"]
            print(f"\n{'─'*W}  [{cur}]")
            print(header)
            print("─" * W)
        print(f"{r['scene_group']:<10} {r['method']:<14}"
              f" {r['clips']:>6} {r['frames']:>7}"
              f" {r['unique_ids']:>8}"
              f" {r['avg_len']:>8.1f} {r['med_len']:>8.1f}"
              f" {r['min_len']:>7} {r['max_len']:>7}")


def print_corrections_table(rows: list[dict]) -> None:
    W = 100
    header = (f"{'Scene':<10} {'Method':<14}"
              f" {'ΔUniqIDs':>9} {'ΔAvgLen':>8}"
              f" {'GhostDel':>9} {'Merged':>7}"
              f" {'IDFixes':>8} {'Interp':>7}")
    cur = None
    for r in rows:
        if r["scene_group"] != cur:
            cur = r["scene_group"]
            print(f"\n{'─'*W}  [{cur}]")
            print(header)
            print("─" * W)
        print(f"{r['scene_group']:<10} {r['sam3_method']:<14}"
              f" {r['delta_unique_ids']:>+9} {r['delta_avg_len']:>+8.1f}"
              f" {r['ghost_deleted']:>9} {r['tracks_merged']:>7}"
              f" {r['id_switches_fixed']:>8} {r['interpolated']:>7}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    print("Discovering clips …")
    clips = discover_clips()
    print(f"  {len(clips)} clips  ·  {len(SOURCES)} sources")

    print("Aggregating stats and running correction analysis …")
    stats, corr = aggregate(clips)

    stat_rows = summarize_stats(stats)
    corr_rows = summarize_corrections(stats, corr)

    print("\n\n══ TABLE 1 — Track statistics ══")
    print_stats_table(stat_rows)

    print("\n\n══ TABLE 2 — SAM3 → Human correction deltas ══")
    print("  Positive Δ = SAM3 had MORE  |  Negative Δ = SAM3 had FEWER")
    print("  ΔAvgLen: positive = human tracks are longer (better continuity)")
    print_corrections_table(corr_rows)

    print("\n\nGenerating centroid heatmaps …")
    centroids = collect_centroids(clips)
    plot_heatmaps(centroids)

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv1 = RESULTS_DIR / "compare_mot_labels.csv"
    cols1 = ["scene_group", "method", "clips", "frames",
             "unique_ids", "total_tracks", "avg_len", "med_len", "min_len", "max_len"]
    with open(csv1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols1)
        w.writeheader()
        w.writerows(stat_rows)

    csv2 = RESULTS_DIR / "compare_mot_corrections.csv"
    cols2 = ["scene_group", "sam3_method", "delta_unique_ids", "delta_tracks",
             "delta_avg_len", "ghost_deleted", "tracks_merged",
             "id_switches_fixed", "interpolated"]
    with open(csv2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols2)
        w.writeheader()
        w.writerows(corr_rows)

    json_path = RESULTS_DIR / "compare_mot_labels.json"
    with open(json_path, "w") as f:
        json.dump({
            "generated":   datetime.now(timezone.utc).isoformat(),
            "sources":     {k: str(v) for k, v in SOURCES.items()},
            "stats":       stat_rows,
            "corrections": corr_rows,
        }, f, indent=2)

    print(f"\nCSV (stats)       → {csv1}")
    print(f"CSV (corrections) → {csv2}")
    print(f"JSON              → {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
