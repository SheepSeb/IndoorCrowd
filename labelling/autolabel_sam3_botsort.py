#!/usr/bin/env python3
"""Auto-label pipeline: SAM3 detections → BoT-SORT → post-processing → MOT gt.txt.

Replaces ByteTrack with BoT-SORT (Aharon et al., 2022), which adds:
  - Global Motion Compensation (GMC / sparseOptFlow) for camera-motion robustness
  - Kalman-filter state prediction (same as ByteTrack)
  - Two-stage high/low confidence association

Post-processing pipeline (applied after tracking):
  1. Per-frame NMS          — aggressive IoU suppression to kill duplicate boxes
  2. Confidence filter      — drop boxes below min_conf
  3. Short-track removal    — drop tracks shorter than min_track_len frames
  4. IoU gap-linking        — merge split tracklets across short temporal gaps
  5. Short-track removal    — second pass after merging
  6. Final per-frame NMS    — catch any remaining duplicates introduced by merging
  7. Compact ID remapping   — renumber surviving tracks 1, 2, 3, …

Output layout (MOT Challenge format)
--------------------------------------
  dataset/MOT_labels_sam3_botsort/
    {split}/
      {scene}/
        gt/gt.txt       ← frame,id,x,y,w,h,conf,-1,-1,-1
        seqinfo.ini

Usage
-----
  uv run python labelling/autolabel_sam3_botsort.py

  uv run python labelling/autolabel_sam3_botsort.py \\
      --sam3-labels-dir dataset/labels_3fps \\
      --raw-frames-dir  dataset/raw_frames_3fps \\
      --output-dir      dataset/MOT_labels_sam3_botsort

  # Tune post-processing
  uv run python labelling/autolabel_sam3_botsort.py \\
      --nms-iou-thresh  0.40 \\
      --min-conf        0.40 \\
      --min-track-len   3    \\
      --max-gap         6    \\
      --link-iou-thresh 0.20

  # Tune BoT-SORT
  uv run python labelling/autolabel_sam3_botsort.py \\
      --track-high-thresh 0.35 \\
      --track-low-thresh  0.05 \\
      --new-track-thresh  0.40 \\
      --track-buffer      30   \\
      --match-thresh      0.80 \\
      --gmc-method        sparseOptFlow
"""

from __future__ import annotations

import argparse
import configparser
import fnmatch
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── BoT-SORT detection wrapper ────────────────────────────────────────────────


class _Detections:
    """Minimal stand-in for ultralytics Boxes, accepted by BOTSORT.update()."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        assert xyxy.ndim == 2 and xyxy.shape[1] == 4
        self._xyxy = xyxy.astype(np.float32)
        self.conf = torch.from_numpy(conf.astype(np.float32))
        self.cls = torch.from_numpy(cls.astype(np.float32))
        # centre-format xywh required by init_track
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx):
        idx_np = idx.numpy() if isinstance(idx, torch.Tensor) else idx
        return _Detections(
            self._xyxy[idx_np],
            self.conf.numpy()[idx_np],
            self.cls.numpy()[idx_np],
        )


_EMPTY = _Detections(
    np.empty((0, 4), np.float32),
    np.empty(0, np.float32),
    np.empty(0, np.float32),
)


def _make_botsort(args) -> "BOTSORT":
    from ultralytics.trackers.bot_sort import BOTSORT

    ns = types.SimpleNamespace(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        proximity_thresh=args.proximity_thresh,
        appearance_thresh=0.25,
        with_reid=False,
        gmc_method=args.gmc_method,
        model="auto",
        fuse_score=True,
    )
    return BOTSORT(ns, frame_rate=args.frame_rate)


# ── IoU utilities ─────────────────────────────────────────────────────────────


def _iou_xywh(a: list[float], b: list[float]) -> float:
    """IoU of two [x, y, w, h] boxes (top-left corner format)."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 1e-6 else 0.0


def _nms_frame(
    boxes: list[tuple[int, list[float], float]],
    iou_thresh: float,
) -> set[int]:
    """
    Greedy NMS over one frame's boxes.

    Args:
        boxes: [(track_id, [x,y,w,h], conf), ...]
        iou_thresh: suppress if IoU > this

    Returns:
        Set of track_ids whose box in this frame should be *kept*.
    """
    if not boxes:
        return set()
    # Sort by confidence descending
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][2], reverse=True)
    suppressed = [False] * len(boxes)
    kept: set[int] = set()
    for i in order:
        if suppressed[i]:
            continue
        kept.add(boxes[i][0])
        for j in order:
            if i == j or suppressed[j]:
                continue
            if _iou_xywh(boxes[i][1], boxes[j][1]) > iou_thresh:
                suppressed[j] = True
    return kept


# ── Post-processing steps ─────────────────────────────────────────────────────


TrackData = dict[int, list[dict]]  # {track_id: [{frame, bbox, conf}, ...]}


def _build_by_frame(tracks: TrackData) -> dict[int, list[tuple[int, list[float], float]]]:
    """Invert tracks → {frame: [(tid, bbox, conf), ...]}."""
    by_frame: dict[int, list] = defaultdict(list)
    for tid, rows in tracks.items():
        for r in rows:
            by_frame[r["frame"]].append((tid, r["bbox"], r["conf"]))
    return by_frame


def step_nms(tracks: TrackData, iou_thresh: float) -> TrackData:
    """Per-frame greedy NMS: remove a track's box for frames where it is suppressed."""
    by_frame = _build_by_frame(tracks)
    # Which (frame, tid) entries survive?
    survivors: set[tuple[int, int]] = set()
    for fnum, boxes in by_frame.items():
        for tid in _nms_frame(boxes, iou_thresh):
            survivors.add((fnum, tid))
    # Rebuild tracks keeping only surviving boxes
    out: TrackData = {}
    for tid, rows in tracks.items():
        kept = [r for r in rows if (r["frame"], tid) in survivors]
        if kept:
            out[tid] = kept
    return out


def step_conf_filter(tracks: TrackData, min_conf: float) -> TrackData:
    """Drop individual boxes below min_conf; remove tracks that become empty."""
    out: TrackData = {}
    for tid, rows in tracks.items():
        kept = [r for r in rows if r["conf"] >= min_conf]
        if kept:
            out[tid] = kept
    return out


def step_short_track_removal(tracks: TrackData, min_len: int) -> TrackData:
    """Remove tracks that appear in fewer than min_len frames."""
    return {tid: rows for tid, rows in tracks.items() if len(rows) >= min_len}


def step_gap_linking(
    tracks: TrackData,
    max_gap: int,
    link_iou_thresh: float,
) -> TrackData:
    """
    Merge pairs of tracklets (A → B) where:
      - A ends before B starts
      - gap between them ≤ max_gap frames
      - IoU(last box of A, first box of B) ≥ link_iou_thresh

    Linking is greedy: processes pairs ordered by gap size (shortest first),
    and a track can only be merged once as a recipient.
    """
    # Summarise each track: first frame, last frame, first/last bbox
    summary: dict[int, dict] = {}
    for tid, rows in tracks.items():
        srows = sorted(rows, key=lambda r: r["frame"])
        summary[tid] = {
            "first": srows[0]["frame"],
            "last": srows[-1]["frame"],
            "first_bbox": srows[0]["bbox"],
            "last_bbox": srows[-1]["bbox"],
        }

    # Build candidate merge pairs (A absorbs B)
    tids = sorted(summary)
    candidates: list[tuple[int, int, int]] = []  # (gap, A, B)
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            sa, sb = summary[a], summary[b]
            # A must end before B starts
            if sa["last"] >= sb["first"]:
                continue
            gap = sb["first"] - sa["last"]
            if gap > max_gap:
                continue
            iou = _iou_xywh(sa["last_bbox"], sb["first_bbox"])
            if iou >= link_iou_thresh:
                candidates.append((gap, a, b))

    # Sort by gap (prefer small gaps)
    candidates.sort()

    # Greedy merge: each track can be merged into at most one other
    merged_into: dict[int, int] = {}  # source → target

    def _root(tid: int) -> int:
        while tid in merged_into:
            tid = merged_into[tid]
        return tid

    for gap, a, b in candidates:
        ra, rb = _root(a), _root(b)
        if ra == rb:
            continue  # already same track
        # Absorb rb into ra
        merged_into[rb] = ra

    # Apply merges
    out: TrackData = defaultdict(list)
    for tid, rows in tracks.items():
        target = _root(tid)
        out[target].extend(rows)

    # Re-sort rows by frame within each merged track
    return {tid: sorted(rows, key=lambda r: r["frame"]) for tid, rows in out.items()}


def step_remap_ids(tracks: TrackData) -> TrackData:
    """Renumber track IDs compactly starting from 1."""
    new_id = {old: new for new, old in enumerate(sorted(tracks), start=1)}
    return {new_id[tid]: rows for tid, rows in tracks.items()}


def postprocess(
    tracks: TrackData,
    nms_iou_thresh: float,
    min_conf: float,
    min_track_len: int,
    max_gap: int,
    link_iou_thresh: float,
) -> tuple[TrackData, dict]:
    """Full post-processing pipeline. Returns (tracks, stats)."""
    def _counts(t): return {"tracks": len(t), "boxes": sum(len(r) for r in t.values())}

    s0 = _counts(tracks)

    tracks = step_nms(tracks, nms_iou_thresh)
    s1 = _counts(tracks)

    tracks = step_conf_filter(tracks, min_conf)
    s2 = _counts(tracks)

    tracks = step_short_track_removal(tracks, min_track_len)
    s3 = _counts(tracks)

    tracks = step_gap_linking(tracks, max_gap, link_iou_thresh)
    s4 = _counts(tracks)

    tracks = step_short_track_removal(tracks, min_track_len)
    s5 = _counts(tracks)

    tracks = step_nms(tracks, nms_iou_thresh)
    s6 = _counts(tracks)

    tracks = step_remap_ids(tracks)

    stats = {
        "raw":          s0,
        "after_nms1":   s1,
        "after_conf":   s2,
        "after_short1": s3,
        "after_link":   s4,
        "after_short2": s5,
        "after_nms2":   s6,
        "final":        _counts(tracks),
    }
    return tracks, stats


# ── Scene processing ──────────────────────────────────────────────────────────


def frame_number_from_name(file_name: str) -> int:
    stem = Path(file_name).stem
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else 0


def run_scene(
    coco_path: Path,
    scene_name: str,
    raw_frames_dir: Path,
    output_dir: Path,
    args,
) -> dict:
    with open(coco_path) as f:
        coco = json.load(f)

    images = coco.get("images", [])
    if not images:
        return {"scene": scene_name, "frames": 0, "tracks": 0, "rows": 0}

    # Group SAM3 detections by image_id
    anns_by_img: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        x, y, w, h = ann["bbox"]
        xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
        anns_by_img[ann["image_id"]].append((xyxy, float(ann.get("score", 1.0))))

    images_sorted = sorted(images, key=lambda img: img["file_name"])
    img_w = images_sorted[0].get("width", 0)
    img_h = images_sorted[0].get("height", 0)

    # Write seqinfo.ini
    split_part = images_sorted[0]["file_name"].split("/")[0]
    scene_frames_dir = raw_frames_dir / split_part / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_cfg = configparser.ConfigParser()
    seq_cfg["Sequence"] = {
        "name": scene_name,
        "imDir": str(scene_frames_dir.resolve()),
        "frameRate": str(args.frame_rate),
        "seqLength": str(len(images_sorted)),
        "imWidth": str(img_w),
        "imHeight": str(img_h),
        "imExt": ".jpg",
    }
    with open(output_dir / "seqinfo.ini", "w") as f:
        seq_cfg.write(f)

    # Run BoT-SORT
    tracker = _make_botsort(args)

    raw_tracks: TrackData = defaultdict(list)

    for img_info in images_sorted:
        img_id = img_info["id"]
        fnum = frame_number_from_name(img_info["file_name"])

        dets_list = anns_by_img.get(img_id, [])
        if dets_list:
            xyxy = np.stack([d[0] for d in dets_list])
            conf = np.array([d[1] for d in dets_list], dtype=np.float32)
            cls = np.zeros(len(dets_list), dtype=np.float32)
            dets = _Detections(xyxy, conf, cls)
        else:
            dets = _EMPTY

        # Load grayscale frame for GMC (sparseOptFlow)
        frame_path = raw_frames_dir / img_info["file_name"]
        if frame_path.exists():
            gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        else:
            gray = np.zeros((img_h, img_w), dtype=np.uint8)

        result = tracker.update(dets, img=gray)  # (N, 8): x1,y1,x2,y2,tid,conf,cls,idx

        for row in result:
            x1, y1, x2, y2, tid, conf_val = (
                float(row[0]), float(row[1]), float(row[2]), float(row[3]),
                int(row[4]), float(row[5]),
            )
            raw_tracks[tid].append({
                "frame": fnum,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "conf": conf_val,
            })

    # Post-processing
    tracks, pp_stats = postprocess(
        dict(raw_tracks),
        nms_iou_thresh=args.nms_iou_thresh,
        min_conf=args.min_conf,
        min_track_len=args.min_track_len,
        max_gap=args.max_gap,
        link_iou_thresh=args.link_iou_thresh,
    )

    # Write gt.txt
    gt_rows: list[str] = []
    for tid, rows in sorted(tracks.items()):
        for r in sorted(rows, key=lambda x: x["frame"]):
            x, y, w, h = r["bbox"]
            gt_rows.append(
                f"{r['frame']},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f}"
                f",{r['conf']:.4f},-1,-1,-1"
            )
    gt_rows.sort(key=lambda s: (int(s.split(",")[0]), int(s.split(",")[1])))

    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "gt.txt").write_text("\n".join(gt_rows) + ("\n" if gt_rows else ""))

    return {
        "scene": scene_name,
        "frames": len(images_sorted),
        "tracks": len(tracks),
        "rows": len(gt_rows),
        "pp": pp_stats,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAM3 detections + BoT-SORT + post-processing → MOT gt.txt"
    )

    # Paths
    parser.add_argument("--sam3-labels-dir", type=Path, default=Path("dataset/labels_3fps"))
    parser.add_argument("--raw-frames-dir",  type=Path, default=Path("dataset/raw_frames_3fps"))
    parser.add_argument("--output-dir",      type=Path, default=Path("dataset/MOT_labels_sam3_botsort"))

    # BoT-SORT parameters
    parser.add_argument("--track-high-thresh", type=float, default=0.35,
                        help="High-conf detections → stage-1 association.")
    parser.add_argument("--track-low-thresh",  type=float, default=0.05,
                        help="Low-conf detections → stage-2 association.")
    parser.add_argument("--new-track-thresh",  type=float, default=0.40,
                        help="Minimum score to start a new track.")
    parser.add_argument("--track-buffer",      type=int,   default=30,
                        help="Frames a lost track is kept alive.")
    parser.add_argument("--match-thresh",      type=float, default=0.80,
                        help="IoU threshold for Hungarian assignment.")
    parser.add_argument("--proximity-thresh",  type=float, default=0.50,
                        help="Proximity threshold for appearance gating.")
    parser.add_argument("--gmc-method", type=str, default="sparseOptFlow",
                        choices=["sparseOptFlow", "orb", "sift", "ecc", "off"],
                        help="Global Motion Compensation method.")
    parser.add_argument("--frame-rate", type=int, default=3)

    # Post-processing parameters
    parser.add_argument("--nms-iou-thresh",  type=float, default=0.40,
                        help="IoU threshold for per-frame NMS (aggressive: lower = stricter).")
    parser.add_argument("--min-conf",        type=float, default=0.40,
                        help="Drop boxes below this confidence after tracking.")
    parser.add_argument("--min-track-len",   type=int,   default=3,
                        help="Minimum frames a track must span to survive.")
    parser.add_argument("--max-gap",         type=int,   default=6,
                        help="Maximum frame gap to attempt IoU-based tracklet linking.")
    parser.add_argument("--link-iou-thresh", type=float, default=0.20,
                        help="Minimum IoU between last/first box to link two tracklets.")
    parser.add_argument("--verbose-pp", action="store_true",
                        help="Print per-scene post-processing breakdown.")

    # Scene selection
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--exclude", nargs="+", default=None, metavar="PATTERN")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.sam3_labels_dir.exists():
        print(f"ERROR: SAM3 labels dir not found: {args.sam3_labels_dir}", file=sys.stderr)
        return 1

    print(f"SAM3 labels : {args.sam3_labels_dir}")
    print(f"Output      : {args.output_dir}")
    print(f"BoT-SORT    : high={args.track_high_thresh} low={args.track_low_thresh} "
          f"new={args.new_track_thresh} buffer={args.track_buffer} "
          f"match={args.match_thresh} gmc={args.gmc_method}")
    print(f"Post-proc   : nms_iou={args.nms_iou_thresh} min_conf={args.min_conf} "
          f"min_len={args.min_track_len} max_gap={args.max_gap} "
          f"link_iou={args.link_iou_thresh}")

    split_dirs = sorted(
        d for d in args.sam3_labels_dir.iterdir()
        if d.is_dir() and d.name != "session_logs"
    )
    if args.splits:
        split_dirs = [s for s in split_dirs if s.name in set(args.splits)]
    if not split_dirs:
        print("No splits found.", file=sys.stderr)
        return 1

    total_scenes = total_tracks = total_rows = 0

    for split_dir in split_dirs:
        json_files = sorted(split_dir.glob("*.json"))
        if args.scenes:
            json_files = [j for j in json_files if j.stem in set(args.scenes)]
        if args.exclude:
            json_files = [
                j for j in json_files
                if not any(fnmatch.fnmatch(j.stem, pat) for pat in args.exclude)
            ]

        print(f"\n── {split_dir.name} ({len(json_files)} scenes) ──")

        for coco_path in tqdm(json_files, desc=split_dir.name, unit="scene"):
            scene_name = coco_path.stem
            out_scene = args.output_dir / split_dir.name / scene_name
            gt_file = out_scene / "gt" / "gt.txt"

            if gt_file.exists() and not args.overwrite:
                tqdm.write(f"  skip (exists): {scene_name}")
                continue

            stats = run_scene(
                coco_path=coco_path,
                scene_name=scene_name,
                raw_frames_dir=args.raw_frames_dir,
                output_dir=out_scene,
                args=args,
            )

            pp = stats["pp"]
            tqdm.write(
                f"  {stats['scene']}: "
                f"{stats['frames']} frames | "
                f"{stats['tracks']} tracks | "
                f"{stats['rows']} annotations"
            )
            if args.verbose_pp:
                tqdm.write(
                    f"    raw {pp['raw']['tracks']}t/{pp['raw']['boxes']}b → "
                    f"nms1 {pp['after_nms1']['boxes']}b → "
                    f"conf {pp['after_conf']['boxes']}b → "
                    f"short1 {pp['after_short1']['tracks']}t → "
                    f"link {pp['after_link']['tracks']}t → "
                    f"short2 {pp['after_short2']['tracks']}t → "
                    f"nms2 {pp['after_nms2']['boxes']}b → "
                    f"final {pp['final']['tracks']}t/{pp['final']['boxes']}b"
                )

            total_scenes += 1
            total_tracks += stats["tracks"]
            total_rows += stats["rows"]

    print(f"\n{'─' * 60}")
    print(f"Done.  Scenes: {total_scenes}  |  Tracks: {total_tracks}  |  Annotations: {total_rows}")
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
