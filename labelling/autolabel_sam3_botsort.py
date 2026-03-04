#!/usr/bin/env python3
"""SAM3 detection + BoT-SORT tracking → MOT gt.txt for raw_frames_5fps.

Pipeline
--------
For each scene in dataset/raw_frames_5fps:
  1. Split frames into parts of ~PART_SIZE frames.
  2. Write a temporary MP4 per part (SAM3VideoSemanticPredictor requires video).
  3. Run SAM3VideoSemanticPredictor with text="person", collect raw per-frame
     bounding boxes (SAM3 internal track IDs are discarded).
  4. Run BoT-SORT across all frames of the scene (combining parts) to assign
     temporally consistent IDs with Global Motion Compensation.
  5. Apply 7-step post-processing (NMS → conf filter → short tracks → gap
     linking → short tracks → NMS → compact ID remapping).
  6. Write scene-level MOT gt.txt + seqinfo.ini.

Output layout
-------------
  dataset/MOT_labels_sam3_botsort_5fps/
    {split}/
      {scene}/
        gt/gt.txt        ← frame,id,x,y,w,h,conf,-1,-1,-1  (1-based frames)
        seqinfo.ini

Usage
-----
  # All splits / scenes
  uv run python labelling/autolabel_sam3_botsort_5fps.py

  # Custom paths
  uv run python labelling/autolabel_sam3_botsort_5fps.py \\
      --raw-frames-dir dataset/raw_frames_5fps \\
      --output-dir     dataset/MOT_labels_sam3_botsort_5fps

  # Specific split / scene
  uv run python labelling/autolabel_sam3_botsort_5fps.py \\
      --splits train \\
      --scenes acs_s1_recording_2026-02-23_17-15-46 \\
      --overwrite

  # Tune BoT-SORT
  uv run python labelling/autolabel_sam3_botsort_5fps.py \\
      --track-high-thresh 0.35 \\
      --track-low-thresh  0.05 \\
      --new-track-thresh  0.40 \\
      --track-buffer      50   \\
      --match-thresh      0.80 \\
      --gmc-method        sparseOptFlow

  # Tune post-processing
  uv run python labelling/autolabel_sam3_botsort_5fps.py \\
      --nms-iou-thresh  0.40 \\
      --min-conf        0.40 \\
      --min-track-len   5    \\
      --max-gap         10   \\
      --link-iou-thresh 0.20
"""

from __future__ import annotations

import argparse
import configparser
import fnmatch
import sys
import types
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_RATE = 5  # dataset is sampled at 5 fps


# ── BoT-SORT detection wrapper ────────────────────────────────────────────────


class _Detections:
    """Minimal stand-in for ultralytics Boxes, accepted by BOTSORT.update()."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        assert xyxy.ndim == 2 and xyxy.shape[1] == 4
        self._xyxy = xyxy.astype(np.float32)
        self.conf = torch.from_numpy(conf.astype(np.float32))
        self.cls = torch.from_numpy(cls.astype(np.float32))
        # Centre-format xywh required by BOTSORT.init_track
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


def _make_botsort(args):
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
    """Greedy NMS over one frame's boxes. Returns set of surviving track IDs."""
    if not boxes:
        return set()
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


def _build_by_frame(
    tracks: TrackData,
) -> dict[int, list[tuple[int, list[float], float]]]:
    by_frame: dict[int, list] = defaultdict(list)
    for tid, rows in tracks.items():
        for r in rows:
            by_frame[r["frame"]].append((tid, r["bbox"], r["conf"]))
    return by_frame


def step_nms(tracks: TrackData, iou_thresh: float) -> TrackData:
    by_frame = _build_by_frame(tracks)
    survivors: set[tuple[int, int]] = set()
    for fnum, boxes in by_frame.items():
        for tid in _nms_frame(boxes, iou_thresh):
            survivors.add((fnum, tid))
    out: TrackData = {}
    for tid, rows in tracks.items():
        kept = [r for r in rows if (r["frame"], tid) in survivors]
        if kept:
            out[tid] = kept
    return out


def step_conf_filter(tracks: TrackData, min_conf: float) -> TrackData:
    out: TrackData = {}
    for tid, rows in tracks.items():
        kept = [r for r in rows if r["conf"] >= min_conf]
        if kept:
            out[tid] = kept
    return out


def step_short_track_removal(tracks: TrackData, min_len: int) -> TrackData:
    return {tid: rows for tid, rows in tracks.items() if len(rows) >= min_len}


def step_gap_linking(
    tracks: TrackData,
    max_gap: int,
    link_iou_thresh: float,
) -> TrackData:
    """
    Merge tracklet pairs (A → B) where A ends before B starts,
    gap ≤ max_gap, and IoU(last_box_A, first_box_B) ≥ link_iou_thresh.
    Greedy: shortest gaps first; each track merges at most once.
    """
    summary: dict[int, dict] = {}
    for tid, rows in tracks.items():
        srows = sorted(rows, key=lambda r: r["frame"])
        summary[tid] = {
            "first": srows[0]["frame"],
            "last": srows[-1]["frame"],
            "first_bbox": srows[0]["bbox"],
            "last_bbox": srows[-1]["bbox"],
        }

    tids = sorted(summary)
    candidates: list[tuple[int, int, int]] = []  # (gap, A, B)
    for i, a in enumerate(tids):
        for b in tids[i + 1 :]:
            sa, sb = summary[a], summary[b]
            if sa["last"] >= sb["first"]:
                continue
            gap = sb["first"] - sa["last"]
            if gap > max_gap:
                continue
            iou = _iou_xywh(sa["last_bbox"], sb["first_bbox"])
            if iou >= link_iou_thresh:
                candidates.append((gap, a, b))

    candidates.sort()

    merged_into: dict[int, int] = {}

    def _root(tid: int) -> int:
        while tid in merged_into:
            tid = merged_into[tid]
        return tid

    for gap, a, b in candidates:
        ra, rb = _root(a), _root(b)
        if ra == rb:
            continue
        merged_into[rb] = ra

    out: TrackData = defaultdict(list)
    for tid, rows in tracks.items():
        target = _root(tid)
        out[target].extend(rows)

    return {tid: sorted(rows, key=lambda r: r["frame"]) for tid, rows in out.items()}


def step_remap_ids(tracks: TrackData) -> TrackData:
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
    """Full 7-step post-processing pipeline. Returns (tracks, stats)."""

    def _counts(t):
        return {"tracks": len(t), "boxes": sum(len(r) for r in t.values())}

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
        "raw": s0,
        "after_nms1": s1,
        "after_conf": s2,
        "after_short1": s3,
        "after_link": s4,
        "after_short2": s5,
        "after_nms2": s6,
        "final": _counts(tracks),
    }
    return tracks, stats


# ── Frame / video helpers ─────────────────────────────────────────────────────


def collect_frames(scene_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in scene_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def frame_number(frame_path: Path) -> int:
    digits = "".join(c for c in frame_path.stem if c.isdigit())
    return int(digits) if digits else 0


def write_video(frames: list[Path], out_path: Path, fps: int) -> tuple[int, int]:
    """Write frame images to an MP4 file. Returns (width, height)."""
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in frames:
        img = cv2.imread(str(f))
        if img is not None:
            writer.write(img)
    writer.release()
    return w, h


# ── SAM3 detection per part ───────────────────────────────────────────────────


def detect_part(
    part_frames: list[Path],
    predictor,
    text_prompt: str,
    tmp_dir: Path,
) -> list[tuple[Path, list[tuple[np.ndarray, float]]]]:
    """
    Run SAM3VideoSemanticPredictor on one part of frames.

    Returns [(frame_path, [(xyxy, conf), ...]), ...] in frame order.
    SAM3 internal track IDs are discarded; only bboxes + scores are kept.
    """
    tmp_video = tmp_dir / "_part_tmp.mp4"
    write_video(part_frames, tmp_video, fps=FRAME_RATE)

    # Reset tracker state between parts to prevent ID leakage
    predictor.inference_state = {}

    results_gen = predictor(source=str(tmp_video), text=[text_prompt], stream=True)

    frame_dets: list[tuple[Path, list[tuple[np.ndarray, float]]]] = []
    for frame_path, result in zip(part_frames, results_gen):
        dets: list[tuple[np.ndarray, float]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for box, score in zip(boxes_xyxy, scores):
                dets.append((box.astype(np.float32), float(score)))
        frame_dets.append((frame_path, dets))

    tmp_video.unlink(missing_ok=True)
    return frame_dets


# ── Scene-level processing ────────────────────────────────────────────────────


def run_scene(
    scene_dir: Path,
    output_dir: Path,
    predictor,
    args,
) -> dict:
    frames = collect_frames(scene_dir)
    if not frames:
        return {"scene": scene_dir.name, "frames": 0, "tracks": 0, "rows": 0}

    # Determine image dimensions from first frame
    first_img = cv2.imread(str(frames[0]))
    img_h, img_w = first_img.shape[:2]

    # Write seqinfo.ini
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_cfg = configparser.ConfigParser()
    seq_cfg["Sequence"] = {
        "name": scene_dir.name,
        "imDir": str(scene_dir.resolve()),
        "frameRate": str(FRAME_RATE),
        "seqLength": str(len(frames)),
        "imWidth": str(img_w),
        "imHeight": str(img_h),
        "imExt": frames[0].suffix,
    }
    with open(output_dir / "seqinfo.ini", "w") as f:
        seq_cfg.write(f)

    # --- Step 1: Run SAM3 part-by-part to collect raw per-frame detections ---
    parts = [
        frames[i : i + args.part_size] for i in range(0, len(frames), args.part_size)
    ]

    # Map frame_path → list of (xyxy, conf) detections
    all_frame_dets: dict[Path, list[tuple[np.ndarray, float]]] = {}
    for part_idx, part_frames in enumerate(parts, start=1):
        tqdm.write(
            f"    SAM3 part {part_idx}/{len(parts)}"
            f"  [{part_frames[0].name}…{part_frames[-1].name}]"
            f"  ({len(part_frames)} frames)"
        )
        part_dets = detect_part(part_frames, predictor, args.text_prompt, output_dir)
        for fpath, dets in part_dets:
            all_frame_dets[fpath] = dets

    # --- Step 2: Run BoT-SORT across all frames in scene order ---------------
    tracker = _make_botsort(args)
    raw_tracks: TrackData = defaultdict(list)

    for fnum, frame_path in enumerate(frames, start=1):
        dets_list = all_frame_dets.get(frame_path, [])

        if dets_list:
            xyxy = np.stack([d[0] for d in dets_list])
            conf = np.array([d[1] for d in dets_list], dtype=np.float32)
            cls = np.zeros(len(dets_list), dtype=np.float32)
            dets = _Detections(xyxy, conf, cls)
        else:
            dets = _EMPTY

        # Load grayscale frame for GMC (sparseOptFlow)
        if frame_path.exists():
            gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        else:
            gray = np.zeros((img_h, img_w), dtype=np.uint8)

        result = tracker.update(dets, img=gray)  # (N, 8): x1,y1,x2,y2,tid,conf,cls,idx

        for row in result:
            x1, y1, x2, y2, tid, conf_val = (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                int(row[4]),
                float(row[5]),
            )
            raw_tracks[tid].append(
                {
                    "frame": fnum,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "conf": conf_val,
                }
            )

    # --- Step 3: Post-processing ---------------------------------------------
    tracks, pp_stats = postprocess(
        dict(raw_tracks),
        nms_iou_thresh=args.nms_iou_thresh,
        min_conf=args.min_conf,
        min_track_len=args.min_track_len,
        max_gap=args.max_gap,
        link_iou_thresh=args.link_iou_thresh,
    )

    # --- Step 4: Write gt.txt ------------------------------------------------
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
        "scene": scene_dir.name,
        "frames": len(frames),
        "tracks": len(tracks),
        "rows": len(gt_rows),
        "pp": pp_stats,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAM3 detection + BoT-SORT tracking → MOT gt.txt (5 fps)"
    )

    # Paths
    parser.add_argument(
        "--raw-frames-dir", type=Path, default=Path("dataset/main/raw_frames_5fps")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/main/MOT_labels_sam3_botsort_5fps"),
    )

    # SAM3 parameters
    parser.add_argument("--model", type=str, default="sam3.pt")
    parser.add_argument("--text-prompt", type=str, default="person")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="SAM3 detection confidence threshold."
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--part-size",
        type=int,
        default=55,
        help="Frames per SAM3 video part (50-60 recommended).",
    )

    # BoT-SORT parameters
    parser.add_argument(
        "--track-high-thresh",
        type=float,
        default=0.35,
        help="High-conf detections → stage-1 association.",
    )
    parser.add_argument(
        "--track-low-thresh",
        type=float,
        default=0.05,
        help="Low-conf detections → stage-2 association.",
    )
    parser.add_argument(
        "--new-track-thresh",
        type=float,
        default=0.40,
        help="Minimum score to start a new track.",
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=50,
        help="Frames a lost track is kept alive (higher for 5fps).",
    )
    parser.add_argument(
        "--match-thresh",
        type=float,
        default=0.80,
        help="IoU threshold for Hungarian assignment.",
    )
    parser.add_argument(
        "--proximity-thresh",
        type=float,
        default=0.50,
        help="Proximity threshold for appearance gating.",
    )
    parser.add_argument(
        "--gmc-method",
        type=str,
        default="sparseOptFlow",
        choices=["sparseOptFlow", "orb", "sift", "ecc", "off"],
        help="Global Motion Compensation method.",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=FRAME_RATE,
        help="Frame rate passed to BoT-SORT Kalman filter.",
    )

    # Post-processing parameters
    parser.add_argument("--nms-iou-thresh", type=float, default=0.40)
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.40,
        help="Drop boxes below this confidence after tracking.",
    )
    parser.add_argument(
        "--min-track-len",
        type=int,
        default=5,
        help="Minimum frames a track must span to survive.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=10,
        help="Maximum frame gap for IoU-based tracklet linking.",
    )
    parser.add_argument(
        "--link-iou-thresh",
        type=float,
        default=0.20,
        help="Minimum IoU to link two tracklets across a gap.",
    )
    parser.add_argument(
        "--verbose-pp",
        action="store_true",
        help="Print per-scene post-processing breakdown.",
    )

    # Scene selection
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--exclude", nargs="+", default=None, metavar="PATTERN")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if not args.raw_frames_dir.exists():
        print(
            f"ERROR: raw frames dir not found: {args.raw_frames_dir}", file=sys.stderr
        )
        return 1

    # Load SAM3 model once
    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=args.model,
        imgsz=args.imgsz,
        half=True,
        verbose=False,
    )
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)

    print(f"Model       : {args.model}  (text='{args.text_prompt}'  conf={args.conf})")
    print(f"Part size   : {args.part_size} frames")
    print(
        f"BoT-SORT    : high={args.track_high_thresh} low={args.track_low_thresh} "
        f"new={args.new_track_thresh} buffer={args.track_buffer} "
        f"match={args.match_thresh} gmc={args.gmc_method}"
    )
    print(
        f"Post-proc   : nms_iou={args.nms_iou_thresh} min_conf={args.min_conf} "
        f"min_len={args.min_track_len} max_gap={args.max_gap} "
        f"link_iou={args.link_iou_thresh}"
    )
    print(f"Input       : {args.raw_frames_dir}")
    print(f"Output      : {args.output_dir}")

    # Discover splits / scenes
    split_dirs = sorted(d for d in args.raw_frames_dir.iterdir() if d.is_dir())
    if args.splits:
        split_dirs = [s for s in split_dirs if s.name in set(args.splits)]
    if not split_dirs:
        print("No splits found.", file=sys.stderr)
        return 1

    total_scenes = total_tracks = total_rows = 0

    for split_dir in split_dirs:
        scenes = sorted(d for d in split_dir.iterdir() if d.is_dir())
        if args.scenes:
            scenes = [s for s in scenes if s.name in set(args.scenes)]
        if args.exclude:
            scenes = [
                s
                for s in scenes
                if not any(fnmatch.fnmatch(s.name, pat) for pat in args.exclude)
            ]

        print(f"\n── {split_dir.name} ({len(scenes)} scenes) ──")

        for scene_dir in tqdm(scenes, desc=split_dir.name, unit="scene"):
            out_scene = args.output_dir / split_dir.name / scene_dir.name
            gt_file = out_scene / "gt" / "gt.txt"

            if gt_file.exists() and not args.overwrite:
                tqdm.write(f"  skip (exists): {scene_dir.name}")
                continue

            tqdm.write(f"  {scene_dir.name}")

            stats = run_scene(
                scene_dir=scene_dir,
                output_dir=out_scene,
                predictor=predictor,
                args=args,
            )

            pp = stats["pp"]
            tqdm.write(
                f"  → {stats['frames']} frames | "
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
    print(
        f"Done.  Scenes: {total_scenes}  |  Tracks: {total_tracks}  |  Annotations: {total_rows}"
    )
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
