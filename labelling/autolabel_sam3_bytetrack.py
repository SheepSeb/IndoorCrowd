#!/usr/bin/env python3
"""Auto-label pipeline: SAM3 detections → ByteTrack → MOT gt.txt.

Reads pre-computed SAM3 COCO JSON annotations
(dataset/labels_3fps/{split}/{scene}.json) and runs ByteTrack on the
per-frame bounding boxes to assign consistent track IDs across frames.

This completely decouples detection (SAM3) from tracking (ByteTrack):
no GPU or model inference is needed at tracking time.

Output layout (MOT Challenge format)
--------------------------------------
  dataset/MOT_labels_sam3_bytetrack/
    {split}/
      {scene}/
        gt/
          gt.txt      ← frame,id,x,y,w,h,conf,-1,-1,-1
        seqinfo.ini   ← sequence metadata

Usage
-----
  # Default paths
  uv run python labelling/autolabel_sam3_bytetrack.py

  # Custom paths
  uv run python labelling/autolabel_sam3_bytetrack.py \\
      --sam3-labels-dir dataset/labels_3fps \\
      --raw-frames-dir  dataset/raw_frames_3fps \\
      --output-dir      dataset/MOT_labels_sam3_bytetrack

  # Tune ByteTrack thresholds
  uv run python labelling/autolabel_sam3_bytetrack.py \\
      --track-activation-thresh 0.35 \\
      --track-buffer            30   \\
      --match-thresh            0.80

  # Only re-process specific scenes
  uv run python labelling/autolabel_sam3_bytetrack.py \\
      --scenes acs_s1_recording_2026-02-23_17-15-46 --overwrite
"""

from __future__ import annotations

import argparse
import configparser
import fnmatch
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import supervision as sv
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def frame_number_from_name(file_name: str) -> int:
    """Extract the numeric frame index from a filename like 'frame_00042.jpg'."""
    stem = Path(file_name).stem
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else 0


def run_scene(
    coco_path: Path,
    scene_name: str,
    raw_frames_dir: Path,
    output_dir: Path,
    track_activation_thresh: float,
    track_buffer: int,
    match_thresh: float,
    frame_rate: int,
    min_consecutive_frames: int,
) -> dict:
    """Run ByteTrack on one scene's SAM3 detections and write MOT gt.txt."""
    with open(coco_path) as f:
        coco = json.load(f)

    images = coco.get("images", [])
    if not images:
        return {"scene": scene_name, "frames": 0, "tracks": 0, "rows": 0}

    # Group annotations by image_id
    anns_by_image: dict[int, list[tuple[list[float], float]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        x, y, w, h = ann["bbox"]
        xyxy = [x, y, x + w, y + h]
        anns_by_image[ann["image_id"]].append((xyxy, float(ann.get("score", 1.0))))

    # Sort frames by filename for correct temporal order
    images_sorted = sorted(images, key=lambda img: img["file_name"])
    img_w = images_sorted[0].get("width", 0)
    img_h = images_sorted[0].get("height", 0)

    # Write seqinfo.ini
    scene_frames_dir = raw_frames_dir / images_sorted[0]["file_name"].split("/")[0] / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_cfg = configparser.ConfigParser()
    seq_cfg["Sequence"] = {
        "name": scene_name,
        "imDir": str(scene_frames_dir.resolve()),
        "frameRate": str(frame_rate),
        "seqLength": str(len(images_sorted)),
        "imWidth": str(img_w),
        "imHeight": str(img_h),
        "imExt": ".jpg",
    }
    with open(output_dir / "seqinfo.ini", "w") as f:
        seq_cfg.write(f)

    # Initialise ByteTrack
    tracker = sv.ByteTrack(
        track_activation_threshold=track_activation_thresh,
        lost_track_buffer=track_buffer,
        minimum_matching_threshold=match_thresh,
        frame_rate=frame_rate,
        minimum_consecutive_frames=min_consecutive_frames,
    )
    tracker.reset()

    gt_rows: list[str] = []
    all_track_ids: set[int] = set()

    for img_info in images_sorted:
        img_id = img_info["id"]
        fnum = frame_number_from_name(img_info["file_name"])

        dets = anns_by_image.get(img_id, [])
        if dets:
            xyxy = np.array([d[0] for d in dets], dtype=np.float32)
            scores = np.array([d[1] for d in dets], dtype=np.float32)
            class_ids = np.zeros(len(dets), dtype=int)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)
            class_ids = np.empty(0, dtype=int)

        detections = sv.Detections(xyxy=xyxy, confidence=scores, class_id=class_ids)
        tracked = tracker.update_with_detections(detections)

        if tracked.tracker_id is None or len(tracked) == 0:
            continue

        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            tid = int(tracked.tracker_id[i])
            score = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            gt_rows.append(
                f"{fnum},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1"
            )
            all_track_ids.add(tid)

    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "gt.txt").write_text("\n".join(gt_rows) + ("\n" if gt_rows else ""))

    return {
        "scene": scene_name,
        "frames": len(images_sorted),
        "tracks": len(all_track_ids),
        "rows": len(gt_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAM3 detections + ByteTrack → MOT gt.txt"
    )
    parser.add_argument(
        "--sam3-labels-dir",
        type=Path,
        default=Path("dataset/labels_3fps"),
        help="Directory with SAM3 COCO JSONs: {split}/{scene}.json",
    )
    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        default=Path("dataset/raw_frames_3fps"),
        help="Root directory with raw frames (used for seqinfo.ini only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/MOT_labels_sam3_bytetrack"),
        help="Output root for MOT-format labels.",
    )
    # ByteTrack parameters
    parser.add_argument(
        "--track-activation-thresh",
        type=float,
        default=0.35,
        help="Confidence threshold to activate / initialise a new track.",
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=30,
        help="Frames a lost track is kept alive by the Kalman filter before removal.",
    )
    parser.add_argument(
        "--match-thresh",
        type=float,
        default=0.80,
        help="Minimum IoU for the Hungarian assignment in ByteTrack.",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=3,
        help="Frame rate of the sequences (affects Kalman filter velocity model).",
    )
    parser.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=1,
        help="Minimum consecutive frames before a track is confirmed and returned.",
    )
    # Scene selection
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to process (default: all found in sam3-labels-dir).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Restrict to specific scene names (without .json extension).",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        metavar="PATTERN",
        help="Glob patterns for scene names to skip.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process already labelled scenes.",
    )
    args = parser.parse_args()

    if not args.sam3_labels_dir.exists():
        print(
            f"ERROR: SAM3 labels directory not found: {args.sam3_labels_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"SAM3 labels:  {args.sam3_labels_dir}")
    print(f"Output:       {args.output_dir}")
    print(
        f"ByteTrack:    activation={args.track_activation_thresh}, "
        f"buffer={args.track_buffer}, match={args.match_thresh}, "
        f"fps={args.frame_rate}, min_frames={args.min_consecutive_frames}"
    )

    # Discover splits
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
                track_activation_thresh=args.track_activation_thresh,
                track_buffer=args.track_buffer,
                match_thresh=args.match_thresh,
                frame_rate=args.frame_rate,
                min_consecutive_frames=args.min_consecutive_frames,
            )
            tqdm.write(
                f"  {stats['scene']}: "
                f"{stats['frames']} frames | "
                f"{stats['tracks']} tracks | "
                f"{stats['rows']} annotations"
            )
            total_scenes += 1
            total_tracks += stats["tracks"]
            total_rows += stats["rows"]

    print(f"\n{'─' * 60}")
    print(
        f"Done.  Scenes: {total_scenes}  |  "
        f"Unique tracks: {total_tracks}  |  "
        f"Annotations: {total_rows}"
    )
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
