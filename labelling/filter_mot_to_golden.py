#!/usr/bin/env python3
"""Filter full-sequence MOT gt.txt files down to golden-frame rows.

Golden frames are a non-consecutive subset of the full raw video sequences.
Video trackers (ByteTrack, Tracktor++) must run on the full sequences to
produce coherent track IDs, because temporal continuity is required for
IoU matching and Kalman filtering.  This script is the second step:

  Step 1 (run separately):
    uv run python labelling/autolabel_bytetrack.py \\
        --detector golden \\
        --raw-frames-dir dataset/raw_frames_3fps \\
        --output-dir     dataset/MOT_labels_bytetrack

  Step 2 (this script):
    uv run python labelling/filter_mot_to_golden.py

For each scene the script:
  1. Reads the golden frame numbers from the filenames in
     `golden_frames_3fps_mid/{split}/{scene}/`.
  2. Reads the full-sequence gt.txt produced by the tracker.
  3. Keeps only the rows whose frame number matches a golden frame.
  4. Writes the filtered gt.txt and a matching seqinfo.ini to
     `MOT_labels_golden_mid/{split}/{scene}/`.

Output layout (MOT Challenge format)
-------------------------------------
  dataset/MOT_labels_golden_mid/
    {split}/
      {scene}/
        gt/
          gt.txt      ← filtered rows: frame,id,x,y,w,h,conf,-1,-1,-1
        seqinfo.ini   ← sequence metadata (seqLength = golden frame count)

Usage
-----
  # Default paths
  uv run python labelling/filter_mot_to_golden.py

  # Custom paths
  uv run python labelling/filter_mot_to_golden.py \\
      --tracker-dir dataset/MOT_labels_bytetrack \\
      --golden-dir  dataset/golden_frames_3fps_mid \\
      --output-dir  dataset/MOT_labels_golden_mid \\
      --splits train test challange
"""

from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def frame_number(frame_path: Path) -> int:
    """Extract the integer frame number from a filename like ``frame_00042.jpg``."""
    stem = frame_path.stem  # e.g. "frame_00042"
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else 0


def golden_frames_for_scene(scene_dir: Path) -> list[Path]:
    """Return sorted list of image paths in a golden-frames scene directory."""
    frames = sorted(
        (p for p in scene_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=frame_number,
    )
    return frames


def filter_scene(
    scene_name: str,
    split: str,
    tracker_dir: Path,
    golden_dir: Path,
    output_dir: Path,
) -> dict:
    """Filter one scene's gt.txt to golden frames and write the result.

    Returns a summary dict with keys: scene, golden_frames, matched_rows,
    total_tracker_rows, skipped (bool).
    """
    golden_scene = golden_dir / split / scene_name
    tracker_gt = tracker_dir / split / scene_name / "gt" / "gt.txt"
    out_scene = output_dir / split / scene_name

    # ── Collect golden frame numbers ─────────────────────────────────────────
    frames = golden_frames_for_scene(golden_scene)
    if not frames:
        return {"scene": scene_name, "golden_frames": 0, "skipped": True,
                "reason": "no images in golden scene dir"}

    golden_set: set[int] = {frame_number(f) for f in frames}

    # ── Read tracker gt.txt ───────────────────────────────────────────────────
    if not tracker_gt.exists():
        return {"scene": scene_name, "golden_frames": len(frames), "skipped": True,
                "reason": f"tracker gt.txt not found: {tracker_gt}"}

    all_rows = tracker_gt.read_text().splitlines()
    # Strip blank lines
    all_rows = [r for r in all_rows if r.strip()]

    # ── Filter to golden frames ───────────────────────────────────────────────
    matched: list[str] = []
    for row in all_rows:
        parts = row.split(",")
        if not parts:
            continue
        try:
            fnum = int(parts[0])
        except ValueError:
            continue
        if fnum in golden_set:
            matched.append(row)

    # ── Write output gt.txt ───────────────────────────────────────────────────
    gt_out = out_scene / "gt" / "gt.txt"
    gt_out.parent.mkdir(parents=True, exist_ok=True)
    gt_out.write_text("\n".join(matched) + ("\n" if matched else ""))

    # ── Write seqinfo.ini ─────────────────────────────────────────────────────
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is not None:
        img_h, img_w = first_frame.shape[:2]
    else:
        img_h, img_w = 0, 0

    seq_cfg = configparser.ConfigParser()
    seq_cfg["Sequence"] = {
        "name": scene_name,
        "imDir": str(golden_scene.resolve()),
        "frameRate": "3",
        "seqLength": str(len(frames)),
        "imWidth": str(img_w),
        "imHeight": str(img_h),
        "imExt": frames[0].suffix,
    }
    with open(out_scene / "seqinfo.ini", "w") as f:
        seq_cfg.write(f)

    return {
        "scene": scene_name,
        "golden_frames": len(frames),
        "matched_rows": len(matched),
        "total_tracker_rows": len(all_rows),
        "skipped": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter full-sequence MOT gt.txt to golden-frame rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tracker-dir",
        type=Path,
        default=Path("dataset/MOT_labels_bytetrack"),
        help="Root directory of the full-sequence MOT tracker output.",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path("dataset/golden_frames_3fps_mid"),
        help="Root directory of the golden frames (frame_XXXXX.jpg files).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/MOT_labels_golden_mid"),
        help="Root directory where filtered gt.txt files will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "challange"],
        help="Dataset splits to process.",
    )
    args = parser.parse_args()

    if not args.golden_dir.exists():
        print(f"ERROR: golden-dir not found: {args.golden_dir}", file=sys.stderr)
        return 1
    if not args.tracker_dir.exists():
        print(
            f"WARNING: tracker-dir not found: {args.tracker_dir}\n"
            "Run autolabel_bytetrack.py first to generate full-sequence MOT labels.",
            file=sys.stderr,
        )
        return 1

    total_golden = 0
    total_matched = 0
    warnings: list[str] = []

    for split in args.splits:
        split_dir = args.golden_dir / split
        if not split_dir.exists():
            continue

        scenes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        if not scenes:
            continue

        print(f"\n── {split} ({len(scenes)} scenes) ──")

        for scene_name in scenes:
            result = filter_scene(
                scene_name=scene_name,
                split=split,
                tracker_dir=args.tracker_dir,
                golden_dir=args.golden_dir,
                output_dir=args.output_dir,
            )

            if result["skipped"]:
                msg = f"  SKIP  {scene_name}: {result['reason']}"
                print(msg)
                warnings.append(msg)
                continue

            gf = result["golden_frames"]
            mr = result["matched_rows"]
            tr = result["total_tracker_rows"]
            total_golden += gf
            total_matched += mr

            status = ""
            if mr == 0:
                status = "  *** NO MATCHES — check detector quality ***"
                warnings.append(f"  {split}/{scene_name}: 0 matched rows (golden={gf}, tracker_total={tr})")

            print(f"  {scene_name}: {gf} golden frames, {mr} matched rows "
                  f"(of {tr} tracker rows){status}")

    print(f"\nDone. {total_golden} golden frames across all scenes, "
          f"{total_matched} matched MOT rows written to {args.output_dir}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(w)

    return 0


if __name__ == "__main__":
    sys.exit(main())
