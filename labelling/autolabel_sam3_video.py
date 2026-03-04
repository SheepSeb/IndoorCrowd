#!/usr/bin/env python3
"""SAM3 Video semantic tracking pipeline — auto-annotation.

Based on track_sam3.py.  Extends it to process all scenes automatically:

  1. Splits each scene's 3-fps frames into parts of ~PART_SIZE frames.
  2. Writes a temporary MP4 per part (SAM3VideoSemanticPredictor needs video mode).
  3. Runs SAM3VideoSemanticPredictor with text="person", stream=True.
  4. Saves per-part outputs:
       frames/          symlinks to the source frame images
       gt/gt.txt        MOT Challenge format  (frame,id,x,y,w,h,conf,-1,-1,-1)
       annotations.json COCO-style with per-instance track_id + segmentation
       seqinfo.ini      sequence metadata

Output layout
-------------
  dataset/SAM3_video_tracks/
    {split}/
      {scene}/
        part_001/
          frames/         ← symlinks to raw_frames_3fps images
          gt/gt.txt
          annotations.json
          seqinfo.ini
        part_002/
          ...

Usage
-----
  # Defaults (sam3.pt, "person", 55 frames/part)
  uv run python labelling/autolabel_sam3_video.py

  # Tune
  uv run python labelling/autolabel_sam3_video.py \\
      --part-size 50 \\
      --conf 0.25 \\
      --output-dir dataset/SAM3_video_tracks

  # One scene
  uv run python labelling/autolabel_sam3_video.py \\
      --splits train \\
      --scenes acs_s1_recording_2026-02-23_17-15-46 \\
      --overwrite
"""

from __future__ import annotations

import argparse
import configparser
import fnmatch
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from pycocotools import mask as mask_utils
from tqdm import tqdm

load_dotenv()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_RATE = 3  # dataset is sampled at 3 fps


# ── helpers ───────────────────────────────────────────────────────────────────


def collect_frames(scene_dir: Path) -> list[Path]:
    return sorted(
        p for p in scene_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def frame_number(frame_path: Path) -> int:
    digits = "".join(c for c in frame_path.stem if c.isdigit())
    return int(digits) if digits else 0


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def write_video(frames: list[Path], out_path: Path, fps: int = FRAME_RATE) -> tuple[int, int]:
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


def symlink_frames(src_frames: list[Path], frames_dir: Path) -> None:
    """Create symlinks in frames_dir pointing to the source frames."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for src in src_frames:
        dst = frames_dir / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())


# ── per-part processing ───────────────────────────────────────────────────────


def run_part(
    part_frames: list[Path],
    part_out: Path,
    predictor,
    text_prompt: str,
    part_idx: int,
    scene_name: str,
) -> dict:
    """
    Run SAM3 video tracking on one part and write outputs.

    Returns stats dict with keys: frames, tracks, rows.
    """
    part_out.mkdir(parents=True, exist_ok=True)

    # Symlink source frames for reference / review tool
    symlink_frames(part_frames, part_out / "frames")

    # Write temp video (SAM3VideoSemanticPredictor needs video mode, not image dir)
    tmp_video = part_out / "_part.mp4"
    img_w, img_h = write_video(part_frames, tmp_video, fps=FRAME_RATE)

    # seqinfo.ini
    seq_cfg = configparser.ConfigParser()
    seq_cfg["Sequence"] = {
        "name": f"{scene_name}_part{part_idx:03d}",
        "imDir": str((part_out / "frames").resolve()),
        "frameRate": str(FRAME_RATE),
        "seqLength": str(len(part_frames)),
        "imWidth": str(img_w),
        "imHeight": str(img_h),
        "imExt": part_frames[0].suffix,
    }
    with open(part_out / "seqinfo.ini", "w") as f:
        seq_cfg.write(f)

    # Reset tracker state between parts so IDs don't leak
    predictor.inference_state = {}

    # --- Run SAM3 (same pattern as track_sam3.py) ---
    results_gen = predictor(source=str(tmp_video), text=[text_prompt], stream=True)

    mot_rows: list[str] = []
    coco_images: list[dict] = []
    coco_anns: list[dict] = []
    all_track_ids: set[int] = set()
    ann_id = 1

    for local_fnum, (frame_path, result) in enumerate(
        zip(part_frames, results_gen), start=1
    ):
        coco_images.append({
            "id": local_fnum,
            "file_name": frame_path.name,
            "width": img_w,
            "height": img_h,
            "source_path": str(frame_path),
        })

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        track_ids = (
            result.boxes.id.cpu().numpy().astype(int)
            if result.boxes.id is not None
            else np.arange(len(boxes_xyxy), dtype=int)
        )
        masks_data = (
            result.masks.data.cpu().numpy()
            if result.masks is not None
            else None
        )

        for i, (box, score, tid) in enumerate(zip(boxes_xyxy, scores, track_ids)):
            x1, y1, x2, y2 = box.tolist()
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            tid = int(tid)

            mot_rows.append(
                f"{local_fnum},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f}"
                f",{score:.4f},-1,-1,-1"
            )
            all_track_ids.add(tid)

            # Segmentation mask → COCO RLE
            if masks_data is not None and i < len(masks_data):
                binary = (masks_data[i] > 0.5).astype(np.uint8)
                if binary.shape != (img_h, img_w):
                    binary = cv2.resize(
                        binary, (img_w, img_h), interpolation=cv2.INTER_NEAREST
                    )
                segmentation = mask_to_rle(binary)
                area = float(binary.sum())
            else:
                segmentation = []
                area = round(w * h, 2)

            coco_anns.append({
                "id": ann_id,
                "image_id": local_fnum,
                "category_id": 1,
                "track_id": tid,
                "bbox": [round(v, 2) for v in [x, y, w, h]],
                "area": round(area, 2),
                "segmentation": segmentation,
                "score": round(float(score), 4),
                "iscrowd": 0,
            })
            ann_id += 1

    # Clean up temp video
    tmp_video.unlink(missing_ok=True)

    # Write MOT gt.txt
    gt_dir = part_out / "gt"
    gt_dir.mkdir(exist_ok=True)
    (gt_dir / "gt.txt").write_text(
        "\n".join(mot_rows) + ("\n" if mot_rows else "")
    )

    # Write COCO annotations.json
    with open(part_out / "annotations.json", "w") as f:
        json.dump(
            {
                "images": coco_images,
                "annotations": coco_anns,
                "categories": [
                    {"id": 1, "name": "person", "supercategory": "person"}
                ],
            },
            f,
            indent=2,
        )

    return {
        "frames": len(part_frames),
        "tracks": len(all_track_ids),
        "rows": len(mot_rows),
    }


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAM3 Video semantic tracking → per-part MOT + COCO annotations"
    )
    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        default=Path("dataset/raw_frames_3fps"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/SAM3_video_tracks"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sam3.pt",
        help="SAM3 model weights (auto-downloaded if absent).",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="person",
        help="Text concept to track.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--part-size",
        type=int,
        default=55,
        help="Number of frames per part (50–60 recommended).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to process (default: all found).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Restrict to specific scene names.",
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
        help="Re-process already labelled parts.",
    )
    args = parser.parse_args()

    if not args.raw_frames_dir.exists():
        print(
            f"ERROR: raw frames dir not found: {args.raw_frames_dir}",
            file=sys.stderr,
        )
        return 1

    # ---- load model (once) -------------------------------------------------
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

    print(f"Model:     {args.model}")
    print(f"Prompt:    '{args.text_prompt}'")
    print(f"Part size: {args.part_size} frames  |  conf={args.conf}")
    print(f"Output:    {args.output_dir}")

    # ---- discover splits / scenes ------------------------------------------
    split_dirs = sorted(d for d in args.raw_frames_dir.iterdir() if d.is_dir())
    if args.splits:
        split_dirs = [s for s in split_dirs if s.name in set(args.splits)]
    if not split_dirs:
        print("No splits found.", file=sys.stderr)
        return 1

    total_parts = total_tracks = total_rows = 0

    for split_dir in split_dirs:
        scenes = sorted(d for d in split_dir.iterdir() if d.is_dir())
        if args.scenes:
            scenes = [s for s in scenes if s.name in set(args.scenes)]
        if args.exclude:
            scenes = [
                s for s in scenes
                if not any(fnmatch.fnmatch(s.name, pat) for pat in args.exclude)
            ]

        print(f"\n── {split_dir.name} ({len(scenes)} scenes) ──")

        for scene_dir in tqdm(scenes, desc=split_dir.name, unit="scene"):
            frames = collect_frames(scene_dir)
            if not frames:
                continue

            parts = [
                frames[i: i + args.part_size]
                for i in range(0, len(frames), args.part_size)
            ]
            scene_out = args.output_dir / split_dir.name / scene_dir.name

            for part_idx, part_frames in enumerate(parts, start=1):
                part_out = scene_out / f"part_{part_idx:03d}"
                gt_file = part_out / "gt" / "gt.txt"

                if gt_file.exists() and not args.overwrite:
                    tqdm.write(
                        f"  skip (exists): {scene_dir.name}/part_{part_idx:03d}"
                    )
                    continue

                tqdm.write(
                    f"  {scene_dir.name} / part_{part_idx:03d}"
                    f"  [{part_frames[0].name}…{part_frames[-1].name}]"
                    f"  ({len(part_frames)} frames)"
                )

                stats = run_part(
                    part_frames=part_frames,
                    part_out=part_out,
                    predictor=predictor,
                    text_prompt=args.text_prompt,
                    part_idx=part_idx,
                    scene_name=scene_dir.name,
                )

                tqdm.write(
                    f"    → {stats['tracks']} tracks | {stats['rows']} annotations"
                )
                total_parts += 1
                total_tracks += stats["tracks"]
                total_rows += stats["rows"]

    print(f"\n{'─' * 60}")
    print(
        f"Done.  Parts: {total_parts}  |  "
        f"Tracks total: {total_tracks}  |  "
        f"Annotations: {total_rows}"
    )
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
