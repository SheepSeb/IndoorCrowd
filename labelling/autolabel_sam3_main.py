#!/usr/bin/env python3
"""SAM3 video tracking → COCO JSON for dataset/main.

Produces per-recording COCO files with bbox + RLE segmentation masks.
Wall-clock timing is saved to timing.json in the output root.

Output layout
-------------
  dataset/main/labels_sam3/
    {scene}/
      {recording}.json
    timing.json

Usage
-----
  uv run python labelling/autolabel_sam3_main.py

  # Specific scenes only
  uv run python labelling/autolabel_sam3_main.py --scenes acs_ec acs_eg

  # Tune
  uv run python labelling/autolabel_sam3_main.py \\
      --conf 0.30 --part-size 60 --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_RATE = 5


def collect_frames(d: Path) -> list[Path]:
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_video(frames: list[Path], out: Path, fps: int) -> tuple[int, int]:
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        img = cv2.imread(str(f))
        if img is not None:
            writer.write(img)
    writer.release()
    return w, h


def mask_to_rle(binary: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def run_recording(
    recording_dir: Path,
    out_json: Path,
    raw_frames_dir: Path,
    predictor,
    args,
) -> dict:
    frames = collect_frames(recording_dir)
    if not frames:
        return {"frames": 0, "annotations": 0, "seconds": 0.0}

    first_img = cv2.imread(str(frames[0]))
    img_h, img_w = first_img.shape[:2]

    parts = [frames[i: i + args.part_size] for i in range(0, len(frames), args.part_size)]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_video = out_json.parent / "_tmp.mp4"

    coco_images: list[dict] = []
    coco_anns: list[dict] = []
    ann_id = 1
    image_id = 1

    t0 = time.perf_counter()

    for part_idx, part_frames in enumerate(parts, 1):
        tqdm.write(
            f"    part {part_idx}/{len(parts)}"
            f"  [{part_frames[0].name}…{part_frames[-1].name}]"
        )
        write_video(part_frames, tmp_video, fps=FRAME_RATE)

        # Reset SAM3 state so IDs don't leak between parts
        predictor.inference_state = {}
        results_gen = predictor(source=str(tmp_video), text=[args.text_prompt], stream=True)

        for frame_path, result in zip(part_frames, results_gen):
            coco_images.append({
                "id": image_id,
                "file_name": str(frame_path.relative_to(raw_frames_dir)),
                "width": img_w,
                "height": img_h,
            })

            if result.boxes is None or len(result.boxes) == 0:
                image_id += 1
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
                if result.masks is not None else None
            )

            for i, (box, score, tid) in enumerate(zip(boxes_xyxy, scores, track_ids)):
                x1, y1, x2, y2 = box.tolist()
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                if masks_data is not None and i < len(masks_data):
                    binary = (masks_data[i] > 0.5).astype(np.uint8)
                    if binary.shape != (img_h, img_w):
                        binary = cv2.resize(
                            binary, (img_w, img_h), interpolation=cv2.INTER_NEAREST
                        )
                    segmentation = mask_to_rle(binary)
                    area = float(binary.sum())
                else:
                    segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                    area = w * h

                coco_anns.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "track_id": int(tid),
                    "bbox": [round(v, 2) for v in [x, y, w, h]],
                    "area": round(area, 2),
                    "segmentation": segmentation,
                    "score": round(float(score), 4),
                    "iscrowd": 0,
                })
                ann_id += 1

            image_id += 1

    elapsed = time.perf_counter() - t0
    tmp_video.unlink(missing_ok=True)

    with open(out_json, "w") as f:
        json.dump(
            {
                "images": coco_images,
                "annotations": coco_anns,
                "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
            },
            f,
        )

    return {"frames": len(frames), "annotations": len(coco_anns), "seconds": elapsed}


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM3 → COCO JSON for dataset/main")
    parser.add_argument("--raw-frames-dir", type=Path,
                        default=Path("dataset/main/raw_frames_5fps"))
    parser.add_argument("--output-dir",     type=Path,
                        default=Path("dataset/main/labels_sam3"))
    parser.add_argument("--model",       type=str,   default="sam3.pt")
    parser.add_argument("--text-prompt", type=str,   default="person")
    parser.add_argument("--conf",        type=float, default=0.25)
    parser.add_argument("--imgsz",       type=int,   default=640)
    parser.add_argument("--part-size",   type=int,   default=55,
                        help="Frames per SAM3 video part (50–60 recommended).")
    parser.add_argument("--scenes",  nargs="+", default=None,
                        help="Restrict to specific scene dirs (e.g. acs_ec acs_eg).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.raw_frames_dir.exists():
        print(f"ERROR: {args.raw_frames_dir} not found", file=sys.stderr)
        return 1

    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    t_load = time.perf_counter()
    predictor = SAM3VideoSemanticPredictor(overrides=dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=args.model,
        imgsz=args.imgsz,
        half=True,
        verbose=False,
    ))
    model_load_s = time.perf_counter() - t_load

    print(f"Model       : {args.model}  conf={args.conf}  (load: {model_load_s:.1f}s)")
    print(f"Input       : {args.raw_frames_dir}")
    print(f"Output      : {args.output_dir}")

    scene_dirs = sorted(d for d in args.raw_frames_dir.iterdir() if d.is_dir())
    if args.scenes:
        scene_dirs = [s for s in scene_dirs if s.name in set(args.scenes)]
    if not scene_dirs:
        print("No scene directories found.", file=sys.stderr)
        return 1

    # Load existing timing.json so we can merge (don't lose previously timed scenes)
    timing_path = args.output_dir / "timing.json"
    if timing_path.exists() and not args.overwrite:
        with open(timing_path) as f:
            timing = json.load(f)
    else:
        timing = {
            "method": "sam3",
            "model": args.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_load_seconds": round(model_load_s, 2),
            "total_seconds": 0.0,
            "scenes": {},
        }

    t_total = time.perf_counter()

    for scene_dir in scene_dirs:
        recordings = sorted(d for d in scene_dir.iterdir() if d.is_dir())
        print(f"\n── {scene_dir.name} ({len(recordings)} recordings) ──")

        for rec_dir in tqdm(recordings, desc=scene_dir.name, unit="rec"):
            out_json = args.output_dir / scene_dir.name / f"{rec_dir.name}.json"
            key = f"{scene_dir.name}/{rec_dir.name}"

            if out_json.exists() and not args.overwrite:
                tqdm.write(f"  skip (exists): {rec_dir.name}")
                continue

            tqdm.write(f"  {rec_dir.name}")
            stats = run_recording(rec_dir, out_json, args.raw_frames_dir, predictor, args)

            timing["scenes"][key] = {
                "frames": stats["frames"],
                "annotations": stats["annotations"],
                "seconds": round(stats["seconds"], 2),
                "fps_processed": (
                    round(stats["frames"] / stats["seconds"], 2)
                    if stats["seconds"] > 0 else 0.0
                ),
            }
            tqdm.write(
                f"  → {stats['frames']} frames | {stats['annotations']} anns"
                f" | {stats['seconds']:.1f}s"
            )

    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\nDone in {timing['total_seconds']:.1f}s")
    print(f"Timing: {timing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
