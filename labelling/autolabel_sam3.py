#!/usr/bin/env python3
"""SAM3 video tracking → scene-level MOT gt.txt for raw_frames_5fps.

Uses SAM3VideoSemanticPredictor (SAM3's built-in tracker) to detect and track
persons.  Because the predictor requires video input and resets its state between
clips, each scene is split into overlapping parts.  Track IDs are offset per part
so they never collide in the merged output.

Output layout
-------------
  dataset/MOT_labels_sam3_5fps/
    {split}/
      {scene}/
        gt/gt.txt        ← frame,id,x,y,w,h,conf,-1,-1,-1  (global 1-based frames)
        seqinfo.ini

Usage
-----
  # All splits / scenes
  uv run python labelling/autolabel_sam3_5fps.py

  # One split / scene
  uv run python labelling/autolabel_sam3_5fps.py \\
      --splits train \\
      --scenes acs_s1_recording_2026-02-23_17-15-46 \\
      --overwrite

  # Tune
  uv run python labelling/autolabel_sam3_5fps.py \\
      --part-size  60   \\
      --conf       0.25 \\
      --min-track-len 3
"""

from __future__ import annotations

import argparse
import configparser
import fnmatch
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_RATE = 5


# ── helpers ───────────────────────────────────────────────────────────────────


def collect_frames(scene_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in scene_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def frame_number(p: Path) -> int:
    digits = "".join(c for c in p.stem if c.isdigit())
    return int(digits) if digits else 0


def write_video(frames: list[Path], out_path: Path, fps: int) -> tuple[int, int]:
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for f in frames:
        img = cv2.imread(str(f))
        if img is not None:
            writer.write(img)
    writer.release()
    return w, h


# ── scene processing ──────────────────────────────────────────────────────────


def run_scene(
    scene_dir: Path,
    output_dir: Path,
    predictor,
    args,
) -> dict:
    frames = collect_frames(scene_dir)
    if not frames:
        return {"scene": scene_dir.name, "frames": 0, "tracks": 0, "rows": 0}

    first_img = cv2.imread(str(frames[0]))
    img_h, img_w = first_img.shape[:2]

    parts = [
        frames[i : i + args.part_size] for i in range(0, len(frames), args.part_size)
    ]

    # Accumulated MOT rows across all parts: {global_frame: [(tid, x,y,w,h, conf), ...]}
    mot_rows: list[str] = []

    # Track ID offset so each part's IDs don't collide with previous parts
    tid_offset = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_video = output_dir / "_tmp_part.mp4"

    for part_idx, part_frames in enumerate(parts, start=1):
        tqdm.write(
            f"    part {part_idx}/{len(parts)}"
            f"  [{part_frames[0].name}…{part_frames[-1].name}]"
            f"  ({len(part_frames)} frames)"
        )

        write_video(part_frames, tmp_video, fps=FRAME_RATE)

        # Reset SAM3 state so IDs don't leak from prior parts
        predictor.inference_state = {}

        results_gen = predictor(
            source=str(tmp_video), text=[args.text_prompt], stream=True
        )

        part_track_ids: set[int] = set()

        for local_fnum, (frame_path, result) in enumerate(
            zip(part_frames, results_gen), start=1
        ):
            global_fnum = frame_number(frame_path)

            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            track_ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else np.arange(len(boxes_xyxy), dtype=int)
            )

            for box, score, tid in zip(boxes_xyxy, scores, track_ids):
                x1, y1, x2, y2 = box.tolist()
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                global_tid = int(tid) + tid_offset
                part_track_ids.add(int(tid))
                mot_rows.append(
                    f"{global_fnum},{global_tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f}"
                    f",{score:.4f},-1,-1,-1"
                )

        max_tid = max(part_track_ids, default=-1)
        tid_offset += max_tid + 1

    tmp_video.unlink(missing_ok=True)

    # Optional: drop tracks shorter than min_track_len
    if args.min_track_len > 1:
        from collections import defaultdict

        by_track: dict[int, list[str]] = defaultdict(list)
        for row in mot_rows:
            tid = int(row.split(",")[1])
            by_track[tid].append(row)
        mot_rows = [
            row
            for tid, rows in by_track.items()
            if len(rows) >= args.min_track_len
            for row in rows
        ]
        # Compact IDs
        surviving = sorted({int(r.split(",")[1]) for r in mot_rows})
        remap = {old: new for new, old in enumerate(surviving, start=1)}
        mot_rows = [
            ",".join(
                [r.split(",")[0], str(remap[int(r.split(",")[1])]), *r.split(",")[2:]]
            )
            for r in mot_rows
        ]

    mot_rows.sort(key=lambda s: (int(s.split(",")[0]), int(s.split(",")[1])))

    gt_dir = output_dir / "gt"
    gt_dir.mkdir(exist_ok=True)
    (gt_dir / "gt.txt").write_text("\n".join(mot_rows) + ("\n" if mot_rows else ""))

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

    unique_tracks = len({int(r.split(",")[1]) for r in mot_rows})
    return {
        "scene": scene_dir.name,
        "frames": len(frames),
        "tracks": unique_tracks,
        "rows": len(mot_rows),
    }


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAM3 video tracking → scene-level MOT gt.txt (5 fps)"
    )
    parser.add_argument(
        "--raw-frames-dir", type=Path, default=Path("dataset/main/raw_frames_5fps")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dataset/MOT_labels_sam3_5fps")
    )
    parser.add_argument("--model", type=str, default="sam3.pt")
    parser.add_argument("--text-prompt", type=str, default="person")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--part-size",
        type=int,
        default=55,
        help="Frames per SAM3 video part (50–60 recommended).",
    )
    parser.add_argument(
        "--min-track-len",
        type=int,
        default=3,
        help="Remove tracks shorter than this many frames.",
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--exclude", nargs="+", default=None, metavar="PATTERN")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.raw_frames_dir.exists():
        print(f"ERROR: not found: {args.raw_frames_dir}", file=sys.stderr)
        return 1

    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    predictor = SAM3VideoSemanticPredictor(
        overrides=dict(
            conf=args.conf,
            task="segment",
            mode="predict",
            model=args.model,
            imgsz=args.imgsz,
            half=True,
            verbose=False,
        )
    )

    print(f"Model       : {args.model}  prompt='{args.text_prompt}'  conf={args.conf}")
    print(
        f"Part size   : {args.part_size} frames  |  min-track-len={args.min_track_len}"
    )
    print(f"Input       : {args.raw_frames_dir}")
    print(f"Output      : {args.output_dir}")

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

            tqdm.write(
                f"  → {stats['frames']} frames | "
                f"{stats['tracks']} tracks | "
                f"{stats['rows']} annotations"
            )
            total_scenes += 1
            total_tracks += stats["tracks"]
            total_rows += stats["rows"]

    print(f"\n{'─' * 60}")
    print(
        f"Done. Scenes: {total_scenes} | Tracks: {total_tracks} | Annotations: {total_rows}"
    )
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
