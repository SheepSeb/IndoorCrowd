#!/usr/bin/env python3
"""
Auto-label dataset/MOT_main/clips_sam with SAM3 in two ways:

  1. sam3_native  — SAM3 video predictor track IDs used directly (no external tracker)
  2. sam3_botsort — SAM3 detections fed into BoT-SORT

Both outputs are produced from the same SAM3 video pass per clip, so the
model is only run once.

Output layout
-------------
  dataset/MOT_main/labels_sam3_native/{split}/{scene}/{clip}/gt/gt.txt
  dataset/MOT_main/labels_sam3_botsort/{split}/{scene}/{clip}/gt/gt.txt
  dataset/MOT_main/timing_autolabel.json

MOT gt.txt format
-----------------
  frame,id,x,y,w,h,conf,-1,-1,-1
  (absolute frame numbers matching frame_XXXXX.jpg filenames, TLWH coords)

Usage
-----
  uv run python labelling/autolabel_mot_clips_sam.py

  # Only test split
  uv run python labelling/autolabel_mot_clips_sam.py --splits test

  # Specific scenes
  uv run python labelling/autolabel_mot_clips_sam.py --scenes acs_s1_recording_2026-02-23_18-08-10

  # Re-process already-done clips
  uv run python labelling/autolabel_mot_clips_sam.py --overwrite
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT   = Path(__file__).resolve().parents[1]
CLIPS_DIR   = REPO_ROOT / "dataset" / "MOT_main" / "clips_sam"
OUT_NATIVE  = REPO_ROOT / "dataset" / "MOT_main" / "labels_sam3_native"
OUT_BOTSORT = REPO_ROOT / "dataset" / "MOT_main" / "labels_sam3_botsort"
TIMING_PATH = REPO_ROOT / "dataset" / "MOT_main" / "timing_autolabel.json"

FRAME_RATE  = 3    # clips are at 3 fps
SAM3_PART   = 55   # frames per SAM3 video part (50–60 recommended)
TEXT_PROMPT = "person"
CONF_THRESH = 0.25
IMG_SIZE    = 640


# ── Helpers ────────────────────────────────────────────────────────────────────


def _fnum(path: Path) -> int:
    """frame_00166.jpg → 166"""
    return int("".join(filter(str.isdigit, path.stem)))


def _write_video(frames: list[Path], out: Path, fps: int) -> None:
    first  = cv2.imread(str(frames[0]))
    h, w   = first.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        img = cv2.imread(str(f))
        if img is not None:
            writer.write(img)
    writer.release()


def _write_mot(rows: list[tuple], path: Path) -> None:
    """rows = [(frame, tid, x, y, w, h, conf), ...]  →  MOT gt.txt"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame, tid, x, y, w, h, conf in sorted(rows):
            f.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")


# ── BoT-SORT wrapper (same as benchmark_mot.py) ────────────────────────────────


class _Dets:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        import torch
        self._xyxy = xyxy.astype(np.float32)
        self.conf  = torch.from_numpy(conf.astype(np.float32))
        self.cls   = torch.zeros(len(conf), dtype=torch.float32)
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
        w  = xyxy[:, 2] - xyxy[:, 0]
        h  = xyxy[:, 3] - xyxy[:, 1]
        self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx):
        import torch
        i = idx.numpy() if isinstance(idx, torch.Tensor) else idx
        return _Dets(self._xyxy[i], self.conf.numpy()[i])


_EMPTY_DETS = _Dets(np.empty((0, 4), np.float32), np.empty(0, np.float32))


def _make_botsort(fps: int):
    from ultralytics.trackers.bot_sort import BOTSORT
    args = types.SimpleNamespace(
        track_high_thresh=0.35, track_low_thresh=0.10,
        new_track_thresh=0.40, track_buffer=fps * 2,
        match_thresh=0.80, proximity_thresh=0.50,
        appearance_thresh=0.25, with_reid=False,
        gmc_method="sparseOptFlow", model="auto", fuse_score=True,
    )
    return BOTSORT(args, frame_rate=fps)


# ── Per-clip processing ────────────────────────────────────────────────────────


def process_clip(
    clip_dir: Path,
    predictor,
    out_native: Path,
    out_botsort: Path,
) -> dict:
    """
    Run SAM3 on the clip, write both MOT gt.txt files.
    Returns stats dict.
    """
    frames = sorted(clip_dir.glob("frames/frame_*.jpg"))
    if not frames:
        return {"frames": 0, "anns_native": 0, "anns_botsort": 0, "seconds": 0.0}

    parts   = [frames[i: i + SAM3_PART] for i in range(0, len(frames), SAM3_PART)]
    botsort = _make_botsort(FRAME_RATE)

    rows_native:  list[tuple] = []
    rows_botsort: list[tuple] = []

    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video = Path(tmpdir) / "_sam3.mp4"
        frame_idx = 0

        for part_frames in parts:
            _write_video(part_frames, tmp_video, fps=FRAME_RATE)
            predictor.inference_state = {}

            for result in predictor(source=str(tmp_video), text=[TEXT_PROMPT], stream=True):
                frame_path = frames[frame_idx]
                fnum       = _fnum(frame_path)
                frame_idx += 1

                if result.boxes is None or len(result.boxes) == 0:
                    continue

                xyxy  = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                confs = result.boxes.conf.cpu().numpy().astype(np.float32)
                tids  = (
                    result.boxes.id.cpu().numpy().astype(int)
                    if result.boxes.id is not None
                    else np.arange(len(xyxy), dtype=int)
                )

                # Method 1: native SAM3 track IDs
                for box, conf, tid in zip(xyxy, confs, tids):
                    x1, y1, x2, y2 = box
                    rows_native.append(
                        (fnum, int(tid), float(x1), float(y1),
                         float(x2 - x1), float(y2 - y1), float(conf))
                    )

                # Method 2: BoT-SORT
                img_bgr = cv2.imread(str(frame_path))
                if img_bgr is not None:
                    dets  = _Dets(xyxy, confs)
                    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    tracks = botsort.update(dets, img=gray)
                    for row in tracks:
                        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                        tid_bs = int(row[4])
                        conf_bs = float(row[5]) if len(row) > 5 else 1.0
                        rows_botsort.append(
                            (fnum, tid_bs, x1, y1, x2 - x1, y2 - y1, conf_bs)
                        )

    elapsed = time.perf_counter() - t0

    _write_mot(rows_native,  out_native  / "gt" / "gt.txt")
    _write_mot(rows_botsort, out_botsort / "gt" / "gt.txt")

    return {
        "frames":       len(frames),
        "anns_native":  len(rows_native),
        "anns_botsort": len(rows_botsort),
        "seconds":      round(elapsed, 2),
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-label MOT_main/clips_sam with SAM3 native + SAM3+BoT-SORT"
    )
    parser.add_argument("--clips-dir",   type=Path, default=CLIPS_DIR)
    parser.add_argument("--out-native",  type=Path, default=OUT_NATIVE)
    parser.add_argument("--out-botsort", type=Path, default=OUT_BOTSORT)
    parser.add_argument("--model",   type=str,   default="sam3.pt")
    parser.add_argument("--conf",    type=float, default=CONF_THRESH)
    parser.add_argument("--imgsz",   type=int,   default=IMG_SIZE)
    parser.add_argument("--splits",  nargs="+",  default=None,
                        help="Splits to process (default: all found under clips-dir).")
    parser.add_argument("--scenes",  nargs="+",  default=None,
                        help="Filter to specific scene directory names.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process clips that already have output.")
    args = parser.parse_args()

    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    t_load = time.perf_counter()
    predictor = SAM3VideoSemanticPredictor(overrides=dict(
        conf=args.conf, task="segment", mode="predict",
        model=args.model, imgsz=args.imgsz, half=True, verbose=False,
    ))
    model_load_s = time.perf_counter() - t_load
    print(f"Model : {args.model}  conf={args.conf}  imgsz={args.imgsz}"
          f"  (loaded in {model_load_s:.1f}s)")
    print(f"Input : {args.clips_dir}")
    print(f"Output: {args.out_native}  |  {args.out_botsort}")

    # Merge with existing timing if not overwriting everything
    if TIMING_PATH.exists() and not args.overwrite:
        with open(TIMING_PATH) as f:
            timing = json.load(f)
    else:
        timing = {
            "method":             "sam3_native + sam3_botsort",
            "model":              args.model,
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "model_load_seconds": round(model_load_s, 2),
            "clips":              {},
        }

    splits = args.splits or [
        d.name for d in sorted(args.clips_dir.iterdir()) if d.is_dir()
    ]

    for split in splits:
        split_dir = args.clips_dir / split
        if not split_dir.exists():
            print(f"[WARN] split dir not found: {split_dir}")
            continue

        scene_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        if args.scenes:
            scene_dirs = [s for s in scene_dirs if s.name in set(args.scenes)]

        for scene_dir in scene_dirs:
            clip_dirs = sorted(c for c in scene_dir.iterdir() if c.is_dir())
            print(f"\n── {split}/{scene_dir.name}  ({len(clip_dirs)} clips)")

            for clip_dir in tqdm(clip_dirs, unit="clip"):
                key = f"{split}/{scene_dir.name}/{clip_dir.name}"
                out_n = args.out_native  / split / scene_dir.name / clip_dir.name
                out_b = args.out_botsort / split / scene_dir.name / clip_dir.name

                if (out_n / "gt" / "gt.txt").exists() \
                        and (out_b / "gt" / "gt.txt").exists() \
                        and not args.overwrite:
                    tqdm.write(f"  skip (exists): {clip_dir.name}")
                    continue

                tqdm.write(f"  {clip_dir.name}")
                try:
                    stats = process_clip(clip_dir, predictor, out_n, out_b)
                except Exception as exc:
                    tqdm.write(f"  ERROR: {exc}")
                    continue

                timing["clips"][key] = {
                    **stats,
                    "fps_processed": round(
                        stats["frames"] / stats["seconds"], 2
                    ) if stats["seconds"] > 0 else 0.0,
                }
                tqdm.write(
                    f"  → {stats['frames']} frames"
                    f" | native={stats['anns_native']} anns"
                    f" | botsort={stats['anns_botsort']} anns"
                    f" | {stats['seconds']:.1f}s"
                )

    TIMING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TIMING_PATH, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\nDone. Timing → {TIMING_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
