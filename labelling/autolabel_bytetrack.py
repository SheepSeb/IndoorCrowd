#!/usr/bin/env python3
"""Multi-object tracking pipeline — YOLO + ByteTrack.

ByteTrack (Zhang et al., 2022) improves over simple IoU tracking by using a
two-stage association pass:
  1. High-confidence detections  → IoU + Kalman filter matching to active tracks.
  2. Low-confidence detections   → IoU matching to tracks that survived stage 1 unmatched.

This handles partially-occluded or low-contrast persons that a single threshold
would otherwise discard, and uses a Kalman filter to predict track positions
across frames, giving more stable IDs than the Tracktor++ IoU-only approach.

Tracking is handled by the ultralytics built-in ByteTrack implementation
(no extra dependencies beyond what is already in pyproject.toml).

Output layout (MOT Challenge format)
-------------------------------------
  dataset/MOT_labels_bytetrack/
    {split}/
      {scene}/
        gt/
          gt.txt      ← frame,id,x,y,w,h,conf,-1,-1,-1
        seqinfo.ini   ← sequence metadata

Usage
-----
  # Default: YOLO11x + ByteTrack
  uv run python labelling/autolabel_bytetrack.py

  # Smaller/faster model
  uv run python labelling/autolabel_bytetrack.py --yolo-model yolo11n.pt

  # List available golden_mid training runs
  uv run python labelling/autolabel_bytetrack.py --list-runs

  # Custom golden_mid detector (trained on golden labels)
  uv run python labelling/autolabel_bytetrack.py \\
      --detector golden --golden-run yolo11n2 --golden-checkpoint best

  # YOLO-World: open-vocabulary zero-shot detector (better recall on edge cases)
  uv run python labelling/autolabel_bytetrack.py --detector yolo-world
  uv run python labelling/autolabel_bytetrack.py \\
      --detector yolo-world --yolo-world-model yolov8x-worldv2.pt

  # Full options
  uv run python labelling/autolabel_bytetrack.py \\
      --raw-frames-dir dataset/raw_frames_3fps \\
      --output-dir     dataset/MOT_labels_bytetrack \\
      --yolo-model     yolo11x.pt \\
      --track-high-thresh 0.35 \\
      --track-low-thresh  0.10 \\
      --new-track-thresh  0.50 \\
      --track-buffer      30   \\
      --match-thresh      0.80

"""

from __future__ import annotations

import argparse
import configparser
import fnmatch
import sys
import tempfile
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PERSON_CLASS_ID = 0
GOLDEN_RUNS_DIR = Path("train/golden_mid/runs/detect")


# ── Golden-run helpers (shared with autolabel_mot_tracktor.py) ────────────────


def find_latest_run(runs_dir: Path) -> str | None:
    """Return the name of the most recently modified run directory, or None."""
    if not runs_dir.exists():
        return None
    candidates = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "weights" / "best.pt").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime).name


def list_golden_runs(runs_dir: Path) -> None:
    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        return
    print(f"Available runs in {runs_dir}:\n")
    print(f"  {'Run':<20}  {'Model':<12}  {'Epochs':>6}  {'imgsz':>5}  Checkpoints")
    print(f"  {'-'*20}  {'-'*12}  {'-'*6}  {'-'*5}  -----------")
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        weights_dir = run_dir / "weights"
        ckpts = sorted(p.name for p in weights_dir.glob("*.pt")) if weights_dir.exists() else []
        model_name, epochs, imgsz = "?", "?", "?"
        args_yaml = run_dir / "args.yaml"
        if args_yaml.exists():
            with open(args_yaml) as f:
                a = yaml.safe_load(f)
            model_name = Path(a.get("model", "?")).stem
            epochs = str(a.get("epochs", "?"))
            imgsz = str(a.get("imgsz", "?"))
        print(f"  {run_dir.name:<20}  {model_name:<12}  {epochs:>6}  {imgsz:>5}  {', '.join(ckpts) or '(none)'}")


def load_golden_run(runs_dir: Path, run_name: str, checkpoint: str) -> tuple:
    from ultralytics import YOLO

    run_dir = runs_dir / run_name
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run '{run_name}' not found in {runs_dir}. "
            f"Use --list-runs to see available runs."
        )

    if checkpoint in ("best", "last"):
        ckpt_path = run_dir / "weights" / f"{checkpoint}.pt"
    else:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = run_dir / checkpoint

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    imgsz = 640
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        with open(args_yaml) as f:
            run_args = yaml.safe_load(f)
        imgsz = int(run_args.get("imgsz", 640))

    return YOLO(str(ckpt_path)), imgsz


# ── ByteTrack config ──────────────────────────────────────────────────────────


def write_tracker_config(
    track_high_thresh: float,
    track_low_thresh: float,
    new_track_thresh: float,
    track_buffer: int,
    match_thresh: float,
) -> str:
    """Write a ByteTrack YAML config to a temp file and return its path."""
    cfg = {
        "tracker_type": "bytetrack",
        "track_high_thresh": track_high_thresh,
        "track_low_thresh": track_low_thresh,
        "new_track_thresh": new_track_thresh,
        "track_buffer": track_buffer,
        "match_thresh": match_thresh,
        "fuse_score": True,
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="bytetrack_cfg_"
    )
    yaml.dump(cfg, tmp)
    tmp.flush()
    return tmp.name


# ── Scene processing ──────────────────────────────────────────────────────────


def collect_frames(scene_dir: Path) -> list[Path]:
    return sorted(
        p for p in scene_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def frame_number(frame_path: Path) -> int:
    stem = frame_path.stem  # e.g. "frame_00042"
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else 0


def run_scene(
    scene_dir: Path,
    output_dir: Path,
    model,
    tracker_cfg: str,
    track_low_thresh: float,
    imgsz: int,
) -> dict:
    """
    Track persons through all frames of one scene using ByteTrack.

    Parameters
    ----------
    track_low_thresh : float
        Passed as the YOLO detection confidence threshold so that ByteTrack
        receives both high- and low-confidence candidates for its two-stage
        association.  ByteTrack's own ``track_high_thresh`` then splits them.
    """
    frames = collect_frames(scene_dir)
    if not frames:
        return {"scene": scene_dir.name, "frames": 0, "tracks": 0, "rows": 0}

    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / "gt.txt"

    first = cv2.imread(str(frames[0]))
    if first is None:
        return {"scene": scene_dir.name, "frames": 0, "tracks": 0, "rows": 0}
    img_h, img_w = first.shape[:2]

    seq_cfg = configparser.ConfigParser()
    seq_cfg["Sequence"] = {
        "name": scene_dir.name,
        "imDir": str(scene_dir.resolve()),
        "frameRate": "3",
        "seqLength": str(len(frames)),
        "imWidth": str(img_w),
        "imHeight": str(img_h),
        "imExt": frames[0].suffix,
    }
    with open(output_dir / "seqinfo.ini", "w") as f:
        seq_cfg.write(f)

    # Reset tracker state so IDs don't leak across scenes.
    model.predictor = None

    mot_rows: list[str] = []
    all_track_ids: set[int] = set()

    for frame_path in frames:
        fnum = frame_number(frame_path)
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        # Pass track_low_thresh to YOLO so that ByteTrack receives both
        # high- and low-confidence detections for its two-stage association.
        results = model.track(
            frame_bgr,
            persist=True,
            tracker=tracker_cfg,
            classes=[PERSON_CLASS_ID],
            conf=track_low_thresh,
            imgsz=imgsz,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()

        for box, tid, score in zip(boxes_xyxy, track_ids, scores):
            x1, y1, x2, y2 = box
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            mot_rows.append(
                f"{fnum},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1"
            )
            all_track_ids.add(int(tid))

    gt_path.write_text("\n".join(mot_rows) + ("\n" if mot_rows else ""))
    return {
        "scene": scene_dir.name,
        "frames": len(frames),
        "tracks": len(all_track_ids),
        "rows": len(mot_rows),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YOLO + ByteTrack person tracking pipeline."
    )
    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        default=Path("dataset/raw_frames_3fps"),
        help="Root directory with {split}/{scene}/frame_XXXXX.jpg structure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/MOT_labels_bytetrack"),
        help="Output root for MOT-format labels.",
    )
    # Detector
    parser.add_argument(
        "--detector",
        choices=["yolo", "golden", "yolo-world"],
        default="golden",
        help=(
            "'golden' (default): latest model from train/golden_mid/runs/detect. "
            "'yolo': generic YOLO model. "
            "'yolo-world': open-vocabulary zero-shot detector (better recall on edge cases)."
        ),
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo11x.pt",
        help="YOLO model weights (used when --detector yolo). Auto-downloaded if absent.",
    )
    parser.add_argument(
        "--yolo-world-model",
        type=str,
        default="yolov8x-worldv2.pt",
        help="YOLO-World model weights (used when --detector yolo-world). Auto-downloaded if absent.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help=(
            "Inference image size. "
            "Defaults to imgsz from the run's args.yaml for --detector golden, "
            "1280 for --detector yolo, or 640 for --detector yolo-world."
        ),
    )
    # Golden-run args
    parser.add_argument("--golden-runs-dir", type=Path, default=GOLDEN_RUNS_DIR)
    parser.add_argument(
        "--golden-run",
        type=str,
        default=None,
        help="Name of the training run (e.g. 'yolo11n2'). Required for --detector golden.",
    )
    parser.add_argument(
        "--golden-checkpoint",
        type=str,
        default="best",
        help="Checkpoint to load: 'best' (default), 'last', or path relative to the run dir.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List available golden_mid runs and exit.",
    )
    # ByteTrack thresholds
    parser.add_argument(
        "--track-high-thresh",
        type=float,
        default=0.35,
        help="High-confidence threshold: detections above this are used in stage-1 association.",
    )
    parser.add_argument(
        "--track-low-thresh",
        type=float,
        default=0.10,
        help=(
            "Low-confidence threshold: detections between low and high thresholds "
            "get a second-pass association attempt with unmatched tracks."
        ),
    )
    parser.add_argument(
        "--new-track-thresh",
        type=float,
        default=0.50,
        help="Minimum detection score to initialise a new track.",
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
        help="IoU threshold for the Hungarian assignment in ByteTrack.",
    )
    # Scene selection
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Splits to process (default: all found).")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Restrict to specific scene names.")
    parser.add_argument("--exclude", nargs="+", default=None, metavar="PATTERN",
                        help="Glob patterns for scene names to skip (e.g. 'acs_s1_*').")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process already labelled scenes.")
    args = parser.parse_args()

    if args.list_runs:
        list_golden_runs(args.golden_runs_dir)
        return 0

    if not args.raw_frames_dir.exists():
        print(f"ERROR: raw frames directory not found: {args.raw_frames_dir}", file=sys.stderr)
        return 1

    if args.detector == "golden":
        from ultralytics import YOLO
        if not args.golden_run:
            args.golden_run = find_latest_run(args.golden_runs_dir)
        if not args.golden_run:
            print(
                "ERROR: no runs found in {args.golden_runs_dir}. "
                "Train a model first or use --detector yolo.",
                file=sys.stderr,
            )
            return 1
        model, auto_imgsz = load_golden_run(
            args.golden_runs_dir, args.golden_run, args.golden_checkpoint
        )
        imgsz = args.imgsz or auto_imgsz
        print(
            f"Detector: Golden run '{args.golden_run}' "
            f"checkpoint='{args.golden_checkpoint}', imgsz={imgsz}"
        )
    elif args.detector == "yolo-world":
        from ultralytics import YOLOWorld
        imgsz = args.imgsz or 640
        model = YOLOWorld(args.yolo_world_model)
        model.set_classes(["person"])
        print(f"Detector: YOLO-World ({args.yolo_world_model}), imgsz={imgsz}")
    else:
        from ultralytics import YOLO
        imgsz = args.imgsz or 1280
        print(f"Detector: YOLO ({args.yolo_model}), imgsz={imgsz}")
        model = YOLO(args.yolo_model)

    tracker_cfg = write_tracker_config(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
    )
    print(
        f"ByteTrack: high={args.track_high_thresh}, low={args.track_low_thresh}, "
        f"new={args.new_track_thresh}, buffer={args.track_buffer}, match={args.match_thresh}"
    )

    splits = sorted(d for d in args.raw_frames_dir.iterdir() if d.is_dir())
    if args.splits:
        splits = [s for s in splits if s.name in set(args.splits)]
    if not splits:
        print("No splits found.", file=sys.stderr)
        return 1

    total_scenes = total_tracks = total_rows = 0

    for split_dir in splits:
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
            out_scene = args.output_dir / split_dir.name / scene_dir.name
            gt_file = out_scene / "gt" / "gt.txt"
            if gt_file.exists() and not args.overwrite:
                tqdm.write(f"  skip (exists): {scene_dir.name}")
                continue

            stats = run_scene(
                scene_dir=scene_dir,
                output_dir=out_scene,
                model=model,
                tracker_cfg=tracker_cfg,
                track_low_thresh=args.track_low_thresh,
                imgsz=imgsz,
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

    print(f"\n{'─'*60}")
    print(f"Done.  Scenes: {total_scenes}  |  Unique tracks: {total_tracks}  |  Annotations: {total_rows}")
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
