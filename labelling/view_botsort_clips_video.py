#!/usr/bin/env python3
"""Play BoT-SORT MOT clips as videos with overlaid tracks.

Understands the clip layout produced in `dataset/MOT_labels_sam3_botsort/clips`:

    {clips_dir}/{split}/{scene}/{clip_name}/
        frames/        ← JPEG frames
        gt/gt.txt      ← MOT rows: frame,id,x,y,w,h,conf,...
        seqinfo.ini    ← (optional, not required here)

Usage
-----
Interactive viewer (requires GUI-enabled OpenCV):
    uv run python labelling/view_botsort_clips_video.py

Filter by split / scene:
    uv run python labelling/view_botsort_clips_video.py --split train
    uv run python labelling/view_botsort_clips_video.py --scene acs_s1_recording_2026-02-23_17-31-29

Export MP4s (works in headless / non-GUI envs):
    uv run python labelling/view_botsort_clips_video.py --export-dir out/botsort_clips_mp4

Controls (interactive mode)
---------------------------
    Space : pause / resume
    n     : next clip
    b     : step one frame back
    q/ESC : quit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PALETTE: list[tuple[int, int, int]] = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (128, 0, 0),
    (170, 255, 195),
    (0, 0, 128),
]


def track_color(track_id: int) -> tuple[int, int, int]:
    return PALETTE[int(track_id) % len(PALETTE)]


def frame_number_from_name(file_name: str) -> int:
    """Extract numeric frame index from a filename like 'frame_00123.jpg'."""
    stem = Path(file_name).stem
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else 0


@dataclass
class Clip:
    split: str
    scene: str
    name: str
    frames: List[Path]
    labels_by_frame: Dict[int, List[dict]]

    @property
    def display_name(self) -> str:
        return f"{self.split}/{self.scene}/{self.name}"


def parse_mot_file(gt_file: Path) -> Dict[int, List[dict]]:
    """Parse MOT gt.txt into {frame_number: [rows...]}."""
    by_frame: Dict[int, List[dict]] = {}
    if not gt_file.exists():
        return by_frame
    for line in gt_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame = int(parts[0])
        tid = int(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        conf = float(parts[6]) if len(parts) > 6 else 1.0
        row = {"id": tid, "x": x, "y": y, "w": w, "h": h, "conf": conf}
        by_frame.setdefault(frame, []).append(row)
    return by_frame


def discover_clips(
    clips_dir: Path,
    split_filter: str | None,
    scene_filter: str | None,
) -> list[Clip]:
    clips: list[Clip] = []
    if not clips_dir.exists():
        return clips

    for split_dir in sorted(clips_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        if split_filter and split != split_filter:
            continue

        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene = scene_dir.name
            if scene_filter and scene_filter not in scene:
                continue

            for clip_dir in sorted(scene_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                frames_dir = clip_dir / "frames"
                gt_file = clip_dir / "gt" / "gt.txt"
                if not frames_dir.is_dir() or not gt_file.exists():
                    continue

                frame_paths = sorted(
                    p
                    for p in frames_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
                if not frame_paths:
                    continue

                labels_by_frame = parse_mot_file(gt_file)
                clips.append(
                    Clip(
                        split=split,
                        scene=scene,
                        name=clip_dir.name,
                        frames=frame_paths,
                        labels_by_frame=labels_by_frame,
                    )
                )

    return clips


def draw_overlays(img, rows: list[dict]) -> None:
    for r in rows:
        tid = r["id"]
        x, y = int(r["x"]), int(r["y"])
        w, h = int(r["w"]), int(r["h"])
        color = track_color(tid)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"ID {tid}"
        (tw, th), bl = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        ly = max(y, th + bl + 4)
        cv2.rectangle(
            img,
            (x, ly - th - bl - 4),
            (x + tw + 6, ly),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x + 3, ly - bl - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def play_clips(clips: list[Clip], fps: float) -> None:
    if not clips:
        print("No clips found.")
        return

    # Require GUI-capable OpenCV
    if not (
        hasattr(cv2, "namedWindow")
        and hasattr(cv2, "imshow")
        and hasattr(cv2, "waitKey")
    ):
        raise RuntimeError(
            "This OpenCV build has no GUI support (namedWindow / imshow missing).\n"
            "Either install a GUI-enabled OpenCV (opencv-python), or run this script "
            "with `--export-dir <out_dir>` to export MP4 files instead."
        )

    delay_ms = max(1, int(1000 / max(fps, 1e-3)))

    print(
        "Controls: [Space]=pause/resume  [n]=next clip  [b]=back one frame  [q/ESC]=quit"
    )

    cv2.namedWindow("BoT-SORT Clips", cv2.WINDOW_NORMAL)

    for clip_idx, clip in enumerate(clips):
        print(
            f"\nPlaying clip {clip_idx + 1}/{len(clips)}: {clip.display_name} "
            f"({len(clip.frames)} frames)"
        )
        i = 0
        paused = False

        while 0 <= i < len(clip.frames):
            frame_path = clip.frames[i]
            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"Failed to read frame: {frame_path}")
                i += 1
                continue

            fnum = frame_number_from_name(frame_path.name)
            rows = clip.labels_by_frame.get(fnum, [])
            if rows:
                draw_overlays(img, rows)

            title = f"{clip.display_name}  [{i + 1}/{len(clip.frames)}]  frame#{fnum}"
            cv2.setWindowTitle("BoT-SORT Clips", title)
            cv2.imshow("BoT-SORT Clips", img)

            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            if key == ord(" "):
                paused = not paused
                continue
            if key == ord("n"):
                break
            if key == ord("b"):
                i = max(0, i - 1)
                continue
            if not paused:
                i += 1

    cv2.destroyAllWindows()


def export_clips(clips: list[Clip], fps: float, export_dir: Path) -> None:
    if not clips:
        print("No clips found.")
        return

    export_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for clip in clips:
        # Build a safe filename: split_scene_clipname.mp4
        base = f"{clip.split}_{clip.scene}_{clip.name}".replace("/", "_")
        out_path = export_dir / f"{base}.mp4"
        print(f"Exporting {clip.display_name} → {out_path}")

        writer = None
        for frame_path in clip.frames:
            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"  Skipping unreadable frame: {frame_path}")
                continue

            fnum = frame_number_from_name(frame_path.name)
            rows = clip.labels_by_frame.get(fnum, [])
            if rows:
                draw_overlays(img, rows)

            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path),
                    fourcc,
                    fps,
                    (w, h),
                )
                if not writer.isOpened():
                    print(f"  ERROR: failed to open VideoWriter for {out_path}")
                    writer = None
                    break

            writer.write(img)

        if writer is not None:
            writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play BoT-SORT MOT clips as videos with track overlays."
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("dataset/MOT_labels_sam3_botsort/clips"),
        help="Root clips directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split filter (e.g. train, test, challange).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Optional substring filter for scene name.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Playback frames per second.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="If set, export each clip to an MP4 instead of opening a GUI window.",
    )
    args = parser.parse_args()

    clips = discover_clips(args.clips_dir, args.split, args.scene)
    print(f"Found {len(clips)} clip(s) under {args.clips_dir}")
    if args.export_dir is not None:
        export_clips(clips, fps=args.fps, export_dir=args.export_dir)
    else:
        try:
            play_clips(clips, fps=args.fps)
        except RuntimeError as exc:
            print(str(exc))
            print(
                "Hint: re-run with `--export-dir <out_dir>` to generate MP4 files."
            )


if __name__ == "__main__":
    main()

