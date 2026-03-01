#!/usr/bin/env python3
"""
Sample frames from videos at a fixed FPS and save them as JPEG images.

Default behavior:
- Input videos:  dataset/raw_video
- Output frames: dataset/raw_frames
- Sample rate:   5 fps
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def collect_videos(input_dir: Path) -> list[Path]:
    """Collect supported video files recursively."""
    videos: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    return videos


def frame_should_be_kept(frame_index: int, source_fps: float, target_fps: float) -> bool:
    """Select frame by timestamp to approximate a fixed target FPS."""
    if source_fps <= 0 or target_fps <= 0:
        return False
    return int(frame_index * target_fps / source_fps) != int((frame_index - 1) * target_fps / source_fps)


def extract_video_frames(
    video_path: Path,
    input_root: Path,
    output_root: Path,
    target_fps: float,
    overwrite: bool,
) -> tuple[int, int]:
    """
    Extract frames from a single video.

    Returns:
        (saved_frames, total_frames_read)
    """
    relative_video = video_path.relative_to(input_root)
    destination_dir = output_root / relative_video.parent / video_path.stem

    if overwrite and destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    saved_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1

        if frame_should_be_kept(frame_index, source_fps, target_fps):
            saved_index += 1
            output_file = destination_dir / f"frame_{saved_index:05d}.jpg"
            cv2.imwrite(str(output_file), frame)

    capture.release()
    return saved_index, frame_index


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "raw_video"
    default_output = script_dir / "raw_frames"

    parser = argparse.ArgumentParser(
        description="Sample frames from videos and save them as JPEG images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input root directory containing videos (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output root directory for sampled frames (default: {default_output})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target frame sampling rate in fps (default: 5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output folders for each video.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    target_fps = args.fps

    if target_fps <= 0:
        print("--fps must be > 0", file=sys.stderr)
        return 1
    if not input_root.is_dir():
        print(f"Input directory not found: {input_root}", file=sys.stderr)
        return 1

    videos = collect_videos(input_root)
    if not videos:
        print(f"No supported video files found in: {input_root}", file=sys.stderr)
        return 1

    print(f"Found {len(videos)} videos in {input_root}")
    print(f"Sampling at {target_fps:g} fps -> {output_root}")

    total_saved = 0
    total_read = 0
    failures = 0

    for video_path in videos:
        relative_video = video_path.relative_to(input_root)
        print(f"\nProcessing: {relative_video}")
        try:
            saved, read = extract_video_frames(
                video_path=video_path,
                input_root=input_root,
                output_root=output_root,
                target_fps=target_fps,
                overwrite=args.overwrite,
            )
            total_saved += saved
            total_read += read
            print(f"  Saved {saved} / {read} frames")
        except RuntimeError as exc:
            failures += 1
            print(f"  ERROR: {exc}", file=sys.stderr)

    print("\nDone.")
    print(f"Videos processed: {len(videos) - failures}/{len(videos)}")
    print(f"Frames saved: {total_saved}")
    print(f"Frames read: {total_read}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
