#!/usr/bin/env python3
"""
Analyse raw videos for data volume and frame counts at different sample rates (e.g. 1, 5, 10, 25 fps).
"""

import argparse
import sys
from pathlib import Path

import cv2


DEFAULT_SAMPLE_RATES = (1, 5, 10, 15, 25)

# Default folder to scan when no paths given (project root / dataset)
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


def get_video_info(path: Path) -> dict | None:
    """Open video and return basic properties. Returns None if open fails."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration_s = total_frames / fps if fps > 0 else 0.0
    return {
        "path": path,
        "total_frames": total_frames,
        "fps": fps,
        "duration_s": duration_s,
        "width": width,
        "height": height,
    }


def sampled_frame_count(total_frames: int, video_fps: float, target_fps: float) -> int:
    """
    Number of frames we keep when sampling at target_fps from a video with video_fps.
    We take every Nth frame where N = round(video_fps / target_fps).
    """
    if video_fps <= 0 or target_fps <= 0:
        return 0
    step = max(1, round(video_fps / target_fps))
    return (total_frames + step - 1) // step


def analyse_at_sample_rates(
    total_frames: int,
    video_fps: float,
    duration_s: float,
    sample_rates: tuple[int, ...],
) -> list[dict]:
    """For each sample rate, compute number of frames and effective duration."""
    results = []
    for target_fps in sample_rates:
        n = sampled_frame_count(total_frames, video_fps, target_fps)
        effective_duration = n / target_fps if target_fps > 0 else 0.0
        step = max(1, round(video_fps / target_fps)) if video_fps > 0 and target_fps > 0 else 1
        results.append({
            "target_fps": target_fps,
            "sampled_frames": n,
            "effective_duration_s": effective_duration,
            "frame_step": step,
        })
    return results


def _should_skip_dir(dir_name: str) -> bool:
    """True if this directory should be ignored (uncut or z_*)."""
    return dir_name == "uncut" or dir_name.startswith("z_")


def collect_video_paths(paths: list[Path]) -> list[Path]:
    """Expand paths to a list of .mp4 files. For directories, recurse into subfolders
    but ignore folders named 'uncut' and 'z_*'."""
    out = []
    for p in paths:
        p = p.resolve()
        if p.is_file():
            if p.suffix.lower() == ".mp4":
                out.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*.mp4")):
                try:
                    rel = f.relative_to(p)
                except ValueError:
                    continue
                # Skip if any parent directory is uncut or z_*
                if len(rel.parts) > 1 and any(
                    _should_skip_dir(part) for part in rel.parts[:-1]
                ):
                    continue
                out.append(f)
    return out


def run_analysis(
    video_paths: list[Path],
    sample_rates: tuple[int, ...] = DEFAULT_SAMPLE_RATES,
) -> list[dict]:
    """Run analysis for each video and each sample rate. Returns list of per-video results."""
    all_results = []
    for path in video_paths:
        info = get_video_info(path)
        if info is None:
            all_results.append({"path": path, "error": "Could not open video"})
            continue
        rates = analyse_at_sample_rates(
            info["total_frames"],
            info["fps"],
            info["duration_s"],
            sample_rates,
        )
        all_results.append({
            "path": path,
            "total_frames": info["total_frames"],
            "fps": info["fps"],
            "duration_s": info["duration_s"],
            "width": info["width"],
            "height": info["height"],
            "sample_rates": rates,
        })
    return all_results


def compute_overall(results: list[dict]) -> dict | None:
    """Aggregate totals across all successfully analysed videos. Returns None if no valid results."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None
    total_frames = sum(r["total_frames"] for r in valid)
    total_duration_s = sum(r["duration_s"] for r in valid)
    by_fps = {}
    for r in valid:
        for sr in r["sample_rates"]:
            fps = sr["target_fps"]
            if fps not in by_fps:
                by_fps[fps] = {"sampled_frames": 0, "effective_duration_s": 0.0}
            by_fps[fps]["sampled_frames"] += sr["sampled_frames"]
            by_fps[fps]["effective_duration_s"] += sr["effective_duration_s"]
    return {
        "video_count": len(valid),
        "total_frames": total_frames,
        "total_duration_s": total_duration_s,
        "by_fps": by_fps,
    }


def print_report(results: list[dict], sample_rates: tuple[int, ...]) -> None:
    """Print a human-readable report to stdout."""
    for r in results:
        if "error" in r:
            print(f"\n{r['path']}: {r['error']}")
            continue
        print(f"\n{r['path']}")
        print(f"  Original: {r['total_frames']} frames @ {r['fps']:.2f} fps, {r['duration_s']:.2f}s, {r['width']}x{r['height']}")
        print("  At sample rate:")
        for sr in r["sample_rates"]:
            print(
                f"    {sr['target_fps']} fps -> {sr['sampled_frames']} frames "
                f"(step={sr['frame_step']}), ~{sr['effective_duration_s']:.2f}s"
            )
    overall = compute_overall(results)
    if overall is not None:
        print("\n" + "=" * 60)
        print("OVERALL (all videos)")
        print(f"  Videos: {overall['video_count']}")
        print(f"  Original total: {overall['total_frames']} frames, {overall['total_duration_s']:.2f}s")
        print("  At sample rate:")
        for fps in sample_rates:
            if fps in overall["by_fps"]:
                b = overall["by_fps"][fps]
                print(f"    {fps} fps -> {b['sampled_frames']} frames, ~{b['effective_duration_s']:.2f}s")
        print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse raw videos for data volume at different sample rates (1, 5, 10, 25 fps)."
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        default=[DATASET_DIR],
        metavar="PATH",
        help=f"Folder(s) to scan for .mp4 (recursive; ignores 'uncut' and 'z_*' dirs), or video file(s). Default: {DATASET_DIR}",
    )
    parser.add_argument(
        "--fps",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_RATES),
        metavar="FPS",
        help=f"Sample rates in fps (default: {' '.join(map(str, DEFAULT_SAMPLE_RATES))})",
    )
    args = parser.parse_args()
    sample_rates = tuple(sorted(set(args.fps)))
    if not sample_rates or any(f <= 0 for f in sample_rates):
        print("Invalid --fps; use positive integers.", file=sys.stderr)
        return 1
    video_paths = collect_video_paths(args.paths)
    if not video_paths:
        if args.paths == [DATASET_DIR] and not DATASET_DIR.is_dir():
            print(f"Default dataset folder not found: {DATASET_DIR}", file=sys.stderr)
        else:
            print("No .mp4 files found (folders 'uncut' and 'z_*' are skipped).", file=sys.stderr)
        return 1
    results = run_analysis(video_paths, sample_rates)
    print_report(results, sample_rates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
