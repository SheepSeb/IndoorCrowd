#!/usr/bin/env python3
"""
Build a golden dataset by selecting up to N frames overall.

Default behavior:
- Input scenes:  dataset/raw_frames_5fps
- Output scenes: dataset/golden_frames_5fps_mid
- Label inputs:
  - dataset/main/labels_sam3
- Label outputs:
  - dataset/main/labels_sam3_golden_mid
- Max frames:    600 (overall cap across all scenes)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_input = repo_root / "dataset" / "main" / "raw_frames_5fps"
    default_output = repo_root / "dataset" / "main" / "golden_frames_5fps_mid"
    default_label_inputs = [
        repo_root / "dataset" / "main" / "labels_sam3",
    ]
    default_label_outputs = [
        repo_root / "dataset" / "main" / "labels_sam3_golden_mid",
    ]

    parser = argparse.ArgumentParser(
        description=(
            "Create a golden dataset from scene folders by taking up to a maximum "
            "number of frames overall, centered toward each scene middle."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input root containing scene folders (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output root for selected golden frames (default: {default_output})",
    )
    parser.add_argument(
        "--label-inputs",
        type=Path,
        nargs="*",
        default=default_label_inputs,
        help=(
            "Input label roots to filter/copy for selected frames "
            "(default: labels_sam3)"
        ),
    )
    parser.add_argument(
        "--label-outputs",
        type=Path,
        nargs="*",
        default=default_label_outputs,
        help=(
            "Output roots for filtered labels, same order as --label-inputs "
            "(default: labels_sam3_golden_mid)"
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Maximum frames to keep overall across all scenes (default: 600)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete destination output folders before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be selected without copying files.",
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_scene_dirs(input_root: Path) -> list[Path]:
    """
    A scene dir is any directory that directly contains image files.
    """
    scenes: list[Path] = []
    for directory in sorted(input_root.rglob("*")):
        if not directory.is_dir():
            continue
        if any(is_image(child) for child in directory.iterdir()):
            scenes.append(directory)
    return scenes


def middle_priority_order(files: list[Path]) -> list[Path]:
    """
    Return frames ordered from middle-out (center first, then expanding).
    """
    if not files:
        return []
    middle = (len(files) - 1) / 2.0
    ordered_indices = sorted(
        range(len(files)),
        key=lambda index: (abs(index - middle), index),
    )
    return [files[index] for index in ordered_indices]


def select_frames_overall(
    scene_dirs: list[Path],
    scene_frames: dict[Path, list[Path]],
    max_frames: int,
) -> dict[Path, list[Path]]:
    """
    Select up to max_frames globally, balancing scenes in round-robin order,
    while each scene contributes frames from its middle first.
    """
    prioritized = {
        scene_dir: middle_priority_order(scene_frames[scene_dir])
        for scene_dir in scene_dirs
    }
    selected: dict[Path, list[Path]] = {scene_dir: [] for scene_dir in scene_dirs}

    total_selected = 0
    while total_selected < max_frames:
        selected_in_round = 0
        for scene_dir in scene_dirs:
            scene_selected = selected[scene_dir]
            scene_prioritized = prioritized[scene_dir]
            next_index = len(scene_selected)
            if next_index >= len(scene_prioritized):
                continue

            scene_selected.append(scene_prioritized[next_index])
            total_selected += 1
            selected_in_round += 1

            if total_selected >= max_frames:
                break

        if selected_in_round == 0:
            break

    return selected


def copy_selected_frames(
    selected_frames: list[Path],
    scene_dir: Path,
    input_root: Path,
    output_root: Path,
    dry_run: bool,
) -> int:
    relative_scene = scene_dir.relative_to(input_root)
    destination_scene = output_root / relative_scene

    if not dry_run:
        destination_scene.mkdir(parents=True, exist_ok=True)

    copied = 0
    for frame_path in selected_frames:
        destination = destination_scene / frame_path.name
        if dry_run:
            copied += 1
            continue
        shutil.copy2(frame_path, destination)
        copied += 1
    return copied


def filter_coco_for_selected_frames(
    label_data: dict,
    selected_relpaths: set[str],
) -> dict:
    images = label_data.get("images", [])
    annotations = label_data.get("annotations", [])

    selected_images = []
    selected_image_ids = set()
    for image in images:
        file_name = image.get("file_name")
        if isinstance(file_name, str) and file_name in selected_relpaths:
            selected_images.append(image)
            selected_image_ids.add(image.get("id"))

    selected_annotations = [
        annotation
        for annotation in annotations
        if annotation.get("image_id") in selected_image_ids
    ]

    filtered = dict(label_data)
    filtered["images"] = selected_images
    filtered["annotations"] = selected_annotations
    return filtered


def copy_filtered_scene_labels(
    selected_frames: list[Path],
    scene_dir: Path,
    input_root: Path,
    label_inputs: list[Path],
    label_outputs: list[Path],
    dry_run: bool,
) -> tuple[int, int]:
    """
    Copy scene-level label JSONs and keep only selected-frame annotations.

    Returns:
        (label_files_written, missing_label_files)
    """
    relative_scene = scene_dir.relative_to(input_root)
    selected_relpaths = {
        str((relative_scene / frame_path.name).as_posix())
        for frame_path in selected_frames
    }

    written = 0
    missing = 0
    label_relpath = relative_scene.parent / f"{relative_scene.name}.json"

    for label_input_root, label_output_root in zip(label_inputs, label_outputs):
        source_label = label_input_root / label_relpath
        destination_label = label_output_root / label_relpath

        if not source_label.is_file():
            missing += 1
            print(f"  ! Missing label file: {source_label}", file=sys.stderr)
            continue

        if dry_run:
            written += 1
            continue

        with source_label.open("r", encoding="utf-8") as source_file:
            label_data = json.load(source_file)
        filtered = filter_coco_for_selected_frames(label_data, selected_relpaths)

        destination_label.parent.mkdir(parents=True, exist_ok=True)
        with destination_label.open("w", encoding="utf-8") as destination_file:
            json.dump(filtered, destination_file, ensure_ascii=False, indent=2)
            destination_file.write("\n")
        written += 1

    return written, missing


def main() -> int:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    label_inputs = [path.resolve() for path in args.label_inputs]
    label_outputs = [path.resolve() for path in args.label_outputs]

    if args.max_frames <= 0:
        print("--max-frames must be > 0", file=sys.stderr)
        return 1
    if len(label_inputs) != len(label_outputs):
        print(
            "--label-inputs and --label-outputs must have the same number of paths",
            file=sys.stderr,
        )
        return 1
    if not input_root.is_dir():
        print(f"Input directory not found: {input_root}", file=sys.stderr)
        return 1
    for label_input in label_inputs:
        if not label_input.is_dir():
            print(f"Label input directory not found: {label_input}", file=sys.stderr)
            return 1

    scene_dirs = collect_scene_dirs(input_root)
    if not scene_dirs:
        print(f"No scene folders with images found in: {input_root}", file=sys.stderr)
        return 1

    scene_frames = {
        scene_dir: sorted(child for child in scene_dir.iterdir() if is_image(child))
        for scene_dir in scene_dirs
    }
    total_available = sum(len(frames) for frames in scene_frames.values())
    selections = select_frames_overall(
        scene_dirs=scene_dirs,
        scene_frames=scene_frames,
        max_frames=args.max_frames,
    )

    if args.overwrite and output_root.exists() and not args.dry_run:
        shutil.rmtree(output_root)
    if args.overwrite and not args.dry_run:
        for label_output in label_outputs:
            if label_output.exists():
                shutil.rmtree(label_output)

    print(f"Found {len(scene_dirs)} scene folders in {input_root}")
    print(
        f"Selecting up to {args.max_frames} frames overall "
        f"(available: {total_available})"
    )
    if args.dry_run:
        print("Dry run enabled (no files will be copied).")
    else:
        print(f"Writing golden dataset to {output_root}")
        print("Writing filtered labels to:")
        for label_output in label_outputs:
            print(f"  - {label_output}")

    total_copied = 0
    total_label_files = 0
    total_missing_labels = 0

    for scene_dir in scene_dirs:
        frames = scene_frames[scene_dir]
        selected = selections[scene_dir]

        copied = copy_selected_frames(
            selected_frames=selected,
            scene_dir=scene_dir,
            input_root=input_root,
            output_root=output_root,
            dry_run=args.dry_run,
        )
        total_copied += copied

        label_written, label_missing = copy_filtered_scene_labels(
            selected_frames=selected,
            scene_dir=scene_dir,
            input_root=input_root,
            label_inputs=label_inputs,
            label_outputs=label_outputs,
            dry_run=args.dry_run,
        )
        total_label_files += label_written
        total_missing_labels += label_missing

        relative_scene = scene_dir.relative_to(input_root)
        print(
            f"- {relative_scene}: selected {len(selected)} / {len(frames)} "
            f"(middle-centered), labels written: {label_written}"
        )

    print("\nDone.")
    print(f"Scenes processed: {len(scene_dirs)}")
    print(f"Frames selected: {total_copied}")
    print(f"Label files written: {total_label_files}")
    if total_missing_labels:
        print(f"Missing label files: {total_missing_labels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
