#!/usr/bin/env python3
"""Merge per-part SAM3_video_tracks outputs into one file per scene.

Each scene in SAM3_video_tracks/{split}/{scene}/ contains part_001/, part_002/, …
Each part has:
  frames/        — symlinks to source frames (file name encodes global frame number)
  gt/gt.txt      — MOT format with *local* 1-based frame numbers
  annotations.json — COCO with local image IDs
  seqinfo.ini

This script merges all parts of a scene into:
  {scene}/gt/gt.txt        — MOT with global frame numbers, non-colliding track IDs
  {scene}/seqinfo.ini      — covers full scene
  {scene}/annotations.json — merged COCO with global image IDs

Track IDs are offset per part to avoid collisions (part 1 keeps its IDs, part 2
IDs are shifted by max_part1_id + 1, etc.).  Cross-part track linking is NOT
performed; that is a separate step.

Usage
-----
  # Merge all splits and scenes
  uv run python labelling/merge_sam3_parts.py

  # Specific split / scene
  uv run python labelling/merge_sam3_parts.py --splits challange --scenes acs_s1_recording_2026-02-23_18-06-04

  # Overwrite existing merged output
  uv run python labelling/merge_sam3_parts.py --overwrite
"""

from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path

from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def frame_number_from_name(name: str) -> int:
    digits = "".join(c for c in Path(name).stem if c.isdigit())
    return int(digits) if digits else 0


def load_part_frame_map(part_dir: Path) -> dict[int, tuple[int, str]]:
    """
    Return {local_frame_idx: (global_frame_number, filename)} for a part.

    The frames/ directory contains symlinks whose names encode the global frame
    number (e.g. frame_00056.jpg → global frame 56).  They are sorted and
    assigned local 1-based indices matching SAM3's output.
    """
    frames_dir = part_dir / "frames"
    if not frames_dir.exists():
        return {}
    frame_files = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return {
        local_idx: (frame_number_from_name(p.name), p.name)
        for local_idx, p in enumerate(frame_files, start=1)
    }


def merge_scene(scene_dir: Path, overwrite: bool) -> dict:
    """Merge all part_* subdirectories of scene_dir into scene-level files."""

    gt_out = scene_dir / "gt" / "gt.txt"
    if gt_out.exists() and not overwrite:
        return {"scene": scene_dir.name, "skipped": True}

    part_dirs = sorted(
        d for d in scene_dir.iterdir()
        if d.is_dir() and d.name.startswith("part_")
    )
    if not part_dirs:
        return {"scene": scene_dir.name, "skipped": True, "reason": "no parts found"}

    merged_mot: list[str] = []
    merged_images: list[dict] = []
    merged_anns: list[dict] = []

    global_image_id = 1
    global_ann_id = 1
    track_id_offset = 0
    total_frames = 0
    img_w = img_h = 0
    imext = ".jpg"
    imdir = ""
    framerate = 3

    for part_dir in part_dirs:
        frame_map = load_part_frame_map(part_dir)
        if not frame_map:
            continue

        # Read seqinfo for metadata
        seqinfo_path = part_dir / "seqinfo.ini"
        if seqinfo_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(seqinfo_path)
            seq = cfg["Sequence"]
            img_w = int(seq.get("imwidth", img_w))
            img_h = int(seq.get("imheight", img_h))
            imext = seq.get("imext", imext)
            framerate = int(seq.get("framerate", framerate))
            if not imdir:
                # Point imDir to the raw frames directory by walking up from the symlink target
                sample_link = part_dir / "frames" / list(sorted(
                    p for p in (part_dir / "frames").iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS
                ))[0].name
                imdir = str(sample_link.resolve().parent)

        # Compute max track ID seen so far in this part (for next offset)
        part_track_ids: set[int] = set()

        # --- MOT gt.txt ---
        gt_path = part_dir / "gt" / "gt.txt"
        if gt_path.exists():
            for line in gt_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                local_fnum = int(parts[0])
                local_tid = int(parts[1])
                rest = ",".join(parts[2:])

                info = frame_map.get(local_fnum)
                if info is None:
                    continue
                global_fnum, _ = info
                global_tid = local_tid + track_id_offset
                part_track_ids.add(local_tid)

                merged_mot.append(f"{global_fnum},{global_tid},{rest}")

        # --- annotations.json ---
        ann_path = part_dir / "annotations.json"
        if ann_path.exists():
            coco = json.loads(ann_path.read_text())

            # Build local image_id → global frame number
            local_img_id_to_global: dict[int, int] = {}
            for img in coco.get("images", []):
                local_fnum = img["id"]
                info = frame_map.get(local_fnum)
                if info is None:
                    continue
                global_fnum, fname = info
                local_img_id_to_global[local_fnum] = global_image_id
                merged_images.append({
                    "id": global_image_id,
                    "file_name": fname,
                    "width": img.get("width", img_w),
                    "height": img.get("height", img_h),
                    "source_path": img.get("source_path", ""),
                })
                global_image_id += 1

            for ann in coco.get("annotations", []):
                mapped_img_id = local_img_id_to_global.get(ann["image_id"])
                if mapped_img_id is None:
                    continue
                local_tid = ann.get("track_id", 0)
                part_track_ids.add(local_tid)
                merged_anns.append({
                    **ann,
                    "id": global_ann_id,
                    "image_id": mapped_img_id,
                    "track_id": local_tid + track_id_offset,
                })
                global_ann_id += 1

        total_frames += len(frame_map)
        max_local_tid = max(part_track_ids, default=-1)
        track_id_offset += max_local_tid + 1

    if not merged_mot and not merged_images:
        return {"scene": scene_dir.name, "skipped": True, "reason": "no data"}

    # Sort MOT rows by (frame, track_id)
    merged_mot.sort(key=lambda s: (int(s.split(",")[0]), int(s.split(",")[1])))

    # Write gt.txt
    gt_dir = scene_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_out.write_text("\n".join(merged_mot) + ("\n" if merged_mot else ""))

    # Write seqinfo.ini
    seq_cfg = configparser.ConfigParser()
    seq_cfg["Sequence"] = {
        "name": scene_dir.name,
        "imDir": imdir,
        "frameRate": str(framerate),
        "seqLength": str(total_frames),
        "imWidth": str(img_w),
        "imHeight": str(img_h),
        "imExt": imext,
    }
    with open(scene_dir / "seqinfo.ini", "w") as f:
        seq_cfg.write(f)

    # Write annotations.json
    categories = [{"id": 1, "name": "person", "supercategory": "person"}]
    with open(scene_dir / "annotations.json", "w") as f:
        json.dump(
            {"images": merged_images, "annotations": merged_anns, "categories": categories},
            f,
            indent=2,
        )

    unique_tracks = len({int(r.split(",")[1]) for r in merged_mot})
    return {
        "scene": scene_dir.name,
        "skipped": False,
        "frames": total_frames,
        "parts": len(part_dirs),
        "tracks": unique_tracks,
        "rows": len(merged_mot),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge per-part SAM3_video_tracks into single scene-level files"
    )
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=Path("dataset/SAM3_video_tracks"),
    )
    parser.add_argument("--splits",  nargs="+", default=None)
    parser.add_argument("--scenes",  nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.tracks_dir.exists():
        print(f"ERROR: not found: {args.tracks_dir}", file=sys.stderr)
        return 1

    split_dirs = sorted(d for d in args.tracks_dir.iterdir() if d.is_dir())
    if args.splits:
        split_dirs = [s for s in split_dirs if s.name in set(args.splits)]

    total_scenes = 0
    for split_dir in split_dirs:
        scene_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        if args.scenes:
            scene_dirs = [s for s in scene_dirs if s.name in set(args.scenes)]

        print(f"\n── {split_dir.name} ({len(scene_dirs)} scenes) ──")

        for scene_dir in tqdm(scene_dirs, desc=split_dir.name, unit="scene"):
            result = merge_scene(scene_dir, overwrite=args.overwrite)
            if result.get("skipped"):
                reason = result.get("reason", "exists")
                tqdm.write(f"  skip ({reason}): {result['scene']}")
            else:
                tqdm.write(
                    f"  {result['scene']}: "
                    f"{result['parts']} parts → "
                    f"{result['frames']} frames | "
                    f"{result['tracks']} tracks | "
                    f"{result['rows']} annotations"
                )
                total_scenes += 1

    print(f"\nDone. Merged {total_scenes} scene(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
