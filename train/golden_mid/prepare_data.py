"""
Convert dataset/golden_frames_3fps_mid/seb.json → YOLO format.

Creates two dataset trees under yolo_dataset/:
  detect/   – bounding-box labels  (class cx cy w h)
  segment/  – polygon-mask labels  (class x1 y1 x2 y2 …)

Split mapping  (from seb.json path prefix):
  train/ + challange/  →  YOLO train
  test/                →  YOLO val

Images are symlinked (no copies).  Run once; re-run is idempotent.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import yaml

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "dataset" / "golden_frames_3fps_mid"
SEBE_JSON = GOLDEN_DIR / "seb.json"
OUT_DIR = Path(__file__).parent / "yolo_dataset"

# ── helpers ──────────────────────────────────────────────────────────────────

def rle_to_polygon(rle: dict, min_points: int = 6) -> list[list[float]] | None:
    """Decode COCO RLE mask → list of flattened polygon contours (or None)."""
    mask = mask_util.decode(rle).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        c = c.squeeze(1)        # (N, 2)
        if len(c) < min_points // 2:
            continue
        polys.append(c.flatten().tolist())
    return polys if polys else None


def bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> str:
    """COCO xywh → YOLO cx cy w h (normalised), class 0."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    return f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}"


def poly_to_yolo(points: list[float], img_w: int, img_h: int) -> str:
    """Flat [x,y,x,y,...] → YOLO seg line (class + normalised coords)."""
    normed = []
    for i, v in enumerate(points):
        normed.append(v / (img_w if i % 2 == 0 else img_h))
    coord_str = " ".join(f"{v:.6f}" for v in normed)
    return f"0 {coord_str}"


def symlink_force(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(SEBE_JSON) as f:
        coco = json.load(f)

    # index
    id2img: dict[int, dict] = {img["id"]: img for img in coco["images"]}
    id2anns: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        id2anns[ann["image_id"]].append(ann)

    # split: path prefix → yolo split name
    def yolo_split(file_name: str) -> str:
        prefix = file_name.split("/")[0]
        return "train" if prefix in ("train", "challange") else "val"

    # counters
    stats = {"train": 0, "val": 0, "skipped_seg": 0}

    for img_info in coco["images"]:
        img_id = img_info["id"]
        rel_path = img_info["file_name"]            # e.g. train/scene/frame.jpg
        img_w, img_h = img_info["width"], img_info["height"]
        split = yolo_split(rel_path)
        stats[split] += 1

        src_img = GOLDEN_DIR / rel_path
        stem = Path(rel_path).stem                  # frame_XXXXX
        # use scene name to avoid collisions across scenes
        scene_dir = str(Path(rel_path).parent)      # train/scene_name
        scene_flat = scene_dir.replace("/", "_")    # train_scene_name

        label_name = f"{scene_flat}_{stem}.txt"
        link_name  = f"{scene_flat}_{stem}.jpg"

        # ── detect ──────────────────────────────────────────────────────────
        det_img_dir   = OUT_DIR / "detect" / "images" / split
        det_label_dir = OUT_DIR / "detect" / "labels" / split
        det_img_dir.mkdir(parents=True, exist_ok=True)
        det_label_dir.mkdir(parents=True, exist_ok=True)

        symlink_force(src_img, det_img_dir / link_name)

        det_lines = [bbox_to_yolo(ann["bbox"], img_w, img_h)
                     for ann in id2anns[img_id]]
        (det_label_dir / label_name).write_text("\n".join(det_lines))

        # ── segment ─────────────────────────────────────────────────────────
        seg_img_dir   = OUT_DIR / "segment" / "images" / split
        seg_label_dir = OUT_DIR / "segment" / "labels" / split
        seg_img_dir.mkdir(parents=True, exist_ok=True)
        seg_label_dir.mkdir(parents=True, exist_ok=True)

        symlink_force(src_img, seg_img_dir / link_name)

        seg_lines = []
        for ann in id2anns[img_id]:
            seg = ann.get("segmentation")
            if seg and isinstance(seg, dict) and "counts" in seg:
                polys = rle_to_polygon(seg)
                if polys:
                    # keep largest contour only (main person blob)
                    largest = max(polys, key=len)
                    seg_lines.append(poly_to_yolo(largest, img_w, img_h))
                else:
                    stats["skipped_seg"] += 1
                    seg_lines.append(bbox_to_yolo(ann["bbox"], img_w, img_h)
                                     .replace("0 ", "0 ", 1))  # fallback to bbox
            else:
                # fallback: use bbox as a rectangular polygon
                x, y, w, h = ann["bbox"]
                rect = [x, y, x+w, y, x+w, y+h, x, y+h]
                seg_lines.append(poly_to_yolo(rect, img_w, img_h))

        (seg_label_dir / label_name).write_text("\n".join(seg_lines))

    # ── data YAML files ──────────────────────────────────────────────────────
    for task in ("detect", "segment"):
        yaml_data = {
            "path": str((OUT_DIR / task).resolve()),
            "train": "images/train",
            "val":   "images/val",
            "nc": 1,
            "names": ["person"],
        }
        yaml_path = OUT_DIR / task / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        print(f"Wrote {yaml_path}")

    print(f"\nDone — train: {stats['train']} images, val: {stats['val']} images")
    print(f"Segmentation fallbacks (no contour found): {stats['skipped_seg']}")


if __name__ == "__main__":
    main()
