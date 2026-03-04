"""
Convert dataset/labels_3fps/ (full SAM3 COCO annotations) → YOLO format.

Creates two dataset trees under yolo_dataset/:
  detect/   – bounding-box labels  (class cx cy w h)
  segment/  – polygon-mask labels  (class x1 y1 x2 y2 …)

Images are from dataset/raw_frames_3fps/ and are symlinked (no copies).

Split mapping:
  train/ + challange/  →  YOLO train
  test/                →  YOLO val

Run once; re-run is idempotent.

Usage
-----
  uv run python train/sam3/prepare_data.py
  uv run python train/sam3/prepare_data.py --min-score 0.5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import yaml

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
LABELS_DIR = REPO_ROOT / "dataset" / "labels_3fps"
FRAMES_DIR = REPO_ROOT / "dataset" / "raw_frames_3fps"
OUT_DIR    = Path(__file__).parent / "yolo_dataset"

SPLITS = [
    ("train",     "train"),
    ("challange", "train"),   # challenge → YOLO train
    ("test",      "val"),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def rle_to_polygon(rle: dict, min_points: int = 6) -> list[list[float]] | None:
    """Decode COCO RLE mask → list of flattened polygon contours (or None)."""
    mask = mask_util.decode(rle).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        c = c.squeeze(1)
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
    normed = [v / (img_w if i % 2 == 0 else img_h) for i, v in enumerate(points)]
    return f"0 {' '.join(f'{v:.6f}' for v in normed)}"


def symlink_force(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SAM3 COCO labels (labels_3fps/) to YOLO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Skip annotations with SAM3 score below this value (0 = keep all).",
    )
    args = parser.parse_args()

    stats = {"train": 0, "val": 0, "skipped_score": 0, "skipped_seg": 0}

    for split, yolo_split in SPLITS:
        json_dir = LABELS_DIR / split
        if not json_dir.exists():
            print(f"  skip missing split: {json_dir}")
            continue

        for json_file in sorted(json_dir.glob("*.json")):
            with open(json_file) as f:
                coco = json.load(f)

            id2img: dict[int, dict] = {img["id"]: img for img in coco["images"]}
            id2anns: dict[int, list] = defaultdict(list)
            for ann in coco["annotations"]:
                if args.min_score > 0 and ann.get("score", 1.0) < args.min_score:
                    stats["skipped_score"] += 1
                    continue
                id2anns[ann["image_id"]].append(ann)

            for img_info in coco["images"]:
                img_id  = img_info["id"]
                rel     = img_info["file_name"]   # e.g. "train/scene/frame_00001.jpg"
                img_w   = img_info["width"]
                img_h   = img_info["height"]
                stats[yolo_split] += 1

                src_img  = FRAMES_DIR / rel
                scene    = Path(rel).parent.name   # e.g. "acs_s1_..."
                stem     = Path(rel).stem           # e.g. "frame_00001"
                prefix   = f"{split}_{scene}_{stem}"
                label_name = f"{prefix}.txt"
                link_name  = f"{prefix}.jpg"

                # ── detect ───────────────────────────────────────────────────
                det_img_dir   = OUT_DIR / "detect" / "images" / yolo_split
                det_label_dir = OUT_DIR / "detect" / "labels" / yolo_split
                det_img_dir.mkdir(parents=True, exist_ok=True)
                det_label_dir.mkdir(parents=True, exist_ok=True)

                symlink_force(src_img, det_img_dir / link_name)

                det_lines = [bbox_to_yolo(ann["bbox"], img_w, img_h)
                             for ann in id2anns[img_id]]
                (det_label_dir / label_name).write_text("\n".join(det_lines))

                # ── segment ──────────────────────────────────────────────────
                seg_img_dir   = OUT_DIR / "segment" / "images" / yolo_split
                seg_label_dir = OUT_DIR / "segment" / "labels" / yolo_split
                seg_img_dir.mkdir(parents=True, exist_ok=True)
                seg_label_dir.mkdir(parents=True, exist_ok=True)

                symlink_force(src_img, seg_img_dir / link_name)

                seg_lines = []
                for ann in id2anns[img_id]:
                    seg = ann.get("segmentation")
                    if seg and isinstance(seg, dict) and "counts" in seg:
                        polys = rle_to_polygon(seg)
                        if polys:
                            largest = max(polys, key=len)
                            seg_lines.append(poly_to_yolo(largest, img_w, img_h))
                        else:
                            stats["skipped_seg"] += 1
                            # fallback: bbox as rectangular polygon
                            x, y, w, h = ann["bbox"]
                            rect = [x, y, x+w, y, x+w, y+h, x, y+h]
                            seg_lines.append(poly_to_yolo(rect, img_w, img_h))
                    else:
                        x, y, w, h = ann["bbox"]
                        rect = [x, y, x+w, y, x+w, y+h, x, y+h]
                        seg_lines.append(poly_to_yolo(rect, img_w, img_h))

                (seg_label_dir / label_name).write_text("\n".join(seg_lines))

    # ── data YAML files ───────────────────────────────────────────────────────
    for task in ("detect", "segment"):
        yaml_data = {
            "path":  str((OUT_DIR / task).resolve()),
            "train": "images/train",
            "val":   "images/val",
            "nc": 1,
            "names": ["person"],
        }
        yaml_path = OUT_DIR / task / "data.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        print(f"Wrote {yaml_path}")

    print(f"\nDone — train: {stats['train']} images, val: {stats['val']} images")
    if args.min_score > 0:
        print(f"Skipped (score < {args.min_score}): {stats['skipped_score']} annotations")
    print(f"Segmentation fallbacks (no contour): {stats['skipped_seg']}")


if __name__ == "__main__":
    main()
