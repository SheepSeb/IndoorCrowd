#!/usr/bin/env python3
"""
Qualitative annotation comparison grid.

Automatically selects N challenging frames (most crowded + occluded, one per
scene group for diversity) and renders them as a grid:

  rows  = selected frames
  cols  = Raw image | SAM3 | GroundingSAM | Human GT

Each annotation column shows semi-transparent instance masks with crisp
bounding-box outlines and a small detection count in the corner.

Output
------
  analysis/figures/qualitative_comparison.png

Usage
-----
  uv run python analysis/qualitative_comparison.py
  uv run python analysis/qualitative_comparison.py --n-frames 2
  uv run python analysis/qualitative_comparison.py --frames \
      "test/acs_s1_.../frame_00031.jpg" "test/ie_.../frame_00086.jpg"
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).resolve().parents[1]
DATA_ROOT  = REPO_ROOT / "dataset" / "golden_frames_3fps_mid"

GT_JSON      = DATA_ROOT / "seb.json"
SAM3_DIR     = DATA_ROOT / "labels_3fps_golden_mid"
GROUNDED_DIR = DATA_ROOT / "labels_grounding_sam_3fps_golden_mid"

FIGURES_DIR = Path(__file__).parent / "figures"

# ── Constants ──────────────────────────────────────────────────────────────────

N_FRAMES_DEFAULT = 3

# IoU between two GT boxes to call them "occluding"
OCC_IOU_THRESH = 0.10

# How much to blend the mask fill over the image (0 = invisible, 1 = opaque)
MASK_ALPHA = 0.42

COL_LABELS = ["Raw image", "SAM3", "GroundingSAM", "Human GT"]

# 20 perceptually distinct BGR colours (tab20 palette)
_TAB20 = (
    plt.cm.tab20(np.linspace(0, 1, 20))[:, :3] * 255
).astype(np.uint8)


def _instance_color(idx: int) -> tuple[int, int, int]:
    """Cycle through tab20, return BGR tuple."""
    r, g, b = _TAB20[idx % 20]
    return (int(b), int(g), int(r))


# ── COCO utilities ─────────────────────────────────────────────────────────────


def _decode_seg(seg, h: int, w: int) -> np.ndarray:
    """Decode COCO segmentation (RLE dict or polygon list) → uint8 mask."""
    if seg is None:
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(seg, dict):
        rle = dict(seg)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return mask_utils.decode(rle).astype(np.uint8)
    rles = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)


def load_coco(json_path: Path) -> Tuple[Dict[str, Tuple[int, int]],
                                        Dict[str, List[dict]]]:
    """{file_name: (h,w)}, {file_name: [ann, ...]}"""
    with open(json_path) as f:
        coco = json.load(f)
    id2img = {img["id"]: img for img in coco.get("images", [])}
    dims:  Dict[str, Tuple[int, int]] = {}
    anns:  Dict[str, List[dict]]      = defaultdict(list)
    for img in coco["images"]:
        dims[img["file_name"]] = (int(img["height"]), int(img["width"]))
        anns.setdefault(img["file_name"], [])
    for ann in coco.get("annotations", []):
        img = id2img.get(ann["image_id"])
        if img:
            anns[img["file_name"]].append(ann)
    return dims, dict(anns)


def load_method_dir(root: Path) -> Tuple[Dict[str, Tuple[int, int]],
                                         Dict[str, List[dict]]]:
    """Load every split/*.json under root and merge into one index."""
    dims_all: Dict[str, Tuple[int, int]] = {}
    anns_all: Dict[str, List[dict]]      = defaultdict(list)
    if not root.exists():
        return dims_all, dict(anns_all)
    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        for jf in sorted(split_dir.glob("*.json")):
            d, a = load_coco(jf)
            dims_all.update(d)
            for fn, al in a.items():
                anns_all[fn].extend(al)
    return dims_all, dict(anns_all)


# ── Scene grouping ─────────────────────────────────────────────────────────────


def scene_of(file_name: str) -> str:
    if "acs_s1" in file_name:  return "acs_ec"
    if "acs_s2" in file_name:  return "acs_eg"
    if "/ie_"   in file_name or file_name.startswith("ie_"):  return "ie_central"
    if "rectorat" in file_name: return "r_central"
    return "other"


# ── Frame selection ────────────────────────────────────────────────────────────


def _occ_score(anns: List[dict]) -> float:
    """Fraction of box pairs with IoU > OCC_IOU_THRESH."""
    n = len(anns)
    if n < 2:
        return 0.0
    pairs = n * (n - 1) / 2
    occ = 0
    for i in range(n):
        xi, yi, wi, hi = anns[i]["bbox"]
        xi2, yi2 = xi + wi, yi + hi
        for j in range(i + 1, n):
            xj, yj, wj, hj = anns[j]["bbox"]
            xj2, yj2 = xj + wj, yj + hj
            ix = max(0, min(xi2, xj2) - max(xi, xj))
            iy = max(0, min(yi2, yj2) - max(yi, yj))
            inter = ix * iy
            union = wi * hi + wj * hj - inter
            if union > 0 and inter / union > OCC_IOU_THRESH:
                occ += 1
    return occ / pairs


def select_frames(
    gt_anns: Dict[str, List[dict]],
    available: set[str],
    n: int,
) -> List[str]:
    """
    Pick n frames from 'available' with highest challenge score
    (GT person count × (1 + occlusion fraction)), one per scene group.
    Falls back to top-scoring if scene diversity not achievable.
    """
    scored = []
    for fn in available:
        anns = gt_anns.get(fn, [])
        if len(anns) < 2:
            continue
        score = len(anns) * (1 + _occ_score(anns))
        scored.append((score, fn, scene_of(fn)))
    scored.sort(reverse=True)

    # One per scene first
    selected: List[str] = []
    seen_scenes: set[str] = set()
    for score, fn, sg in scored:
        if sg not in seen_scenes:
            selected.append(fn)
            seen_scenes.add(sg)
        if len(selected) == n:
            return selected

    # Fill remainder with highest-scoring not yet picked
    picked = set(selected)
    for _, fn, _ in scored:
        if fn not in picked:
            selected.append(fn)
            picked.add(fn)
        if len(selected) == n:
            break
    return selected[:n]


# ── Rendering ──────────────────────────────────────────────────────────────────


def render(
    img_bgr: np.ndarray,
    anns: List[dict],
    h: int,
    w: int,
) -> np.ndarray:
    """
    Overlay coloured instance masks + bbox outlines on img_bgr.
    Returns an RGB uint8 array.
    """
    fill = img_bgr.copy()
    for idx, ann in enumerate(anns):
        color_bgr = _instance_color(idx)
        seg = ann.get("segmentation")
        if seg:
            mask = _decode_seg(seg, h, w).astype(bool)
            fill[mask] = color_bgr

    blended = cv2.addWeighted(img_bgr, 1 - MASK_ALPHA, fill, MASK_ALPHA, 0)

    # Crisp bbox outlines drawn after blending
    for idx, ann in enumerate(anns):
        color_bgr = _instance_color(idx)
        x, y, bw, bh = (int(v) for v in ann["bbox"])
        cv2.rectangle(blended, (x, y), (x + bw, y + bh), color_bgr, 2)

    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def annotate_count(ax, n: int) -> None:
    """Stamp 'n = X' in the bottom-right corner of an axes."""
    ax.text(
        0.98, 0.02, f"n = {n}",
        transform=ax.transAxes,
        fontsize=7, color="white", fontweight="bold",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                  alpha=0.55, edgecolor="none"),
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qualitative annotation comparison grid"
    )
    parser.add_argument("--n-frames", type=int, default=N_FRAMES_DEFAULT,
                        help="Number of frames to show (default: 3).")
    parser.add_argument("--frames", nargs="+", default=None,
                        help="Override auto-selection with explicit file_names.")
    parser.add_argument("--out", type=Path,
                        default=FIGURES_DIR / "qualitative_comparison.png")
    args = parser.parse_args()

    # ── Load annotations ───────────────────────────────────────────────────────
    print("Loading GT …")
    gt_dims, gt_anns = load_coco(GT_JSON)

    print("Loading SAM3 …")
    sam3_dims, sam3_anns = load_method_dir(SAM3_DIR)

    print("Loading GroundingSAM …")
    grounded_dims, grounded_anns = load_method_dir(GROUNDED_DIR)

    # Frames present in all three sources
    common = set(gt_dims) & set(sam3_dims) & set(grounded_dims)
    print(f"  {len(common)} frames have annotations from all three sources")

    if not common:
        print("ERROR: no common frames found — check dataset paths.")
        return 1

    # ── Select challenging frames ──────────────────────────────────────────────
    if args.frames:
        selected = args.frames
    else:
        selected = select_frames(gt_anns, common, args.n_frames)

    print(f"Selected {len(selected)} frames:")
    for fn in selected:
        n_gt = len(gt_anns.get(fn, []))
        print(f"  [{scene_of(fn):10s}] {n_gt:2d} GT persons  {fn}")

    # ── Build figure ───────────────────────────────────────────────────────────
    n_rows = len(selected)
    n_cols = 4

    # Infer panel height from first image
    sample_path = DATA_ROOT / selected[0]
    if not sample_path.exists():
        print(f"ERROR: image not found: {sample_path}")
        return 1
    _sample = cv2.imread(str(sample_path))
    img_h, img_w = _sample.shape[:2]
    panel_w = 3.5                               # inches per column
    panel_h = panel_w * img_h / img_w          # preserve aspect ratio

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows + 0.45),
        gridspec_kw={"wspace": 0.02, "hspace": 0.04},
    )
    fig.patch.set_facecolor("white")

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column headers on first row only
    for col_idx, label in enumerate(COL_LABELS):
        axes[0, col_idx].set_title(label, fontsize=9, fontweight="bold", pad=5)

    for row_idx, fn in enumerate(selected):
        img_path = DATA_ROOT / fn
        img_bgr  = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [WARN] could not load {img_path}")
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        a_sam3     = sam3_anns.get(fn,     [])
        a_grounded = grounded_anns.get(fn, [])
        a_gt       = gt_anns.get(fn,       [])

        panels = [
            (img_rgb,                            None),
            (render(img_bgr, a_sam3,     h, w),  len(a_sam3)),
            (render(img_bgr, a_grounded, h, w),  len(a_grounded)),
            (render(img_bgr, a_gt,       h, w),  len(a_gt)),
        ]

        for col_idx, (panel_img, count) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel_img)
            ax.axis("off")
            if count is not None:
                annotate_count(ax, count)

        # Scene label on left margin of raw-image column
        sg = scene_of(fn).replace("_", " ").upper()
        axes[row_idx, 0].set_ylabel(
            sg, fontsize=8, rotation=90, labelpad=4, va="center",
        )
        axes[row_idx, 0].yaxis.set_label_position("left")
        axes[row_idx, 0].yaxis.label.set_visible(True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nFigure → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
