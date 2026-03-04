#!/usr/bin/env python3
"""
Evaluate SAM3 / GroundingSAM / EfficientGroundingSAM labels against human (seb.json)
for the golden_frames_3fps_mid dataset.

Metrics per method and per scene group:
- mAP (bbox-based, AP@0.50 and AP@0.75, plus mean of both)
- Detection precision/recall (bbox IoU ≥ 0.5)
- Mean mask IoU over matched instance pairs
- Cohen's kappa (pixel-wise foreground/background agreement)

Scene groups
------------
- acs_s1_*   → acs_ec
- acs_s2_*   → acs_eg
- ie_*      → ie_central
- rectorat_*→ r_central

Usage
-----
    uv run python analysis/human_vs_models_golden_mid.py

You can override paths if needed:
    uv run python analysis/human_vs_models_golden_mid.py \\
        --gt-json dataset/golden_frames_3fps_mid/seb.json \\
        --sam3-dir dataset/golden_frames_3fps_mid/labels_3fps_golden_mid \\
        --grounded-dir dataset/golden_frames_3fps_mid/labels_grounding_sam_3fps_golden_mid \\
        --efficient-dir dataset/golden_frames_3fps_mid/labels_efficient_grounded_sam_3fps_golden_mid
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_SAM3 = "SAM3"
METHOD_GROUNDED = "GroundingSAM"
METHOD_EFFICIENT = "EfficientGroundingSAM"

METHODS = [METHOD_SAM3, METHOD_GROUNDED, METHOD_EFFICIENT]

# IoU thresholds used for AP/mAP
AP_IOU_THRESHOLDS = [0.50, 0.75]

# Default bbox IoU threshold for counting a match / mask IoU / P/R
DEFAULT_BBOX_IOU_THRESHOLD = 0.50

# IoU threshold to declare boxes "occluding" each other
OCCLUSION_IOU_THRESHOLD = 0.10

# Resolution of the per-scene crowd heatmap (higher = less blocky before smoothing)
HEATMAP_RES = 150

SCENE_GROUPS = ["acs_ec", "acs_eg", "ie_central", "r_central"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def scene_group_from_filename(file_name: str) -> str | None:
    """Map raw file_name to a coarse scene group."""
    if "acs_s1" in file_name:
        return "acs_ec"
    if "acs_s2" in file_name:
        return "acs_eg"
    if "/ie_" in file_name or file_name.startswith("ie_"):
        return "ie_central"
    if "rectorat_" in file_name:
        return "r_central"
    return None


def load_coco_by_filename(
    coco_path: Path,
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, List[dict]]]:
    """Load a COCO-style JSON and index annotations by file_name.

    Returns
    -------
    dims_by_filename : {file_name: (height, width)}
    anns_by_filename : {file_name: [annotation_dict, ...]}
    """
    with open(coco_path) as f:
        coco = json.load(f)

    image_info_by_id: Dict[int, Tuple[str, int, int]] = {
        img["id"]: (img["file_name"], int(img["height"]), int(img["width"]))
        for img in coco.get("images", [])
    }
    dims_by_filename: Dict[str, Tuple[int, int]] = {}
    anns_by_filename: Dict[str, List[dict]] = {}

    for image_id, (file_name, h, w) in image_info_by_id.items():
        dims_by_filename[file_name] = (h, w)
        anns_by_filename[file_name] = []

    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id not in image_info_by_id:
            continue
        file_name, _, _ = image_info_by_id[image_id]
        anns_by_filename[file_name].append(ann)

    return dims_by_filename, anns_by_filename


def load_method_dir(
    labels_root: Path,
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, List[dict]]]:
    """Load all COCO scenes from a labels directory (train/test/... JSON files)."""
    dims_total: Dict[str, Tuple[int, int]] = {}
    anns_total: Dict[str, List[dict]] = {}

    if not labels_root.exists():
        return dims_total, anns_total

    for split_dir in sorted(labels_root.iterdir()):
        if not split_dir.is_dir() or split_dir.name == "session_logs":
            continue
        for json_path in sorted(split_dir.glob("*.json")):
            dims, anns_by_file = load_coco_by_filename(json_path)
            for fname, hw in dims.items():
                if fname not in dims_total:
                    dims_total[fname] = hw
            for fname, anns in anns_by_file.items():
                if not anns:
                    # Ensure we know the file exists even if it is empty
                    anns_total.setdefault(fname, [])
                else:
                    anns_total.setdefault(fname, []).extend(anns)

    return dims_total, anns_total


def decode_mask(segmentation: dict | list, h: int, w: int) -> np.ndarray:
    """Decode a COCO segmentation (RLE or polygon) into a uint8 mask."""
    if segmentation is None:
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return mask_utils.decode(rle).astype(np.uint8)
    rles = mask_utils.frPyObjects(segmentation, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)


def combined_mask(anns: List[dict], h: int, w: int) -> np.ndarray:
    """Binary mask for union of all instance segmentations in anns."""
    rles: List[dict] = []
    for ann in anns:
        seg = ann.get("segmentation")
        if seg is None:
            continue
        if isinstance(seg, dict):
            rle = dict(seg)
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode("utf-8")
            rles.append(rle)
        else:
            rles.extend(mask_utils.frPyObjects(seg, h, w))

    if not rles:
        return np.zeros((h, w), dtype=bool)

    merged = mask_utils.merge(rles)
    mask = mask_utils.decode(merged)
    if mask.ndim == 3:
        mask = mask.any(axis=2)
    return mask.astype(bool)


def bbox_iou_xywh(a: List[float], b: List[float]) -> float:
    """Axis-aligned IoU for [x, y, w, h] boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = (aw * ah) + (bw * bh) - inter_area
    return 0.0 if union <= 0 else inter_area / union


def greedy_match_by_bbox_iou(
    anns_a: List[dict],
    anns_b: List[dict],
    iou_threshold: float,
) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching of annotations by bbox IoU."""
    candidates: List[Tuple[float, int, int]] = []
    for i, ann_a in enumerate(anns_a):
        for j, ann_b in enumerate(anns_b):
            iou = bbox_iou_xywh(ann_a["bbox"], ann_b["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, i, j))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, iou))
    return matches


def safe_ratio(num: int | float, den: int | float) -> float:
    return 0.0 if den <= 0 else float(num) / float(den)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """COCO-style area under the precision–recall curve."""
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MethodSceneMetrics:
    method: str
    scene_group: str
    n_images: int
    n_gt: int
    n_pred: int
    ap_by_thresh: dict = field(default_factory=dict)
    precision_by_thresh: dict = field(default_factory=dict)
    recall_by_thresh: dict = field(default_factory=dict)
    mean_mask_iou: float | None = None
    cohen_kappa: float | None = None

    @property
    def map(self) -> float:
        if not self.ap_by_thresh:
            return 0.0
        return float(sum(self.ap_by_thresh.values()) / len(self.ap_by_thresh))


@dataclass
class SceneStructureStats:
    """Scene-level statistics from human (GT) annotations only."""

    scene_group: str
    n_images: int

    # People-per-frame
    person_count_mean: float
    person_count_std: float
    person_count_min: int
    person_count_max: int
    density_counts: dict
    density_fractions: dict

    # Bounding-box scale
    rel_scale_mean: float  # sqrt(area)/sqrt(frame_area)
    rel_scale_std: float
    abs_scale_mean: float  # sqrt(area) in pixels
    abs_scale_std: float
    size_bin_counts: dict  # small/medium/large counts
    size_bin_fractions: dict

    # Aspect ratio and occlusion proxy
    aspect_ratio_mean: float  # h / w
    aspect_ratio_std: float
    occluded_instances: int
    total_instances: int
    occlusion_fraction: float

    # Path to saved heatmap figure (if any)
    heatmap_path: str | None = None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def collect_predictions(anns_by_file: Dict[str, List[dict]]) -> List[dict]:
    preds: List[dict] = []
    for fname, anns in anns_by_file.items():
        for ann in anns:
            preds.append(
                {
                    "file_name": fname,
                    "bbox": ann["bbox"],
                    "score": float(ann.get("score", 1.0)),
                    "ann": ann,
                }
            )
    return preds


def evaluate_detection(
    preds: List[dict],
    gt_anns_by_file: Dict[str, List[dict]],
    group_files: set[str],
    iou_thresholds: List[float],
) -> Tuple[dict, dict, dict, int, int]:
    """Compute AP / precision / recall for a subset of files."""
    preds_group = [p for p in preds if p["file_name"] in group_files]
    preds_group.sort(key=lambda x: x["score"], reverse=True)

    gt_group: Dict[str, List[dict]] = {
        fname: gt_anns_by_file.get(fname, []) for fname in group_files
    }

    n_gt = sum(len(v) for v in gt_group.values())
    n_pred = len(preds_group)

    ap_by_t: dict[float, float] = {}
    precision_by_t: dict[float, float] = {}
    recall_by_t: dict[float, float] = {}

    if n_gt == 0 or n_pred == 0:
        for t in iou_thresholds:
            ap_by_t[t] = 0.0
            precision_by_t[t] = 0.0
            recall_by_t[t] = 0.0
        return ap_by_t, precision_by_t, recall_by_t, n_gt, n_pred

    for t in iou_thresholds:
        gt_used: Dict[str, List[bool]] = {
            fname: [False] * len(anns) for fname, anns in gt_group.items()
        }
        tp_flags: List[int] = []

        for p in preds_group:
            fname = p["file_name"]
            anns_gt = gt_group.get(fname, [])
            if not anns_gt:
                tp_flags.append(0)
                continue

            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(anns_gt):
                if gt_used[fname][j]:
                    continue
                iou = bbox_iou_xywh(p["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= t and best_j >= 0:
                gt_used[fname][best_j] = True
                tp_flags.append(1)
            else:
                tp_flags.append(0)

        tp_cum = np.cumsum(tp_flags, dtype=float)
        fp_cum = np.cumsum([1 - x for x in tp_flags], dtype=float)

        if tp_cum.size == 0:
            ap_by_t[t] = 0.0
            precision_by_t[t] = 0.0
            recall_by_t[t] = 0.0
            continue

        recalls = tp_cum / float(n_gt)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        ap_by_t[t] = compute_ap(recalls, precisions)
        precision_by_t[t] = float(precisions[-1])
        recall_by_t[t] = float(recalls[-1])

    return ap_by_t, precision_by_t, recall_by_t, n_gt, n_pred


def compute_mean_mask_iou(
    pred_anns_by_file: Dict[str, List[dict]],
    gt_anns_by_file: Dict[str, List[dict]],
    gt_dims_by_file: Dict[str, Tuple[int, int]],
    group_files: set[str],
    bbox_iou_threshold: float,
) -> float | None:
    vals: List[float] = []
    for fname in group_files:
        anns_p = pred_anns_by_file.get(fname, [])
        anns_g = gt_anns_by_file.get(fname, [])
        if not anns_p or not anns_g:
            continue
        h, w = gt_dims_by_file[fname]
        matches = greedy_match_by_bbox_iou(anns_p, anns_g, bbox_iou_threshold)
        for i, j, _ in matches:
            seg_p = anns_p[i].get("segmentation")
            seg_g = anns_g[j].get("segmentation")
            if seg_p is None or seg_g is None:
                continue
            mp = decode_mask(seg_p, h, w) > 0
            mg = decode_mask(seg_g, h, w) > 0
            inter = int(np.logical_and(mp, mg).sum())
            union = int(np.logical_or(mp, mg).sum())
            if union > 0:
                vals.append(inter / union)

    if not vals:
        return None
    return float(statistics.fmean(vals))


def compute_cohen_kappa(
    pred_anns_by_file: Dict[str, List[dict]],
    gt_anns_by_file: Dict[str, List[dict]],
    gt_dims_by_file: Dict[str, Tuple[int, int]],
    group_files: set[str],
) -> float | None:
    tp_total = fp_total = fn_total = tn_total = 0
    N_total = 0

    for fname in group_files:
        h, w = gt_dims_by_file[fname]
        anns_g = gt_anns_by_file.get(fname, [])
        anns_p = pred_anns_by_file.get(fname, [])

        if anns_g or anns_p:
            mg = combined_mask(anns_g, h, w)
            mp = combined_mask(anns_p, h, w)
        else:
            mg = np.zeros((h, w), dtype=bool)
            mp = np.zeros((h, w), dtype=bool)

        N = h * w
        tp = int(np.logical_and(mg, mp).sum())
        fp = int(np.logical_and(~mg, mp).sum())
        fn = int(np.logical_and(mg, ~mp).sum())
        tn = N - tp - fp - fn

        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn
        N_total += N

    if N_total == 0:
        return None

    po = (tp_total + tn_total) / N_total
    p_yes_gt = (tp_total + fn_total) / N_total
    p_yes_pred = (tp_total + fp_total) / N_total
    p_no_gt = (fp_total + tn_total) / N_total
    p_no_pred = (fn_total + tn_total) / N_total
    pe = p_yes_gt * p_yes_pred + p_no_gt * p_no_pred

    if 1.0 - pe <= 0:
        return None

    kappa = (po - pe) / (1.0 - pe)
    return float(kappa)


# ---------------------------------------------------------------------------
# Scene-level structure statistics (GT only)
# ---------------------------------------------------------------------------


_SCENE_LABELS = {
    "acs_ec": "(a) ACS EC",
    "acs_eg": "(b) ACS EG",
    "ie_central": "(c) IE Central",
    "r_central": "(d) R Central",
}


def _plot_density_heatmaps(
    heat_by_scene: Dict[str, np.ndarray],
    scene_stats: Dict[str, "SceneStructureStats"],
    heatmap_dir: Path,
    files_by_scene: Dict[str, List[str]] | None = None,
    image_root: Path | None = None,
) -> None:
    """
    Combined 2×2 figure:
      • Per-scene normalisation + PowerNorm(γ=0.4) — sparse scenes still show structure
      • Representative background frame (greyscale, dimmed) per scene
      • RGBA density overlay — transparent where empty, opaque where dense
      • One shared colorbar on the right
      • No tick marks; (a)–(d) scene titles
    """
    import cv2 as _cv2
    from scipy.ndimage import gaussian_filter
    from matplotlib.colors import PowerNorm
    from matplotlib.cm import ScalarMappable

    SIGMA = 2.5
    CMAP = "YlOrRd"
    GAMMA = 0.4

    # ── Gaussian-smooth each heat array ───────────────────────────────────────
    smoothed: Dict[str, np.ndarray] = {
        scene: gaussian_filter(raw.astype(float), sigma=SIGMA)
        for scene, raw in heat_by_scene.items()
    }

    # ── Load a representative background frame per scene ──────────────────────
    bg_by_scene: Dict[str, np.ndarray | None] = {}
    for scene in SCENE_GROUPS:
        bg_by_scene[scene] = None
    if files_by_scene is not None and image_root is not None:
        for scene, files in files_by_scene.items():
            if not files:
                continue
            rep_file = sorted(files)[len(files) // 2]  # median filename
            img_path = image_root / rep_file
            if img_path.exists():
                raw_img = _cv2.imread(str(img_path))
                if raw_img is not None:
                    bg_by_scene[scene] = _cv2.cvtColor(raw_img, _cv2.COLOR_BGR2GRAY)

    # ── One figure per scene ──────────────────────────────────────────────────
    scenes = [s for s in SCENE_GROUPS if s in smoothed]
    norm_obj = PowerNorm(gamma=GAMMA, vmin=0.0, vmax=1.0)
    cmap_obj = plt.cm.get_cmap(CMAP)

    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        h_smooth = smoothed[scene]

        # Per-scene normalise → [0, 1] so sparse scenes retain contrast
        scene_max = h_smooth.max()
        h_norm = h_smooth / scene_max if scene_max > 0 else h_smooth

        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # ── 1. Background frame (greyscale, dimmed) ───────────────────────────
        bg = bg_by_scene.get(scene)
        if bg is not None:
            ax.imshow(
                bg,
                extent=[0, 1, 1, 0],
                aspect="auto",
                cmap="gray",
                vmin=0,
                vmax=255,
                alpha=0.40,
                interpolation="bilinear",
            )

        # ── 2. RGBA density overlay ───────────────────────────────────────────
        rgba = cmap_obj(norm_obj(h_norm))  # (H, W, 4)
        rgba[..., 3] = np.sqrt(h_norm)  # transparent where empty
        ax.imshow(
            rgba,
            origin="upper",
            extent=[0, 1, 1, 0],
            aspect="auto",
            interpolation="bicubic",
        )

        # ── Cosmetics ─────────────────────────────────────────────────────────
        ax.set_xlabel("Normalised width", fontsize=9)
        ax.set_ylabel("Normalised height", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        sm = ScalarMappable(cmap=cmap_obj, norm=norm_obj)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Relative density", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        out_path = heatmap_dir / f"density_heatmap_{scene}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → {out_path}")

        if scene in scene_stats:
            scene_stats[scene].heatmap_path = str(out_path)


def compute_scene_structure_stats(
    gt_dims_by_file: Dict[str, Tuple[int, int]],
    gt_anns_by_file: Dict[str, List[dict]],
    scene_by_file: Dict[str, str],
    heatmap_dir: Path,
    image_root: Path | None = None,
) -> Dict[str, SceneStructureStats]:
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    files_by_scene: Dict[str, List[str]] = {s: [] for s in SCENE_GROUPS}
    for fname, scene in scene_by_file.items():
        if scene in files_by_scene:
            files_by_scene[scene].append(fname)

    scene_stats: Dict[str, SceneStructureStats] = {}
    heat_by_scene: Dict[str, np.ndarray] = {}

    for scene, files in files_by_scene.items():
        if not files:
            continue

        counts: List[int] = []
        density_counts = {"empty": 0, "sparse": 0, "medium": 0, "dense": 0}
        rel_scales: List[float] = []
        abs_scales: List[float] = []
        size_bin_counts = {"small": 0, "medium": 0, "large": 0}
        aspect_ratios: List[float] = []
        occluded_instances = 0
        total_instances = 0
        heat = np.zeros((HEATMAP_RES, HEATMAP_RES), dtype=float)

        for fname in files:
            h, w = gt_dims_by_file[fname]
            anns = gt_anns_by_file.get(fname, [])
            n = len(anns)
            counts.append(n)

            # Frame-level density category
            if n == 0:
                density_counts["empty"] += 1
            elif n <= 3:
                density_counts["sparse"] += 1
            elif n <= 10:
                density_counts["medium"] += 1
            else:
                density_counts["dense"] += 1

            # Occlusion proxy (IoU between boxes)
            total_instances += n
            if n >= 2:
                occluded_flags = [False] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        iou = bbox_iou_xywh(anns[i]["bbox"], anns[j]["bbox"])
                        if iou > OCCLUSION_IOU_THRESHOLD:
                            occluded_flags[i] = True
                            occluded_flags[j] = True
                occluded_instances += sum(1 for f in occluded_flags if f)

            # Per-instance stats
            img_area = float(h * w) if h > 0 and w > 0 else 0.0
            for ann in anns:
                x, y, bw, bh = ann["bbox"]
                area = bw * bh
                abs_size = float(area) ** 0.5
                rel_size = (area / img_area) ** 0.5 if img_area > 0 else 0.0
                rel_scales.append(rel_size)
                abs_scales.append(abs_size)

                # COCO-style size bins in pixels: <32, 32–96, >96
                if abs_size < 32.0:
                    size_bin_counts["small"] += 1
                elif abs_size <= 96.0:
                    size_bin_counts["medium"] += 1
                else:
                    size_bin_counts["large"] += 1

                if bw > 0:
                    aspect_ratios.append(bh / bw)

                # Crowd density heatmap (box centers, normalized)
                if w > 0 and h > 0:
                    cx = (x + bw * 0.5) / w
                    cy = (y + bh * 0.5) / h
                    ix = int(np.clip(cx * (HEATMAP_RES - 1), 0, HEATMAP_RES - 1))
                    iy = int(np.clip(cy * (HEATMAP_RES - 1), 0, HEATMAP_RES - 1))
                    heat[iy, ix] += 1.0

        n_images = len(files)
        person_count_mean = float(statistics.fmean(counts)) if counts else 0.0
        person_count_std = float(statistics.stdev(counts)) if len(counts) > 1 else 0.0
        person_count_min = min(counts) if counts else 0
        person_count_max = max(counts) if counts else 0

        rel_scale_mean = float(statistics.fmean(rel_scales)) if rel_scales else 0.0
        rel_scale_std = (
            float(statistics.stdev(rel_scales)) if len(rel_scales) > 1 else 0.0
        )
        abs_scale_mean = float(statistics.fmean(abs_scales)) if abs_scales else 0.0
        abs_scale_std = (
            float(statistics.stdev(abs_scales)) if len(abs_scales) > 1 else 0.0
        )

        aspect_ratio_mean = (
            float(statistics.fmean(aspect_ratios)) if aspect_ratios else 0.0
        )
        aspect_ratio_std = (
            float(statistics.stdev(aspect_ratios)) if len(aspect_ratios) > 1 else 0.0
        )

        occlusion_fraction = safe_ratio(occluded_instances, total_instances)

        density_fractions = {
            k: safe_ratio(v, n_images) for k, v in density_counts.items()
        }
        size_bin_fractions = {
            k: safe_ratio(v, sum(size_bin_counts.values()))
            for k, v in size_bin_counts.items()
        }

        scene_stats[scene] = SceneStructureStats(
            scene_group=scene,
            n_images=n_images,
            person_count_mean=person_count_mean,
            person_count_std=person_count_std,
            person_count_min=person_count_min,
            person_count_max=person_count_max,
            density_counts=density_counts,
            density_fractions=density_fractions,
            rel_scale_mean=rel_scale_mean,
            rel_scale_std=rel_scale_std,
            abs_scale_mean=abs_scale_mean,
            abs_scale_std=abs_scale_std,
            size_bin_counts=size_bin_counts,
            size_bin_fractions=size_bin_fractions,
            aspect_ratio_mean=aspect_ratio_mean,
            aspect_ratio_std=aspect_ratio_std,
            occluded_instances=occluded_instances,
            total_instances=total_instances,
            occlusion_fraction=occlusion_fraction,
            heatmap_path=None,  # filled in by _plot_density_heatmaps below
        )
        heat_by_scene[scene] = heat

    # ── Two-pass heatmap plotting: per-scene PowerNorm, shared colorbar ───────
    _plot_density_heatmaps(
        heat_by_scene,
        scene_stats,
        heatmap_dir,
        files_by_scene=files_by_scene,
        image_root=image_root,
    )

    return scene_stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(
    metrics_by_scene: Dict[str, Dict[str, MethodSceneMetrics]],
    iou_thresholds: List[float],
) -> None:
    print("\n" + "=" * 100)
    print("HUMAN vs AUTO-LABELS — GOLDEN_FRAMES_3FPS_MID")
    print("=" * 100)
    print(f"IoU thresholds for AP/mAP: {', '.join(f'{t:.2f}' for t in iou_thresholds)}")
    print(
        f"Detection match threshold for P/R & mask IoU: {DEFAULT_BBOX_IOU_THRESHOLD:.2f}"
    )

    for scene in SCENE_GROUPS:
        scene_metrics = metrics_by_scene.get(scene, {})
        if not scene_metrics:
            continue

        print(f"\nScene group: {scene}")
        header = (
            f"{'method':15s} {'imgs':>6s} {'GT':>6s} {'pred':>6s} "
            f"{'AP@0.50':>8s} {'AP@0.75':>8s} {'mAP':>8s} "
            f"{'Prec@0.50':>10s} {'Rec@0.50':>10s} "
            f"{'meanMaskIoU':>11s} {'kappa':>8s}"
        )
        print(header)
        print("-" * len(header))

        for method in METHODS:
            m = scene_metrics.get(method)
            if m is None:
                continue
            ap50 = m.ap_by_thresh.get(0.50, 0.0)
            ap75 = m.ap_by_thresh.get(0.75, 0.0)
            prec50 = m.precision_by_thresh.get(0.50, 0.0)
            rec50 = m.recall_by_thresh.get(0.50, 0.0)
            mean_iou = "n/a" if m.mean_mask_iou is None else f"{m.mean_mask_iou:.3f}"
            kappa = "n/a" if m.cohen_kappa is None else f"{m.cohen_kappa:.3f}"
            print(
                f"{method:15s} "
                f"{m.n_images:6d} {m.n_gt:6d} {m.n_pred:6d} "
                f"{ap50:8.3f} {ap75:8.3f} {m.map:8.3f} "
                f"{prec50:10.3f} {rec50:10.3f} "
                f"{mean_iou:>11s} {kappa:>8s}"
            )

    print("\n" + "=" * 100 + "\n")


def print_scene_structure_report(
    scene_stats_by_scene: Dict[str, SceneStructureStats],
) -> None:
    print("\n" + "=" * 100)
    print("SCENE-LEVEL STRUCTURE STATS FROM HUMAN ANNOTATIONS")
    print("=" * 100)

    for scene in SCENE_GROUPS:
        stats = scene_stats_by_scene.get(scene)
        if stats is None:
            continue

        print(f"\nScene group: {scene}")
        print(
            f"  People per frame: mean={stats.person_count_mean:.2f} "
            f"std={stats.person_count_std:.2f} "
            f"(min={stats.person_count_min}, max={stats.person_count_max})"
        )
        dc = stats.density_fractions
        print(
            "  Frame density: "
            f"empty={dc['empty']:.1%}, "
            f"sparse={dc['sparse']:.1%}, "
            f"medium={dc['medium']:.1%}, "
            f"dense={dc['dense']:.1%}"
        )
        print(
            f"  Bbox scale (sqrt(area)/sqrt(frame_area)): "
            f"mean={stats.rel_scale_mean:.3f}, std={stats.rel_scale_std:.3f}"
        )
        sb = stats.size_bin_fractions
        print(
            "  Size bins (COCO-style): "
            f"small={sb['small']:.1%}, "
            f"medium={sb['medium']:.1%}, "
            f"large={sb['large']:.1%}"
        )
        print(
            f"  Aspect ratio (h/w): mean={stats.aspect_ratio_mean:.3f}, "
            f"std={stats.aspect_ratio_std:.3f}"
        )
        print(
            f"  Occluded instances: {stats.occluded_instances} / "
            f"{stats.total_instances} "
            f"({stats.occlusion_fraction:.1%})"
        )
        if stats.heatmap_path:
            print(f"  Crowd density heatmap: {stats.heatmap_path}")

    print("\n" + "=" * 100 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SAM3 / GroundingSAM / EfficientGroundingSAM labels against "
            "human annotations (seb.json) on golden_frames_3fps_mid."
        )
    )
    parser.add_argument(
        "--gt-json",
        type=Path,
        default=Path("dataset/golden_frames_3fps_mid/seb.json"),
        help="Human-annotated COCO JSON (seb.json).",
    )
    parser.add_argument(
        "--sam3-dir",
        type=Path,
        default=Path("dataset/golden_frames_3fps_mid/labels_3fps_golden_mid"),
    )
    parser.add_argument(
        "--grounded-dir",
        type=Path,
        default=Path(
            "dataset/golden_frames_3fps_mid/labels_grounding_sam_3fps_golden_mid"
        ),
    )
    parser.add_argument(
        "--efficient-dir",
        type=Path,
        default=Path(
            "dataset/golden_frames_3fps_mid/labels_efficient_grounded_sam_3fps_golden_mid"
        ),
    )
    parser.add_argument(
        "--bbox-iou-threshold",
        type=float,
        default=DEFAULT_BBOX_IOU_THRESHOLD,
        help="BBox IoU threshold for counting a match (P/R, mask IoU, kappa).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("analysis/reports/human_vs_models_golden_mid.json"),
        help="Optional JSON dump of all per-scene metrics.",
    )
    args = parser.parse_args()

    if not args.gt_json.exists():
        print(f"Ground truth JSON not found: {args.gt_json}")
        return 1

    # Load human annotations
    gt_dims_by_file, gt_anns_by_file = load_coco_by_filename(args.gt_json)
    if not gt_dims_by_file:
        print(f"No images found in ground truth JSON: {args.gt_json}")
        return 1

    # Scene group per file (from GT)
    scene_by_file: Dict[str, str] = {}
    for fname in gt_dims_by_file.keys():
        grp = scene_group_from_filename(fname)
        if grp is not None:
            scene_by_file[fname] = grp

    # Scene-level structure stats from GT only
    scene_stats_by_scene = compute_scene_structure_stats(
        gt_dims_by_file=gt_dims_by_file,
        gt_anns_by_file=gt_anns_by_file,
        scene_by_file=scene_by_file,
        heatmap_dir=Path("analysis/figures/human_scene_stats"),
        image_root=args.gt_json.parent,
    )

    # Prepare containers for model-vs-human metrics
    metrics_by_scene: Dict[str, Dict[str, MethodSceneMetrics]] = {
        s: {} for s in SCENE_GROUPS
    }

    # Methods and their label roots
    method_dirs = {
        METHOD_SAM3: args.sam3_dir,
        METHOD_GROUNDED: args.grounded_dir,
        METHOD_EFFICIENT: args.efficient_dir,
    }

    for method, root in method_dirs.items():
        if not root.exists():
            print(f"Warning: labels directory for {method} not found: {root}")
            continue

        method_dims_by_file, method_anns_by_file = load_method_dir(root)
        if not method_dims_by_file:
            print(f"Warning: no images found for {method} in {root}")
            continue

        # Only evaluate on images present in both GT and this method
        available_files = set(gt_dims_by_file.keys()) & set(method_dims_by_file.keys())

        # Collect all predictions (for detection AP) once per method
        method_preds = collect_predictions(method_anns_by_file)

        for scene in SCENE_GROUPS:
            group_files = {f for f in available_files if scene_by_file.get(f) == scene}
            if not group_files:
                continue

            ap_by_t, prec_by_t, rec_by_t, n_gt, n_pred = evaluate_detection(
                preds=method_preds,
                gt_anns_by_file=gt_anns_by_file,
                group_files=group_files,
                iou_thresholds=AP_IOU_THRESHOLDS,
            )

            mean_mask_iou = compute_mean_mask_iou(
                pred_anns_by_file=method_anns_by_file,
                gt_anns_by_file=gt_anns_by_file,
                gt_dims_by_file=gt_dims_by_file,
                group_files=group_files,
                bbox_iou_threshold=args.bbox_iou_threshold,
            )

            kappa = compute_cohen_kappa(
                pred_anns_by_file=method_anns_by_file,
                gt_anns_by_file=gt_anns_by_file,
                gt_dims_by_file=gt_dims_by_file,
                group_files=group_files,
            )

            metrics_by_scene[scene][method] = MethodSceneMetrics(
                method=method,
                scene_group=scene,
                n_images=len(group_files),
                n_gt=n_gt,
                n_pred=n_pred,
                ap_by_thresh=ap_by_t,
                precision_by_thresh=prec_by_t,
                recall_by_thresh=rec_by_t,
                mean_mask_iou=mean_mask_iou,
                cohen_kappa=kappa,
            )

    print_report(metrics_by_scene, AP_IOU_THRESHOLDS)
    print_scene_structure_report(scene_stats_by_scene)

    # Optional JSON dump (model metrics + scene-level stats)
    if args.output_json is not None:
        payload = {
            "model_metrics": {
                scene: {
                    method: asdict(metrics) for method, metrics in per_method.items()
                }
                for scene, per_method in metrics_by_scene.items()
            },
            "scene_stats": {
                scene: asdict(stats) for scene, stats in scene_stats_by_scene.items()
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Detailed JSON metrics written to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
