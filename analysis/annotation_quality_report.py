#!/usr/bin/env python3
"""Research-quality comparison of SAM3, GroundingSAM and EfficientGroundingSAM annotations.

Produces
--------
- Console summary with all key metrics
- PDF/PNG figures ready for inclusion in a paper
- LaTeX table snippets (printed and saved to .tex)
- Detailed JSON report

Usage
-----
    uv run python analysis/annotation_quality_report.py

    # Custom label directories
    uv run python analysis/annotation_quality_report.py \\
        --sam3-dir dataset/labels_3fps \\
        --grounded-dir dataset/labels_grounding_sam_3fps \\
        --efficient-dir dataset/labels_efficient_grounded_sam_3fps \\
        --output-dir analysis/paper_figures \\
        --bbox-iou-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pycocotools import mask as mask_utils

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAM3 = "SAM3"
GROUNDED = "GroundingSAM"
EFFICIENT = "EfficientGroundingSAM"
MODELS = [SAM3, GROUNDED, EFFICIENT]

MODEL_COLORS = {
    SAM3: "#2196F3",       # blue
    GROUNDED: "#4CAF50",   # green
    EFFICIENT: "#FF9800",  # orange
}

PAIRS = [
    (SAM3, GROUNDED),
    (SAM3, EFFICIENT),
    (GROUNDED, EFFICIENT),
]

PAIR_COLORS = ["#9C27B0", "#F44336", "#009688"]

# COCO area thresholds (pixels²)
SMALL_MAX = 32 ** 2
MEDIUM_MAX = 96 ** 2

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def list_scene_files(labels_dir: Path) -> dict[str, dict[str, Path]]:
    """Return {split: {scene_stem: json_path}} for a labels directory."""
    out: dict[str, dict[str, Path]] = {}
    if not labels_dir.exists():
        return out
    for split_dir in sorted(labels_dir.iterdir()):
        if not split_dir.is_dir() or split_dir.name == "session_logs":
            continue
        scenes = {p.stem: p for p in sorted(split_dir.glob("*.json"))}
        if scenes:
            out[split_dir.name] = scenes
    return out


def load_scene(json_path: Path) -> tuple[dict[str, tuple[int, int]], dict[str, list[dict]]]:
    """Load a COCO scene JSON.

    Returns
    -------
    dims_by_filename : {file_name: (height, width)}
    anns_by_filename : {file_name: [annotation_dict, ...]}
    """
    with open(json_path) as f:
        coco = json.load(f)

    id_to_info: dict[int, tuple[str, int, int]] = {
        img["id"]: (img["file_name"], int(img["height"]), int(img["width"]))
        for img in coco.get("images", [])
    }
    dims: dict[str, tuple[int, int]] = {}
    anns: dict[str, list[dict]] = {}
    for img_id, (fname, h, w) in id_to_info.items():
        dims[fname] = (h, w)
        anns[fname] = []
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in id_to_info:
            continue
        fname = id_to_info[img_id][0]
        anns[fname].append(ann)
    return dims, anns


def decode_mask(seg: dict | list, h: int, w: int) -> np.ndarray:
    if isinstance(seg, dict):
        rle = dict(seg)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return mask_utils.decode(rle).astype(np.uint8)
    rles = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def greedy_match(
    anns_a: list[dict], anns_b: list[dict], iou_thresh: float
) -> list[tuple[int, int, float]]:
    candidates = [
        (bbox_iou(a["bbox"], b["bbox"]), i, j)
        for i, a in enumerate(anns_a)
        for j, b in enumerate(anns_b)
        if bbox_iou(a["bbox"], b["bbox"]) >= iou_thresh
    ]
    candidates.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, iou))
    return matches


# ---------------------------------------------------------------------------
# Per-image statistics
# ---------------------------------------------------------------------------


@dataclass
class ImageStats:
    """Per-image metrics for a single model."""
    count: int
    areas_abs: list[float]          # annotation areas in pixels²
    areas_frac: list[float]         # annotation areas as fraction of image
    aspect_ratios: list[float]      # width/height of bboxes
    size_classes: list[str]         # "small" / "medium" / "large"


@dataclass
class PairImageStats:
    """Per-image pairwise comparison metrics."""
    count_a: int
    count_b: int
    matched: int
    bbox_ious: list[float]
    mask_ious: list[float]


def image_stats(anns: list[dict], h: int, w: int) -> ImageStats:
    img_area = h * w
    areas_abs, areas_frac, aspects, sizes = [], [], [], []
    for ann in anns:
        bx, by, bw, bh = ann["bbox"]
        area = bw * bh
        areas_abs.append(area)
        areas_frac.append(area / img_area if img_area > 0 else 0.0)
        aspects.append(bw / bh if bh > 0 else 0.0)
        if area <= SMALL_MAX:
            sizes.append("small")
        elif area <= MEDIUM_MAX:
            sizes.append("medium")
        else:
            sizes.append("large")
    return ImageStats(
        count=len(anns),
        areas_abs=areas_abs,
        areas_frac=areas_frac,
        aspect_ratios=aspects,
        size_classes=sizes,
    )


def pair_image_stats(
    anns_a: list[dict],
    anns_b: list[dict],
    h: int,
    w: int,
    iou_thresh: float,
) -> PairImageStats:
    matches = greedy_match(anns_a, anns_b, iou_thresh)
    bbox_ious = [m[2] for m in matches]
    mask_ious = []
    for i, j, _ in matches:
        seg_a = anns_a[i].get("segmentation")
        seg_b = anns_b[j].get("segmentation")
        if seg_a is None or seg_b is None:
            continue
        ma = decode_mask(seg_a, h, w)
        mb = decode_mask(seg_b, h, w)
        inter = int(((ma > 0) & (mb > 0)).sum())
        union = int(((ma > 0) | (mb > 0)).sum())
        if union > 0:
            mask_ious.append(inter / union)
    return PairImageStats(
        count_a=len(anns_a),
        count_b=len(anns_b),
        matched=len(matches),
        bbox_ious=bbox_ious,
        mask_ious=mask_ious,
    )


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------


@dataclass
class ModelDataset:
    """All per-image stats for one model across the common image set."""
    model: str
    counts: list[int] = field(default_factory=list)
    areas_frac: list[float] = field(default_factory=list)
    aspect_ratios: list[float] = field(default_factory=list)
    size_classes: list[str] = field(default_factory=list)
    n_images: int = 0
    n_images_non_empty: int = 0

    def add(self, stats: ImageStats):
        self.n_images += 1
        self.counts.append(stats.count)
        self.areas_frac.extend(stats.areas_frac)
        self.aspect_ratios.extend(stats.aspect_ratios)
        self.size_classes.extend(stats.size_classes)
        if stats.count > 0:
            self.n_images_non_empty += 1

    @property
    def total_annotations(self) -> int:
        return sum(self.counts)

    @property
    def detection_rate(self) -> float:
        return self.n_images_non_empty / self.n_images if self.n_images > 0 else 0.0

    def summary(self) -> dict:
        c = self.counts
        af = self.areas_frac
        return {
            "n_images": self.n_images,
            "n_images_non_empty": self.n_images_non_empty,
            "detection_rate": round(self.detection_rate, 4),
            "total_annotations": self.total_annotations,
            "annotations_per_image": {
                "mean": round(statistics.fmean(c), 3) if c else 0,
                "median": round(statistics.median(c), 3) if c else 0,
                "stdev": round(statistics.stdev(c), 3) if len(c) > 1 else 0,
                "min": min(c) if c else 0,
                "max": max(c) if c else 0,
            },
            "annotation_area_fraction": {
                "mean": round(statistics.fmean(af), 5) if af else 0,
                "median": round(statistics.median(af), 5) if af else 0,
                "stdev": round(statistics.stdev(af), 5) if len(af) > 1 else 0,
            },
            "size_class_counts": {
                "small": self.size_classes.count("small"),
                "medium": self.size_classes.count("medium"),
                "large": self.size_classes.count("large"),
            },
        }


@dataclass
class PairDataset:
    """All per-image pairwise stats for one model pair."""
    model_a: str
    model_b: str
    counts_a: list[int] = field(default_factory=list)
    counts_b: list[int] = field(default_factory=list)
    matched: list[int] = field(default_factory=list)
    bbox_ious: list[float] = field(default_factory=list)
    mask_ious: list[float] = field(default_factory=list)

    def add(self, stats: PairImageStats):
        self.counts_a.append(stats.count_a)
        self.counts_b.append(stats.count_b)
        self.matched.append(stats.matched)
        self.bbox_ious.extend(stats.bbox_ious)
        self.mask_ious.extend(stats.mask_ious)

    def summary(self) -> dict:
        total_a = sum(self.counts_a)
        total_b = sum(self.counts_b)
        total_m = sum(self.matched)
        bi = self.bbox_ious
        mi = self.mask_ious
        return {
            "total_annotations_a": total_a,
            "total_annotations_b": total_b,
            "matched_pairs": total_m,
            "precision_a": round(total_m / total_a, 4) if total_a else 0,
            "recall_b": round(total_m / total_b, 4) if total_b else 0,
            "f1": round(
                2 * total_m / (total_a + total_b), 4
            ) if (total_a + total_b) else 0,
            "mean_bbox_iou": round(statistics.fmean(bi), 4) if bi else None,
            "median_bbox_iou": round(statistics.median(bi), 4) if bi else None,
            "stdev_bbox_iou": round(statistics.stdev(bi), 4) if len(bi) > 1 else None,
            "mean_mask_iou": round(statistics.fmean(mi), 4) if mi else None,
            "median_mask_iou": round(statistics.median(mi), 4) if mi else None,
            "stdev_mask_iou": round(statistics.stdev(mi), 4) if len(mi) > 1 else None,
        }


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

PAPER_RC = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
}


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


def fig_annotation_counts(
    model_data: dict[str, ModelDataset], out_dir: Path
) -> None:
    """Bar chart: total annotations and mean annotations/image per model."""
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))

        models = MODELS
        x = np.arange(len(models))
        totals = [model_data[m].total_annotations for m in models]
        means = [statistics.fmean(model_data[m].counts) for m in models]
        stds = [statistics.stdev(model_data[m].counts) if len(model_data[m].counts) > 1 else 0 for m in models]
        colors = [MODEL_COLORS[m] for m in models]

        ax = axes[0]
        bars = ax.bar(x, totals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("Grounding", "Grounding\n") for m in models], ha="center")
        ax.set_ylabel("Total annotations")
        ax.set_title("Total Annotations")
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, totals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + totals[-1] * 0.01,
                    f"{v:,}", ha="center", va="bottom", fontsize=9)

        ax = axes[1]
        bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                      linewidth=0.8, capsize=4, error_kw={"elinewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("Grounding", "Grounding\n") for m in models], ha="center")
        ax.set_ylabel("Annotations per image")
        ax.set_title("Annotations per Image (mean ± std)")
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(stds) * 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        fig.suptitle("Annotation Volume Comparison", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        save_fig(fig, out_dir, "fig1_annotation_counts")


def fig_count_distribution(
    model_data: dict[str, ModelDataset], out_dir: Path
) -> None:
    """Violin + box plot of per-image annotation count distribution."""
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7, 5))

        data = [model_data[m].counts for m in MODELS]
        parts = ax.violinplot(data, positions=range(1, len(MODELS) + 1),
                              showmedians=False, showextrema=False)
        for i, (pc, m) in enumerate(zip(parts["bodies"], MODELS)):
            pc.set_facecolor(MODEL_COLORS[m])
            pc.set_alpha(0.5)

        bp = ax.boxplot(data, positions=range(1, len(MODELS) + 1),
                        widths=0.15, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2},
                        whiskerprops={"linewidth": 1.2},
                        capprops={"linewidth": 1.2},
                        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
        for patch, m in zip(bp["boxes"], MODELS):
            patch.set_facecolor(MODEL_COLORS[m])
            patch.set_alpha(0.8)

        ax.set_xticks(range(1, len(MODELS) + 1))
        ax.set_xticklabels(MODELS)
        ax.set_ylabel("Annotations per image")
        ax.set_title("Distribution of Annotations per Image")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        patches = [mpatches.Patch(color=MODEL_COLORS[m], label=m) for m in MODELS]
        ax.legend(handles=patches, loc="upper right")

        fig.tight_layout()
        save_fig(fig, out_dir, "fig2_count_distribution")


def fig_area_distribution(
    model_data: dict[str, ModelDataset], out_dir: Path
) -> None:
    """Overlapping histograms of annotation area as % of image."""
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7, 5))

        bins = np.linspace(0, 0.5, 51)
        for m in MODELS:
            af = np.array(model_data[m].areas_frac) * 100  # as percentage
            af = af[af <= 50]  # clip extreme outliers for readability
            ax.hist(af, bins=bins * 100, alpha=0.55,
                    color=MODEL_COLORS[m], label=m, density=True, edgecolor="none")

        ax.set_xlabel("Annotation area (% of image)")
        ax.set_ylabel("Density")
        ax.set_title("Annotation Area Distribution")
        ax.legend()
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        fig.tight_layout()
        save_fig(fig, out_dir, "fig3_area_distribution")


def fig_size_class(
    model_data: dict[str, ModelDataset], out_dir: Path
) -> None:
    """Stacked bar chart: COCO size class breakdown per model."""
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        size_colors = {"small": "#EF5350", "medium": "#FFA726", "large": "#66BB6A"}
        x = np.arange(len(MODELS))
        width = 0.5
        bottoms = np.zeros(len(MODELS))

        for size in ("small", "medium", "large"):
            vals = []
            for m in MODELS:
                total = max(model_data[m].total_annotations, 1)
                count = model_data[m].size_classes.count(size)
                vals.append(count / total * 100)
            ax.bar(x, vals, width, bottom=bottoms, label=size.capitalize(),
                   color=size_colors[size], edgecolor="white", linewidth=0.5)
            bottoms += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)
        ax.set_ylabel("Fraction of annotations (%)")
        ax.set_title("COCO Size Class Distribution\n(small ≤32², medium ≤96², large >96² px)")
        ax.legend(loc="upper right")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        fig.tight_layout()
        save_fig(fig, out_dir, "fig4_size_class")


def fig_pairwise_iou(
    pair_data: dict[str, PairDataset], out_dir: Path
) -> None:
    """Side-by-side box plots of bbox IoU and mask IoU for each model pair."""
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        pair_keys = [f"{a}_vs_{b}" for a, b in PAIRS]
        pair_labels = [f"{a}\nvs\n{b}" for a, b in PAIRS]

        for ax, iou_key, title in zip(
            axes,
            ["bbox_ious", "mask_ious"],
            ["Bounding-Box IoU", "Mask IoU"],
        ):
            data = [getattr(pair_data[pk], iou_key) for pk in pair_keys]
            bp = ax.boxplot(data, patch_artist=True,
                            medianprops={"color": "black", "linewidth": 2},
                            whiskerprops={"linewidth": 1.2},
                            capprops={"linewidth": 1.2},
                            flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.35})
            for patch, color in zip(bp["boxes"], PAIR_COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

            # overlay mean markers
            for i, vals in enumerate(data, 1):
                if vals:
                    ax.scatter(i, statistics.fmean(vals), marker="D",
                               s=40, color="white", zorder=5, edgecolors="black", linewidths=0.8)

            ax.set_xticks(range(1, len(pair_labels) + 1))
            ax.set_xticklabels(pair_labels, fontsize=9)
            ax.set_ylabel("IoU")
            ax.set_title(title)
            ax.set_ylim(-0.02, 1.05)
            ax.yaxis.grid(True, linestyle="--", alpha=0.5)
            ax.set_axisbelow(True)

        fig.suptitle("Pairwise IoU of Matched Annotations\n(diamond = mean)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out_dir, "fig5_pairwise_iou")


def fig_count_scatter(
    per_image_counts: dict[str, dict[str, int]],
    common_files: list[str],
    out_dir: Path,
) -> None:
    """Scatter plot of per-image annotation counts: SAM3 vs the other two models."""
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        for ax, other in zip(axes, [GROUNDED, EFFICIENT]):
            xs = [per_image_counts[SAM3].get(f, 0) for f in common_files]
            ys = [per_image_counts[other].get(f, 0) for f in common_files]
            lim = max(max(xs, default=0), max(ys, default=0)) + 1

            ax.scatter(xs, ys, alpha=0.25, s=12, color=MODEL_COLORS[other], edgecolors="none")
            ax.plot([0, lim], [0, lim], "k--", linewidth=0.9, alpha=0.5, label="y = x")
            ax.set_xlabel(f"{SAM3} count")
            ax.set_ylabel(f"{other} count")
            ax.set_title(f"{SAM3} vs {other}")
            ax.set_xlim(-0.5, lim)
            ax.set_ylim(-0.5, lim)
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=9)

        fig.suptitle("Per-Image Annotation Count Agreement",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out_dir, "fig6_count_scatter")


def fig_detection_rate(
    model_data: dict[str, ModelDataset], out_dir: Path
) -> None:
    """Horizontal bar: detection rate (fraction of images with ≥1 annotation)."""
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7, 3))
        rates = [model_data[m].detection_rate * 100 for m in MODELS]
        y = np.arange(len(MODELS))
        bars = ax.barh(y, rates, color=[MODEL_COLORS[m] for m in MODELS],
                       edgecolor="white", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(MODELS)
        ax.set_xlabel("Detection rate (%)")
        ax.set_title("Images with ≥1 Annotation (Detection Rate)")
        ax.set_xlim(0, 105)
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, rates):
            ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=10)
        fig.tight_layout()
        save_fig(fig, out_dir, "fig7_detection_rate")


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def fmt(v: float | None, decimals: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{decimals}f}"


def print_model_summary(model_data: dict[str, ModelDataset]) -> None:
    print("\n" + "=" * 80)
    print(" ANNOTATION VOLUME & QUALITY — PER MODEL")
    print("=" * 80)
    header = f"{'Metric':40s}" + "".join(f"{m:>20s}" for m in MODELS)
    print(header)
    print("-" * 80)

    def row(label: str, values: list):
        print(f"{label:40s}" + "".join(f"{str(v):>20s}" for v in values))

    summaries = {m: model_data[m].summary() for m in MODELS}

    row("Common images", [summaries[m]["n_images"] for m in MODELS])
    row("Images with ≥1 annotation", [summaries[m]["n_images_non_empty"] for m in MODELS])
    row("Detection rate", [f"{summaries[m]['detection_rate']:.1%}" for m in MODELS])
    row("Total annotations", [f"{summaries[m]['total_annotations']:,}" for m in MODELS])
    row("Anns/image — mean", [fmt(summaries[m]["annotations_per_image"]["mean"]) for m in MODELS])
    row("Anns/image — median", [fmt(summaries[m]["annotations_per_image"]["median"]) for m in MODELS])
    row("Anns/image — std", [fmt(summaries[m]["annotations_per_image"]["stdev"]) for m in MODELS])
    row("Anns/image — max", [summaries[m]["annotations_per_image"]["max"] for m in MODELS])
    row("Area fraction — mean (%)", [f"{summaries[m]['annotation_area_fraction']['mean']*100:.2f}" for m in MODELS])
    row("Area fraction — median (%)", [f"{summaries[m]['annotation_area_fraction']['median']*100:.2f}" for m in MODELS])
    row("Small annotations", [summaries[m]["size_class_counts"]["small"] for m in MODELS])
    row("Medium annotations", [summaries[m]["size_class_counts"]["medium"] for m in MODELS])
    row("Large annotations", [summaries[m]["size_class_counts"]["large"] for m in MODELS])
    print("=" * 80)


def print_pairwise_summary(pair_data: dict[str, PairDataset]) -> None:
    print("\n" + "=" * 80)
    print(" PAIRWISE AGREEMENT METRICS")
    print("=" * 80)

    for a, b in PAIRS:
        key = f"{a}_vs_{b}"
        pd = pair_data[key]
        s = pd.summary()
        print(f"\n  {a}  vs  {b}")
        print(f"  {'Matched annotation pairs':40s} {s['matched_pairs']:>10,}")
        print(f"  {'Precision (A matched / total A)':40s} {s['precision_a']:>10.3f}")
        print(f"  {'Recall   (B matched / total B)':40s} {s['recall_b']:>10.3f}")
        print(f"  {'F1 score':40s} {s['f1']:>10.3f}")
        print(f"  {'BBox IoU — mean':40s} {fmt(s['mean_bbox_iou']):>10s}")
        print(f"  {'BBox IoU — median':40s} {fmt(s['median_bbox_iou']):>10s}")
        print(f"  {'BBox IoU — std':40s} {fmt(s['stdev_bbox_iou']):>10s}")
        print(f"  {'Mask IoU — mean':40s} {fmt(s['mean_mask_iou']):>10s}")
        print(f"  {'Mask IoU — median':40s} {fmt(s['median_mask_iou']):>10s}")
        print(f"  {'Mask IoU — std':40s} {fmt(s['stdev_mask_iou']):>10s}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------


def build_latex(
    model_data: dict[str, ModelDataset],
    pair_data: dict[str, PairDataset],
) -> str:
    s = {m: model_data[m].summary() for m in MODELS}
    lines = []

    lines.append(r"% ---- Table 1: Per-model annotation statistics ----")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Annotation statistics for each auto-labelling model on the common image set.}")
    lines.append(r"\label{tab:annotation_stats}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{SAM3} & \textbf{GroundingSAM} & \textbf{Eff.\ GroundingSAM} \\")
    lines.append(r"\midrule")

    def tr(label: str, vals: list) -> str:
        return f"{label} & " + " & ".join(str(v) for v in vals) + r" \\"

    lines.append(tr("Images (common)", [f"{s[m]['n_images']:,}" for m in MODELS]))
    lines.append(tr("Detection rate", [f"{s[m]['detection_rate']:.1%}" for m in MODELS]))
    lines.append(tr("Total annotations", [f"{s[m]['total_annotations']:,}" for m in MODELS]))
    lines.append(tr(r"Anns/image (mean$\pm$std)",
        [rf"{s[m]['annotations_per_image']['mean']:.2f}$\pm${s[m]['annotations_per_image']['stdev']:.2f}"
         for m in MODELS]))
    lines.append(tr("Anns/image (median)", [f"{s[m]['annotations_per_image']['median']:.2f}" for m in MODELS]))
    lines.append(tr(r"Area fraction, mean (\%)",
        [f"{s[m]['annotation_area_fraction']['mean']*100:.2f}" for m in MODELS]))
    lines.append(tr("Small / Medium / Large",
        ["{}/{}/{}".format(
            s[m]["size_class_counts"]["small"],
            s[m]["size_class_counts"]["medium"],
            s[m]["size_class_counts"]["large"],
        ) for m in MODELS]))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append(r"% ---- Table 2: Pairwise agreement metrics ----")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pairwise agreement between auto-labelling models. "
                 r"Precision and Recall are computed over matched annotation pairs "
                 r"(greedy bbox-IoU matching). IoU values are means over matched pairs.}")
    lines.append(r"\label{tab:pairwise}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Pair} & \textbf{Matched} & \textbf{Prec.} & \textbf{Rec.} "
                 r"& \textbf{F1} & \textbf{BBox IoU} & \textbf{Mask IoU} \\")
    lines.append(r"\midrule")
    for a, b in PAIRS:
        key = f"{a}_vs_{b}"
        ps = pair_data[key].summary()
        label = f"{a} vs {b}"
        lines.append(
            rf"{label} & {ps['matched_pairs']:,} & {ps['precision_a']:.3f} & "
            rf"{ps['recall_b']:.3f} & {ps['f1']:.3f} & "
            rf"{fmt(ps['mean_bbox_iou'])} & {fmt(ps['mean_mask_iou'])} \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Research-quality annotation quality comparison for SAM3, GroundingSAM, EfficientGroundingSAM."
    )
    parser.add_argument("--sam3-dir", type=Path, default=Path("dataset/labels_3fps"))
    parser.add_argument("--grounded-dir", type=Path, default=Path("dataset/labels_grounding_sam_3fps"))
    parser.add_argument(
        "--efficient-dir",
        type=Path,
        default=Path("dataset/labels_efficient_grounded_sam_3fps"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/paper_figures"),
        help="Directory for figures, LaTeX snippets, and JSON report.",
    )
    parser.add_argument(
        "--bbox-iou-threshold",
        type=float,
        default=0.5,
        help="Minimum bbox IoU to count two annotations as matching (default: 0.5).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Restrict to specific dataset splits (e.g. --splits train test).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap number of images processed (for quick testing).",
    )
    args = parser.parse_args()

    dirs = {SAM3: args.sam3_dir, GROUNDED: args.grounded_dir, EFFICIENT: args.efficient_dir}
    for name, d in dirs.items():
        if not d.exists():
            print(f"ERROR: {name} directory not found: {d}", file=sys.stderr)
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load scene file maps ----
    scene_maps = {m: list_scene_files(dirs[m]) for m in MODELS}

    all_splits = sorted(
        set(scene_maps[SAM3]) & set(scene_maps[GROUNDED]) & set(scene_maps[EFFICIENT])
    )
    if args.splits:
        all_splits = [s for s in all_splits if s in set(args.splits)]

    if not all_splits:
        print("No overlapping splits found.", file=sys.stderr)
        return 1

    # ---- accumulate per-image statistics ----
    model_data: dict[str, ModelDataset] = {m: ModelDataset(model=m) for m in MODELS}
    pair_data: dict[str, PairDataset] = {
        f"{a}_vs_{b}": PairDataset(model_a=a, model_b=b) for a, b in PAIRS
    }
    per_image_counts: dict[str, dict[str, int]] = {m: {} for m in MODELS}
    common_files_all: list[str] = []
    n_processed = 0

    print(f"Processing {len(all_splits)} split(s): {', '.join(all_splits)}")

    for split in all_splits:
        common_scenes = sorted(
            set(scene_maps[SAM3].get(split, {}))
            & set(scene_maps[GROUNDED].get(split, {}))
            & set(scene_maps[EFFICIENT].get(split, {}))
        )
        print(f"  {split}: {len(common_scenes)} scenes")

        for scene in common_scenes:
            loaded = {}
            all_dims = {}
            for m in MODELS:
                dims, anns = load_scene(scene_maps[m][split][scene])
                loaded[m] = anns
                all_dims.update(dims)

            common_files = sorted(
                set(loaded[SAM3]) & set(loaded[GROUNDED]) & set(loaded[EFFICIENT])
            )

            for fname in common_files:
                if args.max_images and n_processed >= args.max_images:
                    break
                h, w = all_dims.get(fname, (720, 1280))

                # per-model stats
                for m in MODELS:
                    ist = image_stats(loaded[m][fname], h, w)
                    model_data[m].add(ist)
                    per_image_counts[m][fname] = ist.count

                # pairwise stats
                for a, b in PAIRS:
                    key = f"{a}_vs_{b}"
                    pst = pair_image_stats(
                        loaded[a][fname], loaded[b][fname], h, w, args.bbox_iou_threshold
                    )
                    pair_data[key].add(pst)

                common_files_all.append(fname)
                n_processed += 1

    print(f"\nTotal common images processed: {n_processed}")

    # ---- console summary ----
    print_model_summary(model_data)
    print_pairwise_summary(pair_data)

    # ---- figures ----
    print(f"\nGenerating figures in {args.output_dir} …")
    fig_annotation_counts(model_data, args.output_dir)
    fig_count_distribution(model_data, args.output_dir)
    fig_area_distribution(model_data, args.output_dir)
    fig_size_class(model_data, args.output_dir)
    fig_pairwise_iou(pair_data, args.output_dir)
    fig_count_scatter(per_image_counts, common_files_all, args.output_dir)
    fig_detection_rate(model_data, args.output_dir)
    print("  fig1_annotation_counts.{pdf,png}  — total & mean annotations per model")
    print("  fig2_count_distribution.{pdf,png} — violin + box of annotations/image")
    print("  fig3_area_distribution.{pdf,png}  — annotation area fraction histograms")
    print("  fig4_size_class.{pdf,png}          — COCO size class stacked bars")
    print("  fig5_pairwise_iou.{pdf,png}        — bbox & mask IoU box plots per pair")
    print("  fig6_count_scatter.{pdf,png}       — per-image count scatter plots")
    print("  fig7_detection_rate.{pdf,png}      — detection rate per model")

    # ---- LaTeX ----
    latex = build_latex(model_data, pair_data)
    latex_path = args.output_dir / "tables.tex"
    latex_path.write_text(latex)
    print(f"\nLaTeX tables written to: {latex_path}")
    print("\n" + latex)

    # ---- JSON report ----
    report = {
        "meta": {
            "common_images": n_processed,
            "bbox_iou_threshold": args.bbox_iou_threshold,
            "splits": all_splits,
            "label_dirs": {m: str(dirs[m]) for m in MODELS},
        },
        "per_model": {m: model_data[m].summary() for m in MODELS},
        "pairwise": {k: v.summary() for k, v in pair_data.items()},
    }
    json_path = args.output_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\nJSON report written to: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
