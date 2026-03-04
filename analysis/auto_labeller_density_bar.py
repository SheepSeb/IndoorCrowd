#!/usr/bin/env python3
"""
Auto-labeller quality vs. density (grouped bar chart).

Reads the golden-mid evaluation report
  analysis/reports/human_vs_models_golden_mid.json
and produces a compact grouped bar chart:

  • X-axis: crowd density bins  (Sparse / Medium / High)
  • Y-axis: AP@0.5  or  Cohen's κ
  • One colour per method: SAM3, GroundingSAM, EfficientGroundingSAM

Density bins are inferred from the human scene statistics in the same JSON:
  - We sort scene groups by mean persons/frame.
  - Lowest-density group → "Sparse"
  - Highest-density group → "High"
  - All remaining groups → "Medium"

Usage
-----
Default (Cohen's κ):
    uv run python analysis/auto_labeller_density_bar.py

AP@0.5 instead:
    uv run python analysis/auto_labeller_density_bar.py --metric ap
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "analysis" / "reports" / "human_vs_models_golden_mid.json"
FIGURES_DIR = REPO_ROOT / "analysis" / "figures"


@dataclass
class DensityMetrics:
    """Aggregated (weighted) metric values for one density bin."""

    # weights are number of GT instances (n_gt) per scene-group/method
    total_weight: float = 0.0
    ap50_sum: float = 0.0
    kappa_sum: float = 0.0

    def add(self, ap50: float, kappa: float, weight: float) -> None:
        if weight <= 0:
            return
        self.total_weight += weight
        self.ap50_sum += ap50 * weight
        self.kappa_sum += kappa * weight

    def ap50_mean(self) -> float:
        return self.ap50_sum / self.total_weight if self.total_weight > 0 else np.nan

    def kappa_mean(self) -> float:
        return self.kappa_sum / self.total_weight if self.total_weight > 0 else np.nan


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Report JSON not found: {path}")
    with path.open() as f:
        return json.load(f)


def assign_density_bins(scene_stats: Dict[str, dict]) -> Dict[str, str]:
    """
    Map scene_group → 'Sparse' | 'Medium' | 'High' based on mean persons/frame.

    Strategy:
      - Sort scene groups by person_count_mean.
      - Lowest-density scene → 'Sparse'
      - Highest-density scene → 'High'
      - All remaining scenes → 'Medium'
    """
    if not scene_stats:
        raise ValueError("scene_stats is empty; run human_vs_models_golden_mid.py first.")

    # Sort by mean crowd size
    ordered = sorted(
        scene_stats.items(),
        key=lambda kv: kv[1].get("person_count_mean", 0.0),
    )
    if len(ordered) < 2:
        # Degenerate case: only one scene group
        sg = ordered[0][0]
        return {sg: "Medium"}

    lowest_scene = ordered[0][0]
    highest_scene = ordered[-1][0]

    mapping: Dict[str, str] = {}
    for scene, _stats in ordered:
        if scene == lowest_scene:
            mapping[scene] = "Sparse"
        elif scene == highest_scene:
            mapping[scene] = "High"
        else:
            mapping[scene] = "Medium"

    return mapping


def aggregate_by_density(
    report: dict,
) -> Dict[str, Dict[str, DensityMetrics]]:
    """
    Returns:
        agg[density_bin][method] = DensityMetrics(...)
    """
    model_metrics: Dict[str, Dict[str, dict]] = report["model_metrics"]
    scene_stats: Dict[str, dict] = report["scene_stats"]

    scene_to_density = assign_density_bins(scene_stats)
    density_bins = ["Sparse", "Medium", "High"]
    methods = ["SAM3", "GroundingSAM", "EfficientGroundingSAM"]

    # Initialise containers
    agg: Dict[str, Dict[str, DensityMetrics]] = {
        d: {m: DensityMetrics() for m in methods} for d in density_bins
    }

    for scene, per_method in model_metrics.items():
        density = scene_to_density.get(scene)
        if density not in density_bins:
            continue

        for method, m in per_method.items():
            if method not in methods:
                continue

            n_gt = float(m.get("n_gt", 0.0))
            ap_by_t = m.get("ap_by_thresh", {})
            ap50 = float(ap_by_t.get("0.5", 0.0))
            kappa = float(m.get("cohen_kappa", 0.0))

            agg[density][method].add(ap50=ap50, kappa=kappa, weight=n_gt)

    return agg


def plot_grouped_bar(
    agg: Dict[str, Dict[str, DensityMetrics]],
    metric: str,
    out_path: Path,
) -> None:
    assert metric in {"kappa", "ap"}, "metric must be 'kappa' or 'ap'"

    density_bins = ["Sparse", "Medium", "High"]
    methods = ["SAM3", "GroundingSAM", "EfficientGroundingSAM"]
    method_colors = {
        "SAM3": "#e07b39",
        "GroundingSAM": "#4a90d9",
        "EfficientGroundingSAM": "#6aaf50",
    }

    x = np.arange(len(density_bins))
    width = 0.22

    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    for i, method in enumerate(methods):
        offsets = (i - (len(methods) - 1) / 2) * width
        vals: List[float] = []
        for d in density_bins:
            dm = agg.get(d, {}).get(method)
            if dm is None:
                vals.append(np.nan)
            else:
                vals.append(dm.kappa_mean() if metric == "kappa" else dm.ap50_mean())

        ax.bar(
            x + offsets,
            vals,
            width,
            label=method,
            color=method_colors.get(method, None),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(density_bins, fontsize=9)
    ax.set_xlabel("Crowd density (persons / frame)", fontsize=10)

    if metric == "kappa":
        ax.set_ylabel("Cohen's κ", fontsize=10)
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylabel("AP@0.5", fontsize=10)
        ax.set_ylim(0.0, 1.0)

    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=8, framealpha=0.9, title="Auto-labeller")

    title_metric = "Cohen's κ" if metric == "kappa" else "AP@0.5"
    fig.suptitle(
        f"Auto-labeller quality vs. crowd density ({title_metric})",
        fontsize=11,
        fontweight="bold",
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Grouped bar chart of auto-labeller quality vs. density "
            "from human_vs_models_golden_mid.json."
        )
    )
    parser.add_argument(
        "--metric",
        choices=["kappa", "ap"],
        default="kappa",
        help="Which metric to plot on the Y-axis (default: kappa).",
    )
    args = parser.parse_args()

    report = load_report(REPORT_PATH)
    agg = aggregate_by_density(report)

    metric_tag = "kappa" if args.metric == "kappa" else "ap50"
    out_name = f"auto_labeller_density_bar_{metric_tag}.png"
    out_path = FIGURES_DIR / out_name

    plot_grouped_bar(agg, metric=args.metric, out_path=out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

