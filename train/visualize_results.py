"""
Visualize training results for all runs under train/golden_mid/runs/.

Produces per-task figures (one per detect/segment):
  - Training + validation loss curves (all models overlaid)
  - Metrics curves: precision, recall, mAP50, mAP50-95
  - Final-epoch summary bar chart

Output: train/golden_mid/figures/
Usage:  uv run python train/golden_mid/visualize_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

HERE = Path(__file__).parent
RUNS_DIR = HERE / "runs"
FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── column groups ─────────────────────────────────────────────────────────────

TRAIN_LOSSES = ["train/box_loss", "train/cls_loss", "train/dfl_loss", "train/seg_loss"]
VAL_LOSSES   = ["val/box_loss",   "val/cls_loss",   "val/dfl_loss",   "val/seg_loss"]

DET_METRICS = {
    "Precision (B)":  "metrics/precision(B)",
    "Recall (B)":     "metrics/recall(B)",
    "mAP50 (B)":      "metrics/mAP50(B)",
    "mAP50-95 (B)":   "metrics/mAP50-95(B)",
}
SEG_METRICS = {
    "Precision (M)":  "metrics/precision(M)",
    "Recall (M)":     "metrics/recall(M)",
    "mAP50 (M)":      "metrics/mAP50(M)",
    "mAP50-95 (M)":   "metrics/mAP50-95(M)",
}

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_runs(task: str) -> dict[str, pd.DataFrame]:
    """Return {model_name: DataFrame} for every completed run of `task`."""
    task_dir = RUNS_DIR / task
    if not task_dir.exists():
        return {}
    runs = {}
    for run_dir in sorted(task_dir.iterdir()):
        csv = run_dir / "results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df.columns = df.columns.str.strip()
            runs[run_dir.name] = df
    return runs


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def plot_loss_curves(runs: dict[str, pd.DataFrame], task: str) -> None:
    """Two-panel figure: train losses (left) and val losses (right)."""
    # collect which loss columns actually appear
    all_train = _present(next(iter(runs.values())), TRAIN_LOSSES)
    all_val   = _present(next(iter(runs.values())), VAL_LOSSES)

    ncols = len(all_train)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7), sharex=True)
    if ncols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"Loss curves — {task}", fontsize=13, fontweight="bold")

    for col_idx, (train_col, val_col) in enumerate(zip(all_train, all_val)):
        label_name = train_col.replace("train/", "").replace("_loss", "")
        ax_tr = axes[0, col_idx]
        ax_val = axes[1, col_idx]

        for i, (model, df) in enumerate(runs.items()):
            color = COLORS[i % len(COLORS)]
            epochs = df["epoch"]
            if train_col in df:
                ax_tr.plot(epochs, df[train_col], color=color, label=model)
            if val_col in df:
                ax_val.plot(epochs, df[val_col], color=color, label=model)

        ax_tr.set_title(f"{label_name} loss")
        ax_tr.set_ylabel("Train loss")
        ax_tr.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax_tr.grid(alpha=0.3)

        ax_val.set_ylabel("Val loss")
        ax_val.set_xlabel("Epoch")
        ax_val.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax_val.grid(alpha=0.3)

    # single legend on last column top panel
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    out = FIGURES_DIR / f"{task}_loss_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_metric_curves(runs: dict[str, pd.DataFrame], task: str) -> None:
    """One row of subplots for box metrics, optional second row for mask metrics."""
    sample_df = next(iter(runs.values()))
    rows_data = [("Box metrics", DET_METRICS)]
    if any(c in sample_df.columns for c in SEG_METRICS.values()):
        rows_data.append(("Mask metrics", SEG_METRICS))

    nrows = len(rows_data)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=True)
    if nrows == 1:
        axes = axes.reshape(1, ncols)

    fig.suptitle(f"Validation metrics — {task}", fontsize=13, fontweight="bold")

    for row_idx, (row_title, metric_dict) in enumerate(rows_data):
        for col_idx, (label, col) in enumerate(metric_dict.items()):
            ax = axes[row_idx, col_idx]
            for i, (model, df) in enumerate(runs.items()):
                if col in df.columns:
                    ax.plot(df["epoch"], df[col], color=COLORS[i % len(COLORS)], label=model)
            ax.set_title(f"{label}")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel(row_title)

    handles, labels_ = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    out = FIGURES_DIR / f"{task}_metric_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_summary_bar(runs: dict[str, pd.DataFrame], task: str) -> None:
    """Bar chart comparing final-epoch key metrics across models."""
    key_metrics: dict[str, str] = {
        "mAP50 (B)":    "metrics/mAP50(B)",
        "mAP50-95 (B)": "metrics/mAP50-95(B)",
        "Prec (B)":     "metrics/precision(B)",
        "Recall (B)":   "metrics/recall(B)",
    }
    sample_df = next(iter(runs.values()))
    if "metrics/mAP50(M)" in sample_df.columns:
        key_metrics["mAP50 (M)"]    = "metrics/mAP50(M)"
        key_metrics["mAP50-95 (M)"] = "metrics/mAP50-95(M)"

    # use row with best mAP50(B) as "final" (avoid last epoch if early-stopped)
    best_col = "metrics/mAP50(B)"
    records = {}
    for model, df in runs.items():
        if best_col in df.columns:
            best_row = df.loc[df[best_col].idxmax()]
        else:
            best_row = df.iloc[-1]
        records[model] = {k: best_row.get(v, float("nan")) for k, v in key_metrics.items()}

    summary = pd.DataFrame(records).T  # models × metrics

    n_models  = len(summary)
    n_metrics = len(summary.columns)
    x = range(n_metrics)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 2), 5))
    for i, (model, row) in enumerate(summary.iterrows()):
        offsets = [xi + (i - n_models / 2 + 0.5) * width for xi in x]
        bars = ax.bar(offsets, row.values, width=width * 0.9,
                      color=COLORS[i % len(COLORS)], label=model)
        for bar, val in zip(bars, row.values):
            if not pd.isna(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(list(x))
    ax.set_xticklabels(summary.columns, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (at best mAP50 epoch)")
    ax.set_title(f"Best-epoch summary — {task}", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / f"{task}_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


def print_summary_table(runs: dict[str, pd.DataFrame], task: str) -> None:
    """Print a compact table of best-epoch metrics to stdout."""
    cols_of_interest = [
        "metrics/precision(B)", "metrics/recall(B)",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "metrics/mAP50(M)", "metrics/mAP50-95(M)",
    ]
    print(f"\n{'='*60}")
    print(f"  {task.upper()} — best-epoch metrics")
    print(f"{'='*60}")
    for model, df in runs.items():
        best_col = "metrics/mAP50(B)"
        row = df.loc[df[best_col].idxmax()] if best_col in df.columns else df.iloc[-1]
        parts = [f"{model:20s}"]
        for c in cols_of_interest:
            if c in df.columns:
                short = c.split("/")[1]
                parts.append(f"{short}={row[c]:.4f}")
        print("  " + "  ".join(parts))
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    plt.rcParams.update({"font.size": 9, "figure.dpi": 100})

    tasks_found = False
    for task in ("detect", "segment"):
        runs = load_runs(task)
        if not runs:
            print(f"[{task}] No completed runs found — skipping.")
            continue
        tasks_found = True
        print(f"\n[{task}] Found {len(runs)} run(s): {', '.join(runs)}")
        plot_loss_curves(runs, task)
        plot_metric_curves(runs, task)
        plot_summary_bar(runs, task)
        print_summary_table(runs, task)

    if not tasks_found:
        print("No runs found. Train some models first with train.py.")
    else:
        print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
