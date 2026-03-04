#!/usr/bin/env python3
"""
Cohen's Kappa vs frame density and relative bounding-box scale.

Answers the question:
  "Do auto-labelers (SAM3-native, SAM3+BoT-SORT) degrade at high crowd
   density (>10 persons/frame) or at small relative scale (<1 % of frame)?"

Method
------
  For each frame, detections from each auto-labeler are greedy-matched to
  human GT boxes (IoU ≥ 0.5).  This yields per-frame TP / FP / FN counts.

  Cohen's Kappa (adapted for detection, no TN):
    N   = TP + FP + FN
    P_o = TP / N                             (observed agreement)
    P_e = (TP+FN)/N · (TP+FP)/N             (chance agreement)
    κ   = (P_o − P_e) / (1 − P_e)

  Frames / boxes are bucketed into:
    Density bins  — persons per frame:  low 1-3 | medium 4-10 | high >10
    Scale bins    — box_area/frame_area: small <1% | medium 1-5% | large >5%

Output
------
  analysis/results/kappa_density.csv
  analysis/results/kappa_scale.csv
  analysis/results/kappa_density_scale.json
  analysis/figures/kappa_density_scale.png

Usage
-----
  uv run python analysis/kappa_density_scale.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parents[1]
MOT_DIR   = REPO_ROOT / "dataset" / "MOT_main"

HUMAN_ROOT   = MOT_DIR / "clips_human"
AUTO_SOURCES = {
    "sam3_native":  MOT_DIR / "labels_sam3_native",
    "sam3_botsort": MOT_DIR / "labels_sam3_botsort",
}

PAPER_NAMES = {
    "acs_s1":   "acs_ec",
    "acs_s2":   "acs_eg",
    "ie":       "ie_central",
    "rectorat": "r_central",
}

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

IOU_THRESH = 0.5

# ── Bins ───────────────────────────────────────────────────────────────────────

DENSITY_BINS = [
    (1,   3,   "Sparse\n(1–3)"),
    (4,   10,  "Medium\n(4–10)"),
    (11,  999, "High\n(>10)"),
]
SCALE_BINS = [
    (0.000, 0.010, "small\n(<1%)"),
    (0.010, 0.050, "med\n(1–5%)"),
    (0.050, 1.000, "large\n(>5%)"),
]

METHOD_COLORS  = {"sam3_native": "#e07b39", "sam3_botsort": "#4a90d9"}
METHOD_LABELS  = {"sam3_native": "SAM3 native", "sam3_botsort": "SAM3+BoT-SORT"}
SCENE_GROUPS   = list(PAPER_NAMES.values())


def scene_group(scene_name: str) -> str:
    key = scene_name.split("_recording_")[0]
    return PAPER_NAMES.get(key, key)


# ── Parsing ────────────────────────────────────────────────────────────────────


def parse_gt_full(path: Path) -> dict[int, dict[int, list[float]]]:
    """{frame: {tid: [x, y, w, h]}}"""
    data: dict[int, dict[int, list[float]]] = defaultdict(dict)
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        frame, tid = int(p[0]), int(p[1])
        data[frame][tid] = [float(p[2]), float(p[3]), float(p[4]), float(p[5])]
    return dict(data)


def parse_auto_with_conf(path: Path) -> dict[int, list[tuple[list[float], float]]]:
    """{frame: [(xywh, conf), ...]}  — reads confidence from column 6."""
    data: dict[int, list] = defaultdict(list)
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 7:
            continue
        frame = int(p[0])
        xywh  = [float(p[2]), float(p[3]), float(p[4]), float(p[5])]
        conf  = float(p[6])
        data[frame].append((xywh, conf))
    return dict(data)


def get_frame_size(clip_dir: Path) -> tuple[int, int]:
    d = clip_dir / "frames"
    if not d.exists():
        return 1920, 1080
    first = next(iter(sorted(d.glob("*.jpg"))), None)
    if first is None:
        return 1920, 1080
    img = cv2.imread(str(first))
    return (img.shape[1], img.shape[0]) if img is not None else (1920, 1080)


# ── IoU matching ───────────────────────────────────────────────────────────────


def _iou(b1: list[float], b2: list[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-6)


def greedy_match(
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
) -> tuple[list[int], list[int]]:
    """
    Greedy max-IoU matching.
    Returns (matched_gt_indices, matched_pred_indices).
    """
    if not gt_boxes or not pred_boxes:
        return [], []
    iou_mat = np.array([[_iou(g, p) for p in pred_boxes] for g in gt_boxes])
    matched_gt, matched_pred = [], []
    used_p = set()
    for gi in np.argsort(-iou_mat.max(axis=1)):
        pi = int(np.argmax(iou_mat[gi]))
        if iou_mat[gi, pi] >= IOU_THRESH and pi not in used_p:
            matched_gt.append(gi)
            matched_pred.append(pi)
            used_p.add(pi)
    return matched_gt, matched_pred


# ── Cohen's Kappa (detection-adapted) ─────────────────────────────────────────


def detection_kappa(tp: int, fp: int, fn: int) -> float:
    N = tp + fp + fn
    if N == 0:
        return float("nan")
    p_o = tp / N
    p_human = (tp + fn) / N
    p_auto  = (tp + fp) / N
    p_e = p_human * p_auto
    if p_e >= 1.0:
        return float("nan")
    return (p_o - p_e) / (1.0 - p_e)


def compute_ap(det_list: list[tuple[float, bool]], n_gt: int) -> float:
    """
    PASCAL-VOC style AP@0.5.
    det_list: [(confidence, is_tp), ...]  (all predictions across all frames in bin)
    n_gt:     total ground-truth boxes in the bin (recall denominator)
    """
    if n_gt == 0 or not det_list:
        return float("nan")
    det_list = sorted(det_list, key=lambda x: -x[0])
    tp_cum, fp_cum = 0, 0
    recalls    = [0.0]
    precisions = [1.0]
    for _, is_tp in det_list:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        recalls.append(tp_cum / n_gt)
        precisions.append(tp_cum / (tp_cum + fp_cum))
    # Monotone decreasing precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    # Area under step function
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def precision_recall_f1(tp, fp, fn) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


# ── Clip discovery ─────────────────────────────────────────────────────────────


def discover_clips() -> list[dict]:
    clips = []
    for split_dir in sorted(HUMAN_ROOT.iterdir()):
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            for clip_dir in sorted(scene_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                rel = Path(split_dir.name) / scene_dir.name / clip_dir.name
                clips.append({
                    "scene_group": scene_group(scene_dir.name),
                    "clip_dir":    clip_dir,
                    "human_gt":    HUMAN_ROOT / rel / "gt" / "gt.txt",
                    "auto_gts":    {m: root / rel / "gt" / "gt.txt"
                                    for m, root in AUTO_SOURCES.items()},
                })
    return clips


# ── Accumulate per-bin stats ───────────────────────────────────────────────────


def density_bin(n: int) -> str | None:
    for lo, hi, label in DENSITY_BINS:
        if lo <= n <= hi:
            return label
    return None


def scale_bin(rel_area: float) -> str | None:
    for lo, hi, label in SCALE_BINS:
        if lo <= rel_area < hi:
            return label
    return None


def build_accumulators(clips: list[dict]) -> dict:
    """
    Returns:
      acc["density"][scene_group][method][bin_label] = {tp, fp, fn}
      acc["scale"]  [scene_group][method][bin_label] = {tp, fp, fn}
    """
    bin_labels_d = [b[2] for b in DENSITY_BINS]
    bin_labels_s = [b[2] for b in SCALE_BINS]

    def _empty(labels):
        return {sg: {m: {lbl: {"tp": 0, "fp": 0, "fn": 0}
                         for lbl in labels}
                     for m in AUTO_SOURCES}
                for sg in SCENE_GROUPS}

    acc = {
        "density": _empty(bin_labels_d),
        "scale":   _empty(bin_labels_s),
    }

    for clip in clips:
        sg = clip["scene_group"]
        if sg not in SCENE_GROUPS:
            continue

        w, h   = get_frame_size(clip["clip_dir"])
        frame_area = w * h
        h_data = parse_gt_full(clip["human_gt"])

        for method, auto_path in clip["auto_gts"].items():
            a_data = parse_gt_full(auto_path)

            all_frames = set(h_data.keys()) | set(a_data.keys())
            for fnum in all_frames:
                h_frame = h_data.get(fnum, {})
                a_frame = a_data.get(fnum, {})

                gt_boxes   = list(h_frame.values())
                pred_boxes = list(a_frame.values())
                n_gt, n_pred = len(gt_boxes), len(pred_boxes)

                m_gt, m_pred = greedy_match(gt_boxes, pred_boxes)
                tp = len(m_gt)
                fp = n_pred - tp
                fn = n_gt  - tp

                # ── density bin (keyed by GT count) ──
                d_lbl = density_bin(n_gt)
                if d_lbl and n_gt > 0:
                    b = acc["density"][sg][method][d_lbl]
                    b["tp"] += tp; b["fp"] += fp; b["fn"] += fn

                # ── scale bins ──
                # TP / FN attributed to GT box relative area
                unmatched_gt = [i for i in range(n_gt) if i not in m_gt]
                for gi in range(n_gt):
                    rel = (gt_boxes[gi][2] * gt_boxes[gi][3]) / frame_area
                    lbl = scale_bin(rel)
                    if lbl is None:
                        continue
                    b = acc["scale"][sg][method][lbl]
                    if gi in m_gt:
                        b["tp"] += 1
                    else:
                        b["fn"] += 1

                # FP attributed to pred box relative area
                unmatched_pred = [i for i in range(n_pred) if i not in m_pred]
                for pi in unmatched_pred:
                    rel = (pred_boxes[pi][2] * pred_boxes[pi][3]) / frame_area
                    lbl = scale_bin(rel)
                    if lbl:
                        acc["scale"][sg][method][lbl]["fp"] += 1

    return acc


# ── AP@0.5 accumulators (confidence-aware, per density bin) ───────────────────


def build_ap_accumulators(clips: list[dict]) -> dict:
    """
    Returns ap_acc[scene_group][method][bin_label] = {
        "det_list": [(conf, is_tp), ...],
        "n_gt":     int,
    }
    Matching is done in confidence order (highest first) within each frame.
    """
    bin_labels_d = [b[2] for b in DENSITY_BINS]

    ap_acc = {
        sg: {
            m: {lbl: {"det_list": [], "n_gt": 0} for lbl in bin_labels_d}
            for m in AUTO_SOURCES
        }
        for sg in SCENE_GROUPS
    }

    for clip in clips:
        sg = clip["scene_group"]
        if sg not in SCENE_GROUPS:
            continue

        h_data = parse_gt_full(clip["human_gt"])

        for method, auto_path in clip["auto_gts"].items():
            a_data = parse_auto_with_conf(auto_path)

            all_frames = set(h_data.keys()) | set(a_data.keys())
            for fnum in all_frames:
                h_frame  = h_data.get(fnum, {})
                a_list   = a_data.get(fnum, [])   # [(xywh, conf), ...]

                n_gt  = len(h_frame)
                d_lbl = density_bin(n_gt)
                if d_lbl is None or n_gt == 0:
                    continue

                gt_boxes = list(h_frame.values())
                # Process predictions highest-confidence first
                a_sorted = sorted(a_list, key=lambda x: -x[1])
                used_gt  = set()
                entries  = []
                for box, conf in a_sorted:
                    best_iou, best_gi = 0.0, -1
                    for gi, gt_box in enumerate(gt_boxes):
                        if gi in used_gt:
                            continue
                        v = _iou(gt_box, box)
                        if v > best_iou:
                            best_iou, best_gi = v, gi
                    is_tp = best_iou >= IOU_THRESH
                    if is_tp:
                        used_gt.add(best_gi)
                    entries.append((conf, is_tp))

                b = ap_acc[sg][method][d_lbl]
                b["det_list"].extend(entries)
                b["n_gt"] += n_gt

    return ap_acc


# ── Flatten to rows ────────────────────────────────────────────────────────────


def flatten_rows(acc: dict, dim: str) -> list[dict]:
    rows = []
    for sg in SCENE_GROUPS:
        for method in AUTO_SOURCES:
            for lbl, b in acc[dim][sg][method].items():
                tp, fp, fn = b["tp"], b["fp"], b["fn"]
                p, r, f1 = precision_recall_f1(tp, fp, fn)
                k = detection_kappa(tp, fp, fn)
                rows.append({
                    "scene_group": sg,
                    "method":      method,
                    "bin":         lbl.replace("\n", " "),
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision":   round(p,  4),
                    "recall":      round(r,  4),
                    "f1":          round(f1, 4),
                    "kappa":       round(k,  4) if not np.isnan(k) else None,
                })
    return rows


# ── Plotting ───────────────────────────────────────────────────────────────────


def figure_density_central(
    acc: dict,
    ap_acc: dict,
    d_bin_labels_raw: list[str],
    out_path: Path,
) -> None:
    """
    Central paper figure: AP@0.5 (left) and Cohen's κ (right) vs crowd density.
    Aggregate line across all scenes; per-scene values shown as faint dots.
    """
    x        = np.arange(len(d_bin_labels_raw))
    x_labels = [lbl.replace("\n", " ") for lbl in d_bin_labels_raw]

    style = {
        "sam3_native":  {"marker": "o", "linestyle": "-",  "linewidth": 2.0, "markersize": 9,  "zorder": 4},
        "sam3_botsort": {"marker": "s", "linestyle": "--", "linewidth": 2.0, "markersize": 8,  "zorder": 4},
    }

    fig, (ax_ap, ax_k) = plt.subplots(
        1, 2, figsize=(8, 3.8),
        gridspec_kw={"wspace": 0.38},
    )

    for method, color in METHOD_COLORS.items():
        ap_agg, k_agg = [], []
        ap_per_scene  = {sg: [] for sg in SCENE_GROUPS}
        k_per_scene   = {sg: [] for sg in SCENE_GROUPS}

        for lbl in d_bin_labels_raw:
            # ── aggregate AP ──
            combined, n_gt_total = [], 0
            for sg in SCENE_GROUPS:
                e = ap_acc[sg][method][lbl]
                combined.extend(e["det_list"])
                n_gt_total += e["n_gt"]
            ap = compute_ap(combined, n_gt_total)
            ap_agg.append(ap if not np.isnan(ap) else np.nan)

            # ── per-scene AP ──
            for sg in SCENE_GROUPS:
                e = ap_acc[sg][method][lbl]
                v = compute_ap(e["det_list"], e["n_gt"])
                ap_per_scene[sg].append(v if not np.isnan(v) else np.nan)

            # ── aggregate κ ──
            tp = fp = fn = 0
            for sg in SCENE_GROUPS:
                b = acc["density"][sg][method][lbl]
                tp += b["tp"]; fp += b["fp"]; fn += b["fn"]
            k = detection_kappa(tp, fp, fn)
            k_agg.append(k if not np.isnan(k) else np.nan)

            # ── per-scene κ ──
            for sg in SCENE_GROUPS:
                b = acc["density"][sg][method][lbl]
                v = detection_kappa(b["tp"], b["fp"], b["fn"])
                k_per_scene[sg].append(v if not np.isnan(v) else np.nan)

        # Per-scene scatter (faint, same colour)
        for sg in SCENE_GROUPS:
            ax_ap.plot(x, ap_per_scene[sg], marker=style[method]["marker"],
                       linestyle=":", color=color, alpha=0.20,
                       markersize=5, linewidth=0.8, zorder=2)
            ax_k.plot(x,  k_per_scene[sg],  marker=style[method]["marker"],
                      linestyle=":", color=color, alpha=0.20,
                      markersize=5, linewidth=0.8, zorder=2)

        # Aggregate line (bold)
        kw = dict(color=color, label=METHOD_LABELS[method], **style[method])
        ax_ap.plot(x, ap_agg, **kw)
        ax_k.plot( x, k_agg,  **kw)

    for ax, ylabel, title in [
        (ax_ap, "AP @ IoU=0.5",   "Average Precision"),
        (ax_k,  "Cohen's κ",      "Cohen's Kappa"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel("Crowd density (persons / frame)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)
        ax.grid(alpha=0.25, axis="y", zorder=0)
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontweight="bold", fontsize=10)

    ax_ap.legend(fontsize=8, loc="lower left", framealpha=0.85)

    fig.suptitle(
        "Auto-labeller quality vs. crowd density",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  figure → {out_path}")


def _get_bin_metrics(
    acc: dict,
    dim: str,
    sg: str,
    method: str,
    raw_labels: list[str],
) -> tuple[list[float], list[float], list[float]]:
    kappas, precisions, recalls = [], [], []
    for raw_lbl in raw_labels:
        b = acc[dim][sg][method].get(raw_lbl, {"tp": 0, "fp": 0, "fn": 0})
        tp, fp, fn = b["tp"], b["fp"], b["fn"]
        k = detection_kappa(tp, fp, fn)
        p, r, _ = precision_recall_f1(tp, fp, fn)
        kappas.append(k if not np.isnan(k) else 0)
        precisions.append(p)
        recalls.append(r)
    return kappas, precisions, recalls


def _fill_subplot(
    ax,
    acc: dict,
    dim: str,
    sg: str,
    raw_labels: list[str],
    tick_labels: list[str],
    xlabel: str,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    x = np.arange(len(raw_labels))
    width = 0.32
    for i, (method, color) in enumerate(METHOD_COLORS.items()):
        kappas, prec, rec = _get_bin_metrics(acc, dim, sg, method, raw_labels)
        offset = (i - 0.5) * width
        ax.bar(x + offset, kappas, width,
               label=METHOD_LABELS[method], color=color, alpha=0.85, zorder=3)
        ax.plot(x + offset, prec, "o--", color=color,
                alpha=0.65, markersize=4, linewidth=1, label="_P")
        ax.plot(x + offset, rec, "s:", color=color,
                alpha=0.65, markersize=4, linewidth=1, label="_R")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    if show_ylabel:
        ax.set_ylabel("κ / P / R", fontsize=8)
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", zorder=2)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", alpha=0.25, zorder=1)
    ax.tick_params(axis="y", labelsize=7)

    if show_legend:
        handles, lbls = ax.get_legend_handles_labels()
        bar_handles = [h for h, l in zip(handles, lbls) if not l.startswith("_")]
        bar_lbls    = [l for l in lbls if not l.startswith("_")]
        ax.legend(bar_handles[:2], bar_lbls[:2],
                  loc="lower right", fontsize=7, framealpha=0.7)


def combined_plot(
    acc: dict,
    d_bin_labels_raw: list[str],
    s_bin_labels_raw: list[str],
    out_path: Path,
) -> None:
    """
    Single figure: rows = scene groups, cols = [density | scale].
    Bars = Cohen's κ; overlaid lines = Precision (○--) / Recall (□·).
    """
    nrows = len(SCENE_GROUPS)
    fig, axes = plt.subplots(
        nrows, 2,
        figsize=(12, 3.2 * nrows),
        gridspec_kw={"wspace": 0.25, "hspace": 0.45},
    )
    if nrows == 1:
        axes = [axes]

    d_tick = [lbl.replace("\n", " ") for lbl in d_bin_labels_raw]
    s_tick = [lbl.replace("\n", " ") for lbl in s_bin_labels_raw]

    for row, sg in enumerate(SCENE_GROUPS):
        ax_d, ax_s = axes[row]

        ax_d.set_title(f"{sg} — density", fontsize=9, fontweight="bold")
        _fill_subplot(ax_d, acc, "density", sg, d_bin_labels_raw, d_tick,
                      "persons / frame", show_ylabel=True,
                      show_legend=(row == 0))

        ax_s.set_title(f"{sg} — scale", fontsize=9, fontweight="bold")
        _fill_subplot(ax_s, acc, "scale", sg, s_bin_labels_raw, s_tick,
                      "box area / frame area", show_ylabel=False,
                      show_legend=False)

    # Shared annotation for line types
    fig.text(
        0.5, 1.002,
        "Bars = Cohen's κ   ○-- = Precision   □· = Recall",
        ha="center", va="bottom", fontsize=9,
        style="italic",
    )
    fig.suptitle(
        "Auto-labeller quality vs. crowd density and detection scale",
        fontsize=11, fontweight="bold", y=1.03,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  figure → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    print("Discovering clips …")
    clips = discover_clips()
    print(f"  {len(clips)} clips")

    print("Building per-bin accumulators …")
    acc = build_accumulators(clips)

    print("Building AP@0.5 accumulators …")
    ap_acc = build_ap_accumulators(clips)

    # Flatten rows using the raw (newline) labels from bins
    d_bin_labels_raw = [b[2] for b in DENSITY_BINS]
    s_bin_labels_raw = [b[2] for b in SCALE_BINS]

    rows_d = flatten_rows(acc, "density")
    rows_s = flatten_rows(acc, "scale")

    # ── Console summary ────────────────────────────────────────────────────────
    def _print_table(rows: list[dict], title: str) -> None:
        print(f"\n\n══ {title} ══")
        W = 88
        hdr = (f"{'Scene':<12} {'Method':<14} {'Bin':<16}"
               f" {'κ':>7} {'P':>7} {'R':>7} {'F1':>7}"
               f" {'TP':>6} {'FP':>6} {'FN':>6}")
        cur = None
        for r in rows:
            sg = r["scene_group"]
            if sg != cur:
                cur = sg
                print(f"\n{'─'*W}  [{sg}]")
                print(hdr)
                print("─" * W)
            k_str = f"{r['kappa']:>7.3f}" if r["kappa"] is not None else "    N/A"
            print(f"{r['scene_group']:<12} {r['method']:<14} {r['bin']:<16}"
                  f" {k_str} {r['precision']:>7.3f} {r['recall']:>7.3f}"
                  f" {r['f1']:>7.3f}"
                  f" {r['tp']:>6} {r['fp']:>6} {r['fn']:>6}")

    _print_table(rows_d, "Cohen's κ vs frame density")
    _print_table(rows_s, "Cohen's κ vs relative box scale")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_density_central(acc, ap_acc, d_bin_labels_raw,
                           FIGURES_DIR / "density_ap_kappa.png")
    combined_plot(acc, d_bin_labels_raw, s_bin_labels_raw,
                  FIGURES_DIR / "kappa_density_scale.png")

    # ── Save CSV / JSON ────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["scene_group", "method", "bin",
            "kappa", "precision", "recall", "f1", "tp", "fp", "fn"]

    csv_d = RESULTS_DIR / "kappa_density.csv"
    with open(csv_d, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows_d)

    csv_s = RESULTS_DIR / "kappa_scale.csv"
    with open(csv_s, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows_s)

    json_path = RESULTS_DIR / "kappa_density_scale.json"
    with open(json_path, "w") as f:
        json.dump({
            "generated":     datetime.now(timezone.utc).isoformat(),
            "iou_threshold": IOU_THRESH,
            "density_bins":  [(lo, hi, lbl.replace("\n", " "))
                              for lo, hi, lbl in DENSITY_BINS],
            "scale_bins":    [(lo, hi, lbl.replace("\n", " "))
                              for lo, hi, lbl in SCALE_BINS],
            "kappa_density": rows_d,
            "kappa_scale":   rows_s,
        }, f, indent=2)

    print(f"\nCSV  density → {csv_d}")
    print(f"CSV  scale   → {csv_s}")
    print(f"JSON         → {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
