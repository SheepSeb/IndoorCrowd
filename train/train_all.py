#!/usr/bin/env python3
"""
Full training pipeline: YOLOv8n (detect + segment), RT-DETR-L (detect),
and YOLOv26n (detect + segment) on the golden_frames_3fps_mid (human-labelled) dataset.

After each model is trained the best weights are evaluated and all metrics
are saved individually (metrics.json per run) and combined in a single report:
  train/golden_mid/runs/report.json
  train/golden_mid/runs/report.csv

Metrics collected
-----------------
  All models   : box mAP@0.5, box mAP@0.5:0.95, Precision, Recall, F1,
                 Inference ms/img, Model size MB, GFLOPs
  Seg models   : + mask mAP@0.5, mask mAP@0.5:0.95, Mean IoU

Usage
-----
  # All models, default 30 epochs / batch 16
  uv run python train/golden_mid/train_all.py

  # Custom settings
  uv run python train/golden_mid/train_all.py --epochs 30 --batch 16

  # Only specific models
  uv run python train/golden_mid/train_all.py --models yolov8n_detect yolov26n_detect
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
DATA_DIR  = HERE / "yolo_dataset"
GOLDEN_DIR = REPO_ROOT / "dataset" / "golden_frames_3fps_mid"
SEBE_JSON  = GOLDEN_DIR / "seb.json"

ALL_MODELS = ["yolov8n_detect", "yolov8n_segment", "rtdetr_l_detect", "yolo26n_detect", "yolo26n_segment"]


# ── Data preparation ───────────────────────────────────────────────────────────


def ensure_data_prepared() -> None:
    detect_yaml = DATA_DIR / "detect" / "data.yaml"
    segment_yaml = DATA_DIR / "segment" / "data.yaml"
    if detect_yaml.exists() and segment_yaml.exists():
        print("YOLO dataset already prepared — skipping prepare_data.py")
        return
    print("Running prepare_data.py …")
    subprocess.run(
        [sys.executable, str(HERE / "prepare_data.py")], check=True
    )


# ── Ultralytics metric extraction ─────────────────────────────────────────────


def _gflops_from_ultralytics(model) -> float | None:
    """Extract GFLOPs from ultralytics model.info()."""
    try:
        info = model.model.info(verbose=True)  # returns (layers, params, grads, flops)
        if info and info[3]:
            return round(float(info[3]), 1)
    except Exception:
        pass
    return None


def _mean_iou_yolo_seg(model, imgsz: int) -> float | None:
    """
    Compute mean mask IoU on the val split for a YOLO-seg model.
    Iterates each val image in seb.json, runs model.predict(), and compares
    predicted masks against GT RLE masks.
    """
    import cv2
    import pycocotools.mask as mask_util

    try:
        with open(SEBE_JSON) as f:
            coco = json.load(f)
    except Exception:
        return None

    val_imgs = [i for i in coco["images"] if i["file_name"].split("/")[0] == "test"]
    id2anns: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        id2anns[ann["image_id"]].append(ann)

    iou_scores: list[float] = []

    for info in val_imgs:
        img_path = GOLDEN_DIR / info["file_name"]
        img_h, img_w = info["height"], info["width"]

        gt_masks: list[np.ndarray] = []
        for ann in id2anns[info["id"]]:
            seg = ann.get("segmentation")
            if seg and isinstance(seg, dict) and "counts" in seg:
                counts = seg["counts"]
                rle = {
                    "size": seg["size"],
                    "counts": counts.encode() if isinstance(counts, str) else counts,
                }
                gt_masks.append(mask_util.decode(rle))
            else:
                x, y, w, h = ann["bbox"]
                binary = np.zeros((img_h, img_w), dtype=np.uint8)
                binary[max(0,int(y)):min(img_h,int(y+h)),
                       max(0,int(x)):min(img_w,int(x+w))] = 1
                gt_masks.append(binary)

        if not gt_masks:
            continue

        results = model.predict(str(img_path), verbose=False, imgsz=imgsz)
        if not results or results[0].masks is None:
            iou_scores.extend([0.0] * len(gt_masks))
            continue

        pred_masks = results[0].masks.data.cpu().numpy()  # (N, H, W) [0..1]
        # Resize to match original image if needed
        if pred_masks.shape[1:] != (img_h, img_w):
            resized = np.stack([
                cv2.resize(pm, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                for pm in pred_masks
            ])
            pred_masks = resized
        pred_bin = (pred_masks > 0.5).astype(np.uint8)

        for gt_mask in gt_masks:
            best = 0.0
            for pm in pred_bin:
                inter = (pm & gt_mask).sum()
                union = (pm | gt_mask).sum()
                best = max(best, inter / (union + 1e-6))
            iou_scores.append(best)

    return round(float(np.mean(iou_scores)), 4) if iou_scores else None


def extract_ultralytics_metrics(
    task: str,
    model_name: str,
    weights_path: Path,
    data_yaml: Path,
    batch: int,
    imgsz: int,
) -> dict:
    """Load best weights, run val, and return full metrics dict."""
    from ultralytics import RTDETR, YOLO

    ModelCls = RTDETR if model_name.startswith("rtdetr") else YOLO
    model    = ModelCls(str(weights_path))

    val_results = model.val(
        data=str(data_yaml), batch=batch, imgsz=imgsz, verbose=False
    )
    box = val_results.box

    precision = float(box.mp)
    recall    = float(box.mr)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    metrics: dict = {
        "task":          task,
        "model":         model_name,
        "box_map50":     round(float(box.map50), 4),
        "box_map5095":   round(float(box.map),   4),
        "mask_map50":    None,
        "mask_map5095":  None,
        "mean_iou":      None,
        "precision":     round(precision, 4),
        "recall":        round(recall,    4),
        "f1":            round(f1,        4),
        "inference_ms":  round(float(val_results.speed.get("inference", 0)), 2),
        "size_mb":       round(weights_path.stat().st_size / 1e6, 2),
        "gflops":        _gflops_from_ultralytics(model),
    }

    if task == "segment" and hasattr(val_results, "seg") and val_results.seg is not None:
        seg = val_results.seg
        metrics["mask_map50"]   = round(float(seg.map50), 4)
        metrics["mask_map5095"] = round(float(seg.map),   4)
        print("  Computing mean IoU on val set …")
        metrics["mean_iou"] = _mean_iou_yolo_seg(model, imgsz)

    return metrics


# ── Per-model training functions ───────────────────────────────────────────────


def _run_ultralytics(
    task: str,
    model_name: str,
    weights_name: str,
    epochs: int,
    batch: int,
    imgsz: int,
) -> dict:
    from ultralytics import RTDETR, YOLO

    data_yaml = DATA_DIR / task / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing {data_yaml} — run prepare_data.py first.")

    ModelCls = RTDETR if model_name.startswith("rtdetr") else YOLO
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    weights_file = REPO_ROOT / f"{weights_name}.pt"
    weights_arg  = str(weights_file) if weights_file.exists() else f"{weights_name}.pt"

    print(f"\n{'='*60}")
    print(f"Task:    {task}")
    print(f"Model:   {weights_arg}")
    print(f"Data:    {data_yaml}")
    print(f"Device:  {device}  |  epochs={epochs}  batch={batch}  imgsz={imgsz}")
    print(f"{'='*60}\n")

    model = ModelCls(weights_arg)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(HERE / "runs" / task),
        name=model_name,
    )

    weights_path = HERE / "runs" / task / model_name / "weights" / "best.pt"
    metrics = extract_ultralytics_metrics(
        task, model_name, weights_path, data_yaml, batch, imgsz
    )
    metrics["weights"] = str(weights_path)

    out_dir = HERE / "runs" / task / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_yolov8n_detect(epochs: int, batch: int, imgsz: int) -> dict:
    return _run_ultralytics("detect", "yolov8n", "yolov8n", epochs, batch, imgsz)


def train_yolov8n_segment(epochs: int, batch: int, imgsz: int) -> dict:
    return _run_ultralytics("segment", "yolov8n-seg", "yolov8n-seg", epochs, batch, imgsz)


def train_rtdetr_detect(epochs: int, batch: int, imgsz: int) -> dict:
    return _run_ultralytics("detect", "rtdetr-l", "rtdetr-l", epochs, batch, imgsz)


def train_yolo26n_detect(epochs: int, batch: int, imgsz: int) -> dict:
    return _run_ultralytics("detect", "yolo26n", "yolo26n", epochs, batch, imgsz)


def train_yolo26n_segment(epochs: int, batch: int, imgsz: int) -> dict:
    return _run_ultralytics("segment", "yolo26n-seg", "yolo26n-seg", epochs, batch, imgsz)


# ── Report saving ──────────────────────────────────────────────────────────────


def save_report(all_metrics: dict[str, dict], epochs: int, batch: int) -> None:
    report_dir = HERE / "runs"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated":  datetime.now(timezone.utc).isoformat(),
        "dataset":    "golden_frames_3fps_mid (seb.json — human labelled)",
        "epochs":     epochs,
        "batch":      batch,
        "models":     all_metrics,
    }

    json_path = report_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport (JSON) → {json_path}")

    # CSV – one row per model
    csv_path = report_dir / "report.csv"
    columns  = [
        "model", "task",
        "box_map50", "box_map5095",
        "mask_map50", "mask_map5095", "mean_iou",
        "precision", "recall", "f1",
        "inference_ms", "size_mb", "gflops",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for key, m in all_metrics.items():
            row = {"model": key, **m}
            w.writerow(row)
    print(f"Report (CSV)  → {csv_path}")

    # Console summary table
    print(f"\n{'─'*100}")
    header = f"{'Model':<22} {'Task':<14} {'box mAP50':>9} {'box mAP':>9} {'mask mAP50':>10} {'mask mAP':>9} {'mIoU':>7} {'P':>7} {'R':>7} {'F1':>7} {'ms':>6} {'MB':>6} {'GFLOPs':>8}"
    print(header)
    print("─" * 100)
    for key, m in all_metrics.items():
        def _f(v):
            return f"{v:.4f}" if isinstance(v, float) else "  N/A "
        gf = f"{m['gflops']:.1f}" if m.get("gflops") is not None else "  N/A"
        print(
            f"{key:<22} {m.get('task',''):<14} "
            f"{_f(m.get('box_map50')):>9} {_f(m.get('box_map5095')):>9} "
            f"{_f(m.get('mask_map50')):>10} {_f(m.get('mask_map5095')):>9} "
            f"{_f(m.get('mean_iou')):>7} "
            f"{_f(m.get('precision')):>7} {_f(m.get('recall')):>7} {_f(m.get('f1')):>7} "
            f"{m.get('inference_ms', 0):>6.1f} {m.get('size_mb', 0):>6.1f} {gf:>8}"
        )
    print("─" * 100)


# ── Main ───────────────────────────────────────────────────────────────────────


_MODEL_CHOICES = {
    "yolov8n_detect":  "YOLOv8n detection",
    "yolov8n_segment": "YOLOv8n segmentation",
    "rtdetr_l_detect": "RT-DETR-L detection",
    "yolo26n_detect":  "YOLOv26n detection",
    "yolo26n_segment": "YOLOv26n segmentation",
}

_TRAIN_FNS = {
    "yolov8n_detect":  lambda e, b, imgsz: train_yolov8n_detect(e, b, imgsz),
    "yolov8n_segment": lambda e, b, imgsz: train_yolov8n_segment(e, b, imgsz),
    "rtdetr_l_detect": lambda e, b, imgsz: train_rtdetr_detect(e, b, imgsz),
    "yolo26n_detect":  lambda e, b, imgsz: train_yolo26n_detect(e, b, imgsz),
    "yolo26n_segment": lambda e, b, imgsz: train_yolo26n_segment(e, b, imgsz),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n + RT-DETR + YOLOv26n on golden_frames_3fps_mid"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument(
        "--models", nargs="+",
        default=list(_MODEL_CHOICES.keys()),
        choices=list(_MODEL_CHOICES.keys()),
        help="Which models to train (default: all).",
    )
    args = parser.parse_args()

    ensure_data_prepared()

    t_start      = time.perf_counter()
    all_metrics: dict[str, dict] = {}

    for key in args.models:
        label = _MODEL_CHOICES[key]
        print(f"\n{'#'*60}")
        print(f"# {label}")
        print(f"{'#'*60}")

        try:
            metrics = _TRAIN_FNS[key](args.epochs, args.batch, args.imgsz)
            all_metrics[key] = metrics
        except Exception as exc:
            print(f"\n[ERROR] {key} failed: {exc}")
            import traceback
            traceback.print_exc()
            all_metrics[key] = {"error": str(exc)}

    total_s = time.perf_counter() - t_start
    print(f"\n\nAll models trained in {total_s / 60:.1f} min")

    if any("error" not in m for m in all_metrics.values()):
        save_report(
            {k: v for k, v in all_metrics.items() if "error" not in v},
            args.epochs, args.batch,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
