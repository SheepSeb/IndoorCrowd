#!/usr/bin/env python3
"""
Train Mask R-CNN (ResNet-50 FPN v2) on dataset/golden_frames_3fps_mid/seb.json.

Reads the COCO JSON directly (RLE segmentation masks).
After training, runs full evaluation and saves metrics.json + results.csv under
runs/instance_seg/maskrcnn_resnet50/.

Split mapping (same as prepare_data.py):
  train/ + challange/  →  train
  test/                →  val

Usage
-----
  uv run python train/golden_mid/train_maskrcnn.py
  uv run python train/golden_mid/train_maskrcnn.py --epochs 30 --batch 4 --lr 0.005

Note on batch size
------------------
Mask R-CNN on 1280×720 images uses ~3 GB GPU memory per sample.
Batch 16 requires ~48 GB VRAM.  Use --batch 2 or 4 on a single consumer GPU.
train_all.py passes --batch 2 by default for Mask R-CNN regardless of the global
--batch setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import pycocotools.mask as mask_util

HERE       = Path(__file__).parent
REPO_ROOT  = HERE.parents[1]
GOLDEN_DIR = REPO_ROOT / "dataset" / "golden_frames_3fps_mid"
SEBE_JSON  = GOLDEN_DIR / "seb.json"

_TRAIN_PREFIXES = {"train", "challange"}


# ── Dataset ────────────────────────────────────────────────────────────────────


class GoldenDataset(Dataset):
    def __init__(self, split: str) -> None:
        assert split in ("train", "val")
        with open(SEBE_JSON) as f:
            coco = json.load(f)

        def _in_split(file_name: str) -> bool:
            prefix = file_name.split("/")[0]
            return (prefix in _TRAIN_PREFIXES) if split == "train" else (prefix == "test")

        self.imgs = [img for img in coco["images"] if _in_split(img["file_name"])]
        self._anns: dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            self._anns[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        info  = self.imgs[idx]
        img_w = info["width"]
        img_h = info["height"]

        img   = Image.open(GOLDEN_DIR / info["file_name"]).convert("RGB")
        img_t = F.to_tensor(img)

        boxes: list[list[float]] = []
        masks: list[np.ndarray]  = []

        for ann in self._anns[info["id"]]:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])

            seg = ann.get("segmentation")
            if seg and isinstance(seg, dict) and "counts" in seg:
                counts = seg["counts"]
                rle = {
                    "size": seg["size"],
                    "counts": counts.encode() if isinstance(counts, str) else counts,
                }
                binary = mask_util.decode(rle)
            else:
                binary = np.zeros((img_h, img_w), dtype=np.uint8)
                x1 = max(0, int(x));       y1 = max(0, int(y))
                x2 = min(img_w, int(x+w)); y2 = min(img_h, int(y+h))
                binary[y1:y2, x1:x2] = 1

            masks.append(binary)

        if not boxes:
            return img_t, {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros(0, dtype=torch.int64),
                "masks":    torch.zeros((0, img_h, img_w), dtype=torch.uint8),
                "image_id": torch.tensor([info["id"]]),
                "area":     torch.zeros(0, dtype=torch.float32),
                "iscrowd":  torch.zeros(0, dtype=torch.int64),
            }

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        area_t  = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])

        return img_t, {
            "boxes":    boxes_t,
            "labels":   torch.ones(len(boxes), dtype=torch.int64),  # 1 = person
            "masks":    masks_t,
            "image_id": torch.tensor([info["id"]]),
            "area":     area_t,
            "iscrowd":  torch.zeros(len(boxes), dtype=torch.int64),
        }


def _collate(batch):
    return tuple(zip(*batch))


# ── Model ──────────────────────────────────────────────────────────────────────


def build_model(num_classes: int = 2) -> torch.nn.Module:
    """Mask R-CNN ResNet-50 FPN v2, COCO pre-trained, heads replaced."""
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    return model


# ── Training helpers ───────────────────────────────────────────────────────────


def train_one_epoch(model, optimizer, loader, device, scaler) -> float:
    model.train()
    total = 0.0
    for imgs, targets in loader:
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def val_loss(model, loader, device) -> float:
    model.train()
    total = 0.0
    for imgs, targets in loader:
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        total  += sum(model(imgs, targets).values()).item()
    model.eval()
    return total / max(len(loader), 1)


# ── Evaluation metrics ─────────────────────────────────────────────────────────


def _build_val_coco():
    """Build a pycocotools COCO object for the val (test) split only."""
    from pycocotools.coco import COCO

    with open(SEBE_JSON) as f:
        full = json.load(f)

    val_imgs = [i for i in full["images"] if i["file_name"].split("/")[0] == "test"]
    val_ids  = {i["id"] for i in val_imgs}
    val_anns = [a for a in full["annotations"] if a["image_id"] in val_ids]

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({"images": val_imgs, "annotations": val_anns,
               "categories": full["categories"]}, tmp)
    tmp.close()

    coco = COCO(tmp.name)
    os.unlink(tmp.name)
    return coco


@torch.no_grad()
def compute_metrics(model, val_loader, device, weights_path: Path) -> dict:
    """Run COCO evaluation and return all report metrics."""
    from pycocotools.cocoeval import COCOeval

    model.eval()

    bbox_preds: list[dict] = []
    segm_preds: list[dict] = []
    iou_scores: list[float] = []

    # Warm-up + timed inference
    t_inf = 0.0
    n_imgs = 0

    for imgs, targets in val_loader:
        imgs_dev = [img.to(device) for img in imgs]

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(imgs_dev)
        if device == "cuda":
            torch.cuda.synchronize()
        t_inf += time.perf_counter() - t0
        n_imgs += len(imgs)

        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"].item())
            boxes  = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            pred_masks = output["masks"].cpu().numpy()  # (N, 1, H, W)
            gt_masks   = target["masks"].numpy()         # (M, H, W)

            for i in range(len(boxes)):
                if scores[i] < 0.05:
                    continue
                x1, y1, x2, y2 = boxes[i]
                bbox_preds.append({
                    "image_id":    image_id,
                    "category_id": int(labels[i]),
                    "bbox":        [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "score":       float(scores[i]),
                })
                binary = (pred_masks[i, 0] > 0.5).astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(binary))
                rle["counts"] = rle["counts"].decode("utf-8")
                segm_preds.append({
                    "image_id":    image_id,
                    "category_id": int(labels[i]),
                    "segmentation": rle,
                    "score":        float(scores[i]),
                })

            # Mean IoU: match each GT to its best-overlapping prediction
            for gt_mask in gt_masks:
                best = 0.0
                for pm in pred_masks:
                    p = (pm[0] > 0.5).astype(np.uint8)
                    inter = (p & gt_mask).sum()
                    union = (p | gt_mask).sum()
                    best = max(best, inter / (union + 1e-6))
                iou_scores.append(best)

    speed_ms = (t_inf / n_imgs * 1000) if n_imgs > 0 else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    coco_gt = _build_val_coco()

    # ── Box AP ─────────────────────────────────────────────────────────────────
    if bbox_preds:
        coco_dt_box  = coco_gt.loadRes(bbox_preds)
        ev_box       = COCOeval(coco_gt, coco_dt_box, "bbox")
        ev_box.evaluate(); ev_box.accumulate(); ev_box.summarize()
        box_map5095  = float(ev_box.stats[0])
        box_map50    = float(ev_box.stats[1])

        # Precision / Recall at best-F1 operating point (IoU=0.5)
        p_curve = ev_box.eval["precision"][0, :, 0, 0, 2]  # (101,)
        r_thrs  = np.array(ev_box.params.recThrs)
        valid   = p_curve > -1
        if valid.any():
            pv, rv = p_curve[valid], r_thrs[valid]
            f1v    = 2 * pv * rv / (pv + rv + 1e-9)
            bi     = int(np.argmax(f1v))
            precision, recall, f1 = float(pv[bi]), float(rv[bi]), float(f1v[bi])
        else:
            precision = recall = f1 = 0.0
    else:
        box_map5095 = box_map50 = precision = recall = f1 = 0.0

    # ── Mask AP ────────────────────────────────────────────────────────────────
    if segm_preds:
        coco_dt_segm = coco_gt.loadRes(segm_preds)
        ev_segm      = COCOeval(coco_gt, coco_dt_segm, "segm")
        ev_segm.evaluate(); ev_segm.accumulate(); ev_segm.summarize()
        mask_map5095 = float(ev_segm.stats[0])
        mask_map50   = float(ev_segm.stats[1])
    else:
        mask_map5095 = mask_map50 = 0.0

    # ── Model size ─────────────────────────────────────────────────────────────
    size_mb = weights_path.stat().st_size / 1e6

    # ── GFLOPs (best-effort via thop) ─────────────────────────────────────────
    gflops = None
    try:
        from thop import profile as thop_profile
        dummy  = [torch.zeros(3, 640, 640).to(device)]
        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        gflops  = round(macs * 2 / 1e9, 1)
    except Exception:
        pass

    return {
        "box_map50":    round(box_map50,    4),
        "box_map5095":  round(box_map5095,  4),
        "mask_map50":   round(mask_map50,   4),
        "mask_map5095": round(mask_map5095, 4),
        "mean_iou":     round(mean_iou,     4),
        "precision":    round(precision,    4),
        "recall":       round(recall,       4),
        "f1":           round(f1,           4),
        "inference_ms": round(speed_ms,     2),
        "size_mb":      round(size_mb,      2),
        "gflops":       gflops,
    }


# ── Main training function (importable) ───────────────────────────────────────


def train(
    epochs: int = 30,
    batch:  int = 4,
    lr:     float = 0.005,
    project: Path | None = None,
    name:   str = "maskrcnn_resnet50",
    workers: int = 4,
) -> dict:
    """Train Mask R-CNN and return full metrics dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Task:    instance_seg")
    print(f"Model:   maskrcnn_resnet50_fpn_v2 (COCO pretrained)")
    print(f"Device:  {device}  |  epochs={epochs}  batch={batch}  lr={lr}")
    print(f"{'='*60}\n")

    ds_train = GoldenDataset("train")
    ds_val   = GoldenDataset("val")
    print(f"Train: {len(ds_train)} images  |  Val: {len(ds_val)} images")

    train_loader = DataLoader(
        ds_train, batch_size=batch, shuffle=True,
        collate_fn=_collate, num_workers=workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        ds_val, batch_size=max(1, batch // 2), shuffle=False,
        collate_fn=_collate, num_workers=workers, pin_memory=(device == "cuda"),
    )

    model     = build_model(num_classes=2).to(device)
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler("cuda") if device == "cuda" else None

    out_dir     = (project or HERE / "runs" / "instance_seg") / name
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_val = float("inf")
    best_path = weights_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss  = train_one_epoch(model, optimizer, train_loader, device, scaler)
        v_loss   = val_loss(model, val_loader, device)
        scheduler.step()

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={tr_loss:.4f}  val={v_loss:.4f}  "
            f"lr={lr_now:.2e}  ({elapsed:.0f}s)"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{tr_loss:.6f}", f"{v_loss:.6f}", f"{lr_now:.8f}"]
            )

        torch.save(model.state_dict(), weights_dir / "last.pt")
        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), best_path)
            print(f"  ↑ new best  val={v_loss:.4f}")

    # ── Full evaluation on best weights ───────────────────────────────────────
    print("\nRunning full evaluation on best weights …")
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    metrics = compute_metrics(model, val_loader, device, best_path)
    metrics["task"]   = "instance_seg"
    metrics["model"]  = "maskrcnn_resnet50_fpn_v2"
    metrics["epochs"] = epochs
    metrics["batch"]  = batch

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved → {out_dir / 'metrics.json'}")
    _print_metrics(metrics)

    return metrics


def _print_metrics(m: dict) -> None:
    print(f"\n{'─'*50}")
    print(f"  box  mAP@0.5       : {m['box_map50']:.4f}")
    print(f"  box  mAP@0.5:0.95  : {m['box_map5095']:.4f}")
    print(f"  mask mAP@0.5       : {m['mask_map50']:.4f}")
    print(f"  mask mAP@0.5:0.95  : {m['mask_map5095']:.4f}")
    print(f"  Mean IoU           : {m['mean_iou']:.4f}")
    print(f"  Precision          : {m['precision']:.4f}")
    print(f"  Recall             : {m['recall']:.4f}")
    print(f"  F1                 : {m['f1']:.4f}")
    print(f"  Inference          : {m['inference_ms']:.1f} ms/img")
    print(f"  Model size         : {m['size_mb']:.1f} MB")
    if m["gflops"] is not None:
        print(f"  GFLOPs             : {m['gflops']:.1f}")
    print(f"{'─'*50}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN on golden_frames_3fps_mid"
    )
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--batch",   type=int,   default=4,
                        help="Batch size (3 GB GPU mem / sample at 1280×720).")
    parser.add_argument("--lr",      type=float, default=0.005)
    parser.add_argument("--workers", type=int,   default=4)
    args = parser.parse_args()
    train(epochs=args.epochs, batch=args.batch, lr=args.lr, workers=args.workers)


if __name__ == "__main__":
    main()
