#!/usr/bin/env python3
"""Gradio app to assist image segmentation with FastSAM.

Features:
- Load images from a specified folder.
- Click to add segmentation masks via FastSAM.
- Click to remove existing masks.
- Navigate images and save progress to a COCO-style JSON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from ultralytics import FastSAM

PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode_rle(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_seg(seg: dict | list, h: int, w: int) -> np.ndarray:
    if isinstance(seg, dict):
        rle = dict(seg)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return mask_utils.decode(rle).astype(np.uint8)
    rles = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)


def xyxy_to_xywh(box_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def collect_images(images_dir: Path) -> list[Path]:
    images = [
        p for p in sorted(images_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images


class Session:
    def __init__(self, fastsam_model: str):
        self.fastsam = FastSAM(fastsam_model)
        self.images_dir: Path | None = None
        self.output_json: Path | None = None
        self.images: list[Path] = []
        self.idx = 0
        self.anns_by_file: dict[str, list[dict]] = {}
        self.next_ann_id = 1
        self.started = datetime.now()
        self.pos_points_by_file: dict[str, list[tuple[int, int]]] = {}
        self.neg_points_by_file: dict[str, list[tuple[int, int]]] = {}
        self.poly_points_by_file: dict[str, list[tuple[int, int]]] = {}

    def load(self, images_dir: Path, output_json: Path):
        self.images_dir = images_dir.resolve()
        self.output_json = output_json.resolve()
        self.images = collect_images(self.images_dir)
        self.idx = 0
        self.anns_by_file = {}
        self.pos_points_by_file = {}
        self.neg_points_by_file = {}
        self.poly_points_by_file = {}
        self.next_ann_id = 1
        if self.output_json.exists():
            self._load_existing(self.output_json)

    def _load_existing(self, path: Path):
        with open(path) as f:
            coco = json.load(f)

        images_by_id = {img["id"]: img["file_name"] for img in coco.get("images", [])}
        max_id = 0
        for ann in coco.get("annotations", []):
            img_id = ann.get("image_id")
            rel = images_by_id.get(img_id)
            if rel is None:
                continue
            self.anns_by_file.setdefault(rel, []).append(ann)
            max_id = max(max_id, int(ann.get("id", 0)))
        self.next_ann_id = max_id + 1

    @property
    def current_image_path(self) -> Path | None:
        if not self.images:
            return None
        return self.images[self.idx]

    def current_rel_file(self) -> str | None:
        if not self.current_image_path or not self.images_dir:
            return None
        return str(self.current_image_path.relative_to(self.images_dir))

    def current_image(self) -> np.ndarray | None:
        p = self.current_image_path
        if p is None:
            return None
        return np.array(Image.open(p).convert("RGB"))

    def anns_current(self) -> list[dict]:
        rel = self.current_rel_file()
        if rel is None:
            return []
        return self.anns_by_file.get(rel, [])

    def points_current(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        rel = self.current_rel_file()
        if rel is None:
            return [], []
        return self.pos_points_by_file.get(rel, []), self.neg_points_by_file.get(rel, [])

    def add_point_current(self, x: int, y: int, positive: bool):
        rel = self.current_rel_file()
        if rel is None:
            return
        points_map = self.pos_points_by_file if positive else self.neg_points_by_file
        points = points_map.setdefault(rel, [])
        points.append((x, y))

    def clear_points_current(self):
        rel = self.current_rel_file()
        if rel is None:
            return
        self.pos_points_by_file[rel] = []
        self.neg_points_by_file[rel] = []

    def polygon_points_current(self) -> list[tuple[int, int]]:
        rel = self.current_rel_file()
        if rel is None:
            return []
        return self.poly_points_by_file.get(rel, [])

    def add_polygon_point_current(self, x: int, y: int):
        rel = self.current_rel_file()
        if rel is None:
            return
        self.poly_points_by_file.setdefault(rel, []).append((x, y))

    def clear_polygon_points_current(self):
        rel = self.current_rel_file()
        if rel is None:
            return
        self.poly_points_by_file[rel] = []

    def add_ann(
        self,
        mask: np.ndarray,
        bbox_xywh: list[float],
        score: float,
        segmentation: dict | list | None = None,
    ):
        rel = self.current_rel_file()
        if rel is None:
            return
        ann = {
            "id": self.next_ann_id,
            "image_id": -1,  # assigned at save time
            "category_id": 1,
            "bbox": [round(float(v), 2) for v in bbox_xywh],
            "area": round(float(mask.sum()), 2),
            "segmentation": encode_rle(mask) if segmentation is None else segmentation,
            "score": round(float(score), 4),
            "iscrowd": 0,
        }
        self.next_ann_id += 1
        self.anns_by_file.setdefault(rel, []).append(ann)

    def delete_smallest_at_point(self, x: int, y: int, remove_all: bool) -> int:
        rel = self.current_rel_file()
        img = self.current_image()
        if rel is None or img is None:
            return 0
        h, w = img.shape[:2]
        anns = self.anns_by_file.get(rel, [])
        hits: list[tuple[float, int]] = []
        for i, ann in enumerate(anns):
            m = decode_seg(ann["segmentation"], h, w)
            if 0 <= y < h and 0 <= x < w and m[y, x] > 0:
                hits.append((float(ann.get("area", float(m.sum()))), i))
        if not hits:
            return 0
        hits.sort(key=lambda t: t[0])
        if remove_all:
            indices = {i for _, i in hits}
        else:
            indices = {hits[0][1]}
        self.anns_by_file[rel] = [a for i, a in enumerate(anns) if i not in indices]
        return len(indices)

    def save_coco(self) -> int:
        if self.images_dir is None or self.output_json is None:
            return 0
        images = []
        annotations = []
        img_id_by_rel: dict[str, int] = {}
        for i, p in enumerate(self.images, 1):
            rel = str(p.relative_to(self.images_dir))
            arr = np.array(Image.open(p).convert("RGB"))
            h, w = arr.shape[:2]
            img_id_by_rel[rel] = i
            images.append({"id": i, "file_name": rel, "width": int(w), "height": int(h)})
        for rel, anns in self.anns_by_file.items():
            if rel not in img_id_by_rel:
                continue
            for ann in anns:
                out = dict(ann)
                out["image_id"] = img_id_by_rel[rel]
                annotations.append(out)

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
        }
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json, "w") as f:
            json.dump(coco, f, indent=2)
        return len(annotations)


def render(img: np.ndarray, anns: list[dict]) -> np.ndarray:
    vis = img.copy()
    h, w = img.shape[:2]
    for i, ann in enumerate(anns):
        c = PALETTE[i % len(PALETTE)]
        mask = decode_seg(ann["segmentation"], h, w)
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = c
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.35, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, c, 2)
        bx, by, bw, bh = (int(v) for v in ann["bbox"])
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), c, 2)
    return vis


def draw_points(vis: np.ndarray, pos_points: list[tuple[int, int]], neg_points: list[tuple[int, int]]) -> np.ndarray:
    out = vis.copy()
    for x, y in pos_points:
        cv2.circle(out, (x, y), 6, (60, 220, 80), -1)
        cv2.circle(out, (x, y), 9, (255, 255, 255), 2)
    for x, y in neg_points:
        cv2.circle(out, (x, y), 6, (230, 70, 70), -1)
        cv2.circle(out, (x, y), 9, (255, 255, 255), 2)
        cv2.line(out, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), 2)
        cv2.line(out, (x - 5, y + 5), (x + 5, y - 5), (255, 255, 255), 2)
    return out


def draw_polygon_preview(vis: np.ndarray, poly_points: list[tuple[int, int]]) -> np.ndarray:
    out = vis.copy()
    if not poly_points:
        return out
    pts = np.array(poly_points, dtype=np.int32)
    cv2.polylines(out, [pts], isClosed=False, color=(80, 180, 255), thickness=2)
    for i, (x, y) in enumerate(poly_points):
        cv2.circle(out, (x, y), 5, (80, 180, 255), -1)
        cv2.circle(out, (x, y), 8, (255, 255, 255), 2)
        cv2.putText(out, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def build_app(session: Session):
    with gr.Blocks(title="FastSAM Segmentation Assistant") as app:
        gr.Markdown("# FastSAM Segmentation Assistant")

        with gr.Row():
            images_dir_tb = gr.Textbox(label="Images folder", placeholder="/path/to/images")
            out_json_tb = gr.Textbox(label="Output COCO JSON", placeholder="/path/to/output.json")
            load_btn = gr.Button("Load Folder", variant="primary")

        with gr.Row():
            frame_slider = gr.Slider(0, 0, step=1, value=0, label="Image index")
            prev_btn = gr.Button("Prev")
            next_btn = gr.Button("Next")
            save_btn = gr.Button("Save")

        with gr.Row():
            with gr.Column(scale=3):
                image_out = gr.Image(type="numpy", interactive=False, label="Image")
                status_md = gr.Markdown("")
            with gr.Column(scale=1):
                click_action = gr.Radio(
                    choices=["Add positive point", "Add negative point", "Add polygon vertex", "Remove"],
                    value="Add positive point",
                    label="Click action",
                )
                apply_prompt_btn = gr.Button("Apply point prompt", variant="primary")
                clear_points_btn = gr.Button("Clear points")
                finalize_polygon_btn = gr.Button("Finalize polygon", variant="primary")
                clear_polygon_btn = gr.Button("Clear polygon")
                remove_all = gr.Checkbox(label="Remove all overlapping at click", value=False)
                conf = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="FastSAM conf")
                iou = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="FastSAM IoU")
                imgsz = gr.Slider(256, 1536, value=1024, step=32, label="FastSAM image size")

        def refresh():
            img = session.current_image()
            if img is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                return blank, "No folder loaded."
            anns = session.anns_current()
            vis = render(img, anns)
            pos_points, neg_points = session.points_current()
            vis = draw_points(vis, pos_points, neg_points)
            poly_points = session.polygon_points_current()
            vis = draw_polygon_preview(vis, poly_points)
            rel = session.current_rel_file()
            return (
                vis,
                f"`{rel}` — image {session.idx + 1}/{len(session.images)} — anns: {len(anns)}"
                f" — +pts: {len(pos_points)} — -pts: {len(neg_points)} — poly pts: {len(poly_points)}",
            )

        def on_load(images_dir: str, out_json: str):
            if not images_dir or not out_json:
                gr.Warning("Provide both images folder and output JSON path.")
                return gr.update(), gr.update(), "Missing paths."
            p = Path(images_dir)
            if not p.exists() or not p.is_dir():
                gr.Warning(f"Images folder not found: {p}")
                return gr.update(), gr.update(), "Invalid images folder."
            session.load(p, Path(out_json))
            vis, status = refresh()
            return gr.update(maximum=max(len(session.images) - 1, 0), value=0), vis, status

        def on_slider(idx: float):
            session.idx = int(idx)
            return refresh()

        def on_prev():
            session.idx = max(0, session.idx - 1)
            vis, status = refresh()
            return gr.update(value=session.idx), vis, status

        def on_next():
            session.idx = min(len(session.images) - 1, session.idx + 1)
            vis, status = refresh()
            return gr.update(value=session.idx), vis, status

        def on_click(
            evt: gr.SelectData, action: str, rm_all: bool, conf_v: float, iou_v: float, imgsz_v: float
        ):
            img = session.current_image()
            if img is None:
                return gr.update(), "No folder loaded."
            h, w = img.shape[:2]
            x, y = evt.index
            x = min(max(int(x), 0), w - 1)
            y = min(max(int(y), 0), h - 1)

            if action == "Remove":
                deleted = session.delete_smallest_at_point(x=x, y=y, remove_all=rm_all)
                if deleted > 0:
                    gr.Info(f"Deleted {deleted} annotation(s).")
                else:
                    gr.Warning("No annotation at click point.")
                return refresh()
            if action == "Add polygon vertex":
                session.add_polygon_point_current(x=x, y=y)
                return refresh()
            if action == "Add negative point":
                session.add_point_current(x=x, y=y, positive=False)
                return refresh()
            session.add_point_current(x=x, y=y, positive=True)
            return on_apply_prompt(conf_v=conf_v, iou_v=iou_v, imgsz_v=imgsz_v)

        def on_apply_prompt(conf_v: float, iou_v: float, imgsz_v: float):
            img = session.current_image()
            if img is None:
                return gr.update(), "No folder loaded."
            pos_points, neg_points = session.points_current()
            if not pos_points:
                gr.Warning("Add at least one positive point first.")
                return refresh()

            preds = session.fastsam(
                img,
                device=DEVICE,
                verbose=False,
                retina_masks=True,
                conf=float(conf_v),
                iou=float(iou_v),
                imgsz=int(imgsz_v),
            )
            res = preds[0]
            if res.masks is None or res.boxes is None:
                gr.Warning("FastSAM returned no masks.")
                return refresh()

            masks = [m.cpu().numpy().astype(np.uint8) for m in res.masks.data]
            boxes = [b.cpu().tolist() for b in res.boxes.xyxy]
            scores = [float(s) for s in res.boxes.conf]

            best = None
            best_score = -1.0
            for m, b, s in zip(masks, boxes, scores):
                # Keep only masks that include all positive points and no negative points.
                if any(m[py, px] == 0 for px, py in pos_points):
                    continue
                if any(m[ny, nx] > 0 for nx, ny in neg_points):
                    continue
                if s > best_score:
                    best = (m, b, s)
                    best_score = s

            if best is None:
                gr.Warning("No segmentation satisfies all prompt points.")
                return refresh()

            mask, box_xyxy, score = best
            session.add_ann(mask=mask, bbox_xywh=xyxy_to_xywh(box_xyxy), score=score)
            session.clear_points_current()
            gr.Info(f"Added annotation (score {score:.3f})")
            return refresh()

        def on_clear_points():
            session.clear_points_current()
            return refresh()

        def on_finalize_polygon():
            img = session.current_image()
            if img is None:
                return gr.update(), "No folder loaded."
            h, w = img.shape[:2]
            poly_points = session.polygon_points_current()
            if len(poly_points) < 3:
                gr.Warning("Add at least 3 polygon points.")
                return refresh()

            pts = np.array(poly_points, dtype=np.int32).reshape((-1, 1, 2))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], color=1)
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                gr.Warning("Polygon produced an empty mask.")
                return refresh()
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            bbox_xywh = [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]
            segmentation = [[coord for x, y in poly_points for coord in (float(x), float(y))]]

            session.add_ann(mask=mask, bbox_xywh=bbox_xywh, score=1.0, segmentation=segmentation)
            session.clear_polygon_points_current()
            gr.Info("Added polygon annotation.")
            return refresh()

        def on_clear_polygon():
            session.clear_polygon_points_current()
            return refresh()

        def on_save():
            n = session.save_coco()
            gr.Info(f"Saved {n} annotations.")
            _, status = refresh()
            return status

        load_btn.click(on_load, [images_dir_tb, out_json_tb], [frame_slider, image_out, status_md])
        frame_slider.release(on_slider, [frame_slider], [image_out, status_md])
        prev_btn.click(on_prev, [], [frame_slider, image_out, status_md])
        next_btn.click(on_next, [], [frame_slider, image_out, status_md])
        image_out.select(on_click, [click_action, remove_all, conf, iou, imgsz], [image_out, status_md])
        apply_prompt_btn.click(on_apply_prompt, [conf, iou, imgsz], [image_out, status_md])
        clear_points_btn.click(on_clear_points, [], [image_out, status_md])
        finalize_polygon_btn.click(on_finalize_polygon, [], [image_out, status_md])
        clear_polygon_btn.click(on_clear_polygon, [], [image_out, status_md])
        save_btn.click(on_save, [], [status_md])

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="FastSAM segmentation assistant.")
    parser.add_argument("--images-dir", type=Path, default=None, help="Optional image folder to pre-load.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("dataset/labels_fastsam_assisted/annotations.json"),
        help="COCO JSON output path.",
    )
    parser.add_argument("--fastsam-model", type=str, default="FastSAM-s.pt")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    session = Session(fastsam_model=args.fastsam_model)
    app = build_app(session)

    if args.images_dir is not None and args.images_dir.exists():
        session.load(args.images_dir, args.output_json)

    app.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css="#component-0 {cursor: crosshair;}",
    )


if __name__ == "__main__":
    main()
