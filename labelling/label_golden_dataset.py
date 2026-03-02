#!/usr/bin/env python3
"""Gradio labelling tool for the golden dataset.

Workflow
--------
1. Point to a folder of images (e.g. dataset/golden_frames_3fps_mid).
2. Click "Load & Auto-detect" — YOLOv8-seg runs on every image and fills in
   initial person segmentations automatically.
3. Navigate frame-by-frame, delete false positives (click on a mask) or add
   missed persons (click on empty space → FastSAM segments at that point).
4. Press Save to write a single COCO-format JSON.

Keyboard shortcuts
------------------
  ← / →   previous / next frame
  Enter    save
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
from ultralytics import FastSAM, YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (128, 0, 0),
    (170, 255, 195),
    (0, 0, 128),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERSON_CLASS_ID = 0  # COCO class index for "person" in YOLO models

KEYBOARD_JS = """
<script>
function handleKeyboard(e) {
    const tag = e.target.tagName.toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    if (e.key === "ArrowLeft") {
        e.preventDefault();
        document.getElementById("btn_prev").click();
    } else if (e.key === "ArrowRight") {
        e.preventDefault();
        document.getElementById("btn_next").click();
    } else if (e.key === "Enter") {
        e.preventDefault();
        document.getElementById("btn_save").click();
    }
}
document.addEventListener("keydown", handleKeyboard, false);
</script>
"""

# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


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


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


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
        label = f"#{ann['id']}  {ann.get('score', 0):.2f}"
        (tw, th2), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (bx, by - th2 - 8), (bx + tw + 6, by), c, -1)
        cv2.putText(vis, label, (bx + 3, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_prompt_point(vis: np.ndarray, x: int, y: int) -> np.ndarray:
    out = vis.copy()
    cv2.circle(out, (x, y), 7, (60, 220, 80), -1)
    cv2.circle(out, (x, y), 10, (255, 255, 255), 2)
    return out


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    def __init__(self, yolo_model: str, fastsam_model: str):
        self._yolo_path = yolo_model
        self._fastsam_path = fastsam_model
        self._yolo: YOLO | None = None
        self._fastsam: FastSAM | None = None

        self.images_dir: Path | None = None
        self.output_json: Path | None = None
        self.images: list[Path] = []
        self.idx: int = 0

        # rel_path → list of annotation dicts
        self.anns_by_file: dict[str, list[dict]] = {}
        self.next_ann_id: int = 1

        # session counters
        self.total_detected: int = 0
        self.total_deleted: int = 0
        self.total_added: int = 0
        self.session_start = datetime.now()

    # ---- lazy model loading ----

    def yolo(self) -> YOLO:
        if self._yolo is None:
            self._yolo = YOLO(self._yolo_path)
        return self._yolo

    def fastsam(self) -> FastSAM:
        if self._fastsam is None:
            self._fastsam = FastSAM(self._fastsam_path)
        return self._fastsam

    # ---- load folder ----

    def load(self, images_dir: Path, output_json: Path):
        self.images_dir = images_dir.resolve()
        self.output_json = output_json.resolve()
        self.images = sorted(
            p for p in self.images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.idx = 0
        self.anns_by_file = {}
        self.next_ann_id = 1
        self.total_detected = 0
        self.total_deleted = 0
        self.total_added = 0
        if self.output_json.exists():
            self._load_existing(self.output_json)

    def _load_existing(self, path: Path):
        with open(path) as f:
            coco = json.load(f)
        id_to_rel = {img["id"]: img["file_name"] for img in coco.get("images", [])}
        max_id = 0
        for ann in coco.get("annotations", []):
            rel = id_to_rel.get(ann.get("image_id"))
            if rel is None:
                continue
            self.anns_by_file.setdefault(rel, []).append(ann)
            max_id = max(max_id, int(ann.get("id", 0)))
        self.next_ann_id = max_id + 1

    # ---- current-image helpers ----

    @property
    def current_path(self) -> Path | None:
        return self.images[self.idx] if self.images else None

    def current_rel(self) -> str | None:
        p = self.current_path
        if p is None or self.images_dir is None:
            return None
        return str(p.relative_to(self.images_dir))

    def current_image(self) -> np.ndarray | None:
        p = self.current_path
        return np.array(Image.open(p).convert("RGB")) if p else None

    def current_anns(self) -> list[dict]:
        rel = self.current_rel()
        return self.anns_by_file.get(rel, []) if rel else []

    # ---- detection ----

    def detect_image(
        self, img: np.ndarray, conf: float, iou: float, imgsz: int
    ) -> list[dict]:
        """Run YOLO person detection on a single image, return annotation dicts."""
        res = self.yolo()(
            img,
            device=DEVICE,
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            retina_masks=True,
            classes=[PERSON_CLASS_ID],
        )[0]

        anns: list[dict] = []
        if res.masks is None or res.boxes is None:
            return anns

        h, w = img.shape[:2]
        for mask_t, box_t, score_t in zip(res.masks.data, res.boxes.xyxy, res.boxes.conf):
            mask = mask_t.cpu().numpy().astype(np.uint8)
            # masks may be at inference resolution — resize to image size
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            box = xyxy_to_xywh(box_t.cpu().tolist())
            score = float(score_t)
            anns.append({
                "id": self.next_ann_id,
                "image_id": -1,
                "category_id": 1,
                "bbox": [round(v, 2) for v in box],
                "area": round(float(mask.sum()), 2),
                "segmentation": encode_rle(mask),
                "score": round(score, 4),
                "iscrowd": 0,
            })
            self.next_ann_id += 1
        return anns

    def run_autodetect(
        self, conf: float, iou: float, imgsz: int, overwrite: bool
    ) -> str:
        """Run YOLO detection on all images. Returns a summary string."""
        if not self.images:
            return "No images loaded."
        detected_total = 0
        skipped = 0
        for i, img_path in enumerate(self.images):
            rel = str(img_path.relative_to(self.images_dir))
            if not overwrite and rel in self.anns_by_file:
                skipped += 1
                continue
            img = np.array(Image.open(img_path).convert("RGB"))
            anns = self.detect_image(img, conf=conf, iou=iou, imgsz=imgsz)
            self.anns_by_file[rel] = anns
            detected_total += len(anns)
        self.total_detected += detected_total
        parts = [f"Auto-detected {detected_total} persons across {len(self.images) - skipped} images."]
        if skipped:
            parts.append(f"Skipped {skipped} already-labelled images (uncheck 'Overwrite' to re-run).")
        return " ".join(parts)

    def detect_current(self, conf: float, iou: float, imgsz: int) -> str:
        """Re-run YOLO on the current frame, replacing its annotations."""
        img = self.current_image()
        rel = self.current_rel()
        if img is None or rel is None:
            return "No image loaded."
        anns = self.detect_image(img, conf=conf, iou=iou, imgsz=imgsz)
        self.anns_by_file[rel] = anns
        self.total_detected += len(anns)
        return f"Detected {len(anns)} person(s) on current frame."

    # ---- add / remove annotations ----

    def add_via_fastsam(
        self, x: int, y: int, conf: float, iou: float, imgsz: int
    ) -> str:
        """Run FastSAM, pick the mask under (x,y), add as annotation."""
        img = self.current_image()
        rel = self.current_rel()
        if img is None or rel is None:
            return "No image loaded."
        h, w = img.shape[:2]

        res = self.fastsam()(
            img,
            device=DEVICE,
            verbose=False,
            retina_masks=True,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
        )[0]

        if res.masks is None or res.boxes is None:
            return "FastSAM returned no masks."

        best_mask, best_box, best_score = None, None, -1.0
        for mask_t, box_t, score_t in zip(res.masks.data, res.boxes.xyxy, res.boxes.conf):
            m = mask_t.cpu().numpy().astype(np.uint8)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            sc = float(score_t)
            if 0 <= y < h and 0 <= x < w and m[y, x] > 0 and sc > best_score:
                best_mask = m
                best_box = box_t.cpu().tolist()
                best_score = sc

        if best_mask is None:
            return "No mask found at click point."

        bbox = xyxy_to_xywh(best_box)
        self.anns_by_file.setdefault(rel, []).append({
            "id": self.next_ann_id,
            "image_id": -1,
            "category_id": 1,
            "bbox": [round(v, 2) for v in bbox],
            "area": round(float(best_mask.sum()), 2),
            "segmentation": encode_rle(best_mask),
            "score": round(best_score, 4),
            "iscrowd": 0,
        })
        self.next_ann_id += 1
        self.total_added += 1
        return f"Added annotation (FastSAM score {best_score:.3f})."

    def delete_at_point(self, x: int, y: int, remove_all: bool) -> str:
        img = self.current_image()
        rel = self.current_rel()
        if img is None or rel is None:
            return "No image loaded."
        h, w = img.shape[:2]
        anns = self.anns_by_file.get(rel, [])
        hits: list[tuple[float, int]] = []
        for i, ann in enumerate(anns):
            m = decode_seg(ann["segmentation"], h, w)
            if 0 <= y < h and 0 <= x < w and m[y, x] > 0:
                hits.append((float(ann.get("area", m.sum())), i))
        if not hits:
            return "No annotation at click point."
        hits.sort(key=lambda t: t[0])
        indices = {i for _, i in hits} if remove_all else {hits[0][1]}
        self.anns_by_file[rel] = [a for i, a in enumerate(anns) if i not in indices]
        self.total_deleted += len(indices)
        return f"Deleted {len(indices)} annotation(s)."

    # ---- save ----

    def save_coco(self) -> tuple[int, int]:
        """Write COCO JSON. Returns (n_images_with_annotations, n_annotations)."""
        if not self.images_dir or not self.output_json:
            return 0, 0
        images_out: list[dict] = []
        anns_out: list[dict] = []
        for img_id, img_path in enumerate(self.images, 1):
            rel = str(img_path.relative_to(self.images_dir))
            arr = np.array(Image.open(img_path).convert("RGB"))
            h, w = arr.shape[:2]
            images_out.append({"id": img_id, "file_name": rel, "width": w, "height": h})
            for ann in self.anns_by_file.get(rel, []):
                out = dict(ann)
                out["image_id"] = img_id
                anns_out.append(out)
        coco = {
            "images": images_out,
            "annotations": anns_out,
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        }
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json, "w") as f:
            json.dump(coco, f, indent=2)
        n_with_anns = sum(1 for rel in self.anns_by_file if self.anns_by_file[rel])
        return n_with_anns, len(anns_out)

    # ---- stats ----

    @property
    def stats_text(self) -> str:
        elapsed = datetime.now() - self.session_start
        m, s = divmod(int(elapsed.total_seconds()), 60)
        rel = self.current_rel() or "—"
        n_anns = len(self.current_anns())
        n_labelled = sum(1 for anns in self.anns_by_file.values() if anns)
        return (
            f"**Session** {m}m {s}s &nbsp;|&nbsp; "
            f"Detected: **{self.total_detected}** &nbsp;|&nbsp; "
            f"Added: **{self.total_added}** &nbsp;|&nbsp; "
            f"Deleted: **{self.total_deleted}**\n\n"
            f"Labelled images: **{n_labelled}** / {len(self.images)} &nbsp;|&nbsp; "
            f"Current frame: **{n_anns}** annotation(s)"
        )


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------


def build_app(session: Session) -> gr.Blocks:
    with gr.Blocks(title="Golden Dataset Labeller", head=KEYBOARD_JS) as app:
        gr.Markdown("# Golden Dataset Labeller")

        # ---- top controls ----
        with gr.Row():
            images_dir_tb = gr.Textbox(
                label="Images folder",
                placeholder="dataset/golden_frames_3fps_mid",
                scale=3,
            )
            output_json_tb = gr.Textbox(
                label="Output COCO JSON",
                placeholder="dataset/labels_golden/annotations.json",
                scale=3,
            )
            load_btn = gr.Button("Load", variant="primary", scale=1)

        with gr.Row():
            detect_conf = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="YOLO conf", scale=2)
            detect_iou = gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="YOLO IoU NMS", scale=2)
            detect_imgsz = gr.Slider(320, 1280, value=640, step=32, label="YOLO image size", scale=2)
            overwrite_chk = gr.Checkbox(label="Overwrite existing", value=False, scale=1)
            autodetect_btn = gr.Button("Auto-detect all", variant="primary", scale=1)
            detect_current_btn = gr.Button("Re-detect current", scale=1)

        detect_status = gr.Markdown("")

        # ---- navigation ----
        with gr.Row():
            frame_slider = gr.Slider(0, 0, step=1, value=0, label="Frame index", scale=5)
            prev_btn = gr.Button("◀ Prev", size="sm", scale=0, elem_id="btn_prev")
            next_btn = gr.Button("Next ▶", size="sm", scale=0, elem_id="btn_next")
            save_btn = gr.Button("💾 Save", variant="primary", size="sm", scale=0, elem_id="btn_save")

        # ---- main area ----
        with gr.Row():
            with gr.Column(scale=3):
                image_out = gr.Image(
                    label="Frame",
                    type="numpy",
                    interactive=False,
                    elem_id="img_out",
                )
                status_md = gr.Markdown("")

            with gr.Column(scale=1, min_width=260):
                stats_md = gr.Markdown("")
                gr.Markdown("---")
                gr.Markdown("### Click action")
                click_action = gr.Radio(
                    choices=["Remove annotation", "Add via FastSAM"],
                    value="Remove annotation",
                    label="What a click does",
                )
                remove_all_chk = gr.Checkbox(label="Remove all overlapping at click", value=False)
                gr.Markdown("---")
                gr.Markdown("### FastSAM (add) settings")
                fs_conf = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Conf")
                fs_iou = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="IoU")
                fs_imgsz = gr.Slider(256, 1536, value=1024, step=32, label="Image size")
                gr.Markdown(
                    "**Remove**: click any coloured mask to delete it.\n\n"
                    "**Add via FastSAM**: click on a missed person to add a mask.\n\n"
                    "← / → to navigate · Enter to save"
                )

        # ---- helpers ----

        def _refresh():
            img = session.current_image()
            if img is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                return blank, "No folder loaded.", session.stats_text
            anns = session.current_anns()
            vis = render(img, anns)
            rel = session.current_rel()
            status = (
                f"`{rel}` — frame **{session.idx + 1}** / {len(session.images)}"
                f" — **{len(anns)}** annotation(s)"
            )
            return vis, status, session.stats_text

        # ---- event handlers ----

        def on_load(images_dir: str, output_json: str):
            if not images_dir or not output_json:
                gr.Warning("Provide both an images folder and an output JSON path.")
                return gr.update(), *_refresh()
            p = Path(images_dir)
            if not p.exists() or not p.is_dir():
                gr.Warning(f"Images folder not found: {p}")
                return gr.update(), *_refresh()
            session.load(p, Path(output_json))
            vis, status, stats = _refresh()
            slider_update = gr.update(maximum=max(len(session.images) - 1, 0), value=0)
            return slider_update, vis, status, stats

        def on_autodetect(conf: float, iou: float, imgsz: float, overwrite: bool):
            if not session.images:
                return "Load a folder first.", *_refresh()
            msg = session.run_autodetect(conf=conf, iou=iou, imgsz=int(imgsz), overwrite=overwrite)
            gr.Info(msg)
            return msg, *_refresh()

        def on_detect_current(conf: float, iou: float, imgsz: float):
            if not session.images:
                return "Load a folder first.", *_refresh()
            msg = session.detect_current(conf=conf, iou=iou, imgsz=int(imgsz))
            return msg, *_refresh()

        def on_slider(idx: float):
            session.idx = int(idx)
            return _refresh()

        def on_prev():
            session.idx = max(0, session.idx - 1)
            vis, status, stats = _refresh()
            return gr.update(value=session.idx), vis, status, stats

        def on_next():
            session.idx = min(len(session.images) - 1, session.idx + 1)
            vis, status, stats = _refresh()
            return gr.update(value=session.idx), vis, status, stats

        def on_save():
            n_imgs, n_anns = session.save_coco()
            gr.Info(f"Saved {n_anns} annotations across {n_imgs} images → {session.output_json}")
            vis, status, stats = _refresh()
            return status, stats

        def on_click(
            evt: gr.SelectData,
            action: str,
            remove_all: bool,
            conf: float,
            iou: float,
            imgsz: float,
        ):
            if not session.images:
                return *_refresh(),
            img = session.current_image()
            if img is None:
                return *_refresh(),
            h, w = img.shape[:2]
            x = min(max(int(evt.index[0]), 0), w - 1)
            y = min(max(int(evt.index[1]), 0), h - 1)

            if action == "Remove annotation":
                msg = session.delete_at_point(x, y, remove_all)
                if "No annotation" in msg:
                    gr.Warning(msg)
                else:
                    gr.Info(msg)
            else:
                msg = session.add_via_fastsam(x, y, conf=conf, iou=iou, imgsz=int(imgsz))
                if "No mask" in msg or "no mask" in msg.lower():
                    gr.Warning(msg)
                else:
                    gr.Info(msg)

            return _refresh()

        # ---- wiring ----
        load_btn.click(
            on_load, [images_dir_tb, output_json_tb],
            [frame_slider, image_out, status_md, stats_md],
        )
        autodetect_btn.click(
            on_autodetect, [detect_conf, detect_iou, detect_imgsz, overwrite_chk],
            [detect_status, image_out, status_md, stats_md],
        )
        detect_current_btn.click(
            on_detect_current, [detect_conf, detect_iou, detect_imgsz],
            [detect_status, image_out, status_md, stats_md],
        )
        frame_slider.release(on_slider, [frame_slider], [image_out, status_md, stats_md])
        prev_btn.click(on_prev, [], [frame_slider, image_out, status_md, stats_md])
        next_btn.click(on_next, [], [frame_slider, image_out, status_md, stats_md])
        save_btn.click(on_save, [], [status_md, stats_md])
        image_out.select(
            on_click,
            [click_action, remove_all_chk, fs_conf, fs_iou, fs_imgsz],
            [image_out, status_md, stats_md],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden dataset labelling tool.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("dataset/golden_frames_3fps_mid"),
        help="Folder of images to label.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("dataset/labels_golden/annotations.json"),
        help="Output COCO JSON path.",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n-seg.pt",
        help="YOLO segmentation model weights (auto-downloaded if not present).",
    )
    parser.add_argument(
        "--fastsam-model",
        type=str,
        default="FastSAM-s.pt",
        help="FastSAM weights for interactive add-mask.",
    )
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    session = Session(yolo_model=args.yolo_model, fastsam_model=args.fastsam_model)
    app = build_app(session)

    # Pre-load folder if it exists
    if args.images_dir.exists():
        session.load(args.images_dir, args.output_json)

    app.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css="#img_out { cursor: crosshair; }",
    )


if __name__ == "__main__":
    main()
