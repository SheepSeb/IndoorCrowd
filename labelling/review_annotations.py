"""Gradio app for reviewing and editing COCO annotations.

Loads per-scene annotation files from a selected annotation source or
from custom paths provided via CLI arguments.

Interaction:
  - Click on an annotation to delete it (default mode)
  - Toggle "Add mode" and click to add a new person via SAM3
  - Left/Right arrow keys navigate between frames
  - Save persists changes to per-scene COCO JSON files
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from pycocotools import mask as mask_utils

load_dotenv()

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

KEYBOARD_HEAD = """
<script>
function handleKeyboard(e) {
    switch (e.target.tagName.toLowerCase()) {
        case "input":
        case "textarea":
        case "select":
            return;
    }
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
document.addEventListener('keydown', handleKeyboard, false);
</script>
"""

_sam3: dict = {"model": None, "processor": None}
_fastsam: dict = {"model": None, "model_path": None}
_device = "cuda" if torch.cuda.is_available() else "cpu"

ANNOTATION_SOURCE_PATHS: dict[str, tuple[Path, Path]] = {
    "sam3_3fps": (Path("dataset/labels_3fps"), Path("dataset/raw_frames_3fps")),
    "grounded_sam_3fps": (
        Path("dataset/labels_grounding_sam_3fps"),
        Path("dataset/raw_frames_3fps"),
    ),
    "efficient_grounded_sam_3fps": (
        Path("dataset/labels_efficient_grounded_sam_3fps"),
        Path("dataset/raw_frames_3fps"),
    ),
}


def load_sam3(model_id: str = "facebook/sam3"):
    if _sam3["model"] is None:
        from transformers import Sam3Model, Sam3Processor

        gr.Info("Loading SAM3 model (first time only) …")
        hf_token = os.environ.get("HF_TOKEN")
        _sam3["model"] = Sam3Model.from_pretrained(model_id, token=hf_token).to(_device).eval()
        _sam3["processor"] = Sam3Processor.from_pretrained(model_id, token=hf_token)
        gr.Info("SAM3 ready.")
    return _sam3["model"], _sam3["processor"]


def load_fastsam(model_path: str = "FastSAM-s.pt"):
    """Lazy-load FastSAM for faster interactive click-based segmentation."""
    if _fastsam["model"] is None or _fastsam["model_path"] != model_path:
        from ultralytics import FastSAM

        gr.Info(f"Loading FastSAM ({model_path}) (first time only) ...")
        _fastsam["model"] = FastSAM(model_path)
        _fastsam["model_path"] = model_path
        gr.Info("FastSAM ready.")
    return _fastsam["model"]


# ---------------------------------------------------------------------------
# Mask codec helpers
# ---------------------------------------------------------------------------

def decode_seg(seg: dict | list, h: int, w: int) -> np.ndarray:
    """Decode a COCO segmentation field (RLE dict or polygon list) to a binary mask."""
    if isinstance(seg, dict):
        rle = dict(seg)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return mask_utils.decode(rle).astype(np.uint8)
    rles = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)


def encode_rle(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(img_rgb: np.ndarray, anns: list[dict], h: int, w: int) -> np.ndarray:
    vis = img_rgb.copy()
    for i, ann in enumerate(anns):
        c = PALETTE[i % len(PALETTE)]
        mask = decode_seg(ann["segmentation"], h, w)

        colored = np.zeros_like(vis)
        colored[mask > 0] = c
        vis = cv2.addWeighted(vis, 1.0, colored, 0.35, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, c, 2)

        bx, by, bw, bh = (int(v) for v in ann["bbox"])
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), c, 2)

        label = f"#{ann['id']}  {ann.get('score', 0):.2f}"
        (tw, th2), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (bx, by - th2 - 8), (bx + tw + 6, by), c, -1)
        cv2.putText(vis, label, (bx + 3, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Hit-testing: find annotation at a click point
# ---------------------------------------------------------------------------

def ann_at_point(anns: list[dict], x: int, y: int, h: int, w: int) -> dict | None:
    """Return the smallest annotation whose mask contains (x, y), or None."""
    best, best_area = None, float("inf")
    for ann in anns:
        mask = decode_seg(ann["segmentation"], h, w)
        if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > 0:
            area = ann.get("area", float(mask.sum()))
            if area < best_area:
                best, best_area = ann, area
    return best


def anns_at_point(anns: list[dict], x: int, y: int, h: int, w: int) -> list[dict]:
    """Return all annotations containing (x, y), smallest area first."""
    hits: list[tuple[float, dict]] = []
    for ann in anns:
        mask = decode_seg(ann["segmentation"], h, w)
        if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > 0:
            area = ann.get("area", float(mask.sum()))
            hits.append((float(area), ann))
    hits.sort(key=lambda t: t[0])
    return [ann for _, ann in hits]


# ---------------------------------------------------------------------------
# State holder  (split -> scene -> frames)
# ---------------------------------------------------------------------------

class Reviewer:
    """Manages per-scene COCO annotation files under labels/{split}/{scene}.json."""

    def __init__(self, labels_dir: Path, raw_frames_dir: Path):
        self.labels_dir = labels_dir
        self.raw_frames_dir = raw_frames_dir
        self.data: dict[str, dict[str, dict]] = {}
        self.modified: set[tuple[str, str]] = set()

        self.session_start = datetime.now()
        self.deleted: int = 0
        self.added: int = 0
        self.saves: int = 0
        self.history: list[dict] = []

        for split_dir in sorted(labels_dir.iterdir()):
            if not split_dir.is_dir() or split_dir.name == "session_logs":
                continue
            split = split_dir.name
            scenes: dict[str, dict] = {}
            for p in sorted(split_dir.glob("*.json")):
                with open(p) as f:
                    coco = json.load(f)
                if "images" in coco and "annotations" in coco:
                    scenes[p.stem] = coco
            if scenes:
                self.data[split] = scenes

        self.split: str | None = next(iter(self.data), None)
        self.scene: str | None = (
            next(iter(self.data.get(self.split, {})), None) if self.split else None
        )
        self.idx: int = 0

    @property
    def splits(self) -> list[str]:
        return list(self.data.keys())

    def scenes_for(self, split: str) -> list[str]:
        return list(self.data.get(split, {}).keys())

    @property
    def coco(self) -> dict | None:
        if self.split and self.scene:
            return self.data.get(self.split, {}).get(self.scene)
        return None

    @property
    def images(self) -> list[dict]:
        return self.coco["images"] if self.coco else []

    @property
    def img_info(self) -> dict | None:
        if 0 <= self.idx < len(self.images):
            return self.images[self.idx]
        return None

    def anns_for_current(self) -> list[dict]:
        info = self.img_info
        if not info or not self.coco:
            return []
        img_id = info["id"]
        return [a for a in self.coco["annotations"] if a["image_id"] == img_id]

    def load_image(self) -> np.ndarray | None:
        info = self.img_info
        if not info:
            return None
        p = self.raw_frames_dir / info["file_name"]
        if not p.exists():
            return None
        return np.array(Image.open(p).convert("RGB"))

    def delete_anns(self, ids: set[int]):
        if not self.coco:
            return
        self.coco["annotations"] = [
            a for a in self.coco["annotations"] if a["id"] not in ids
        ]
        self.modified.add((self.split, self.scene))
        self.deleted += len(ids)
        for ann_id in ids:
            self.history.append({
                "action": "delete",
                "time": datetime.now().isoformat(),
                "split": self.split,
                "scene": self.scene,
                "frame_idx": self.idx,
                "annotation_id": ann_id,
            })

    def add_ann(self, mask: np.ndarray, bbox: list[float], score: float):
        if not self.coco or not self.img_info:
            return
        max_id = max((a["id"] for a in self.coco["annotations"]), default=0)
        new_id = max_id + 1
        self.coco["annotations"].append({
            "id": new_id,
            "image_id": self.img_info["id"],
            "category_id": 1,
            "bbox": [round(v, 2) for v in bbox],
            "area": round(float(mask.sum()), 2),
            "segmentation": encode_rle(mask),
            "score": round(score, 4),
            "iscrowd": 0,
        })
        self.modified.add((self.split, self.scene))
        self.added += 1
        self.history.append({
            "action": "add",
            "time": datetime.now().isoformat(),
            "split": self.split,
            "scene": self.scene,
            "frame_idx": self.idx,
            "annotation_id": new_id,
            "score": round(score, 4),
        })

    def save(self) -> list[str]:
        saved = []
        for split, scene in list(self.modified):
            out = self.labels_dir / split / f"{scene}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(self.data[split][scene], f, indent=2)
            saved.append(f"{split}/{scene}")
        self.modified.clear()
        self.saves += 1
        self._save_session_log()
        return saved

    def _save_session_log(self):
        log_dir = self.labels_dir / "session_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.session_start.strftime("%Y-%m-%d_%H-%M-%S")
        log_path = log_dir / f"{stamp}.json"
        log = {
            "session_start": self.session_start.isoformat(),
            "last_save": datetime.now().isoformat(),
            "total_deleted": self.deleted,
            "total_added": self.added,
            "total_saves": self.saves,
            "history": self.history,
        }
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    @property
    def stats_text(self) -> str:
        elapsed = datetime.now() - self.session_start
        mins = int(elapsed.total_seconds()) // 60
        secs = int(elapsed.total_seconds()) % 60
        return (
            f"**Session** ({mins}m {secs}s)\n\n"
            f"Deleted: **{self.deleted}** · "
            f"Added: **{self.added}** · "
            f"Saves: **{self.saves}**"
        )


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app(rev: Reviewer, model_id: str, fastsam_model: str):
    if not rev.splits:
        with gr.Blocks() as app:
            gr.Markdown(
                "## No annotation files found\n\n"
                "Run `autolabel_sam3.py` first to generate COCO JSON files.\n\n"
                "Expected layout: `dataset/labels/{split}/{scene}.json`"
            )
        return app

    def _refresh():
        img = rev.load_image()
        if img is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return blank, "No image loaded.", rev.stats_text
        info = rev.img_info
        h, w = info["height"], info["width"]
        anns = rev.anns_for_current()
        vis = render(img, anns, h, w)
        mod = ""
        if (rev.split, rev.scene) in rev.modified:
            mod = "  **(unsaved changes)**"
        status = (
            f"**{rev.split} / {rev.scene}** — "
            f"frame {rev.idx + 1} / {len(rev.images)} — "
            f"`{info['file_name']}` — {len(anns)} annotation(s){mod}"
        )
        return vis, status, rev.stats_text

    first_split = rev.splits[0]
    first_scenes = rev.scenes_for(first_split)

    with gr.Blocks(title="COCO Annotation Reviewer") as app:

        # ---- top bar ----
        gr.Markdown("# COCO Annotation Reviewer")
        with gr.Row():
            split_dd = gr.Dropdown(
                rev.splits, value=first_split, label="Split", scale=1,
            )
            scene_dd = gr.Dropdown(
                first_scenes,
                value=first_scenes[0] if first_scenes else None,
                label="Scene",
                scale=2,
            )
            slider = gr.Slider(
                0, max(len(rev.images) - 1, 0), step=1, value=0,
                label="Frame", scale=2,
            )
            prev_btn = gr.Button("Prev", size="sm", scale=0, elem_id="btn_prev")
            next_btn = gr.Button("Next", size="sm", scale=0, elem_id="btn_next")
            save_btn = gr.Button("Save", variant="primary", size="sm", scale=0, elem_id="btn_save")

        # ---- main area ----
        with gr.Row():
            with gr.Column(scale=3):
                img_out = gr.Image(
                    label="Annotated frame",
                    type="numpy",
                    interactive=False,
                    elem_id="img_out",
                )
                status_md = gr.Markdown("")

            with gr.Column(scale=1, min_width=260):
                stats_md = gr.Markdown(rev.stats_text)
                gr.Markdown("---")
                gr.Markdown("### Click Action")
                click_action = gr.Radio(
                    choices=["Remove", "Add segmentation"],
                    value="Remove",
                    label="What click does",
                )
                add_backend = gr.Radio(
                    choices=["FastSAM", "SAM3"],
                    value="FastSAM",
                    label="Add backend",
                )
                remove_all_hits = gr.Checkbox(
                    label="Remove all overlapping annotations at click",
                    value=False,
                )
                add_prompt = gr.Textbox(
                    label="Add prompt (SAM3 text prompt)",
                    value="person",
                    placeholder="e.g. person, backpack, bicycle",
                )
                add_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.01,
                    label="Add detection threshold",
                )
                add_mask_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label="Add mask threshold",
                )
                fastsam_iou = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.01,
                    label="FastSAM IoU threshold",
                )
                fastsam_imgsz = gr.Slider(
                    minimum=256,
                    maximum=1536,
                    value=1024,
                    step=32,
                    label="FastSAM image size",
                )
                gr.Markdown(
                    "**Remove**: click an annotation to delete it.\n\n"
                    "**Add segmentation**: click a missed object to add a new "
                    "segmentation. The app uses the selected backend and picks the "
                    "best mask at (or nearest to) the clicked point.\n\n"
                    "**Left / Right** arrow keys to navigate, **Enter** to save."
                )

        # ---- event handlers ----
        def on_split(s):
            rev.split = s
            scenes = rev.scenes_for(s)
            rev.scene = scenes[0] if scenes else None
            rev.idx = 0
            vis, status, stats = _refresh()
            return (
                gr.update(choices=scenes, value=rev.scene),
                gr.update(maximum=max(len(rev.images) - 1, 0), value=0),
                vis,
                status,
                stats,
            )

        def on_scene(s):
            rev.scene = s
            rev.idx = 0
            vis, status, stats = _refresh()
            return (
                gr.update(maximum=max(len(rev.images) - 1, 0), value=0),
                vis,
                status,
                stats,
            )

        def on_slider(i):
            rev.idx = int(i)
            vis, status, stats = _refresh()
            return vis, status, stats

        def on_prev():
            rev.idx = max(0, rev.idx - 1)
            vis, status, stats = _refresh()
            return gr.update(value=rev.idx), vis, status, stats

        def on_next():
            rev.idx = min(len(rev.images) - 1, rev.idx + 1)
            vis, status, stats = _refresh()
            return gr.update(value=rev.idx), vis, status, stats

        def on_save():
            saved = rev.save()
            _, status, stats = _refresh()
            if saved:
                gr.Info(f"Saved: {', '.join(saved)}")
            else:
                gr.Info("Nothing to save.")
            return status, stats

        def on_click(
            evt: gr.SelectData,
            action: str,
            backend: str,
            remove_all: bool,
            text_prompt: str,
            threshold: float,
            mask_threshold: float,
            fs_iou: float,
            fs_imgsz: float,
        ):
            info = rev.img_info
            if not info:
                return gr.update(), gr.update(), gr.update()

            x, y = evt.index
            h, w = info["height"], info["width"]
            x = min(max(int(x), 0), w - 1)
            y = min(max(int(y), 0), h - 1)

            if action == "Remove":
                anns = rev.anns_for_current()
                hits = anns_at_point(anns, x, y, h, w)
                if hits:
                    if remove_all:
                        ids = {ann["id"] for ann in hits}
                        rev.delete_anns(ids)
                        gr.Info(f"Deleted {len(ids)} annotation(s) at click.")
                    else:
                        hit = hits[0]
                        rev.delete_anns({hit["id"]})
                        gr.Info(f"Deleted annotation #{hit['id']}")
                else:
                    gr.Warning("No annotation at this location.")
                vis, status, stats = _refresh()
                return vis, status, stats

            img_rgb = rev.load_image()
            if img_rgb is None:
                return gr.update(), gr.update(), gr.update()

            prompt = (text_prompt or "").strip() or "person"
            masks = []
            boxes = []
            scores = []

            if backend == "FastSAM":
                fastsam = load_fastsam(fastsam_model)
                preds = fastsam(
                    img_rgb,
                    device=_device,
                    verbose=False,
                    retina_masks=True,
                    conf=float(threshold),
                    iou=float(fs_iou),
                    imgsz=int(fs_imgsz),
                )
                res = preds[0]
                if res.masks is not None and res.boxes is not None:
                    masks = [m for m in res.masks.data]
                    boxes = [b for b in res.boxes.xyxy]
                    scores = [s for s in res.boxes.conf]
            else:
                pil_img = Image.fromarray(img_rgb)
                model, processor = load_sam3(model_id)
                inputs = processor(
                    images=pil_img, text=prompt, return_tensors="pt",
                ).to(_device)
                with torch.no_grad():
                    outputs = model(**inputs)
                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=float(threshold),
                    mask_threshold=float(mask_threshold),
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]
                masks = results.get("masks", [])
                boxes = results.get("boxes", [])
                scores = results.get("scores", [])

            best, best_score = None, -1.0
            for m, b, s in zip(masks, boxes, scores):
                mn = m.cpu().numpy().astype(np.uint8)
                if y < mn.shape[0] and x < mn.shape[1] and mn[y, x] > 0:
                    sc = float(s)
                    if sc > best_score:
                        bl = b.cpu().tolist()
                        best = (mn, [bl[0], bl[1], bl[2] - bl[0], bl[3] - bl[1]], sc)
                        best_score = sc

            if best is None and len(masks) > 0:
                min_dist = float("inf")
                for m, b, s in zip(masks, boxes, scores):
                    mn = m.cpu().numpy().astype(np.uint8)
                    ys, xs = np.where(mn > 0)
                    if len(xs) == 0:
                        continue
                    dist = float(np.min((xs - x) ** 2 + (ys - y) ** 2))
                    if dist < min_dist:
                        min_dist = dist
                        bl = b.cpu().tolist()
                        best = (mn, [bl[0], bl[1], bl[2] - bl[0], bl[3] - bl[1]], float(s))

            if best:
                rev.add_ann(best[0], best[1], best[2])
                if backend == "FastSAM":
                    gr.Info(f"Added annotation via FastSAM (score {best[2]:.3f})")
                else:
                    gr.Info(f"Added annotation via SAM3 (score {best[2]:.3f}, prompt='{prompt}')")
            else:
                if backend == "FastSAM":
                    gr.Warning("No object detected at this location by FastSAM.")
                else:
                    gr.Warning(
                        f"No object detected at this location for prompt '{prompt}'."
                    )

            vis, status, stats = _refresh()
            return vis, status, stats

        def initial_load():
            vis, status, stats = _refresh()
            return vis, status, stats

        # ---- wiring ----
        split_dd.change(on_split, [split_dd], [scene_dd, slider, img_out, status_md, stats_md])
        scene_dd.change(on_scene, [scene_dd], [slider, img_out, status_md, stats_md])
        slider.release(on_slider, [slider], [img_out, status_md, stats_md])
        prev_btn.click(on_prev, [], [slider, img_out, status_md, stats_md])
        next_btn.click(on_next, [], [slider, img_out, status_md, stats_md])
        save_btn.click(on_save, [], [status_md, stats_md])
        img_out.select(
            on_click,
            [
                click_action,
                add_backend,
                remove_all_hits,
                add_prompt,
                add_threshold,
                add_mask_threshold,
                fastsam_iou,
                fastsam_imgsz,
            ],
            [img_out, status_md, stats_md],
        )
        app.load(initial_load, [], [img_out, status_md, stats_md])

    return app


def resolve_paths(
    annotation_source: str, labels_dir: Path, raw_frames_dir: Path
) -> tuple[Path, Path]:
    """Resolve labels/raw-frame paths from a source preset or custom values."""
    if annotation_source == "custom":
        return labels_dir, raw_frames_dir
    return ANNOTATION_SOURCE_PATHS[annotation_source]


def main():
    parser = argparse.ArgumentParser(
        description="Gradio app for reviewing / editing COCO annotations"
    )
    parser.add_argument(
        "--annotation-source",
        choices=["custom", *ANNOTATION_SOURCE_PATHS.keys()],
        default="custom",
        help=(
            "Pick a known annotation source preset. Use 'custom' to rely on "
            "--labels-dir and --raw-frames-dir."
        ),
    )
    parser.add_argument(
        "--labels-dir", type=Path, default=Path("dataset/labels"),
    )
    parser.add_argument(
        "--raw-frames-dir", type=Path, default=Path("dataset/raw_frames"),
    )
    parser.add_argument("--model-id", type=str, default="facebook/sam3")
    parser.add_argument(
        "--fastsam-model",
        type=str,
        default="FastSAM-s.pt",
        help="FastSAM weights path/name for fast click-based segmentation.",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    labels_dir, raw_frames_dir = resolve_paths(
        args.annotation_source, args.labels_dir, args.raw_frames_dir
    )
    print(f"Annotation source: {args.annotation_source}")
    print(f"Labels dir: {labels_dir}")
    print(f"Raw frames dir: {raw_frames_dir}")

    rev = Reviewer(labels_dir, raw_frames_dir)
    app = build_app(rev, args.model_id, args.fastsam_model)
    app.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css="#img_out {cursor: crosshair;}",
        head=KEYBOARD_HEAD,
    )


if __name__ == "__main__":
    main()
