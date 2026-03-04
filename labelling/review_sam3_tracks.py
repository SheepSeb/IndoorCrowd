#!/usr/bin/env python3
"""Gradio viewer + editor for SAM3 video tracking auto-labels.

Loads the output of autolabel_sam3_video.py:
  dataset/SAM3_video_tracks/{split}/{scene}/part_{N}/
    frames/          source frame images (symlinks)
    gt/gt.txt        MOT bounding boxes with track IDs
    annotations.json COCO-style: bbox + RLE mask + track_id per instance

Displays frames with:
  - Semi-transparent segmentation masks (per track colour)
  - Bounding boxes + track ID labels
  - Click a box/mask to highlight that track
  - Track list panel with frame range and box count
  - Reassign / delete / export editing operations

Usage
-----
  uv run python labelling/review_sam3_tracks.py

  uv run python labelling/review_sam3_tracks.py \\
      --tracks-dir dataset/SAM3_video_tracks \\
      --port 7861
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PALETTE: list[tuple[int, int, int]] = [
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

KEYBOARD_JS = """
<script>
function handleKeyboard(e) {
    if (["input","textarea","select"].includes(e.target.tagName.toLowerCase())) return;
    if (e.key === "ArrowLeft")  { e.preventDefault(); document.getElementById("btn_prev").click(); }
    if (e.key === "ArrowRight") { e.preventDefault(); document.getElementById("btn_next").click(); }
    if (e.key === "Enter")      { e.preventDefault(); document.getElementById("btn_save").click(); }
}
document.addEventListener('keydown', handleKeyboard, false);
</script>
"""

MASK_ALPHA = 0.40  # mask overlay transparency


def track_color(track_id: int) -> tuple[int, int, int]:
    return PALETTE[int(track_id) % len(PALETTE)]


def track_color_hex(track_id: int) -> str:
    r, g, b = track_color(track_id)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Rendering ─────────────────────────────────────────────────────────────────


def decode_mask(seg, img_h: int, img_w: int) -> np.ndarray | None:
    """Decode a COCO RLE or polygon segmentation to a binary uint8 mask."""
    if not seg:
        return None
    try:
        if isinstance(seg, dict):
            # RLE — pycocotools expects bytes counts
            rle = {"size": seg["size"], "counts": seg["counts"].encode("utf-8")}
            mask = mask_utils.decode(rle).astype(np.uint8)
        elif isinstance(seg, list):
            # Polygon
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
        else:
            return None
        return mask
    except Exception:
        return None


def render_frame(
    img_rgb: np.ndarray,
    anns: list[dict],
    selected_track_id: int | None,
    show_masks: bool,
) -> np.ndarray:
    """Overlay masks and bounding boxes on the image."""
    vis = img_rgb.copy().astype(np.float32)
    h, w = vis.shape[:2]

    # Draw masks first (behind boxes)
    if show_masks:
        mask_overlay = np.zeros_like(vis)
        for ann in anns:
            tid = ann["track_id"]
            seg = ann.get("segmentation")
            if not seg:
                continue
            binary = decode_mask(seg, h, w)
            if binary is None:
                continue
            c = track_color(tid)
            alpha = MASK_ALPHA * (1.5 if tid == selected_track_id else 1.0)
            alpha = min(alpha, 0.7)
            for ch, cv_ in enumerate(c):
                vis[:, :, ch] = np.where(
                    binary > 0,
                    vis[:, :, ch] * (1 - alpha) + cv_ * alpha,
                    vis[:, :, ch],
                )

    vis = np.clip(vis, 0, 255).astype(np.uint8)

    # Draw boxes (non-selected first, selected on top)
    for ann in sorted(anns, key=lambda a: a["track_id"] == selected_track_id):
        tid = ann["track_id"]
        x, y, bw, bh = [int(v) for v in ann["bbox"]]
        c = track_color(tid)
        is_sel = tid == selected_track_id

        if is_sel:
            cv2.rectangle(vis, (x - 4, y - 4), (x + bw + 4, y + bh + 4), (255, 255, 0), 3)
            thickness = 3
        else:
            thickness = 2

        cv2.rectangle(vis, (x, y), (x + bw, y + bh), c, thickness)

        label = f"ID {tid}"
        fs = 0.55
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ly = max(y, th + bl + 4)
        cv2.rectangle(vis, (x, ly - th - bl - 4), (x + tw + 6, ly), c, -1)
        cv2.putText(
            vis, label, (x + 3, ly - bl - 2),
            cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return vis


def box_at_point(anns: list[dict], px: int, py: int) -> dict | None:
    """Return the smallest bbox annotation containing (px, py), or None."""
    best, best_area = None, float("inf")
    for ann in anns:
        x, y, bw, bh = ann["bbox"]
        if x <= px <= x + bw and y <= py <= y + bh:
            area = bw * bh
            if area < best_area:
                best, best_area = ann, area
    return best


# ── Data loading ──────────────────────────────────────────────────────────────


def load_annotations(ann_file: Path) -> tuple[dict[int, list[dict]], dict]:
    """Load annotations.json → ({image_id: [ann_dict, ...]}, coco_meta)."""
    if not ann_file.exists():
        return {}, {"images": [], "categories": []}
    with open(ann_file) as f:
        coco = json.load(f)
    by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        by_image.setdefault(ann["image_id"], []).append(ann)
    meta = {
        "images": coco.get("images", []),
        "categories": coco.get("categories", []),
    }
    return by_image, meta


def collect_frames_in_dir(d: Path) -> list[Path]:
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_tracks_dir(tracks_dir: Path) -> dict:
    """
    Discover all splits / scenes / parts.

    Returns nested dict:
      data[split][scene][part_key] = {
          "frames":      list[Path],
          "anns_by_img": {image_id: [ann, ...]},
          "coco_meta":   {"images": [...], "categories": [...]},
          "img_id_for_frame": {local_fnum: image_id},
          "part_dir":    Path,
      }
    """
    data: dict = {}
    for split_dir in sorted(tracks_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        scenes: dict = {}
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            parts: dict = {}
            for part_dir in sorted(scene_dir.iterdir()):
                if not part_dir.is_dir() or not part_dir.name.startswith("part_"):
                    continue
                frames_dir = part_dir / "frames"
                if not frames_dir.exists():
                    continue
                frames = collect_frames_in_dir(frames_dir)
                if not frames:
                    continue

                ann_file = part_dir / "annotations.json"
                anns_by_img, coco_meta = load_annotations(ann_file)

                # Build local-frame-number → image_id mapping using the COCO
                # image list (images are numbered 1..N in local order)
                img_id_for_frame: dict[int, int] = {}
                for img_info in coco_meta["images"]:
                    img_id_for_frame[img_info["id"]] = img_info["id"]

                parts[part_dir.name] = {
                    "frames": frames,
                    "anns_by_img": anns_by_img,
                    "coco_meta": coco_meta,
                    "img_id_for_frame": img_id_for_frame,
                    "part_dir": part_dir,
                }
            if parts:
                scenes[scene_dir.name] = parts
        if scenes:
            data[split] = scenes
    return data


# ── Save helpers ──────────────────────────────────────────────────────────────


def save_annotations_json(part_dir: Path, anns_by_img: dict, coco_meta: dict) -> None:
    """Write anns_by_img back to annotations.json."""
    all_anns = [ann for anns in anns_by_img.values() for ann in anns]
    # Re-index annotation IDs
    for i, ann in enumerate(all_anns, start=1):
        ann["id"] = i
    coco_out = {
        "images": coco_meta["images"],
        "categories": coco_meta["categories"],
        "annotations": all_anns,
    }
    ann_file = part_dir / "annotations.json"
    with open(ann_file, "w") as f:
        json.dump(coco_out, f)


def save_gt_txt(part_dir: Path, anns_by_img: dict) -> None:
    """Write MOT gt.txt from anns_by_img."""
    gt_dir = part_dir / "gt"
    gt_dir.mkdir(exist_ok=True)
    rows = []
    for img_id, anns in sorted(anns_by_img.items()):
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            tid = ann["track_id"]
            conf = ann.get("score", 1.0)
            rows.append(f"{img_id},{tid},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},{conf:.4f},-1,-1,-1")
    gt_file = gt_dir / "gt.txt"
    gt_file.write_text("\n".join(rows) + ("\n" if rows else ""))


# ── MP4 export ────────────────────────────────────────────────────────────────


def export_part_mp4(pd: dict, fps: float = 12.0) -> Path:
    """Render every frame with overlays and write to part_dir/clip.mp4."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = pd["part_dir"] / "clip.mp4"
    writer = None

    for local_idx, frame_path in enumerate(pd["frames"]):
        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_id = local_idx + 1
        anns = pd["anns_by_img"].get(img_id, [])
        vis_rgb = render_frame(img_rgb, anns, None, show_masks=True)
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

        if writer is None:
            h, w = vis_bgr.shape[:2]
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for {out_path}")

        writer.write(vis_bgr)

    if writer is not None:
        writer.release()

    return out_path


# ── Gradio app ────────────────────────────────────────────────────────────────


def build_app(data: dict, tracks_dir: Path) -> gr.Blocks:
    if not data:
        with gr.Blocks() as app:
            gr.Markdown(
                "## No SAM3 video tracks found\n\n"
                f"Run `autolabel_sam3_video.py` first.\n\n"
                f"Expected: `{tracks_dir}/{{split}}/{{scene}}/part_N/`"
            )
        return app

    # ── mutable viewer state ──────────────────────────────────────────────────
    state = {
        "split": next(iter(data)),
        "scene": None,
        "part": None,
        "idx": 0,
        "selected_track_id": None,
        "modified": False,
    }

    def _set_scene_and_part(split: str, scene: str | None = None, part: str | None = None):
        state["split"] = split
        scenes = list(data.get(split, {}).keys())
        state["scene"] = scene or (scenes[0] if scenes else None)
        parts = list(data.get(split, {}).get(state["scene"] or "", {}).keys())
        state["part"] = part or (parts[0] if parts else None)
        state["idx"] = 0
        state["selected_track_id"] = None
        state["modified"] = False

    _set_scene_and_part(state["split"])

    def _parts_for(split: str, scene: str) -> list[str]:
        return list(data.get(split, {}).get(scene, {}).keys())

    def _part_data() -> dict | None:
        s, sc, p = state["split"], state["scene"], state["part"]
        return data.get(s, {}).get(sc or "", {}).get(p or "")

    def _anns_for_frame(idx: int) -> list[dict]:
        pd = _part_data()
        if not pd:
            return []
        img_id = idx + 1  # 1-based
        return pd["anns_by_img"].get(img_id, [])

    def _load_image(idx: int) -> np.ndarray | None:
        pd = _part_data()
        if not pd:
            return None
        frames = pd["frames"]
        if not (0 <= idx < len(frames)):
            return None
        fp = frames[idx]
        if not fp.exists():
            return None
        return np.array(Image.open(fp).convert("RGB"))

    def _track_ids() -> list[int]:
        pd = _part_data()
        if not pd:
            return []
        return sorted({
            ann["track_id"]
            for anns in pd["anns_by_img"].values()
            for ann in anns
        })

    def _track_frame_range(tid: int) -> tuple[int, int]:
        pd = _part_data()
        if not pd:
            return 0, 0
        frames = [
            img_id for img_id, anns in pd["anns_by_img"].items()
            if any(a["track_id"] == tid for a in anns)
        ]
        return (min(frames), max(frames)) if frames else (0, 0)

    def _next_free_id() -> int:
        ids = _track_ids()
        return (max(ids) + 1) if ids else 1

    # ── edit operations ───────────────────────────────────────────────────────

    def _op_delete_track(tid: int) -> int:
        pd = _part_data()
        if not pd:
            return 0
        before = sum(len(v) for v in pd["anns_by_img"].values())
        pd["anns_by_img"] = {
            img_id: [a for a in anns if a["track_id"] != tid]
            for img_id, anns in pd["anns_by_img"].items()
        }
        # Remove empty keys
        pd["anns_by_img"] = {k: v for k, v in pd["anns_by_img"].items() if v}
        after = sum(len(v) for v in pd["anns_by_img"].values())
        changed = before - after
        if changed:
            state["modified"] = True
        return changed

    def _op_delete_box(tid: int, img_id: int) -> int:
        pd = _part_data()
        if not pd:
            return 0
        anns = pd["anns_by_img"].get(img_id, [])
        before = len(anns)
        new_anns = [a for a in anns if a["track_id"] != tid]
        changed = before - len(new_anns)
        if new_anns:
            pd["anns_by_img"][img_id] = new_anns
        elif img_id in pd["anns_by_img"]:
            del pd["anns_by_img"][img_id]
        if changed:
            state["modified"] = True
        return changed

    def _op_reassign(old_id: int, new_id: int, scope: str, cur_img_id: int) -> int:
        pd = _part_data()
        if not pd:
            return 0
        changed = 0
        for img_id, anns in pd["anns_by_img"].items():
            if scope == "this_frame" and img_id != cur_img_id:
                continue
            if scope == "from_here" and img_id < cur_img_id:
                continue
            for ann in anns:
                if ann["track_id"] == old_id:
                    ann["track_id"] = new_id
                    changed += 1
        if changed:
            state["modified"] = True
        return changed

    # ── rendering ─────────────────────────────────────────────────────────────

    def _refresh(show_masks: bool = True):
        pd = _part_data()
        if pd is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No data", (220, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            return blank, "No data loaded.", _track_list_html()

        img = _load_image(state["idx"])
        if img is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return blank, "Frame not found.", _track_list_html()

        anns = _anns_for_frame(state["idx"])
        vis = render_frame(img, anns, state["selected_track_id"], show_masks)

        n_frames = len(pd["frames"])
        fnum = state["idx"] + 1
        mod_str = "  **(unsaved changes)**" if state["modified"] else ""
        status = (
            f"**{state['split']} / {state['scene']} / {state['part']}** — "
            f"Frame {fnum}/{n_frames} — "
            f"{len(anns)} instance(s) detected"
            f"{mod_str}"
        )
        return vis, status, _track_list_html()

    def _track_list_html() -> str:
        ids = _track_ids()
        if not ids:
            return "<p style='color:#888'>No tracks in this part.</p>"
        rows_html = []
        for tid in ids:
            fmin, fmax = _track_frame_range(tid)
            count = sum(
                1
                for anns in (_part_data() or {}).get("anns_by_img", {}).values()
                for a in anns
                if a["track_id"] == tid
            )
            hex_c = track_color_hex(tid)
            bg = "background:#333;border:2px solid yellow;" if tid == state["selected_track_id"] else ""
            rows_html.append(
                f'<tr style="{bg}">'
                f'<td style="padding:2px 8px">'
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:{hex_c};border-radius:2px;margin-right:5px;vertical-align:middle"></span>'
                f'<b>{tid}</b></td>'
                f'<td style="padding:2px 8px;color:#ccc">{fmin}–{fmax}</td>'
                f'<td style="padding:2px 8px;color:#ccc">{count}</td>'
                f'</tr>'
            )
        return (
            '<table style="font-size:0.82em;border-collapse:collapse;width:100%;'
            'color:#eee;background:#1a1a1a;border-radius:6px">'
            '<thead><tr>'
            '<th style="padding:4px 8px;text-align:left;border-bottom:1px solid #444">Track</th>'
            '<th style="padding:4px 8px;text-align:left;border-bottom:1px solid #444">Frames</th>'
            '<th style="padding:4px 8px;text-align:left;border-bottom:1px solid #444">Count</th>'
            '</tr></thead>'
            '<tbody>' + "".join(rows_html) + "</tbody></table>"
        )

    def _selection_text() -> str:
        tid = state["selected_track_id"]
        if tid is None:
            return "_Click a box to select a track._"
        fmin, fmax = _track_frame_range(tid)
        count = sum(
            1
            for anns in (_part_data() or {}).get("anns_by_img", {}).values()
            for a in anns
            if a["track_id"] == tid
        )
        return (
            f"**Selected: Track {tid}**  \n"
            f"Frames {fmin}–{fmax} ({count} box(es))  \n"
            f"_Next free ID: {_next_free_id()}_"
        )

    # ── initial values ────────────────────────────────────────────────────────

    first_split = state["split"]
    first_scenes = list(data.get(first_split, {}).keys())
    first_scene = first_scenes[0] if first_scenes else None
    first_parts = _parts_for(first_split, first_scene or "")
    first_part = first_parts[0] if first_parts else None
    n_frames_init = len((_part_data() or {}).get("frames", []))

    # ── build UI ──────────────────────────────────────────────────────────────

    with gr.Blocks(title="SAM3 Track Viewer") as app:
        gr.Markdown("# SAM3 Video Track Viewer & Editor")

        show_masks_state = gr.State(True)

        with gr.Row():
            split_dd = gr.Dropdown(
                list(data.keys()), value=first_split, label="Split", scale=1
            )
            scene_dd = gr.Dropdown(
                first_scenes, value=first_scene, label="Scene", scale=3
            )
            part_dd = gr.Dropdown(
                first_parts, value=first_part, label="Part", scale=1
            )
            mask_toggle = gr.Checkbox(value=True, label="Show masks", scale=0)
            save_btn = gr.Button("💾 Save", variant="primary", size="sm", scale=0, elem_id="btn_save")

        with gr.Row():
            prev_btn = gr.Button("◀ Prev", size="sm", elem_id="btn_prev", scale=0)
            slider = gr.Slider(
                0, max(n_frames_init - 1, 0), step=1, value=0,
                label="Frame", scale=5,
            )
            next_btn = gr.Button("Next ▶", size="sm", elem_id="btn_next", scale=0)

        with gr.Row():
            with gr.Column(scale=3):
                img_out = gr.Image(
                    label="Frame  (click a box to highlight/select track)",
                    type="numpy",
                    interactive=False,
                    elem_id="img_out",
                )
                status_md = gr.Markdown("")

            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Tracks in part")
                track_list = gr.HTML(_track_list_html())
                gr.Markdown(
                    "<small>Each row: **track ID**, frame range, box count.</small>"
                )

                gr.Markdown("---")
                gr.Markdown("### Selection")
                selected_md = gr.Markdown(_selection_text())

                gr.Markdown("---")
                gr.Markdown("### Reassign Track ID")
                new_id_input = gr.Number(
                    label="New Track ID",
                    precision=0,
                    minimum=1,
                    value=None,
                    info="Existing ID = merge; fresh ID = split",
                )
                scope_radio = gr.Radio(
                    choices=["This frame only", "From this frame onwards", "All frames"],
                    value="From this frame onwards",
                    label="Scope",
                )
                with gr.Row():
                    apply_btn = gr.Button("Apply", variant="primary")
                    clear_sel_btn = gr.Button("Clear Selection")

                gr.Markdown("---")
                gr.Markdown("### Delete")
                with gr.Row():
                    del_box_btn = gr.Button("Delete Box", variant="stop", size="sm")
                    del_track_btn = gr.Button("Delete Track", variant="stop", size="sm")

                gr.Markdown("---")
                gr.Markdown("### Export")
                export_btn = gr.Button("Export Part MP4", size="sm")
                log_box = gr.Textbox(
                    label="Log",
                    lines=4,
                    interactive=False,
                    placeholder="Actions will be logged here.",
                )

        # ── event handlers ────────────────────────────────────────────────────

        _img_outs = [img_out, status_md, track_list]

        def on_split(split, show_masks):
            _set_scene_and_part(split)
            scenes = list(data.get(split, {}).keys())
            parts = _parts_for(split, scenes[0] if scenes else "")
            pd = _part_data()
            n = len(pd["frames"]) if pd else 0
            vis, status, tlist = _refresh(show_masks)
            return (
                gr.update(choices=scenes, value=state["scene"]),
                gr.update(choices=parts, value=state["part"]),
                gr.update(maximum=max(n - 1, 0), value=0),
                vis, status, tlist,
                _selection_text(),
            )

        def on_scene(scene, split, show_masks):
            parts = _parts_for(split, scene)
            state["scene"] = scene
            state["part"] = parts[0] if parts else None
            state["idx"] = 0
            state["selected_track_id"] = None
            state["modified"] = False
            pd = _part_data()
            n = len(pd["frames"]) if pd else 0
            vis, status, tlist = _refresh(show_masks)
            return (
                gr.update(choices=parts, value=state["part"]),
                gr.update(maximum=max(n - 1, 0), value=0),
                vis, status, tlist,
                _selection_text(),
            )

        def on_part(part, show_masks):
            state["part"] = part
            state["idx"] = 0
            state["selected_track_id"] = None
            state["modified"] = False
            pd = _part_data()
            n = len(pd["frames"]) if pd else 0
            vis, status, tlist = _refresh(show_masks)
            return (
                gr.update(maximum=max(n - 1, 0), value=0),
                vis, status, tlist,
                _selection_text(),
            )

        def on_slider(i, show_masks):
            state["idx"] = int(i)
            return _refresh(show_masks)

        def on_prev(show_masks):
            state["idx"] = max(0, state["idx"] - 1)
            vis, status, tlist = _refresh(show_masks)
            return gr.update(value=state["idx"]), vis, status, tlist

        def on_next(show_masks):
            pd = _part_data()
            n = len(pd["frames"]) if pd else 0
            state["idx"] = min(n - 1, state["idx"] + 1)
            vis, status, tlist = _refresh(show_masks)
            return gr.update(value=state["idx"]), vis, status, tlist

        def on_mask_toggle(show_masks):
            return _refresh(show_masks)

        def on_click(evt: gr.SelectData, show_masks):
            px, py = int(evt.index[0]), int(evt.index[1])
            anns = _anns_for_frame(state["idx"])
            hit = box_at_point(anns, px, py)
            if hit:
                tid = hit["track_id"]
                state["selected_track_id"] = (
                    None if state["selected_track_id"] == tid else tid
                )
            else:
                state["selected_track_id"] = None
            vis, status, tlist = _refresh(show_masks)
            return vis, status, tlist, _selection_text()

        def on_save(show_masks):
            pd = _part_data()
            if not pd:
                return _refresh(show_masks) + ("No part loaded.",)
            if not state["modified"]:
                return _refresh(show_masks) + ("Nothing to save.",)
            save_annotations_json(pd["part_dir"], pd["anns_by_img"], pd["coco_meta"])
            save_gt_txt(pd["part_dir"], pd["anns_by_img"])
            state["modified"] = False
            vis, status, tlist = _refresh(show_masks)
            msg = f"Saved → {pd['part_dir']}"
            gr.Info("Saved successfully.")
            return vis, status, tlist, msg

        def on_apply(new_id, scope, show_masks):
            if state["selected_track_id"] is None:
                gr.Warning("No track selected — click a box first.")
                return _refresh(show_masks) + (_selection_text(), "")
            if new_id is None:
                gr.Warning("Enter a new Track ID first.")
                return _refresh(show_masks) + (_selection_text(), "")
            old_id = state["selected_track_id"]
            new_id = int(new_id)
            cur_img_id = state["idx"] + 1
            scope_map = {
                "This frame only":         "this_frame",
                "From this frame onwards": "from_here",
                "All frames":              "all",
            }
            changed = _op_reassign(old_id, new_id, scope_map[scope], cur_img_id)
            if changed:
                state["selected_track_id"] = new_id
                msg = f"Reassigned {changed} box(es): Track {old_id} → {new_id} ({scope})"
                gr.Info(msg)
            else:
                msg = f"No boxes matched scope for Track {old_id}."
                gr.Warning(msg)
            vis, status, tlist = _refresh(show_masks)
            return vis, status, tlist, _selection_text(), msg

        def on_clear_sel(show_masks):
            state["selected_track_id"] = None
            vis, status, tlist = _refresh(show_masks)
            return vis, status, tlist, _selection_text()

        def on_del_box(show_masks):
            tid = state["selected_track_id"]
            if tid is None:
                gr.Warning("No track selected.")
                return _refresh(show_masks) + ("",)
            img_id = state["idx"] + 1
            changed = _op_delete_box(tid, img_id)
            if changed:
                msg = f"Deleted box: Track {tid} on frame {img_id}."
                gr.Info(msg)
            else:
                msg = f"Track {tid} has no box on frame {img_id}."
                gr.Warning(msg)
            vis, status, tlist = _refresh(show_masks)
            return vis, status, tlist, msg

        def on_del_track(show_masks):
            tid = state["selected_track_id"]
            if tid is None:
                gr.Warning("No track selected.")
                return _refresh(show_masks) + (_selection_text(), "")
            changed = _op_delete_track(tid)
            state["selected_track_id"] = None
            if changed:
                msg = f"Deleted Track {tid} ({changed} box(es))."
                gr.Info(msg)
            else:
                msg = f"Track {tid} had no boxes to delete."
                gr.Warning(msg)
            vis, status, tlist = _refresh(show_masks)
            return vis, status, tlist, _selection_text(), msg

        def on_export(show_masks):
            pd = _part_data()
            if not pd:
                return _refresh(show_masks) + ("No part loaded.",)
            try:
                out_path = export_part_mp4(pd)
                msg = f"Exported → {out_path}"
                gr.Info(msg)
            except Exception as exc:
                msg = f"Export failed: {exc}"
                gr.Warning(msg)
            return _refresh(show_masks) + (msg,)

        # ── wiring ────────────────────────────────────────────────────────────

        split_dd.change(
            on_split, [split_dd, show_masks_state],
            [scene_dd, part_dd, slider] + _img_outs + [selected_md],
        )
        scene_dd.change(
            on_scene, [scene_dd, split_dd, show_masks_state],
            [part_dd, slider] + _img_outs + [selected_md],
        )
        part_dd.change(
            on_part, [part_dd, show_masks_state],
            [slider] + _img_outs + [selected_md],
        )
        slider.release(on_slider, [slider, show_masks_state], _img_outs)
        prev_btn.click(on_prev, [show_masks_state], [slider] + _img_outs)
        next_btn.click(on_next, [show_masks_state], [slider] + _img_outs)
        mask_toggle.change(on_mask_toggle, [mask_toggle], _img_outs)
        img_out.select(on_click, [mask_toggle], _img_outs + [selected_md])

        save_btn.click(on_save, [mask_toggle], _img_outs + [log_box])
        apply_btn.click(on_apply, [new_id_input, scope_radio, mask_toggle], _img_outs + [selected_md, log_box])
        clear_sel_btn.click(on_clear_sel, [mask_toggle], _img_outs + [selected_md])
        del_box_btn.click(on_del_box, [mask_toggle], _img_outs + [log_box])
        del_track_btn.click(on_del_track, [mask_toggle], _img_outs + [selected_md, log_box])
        export_btn.click(on_export, [mask_toggle], _img_outs + [log_box])

        app.load(lambda m: _refresh(m), [mask_toggle], _img_outs)

    return app


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradio viewer for SAM3 video tracking auto-labels"
    )
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=Path("dataset/SAM3_video_tracks"),
        help="Root directory output by autolabel_sam3_video.py.",
    )
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if not args.tracks_dir.exists():
        print(f"ERROR: tracks directory not found: {args.tracks_dir}")
        raise SystemExit(1)

    print(f"Loading SAM3 tracks from: {args.tracks_dir}")
    data = load_tracks_dir(args.tracks_dir)
    total_parts = sum(
        len(parts)
        for scenes in data.values()
        for parts in scenes.values()
    )
    total_scenes = sum(len(scenes) for scenes in data.values())
    print(
        f"Loaded {total_parts} part(s) across "
        f"{total_scenes} scene(s) in {len(data)} split(s)."
    )

    app = build_app(data, args.tracks_dir)
    app.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css="#img_out { cursor: crosshair; }",
        head=KEYBOARD_JS,
    )


if __name__ == "__main__":
    main()
