#!/usr/bin/env python3
"""Gradio app for reviewing and relabelling MOT-format tracking labels.

Loads MOT labels from dataset/MOT_labels_3fps/{split}/{scene}/gt/gt.txt
and the corresponding raw frames listed in seqinfo.ini.

Interaction
-----------
- Click on a bounding box to select that track
- Choose relabelling scope: "This frame", "From this frame onwards", "All frames"
- Enter a new track ID and click Apply — reassigns the ID (merge by using an
  existing ID; split by using a fresh one)
- Delete a single box or all boxes of a selected track
- Left/Right arrow keys navigate frames; Enter saves

Usage
-----
  uv run python labelling/review_mot_labels.py

  uv run python labelling/review_mot_labels.py \\
      --mot-dir   dataset/MOT_labels_3fps \\
      --port      7860
"""

from __future__ import annotations

import argparse
import configparser
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

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

KEYBOARD_HEAD = """
<script>
function handleKeyboard(e) {
    switch (e.target.tagName.toLowerCase()) {
        case "input": case "textarea": case "select": return;
    }
    if (e.key === "ArrowLeft")  { e.preventDefault(); document.getElementById("btn_prev").click(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); document.getElementById("btn_next").click(); }
    else if (e.key === "Enter") { e.preventDefault(); document.getElementById("btn_save").click(); }
}
document.addEventListener('keydown', handleKeyboard, false);
</script>
"""


def track_color(track_id: int) -> tuple[int, int, int]:
    return PALETTE[(track_id - 1) % len(PALETTE)]


def track_color_hex(track_id: int) -> str:
    r, g, b = track_color(track_id)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Rendering ─────────────────────────────────────────────────────────────────


def render_frame(
    img_rgb: np.ndarray,
    rows: list[dict],
    selected_track_id: int | None = None,
) -> np.ndarray:
    """Draw bounding boxes with track-ID labels; highlight the selected track."""
    vis = img_rgb.copy()

    # Draw non-selected boxes first (so selected renders on top)
    for row in sorted(rows, key=lambda r: r["id"] == selected_track_id):
        tid = row["id"]
        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["w"]), int(row["h"])
        c = track_color(tid)
        is_selected = tid == selected_track_id

        if is_selected:
            # Yellow outer glow for the selected track
            cv2.rectangle(vis, (x - 4, y - 4), (x + w + 4, y + h + 4), (255, 255, 0), 3)
            thickness = 3
        else:
            thickness = 2

        cv2.rectangle(vis, (x, y), (x + w, y + h), c, thickness)

        label = f"ID {tid}"
        font_scale = 0.6
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = max(y, th + baseline + 4)
        cv2.rectangle(vis, (x, label_y - th - baseline - 4), (x + tw + 6, label_y), c, -1)
        cv2.putText(
            vis, label, (x + 3, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return vis


# ── Hit testing ───────────────────────────────────────────────────────────────


def box_at_point(rows: list[dict], px: int, py: int) -> dict | None:
    """Return the smallest bounding box containing (px, py), or None."""
    best: dict | None = None
    best_area = float("inf")
    for row in rows:
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        if x <= px <= x + w and y <= py <= y + h:
            area = w * h
            if area < best_area:
                best, best_area = row, area
    return best


# ── State ─────────────────────────────────────────────────────────────────────


class MOTReviewer:
    """Manages in-memory MOT labels and file I/O for the review tool."""

    def __init__(self, mot_dir: Path):
        self.mot_dir = mot_dir

        # data[split][scene] = {"rows": list[dict], "frames": list[Path],
        #                        "seqinfo": dict, "gt_file": Path}
        self.data: dict[str, dict[str, dict]] = {}
        self.modified: set[tuple[str, str]] = set()

        self.session_start = datetime.now()
        self.edits: int = 0
        self.saves: int = 0
        self.history: list[dict] = []

        self._load_all()

        self.split: str | None = next(iter(self.data), None)
        self.scene: str | None = (
            next(iter(self.data.get(self.split or "", {})), None) if self.split else None
        )
        self.idx: int = 0
        self.selected_track_id: int | None = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        for split_dir in sorted(self.mot_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split = split_dir.name
            scenes: dict[str, dict] = {}
            for scene_dir in sorted(split_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                gt_file = scene_dir / "gt" / "gt.txt"
                seqinfo_file = scene_dir / "seqinfo.ini"
                if not gt_file.exists():
                    continue
                rows = self._parse_gt(gt_file)
                seqinfo = self._parse_seqinfo(seqinfo_file) if seqinfo_file.exists() else {}
                frames = self._collect_frames(seqinfo, scene_dir)
                scenes[scene_dir.name] = {
                    "rows": rows,
                    "frames": frames,
                    "seqinfo": seqinfo,
                    "gt_file": gt_file,
                }
            if scenes:
                self.data[split] = scenes

    @staticmethod
    def _parse_gt(gt_file: Path) -> list[dict]:
        rows: list[dict] = []
        for line in gt_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            rows.append({
                "frame": int(parts[0]),
                "id":    int(parts[1]),
                "x":     float(parts[2]),
                "y":     float(parts[3]),
                "w":     float(parts[4]),
                "h":     float(parts[5]),
                "conf":  float(parts[6]),
                "rest":  ",".join(parts[7:]) if len(parts) > 7 else "-1,-1,-1",
            })
        return rows

    @staticmethod
    def _parse_seqinfo(seqinfo_file: Path) -> dict:
        cfg = configparser.ConfigParser()
        cfg.read(str(seqinfo_file))
        return dict(cfg["Sequence"]) if "Sequence" in cfg else {}

    @staticmethod
    def _collect_frames(seqinfo: dict, scene_dir: Path) -> list[Path]:
        im_dir = seqinfo.get("imdir")
        d = Path(im_dir) if im_dir else scene_dir
        if not d.exists():
            return []
        return sorted(
            p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def splits(self) -> list[str]:
        return list(self.data.keys())

    def scenes_for(self, split: str) -> list[str]:
        return list(self.data.get(split, {}).keys())

    @property
    def scene_data(self) -> dict | None:
        if self.split and self.scene:
            return self.data.get(self.split, {}).get(self.scene)
        return None

    @property
    def rows(self) -> list[dict]:
        d = self.scene_data
        return d["rows"] if d else []

    @property
    def frames(self) -> list[Path]:
        d = self.scene_data
        return d["frames"] if d else []

    @property
    def current_frame(self) -> Path | None:
        return self.frames[self.idx] if 0 <= self.idx < len(self.frames) else None

    def _frame_num(self, frame_path: Path) -> int:
        digits = "".join(c for c in frame_path.stem if c.isdigit())
        return int(digits) if digits else self.idx + 1

    @property
    def current_frame_num(self) -> int | None:
        fp = self.current_frame
        return self._frame_num(fp) if fp else None

    def rows_for_current(self) -> list[dict]:
        fnum = self.current_frame_num
        return [r for r in self.rows if r["frame"] == fnum] if fnum is not None else []

    def load_image(self) -> np.ndarray | None:
        fp = self.current_frame
        if fp and fp.exists():
            return np.array(Image.open(fp).convert("RGB"))
        return None

    def track_ids(self) -> list[int]:
        return sorted({r["id"] for r in self.rows})

    def track_frame_range(self, tid: int) -> tuple[int, int]:
        frames = [r["frame"] for r in self.rows if r["id"] == tid]
        return (min(frames), max(frames)) if frames else (0, 0)

    def next_free_id(self) -> int:
        ids = self.track_ids()
        return (max(ids) + 1) if ids else 1

    # ── Mutations ─────────────────────────────────────────────────────────────

    def reassign(self, old_id: int, new_id: int, scope: str, from_frame: int) -> int:
        """Reassign track IDs in-place. Returns number of changed rows."""
        changed = 0
        for row in self.rows:
            if row["id"] != old_id:
                continue
            if scope == "this_frame" and row["frame"] != from_frame:
                continue
            if scope == "from_here" and row["frame"] < from_frame:
                continue
            row["id"] = new_id
            changed += 1
        if changed:
            self.modified.add((self.split, self.scene))
            self.edits += changed
            self.history.append({
                "action": "reassign",
                "time": datetime.now().isoformat(),
                "split": self.split,
                "scene": self.scene,
                "old_id": old_id,
                "new_id": new_id,
                "scope": scope,
                "from_frame": from_frame,
                "changed": changed,
            })
        return changed

    def delete_box(self, track_id: int, frame_num: int) -> int:
        d = self.scene_data
        if not d:
            return 0
        before = len(d["rows"])
        d["rows"] = [
            r for r in d["rows"]
            if not (r["id"] == track_id and r["frame"] == frame_num)
        ]
        changed = before - len(d["rows"])
        if changed:
            self.modified.add((self.split, self.scene))
            self.edits += changed
            self.history.append({
                "action": "delete_box",
                "time": datetime.now().isoformat(),
                "split": self.split,
                "scene": self.scene,
                "track_id": track_id,
                "frame_num": frame_num,
            })
        return changed

    def delete_track(self, track_id: int) -> int:
        d = self.scene_data
        if not d:
            return 0
        before = len(d["rows"])
        d["rows"] = [r for r in d["rows"] if r["id"] != track_id]
        changed = before - len(d["rows"])
        if changed:
            self.modified.add((self.split, self.scene))
            self.edits += changed
            self.history.append({
                "action": "delete_track",
                "time": datetime.now().isoformat(),
                "split": self.split,
                "scene": self.scene,
                "track_id": track_id,
                "changed": changed,
            })
        return changed

    def save(self) -> list[str]:
        saved = []
        for split, scene in list(self.modified):
            d = self.data[split][scene]
            gt_file: Path = d["gt_file"]
            rows = sorted(d["rows"], key=lambda r: (r["frame"], r["id"]))
            lines = [
                f"{r['frame']},{r['id']},{r['x']:.2f},{r['y']:.2f},"
                f"{r['w']:.2f},{r['h']:.2f},{r['conf']:.4f},{r['rest']}"
                for r in rows
            ]
            gt_file.write_text("\n".join(lines) + ("\n" if lines else ""))
            saved.append(f"{split}/{scene}")
        self.modified.clear()
        self.saves += 1
        return saved

    # ── UI helpers ────────────────────────────────────────────────────────────

    @property
    def stats_text(self) -> str:
        elapsed = datetime.now() - self.session_start
        mins = int(elapsed.total_seconds()) // 60
        secs = int(elapsed.total_seconds()) % 60
        return (
            f"**Session** ({mins}m {secs}s)  ·  "
            f"Edits: **{self.edits}**  ·  "
            f"Saves: **{self.saves}**"
        )

    def track_list_html(self) -> str:
        ids = self.track_ids()
        if not ids:
            return "<p style='color:#888'>No tracks in this scene.</p>"
        rows_html = []
        for tid in ids:
            fmin, fmax = self.track_frame_range(tid)
            count = sum(1 for r in self.rows if r["id"] == tid)
            hex_c = track_color_hex(tid)
            bg = "background:#333; border:2px solid yellow;" if tid == self.selected_track_id else ""
            rows_html.append(
                f'<tr style="{bg}">'
                f'<td style="padding:2px 6px">'
                f'  <span style="display:inline-block;width:12px;height:12px;'
                f'background:{hex_c};border-radius:2px;margin-right:5px;vertical-align:middle"></span>'
                f'  <b>{tid}</b>'
                f'</td>'
                f'<td style="padding:2px 6px;color:#ccc">{fmin}–{fmax}</td>'
                f'<td style="padding:2px 6px;color:#ccc">{count}</td>'
                f'</tr>'
            )
        return (
            '<table style="font-size:0.82em;border-collapse:collapse;width:100%;'
            'color:#eee;background:#1a1a1a;border-radius:6px">'
            '<thead><tr>'
            '<th style="padding:4px 6px;text-align:left;border-bottom:1px solid #444">Track</th>'
            '<th style="padding:4px 6px;text-align:left;border-bottom:1px solid #444">Frames</th>'
            '<th style="padding:4px 6px;text-align:left;border-bottom:1px solid #444">Boxes</th>'
            '</tr></thead>'
            '<tbody>' + "".join(rows_html) + "</tbody></table>"
        )


# ── Gradio app ────────────────────────────────────────────────────────────────


def build_app(rev: MOTReviewer) -> gr.Blocks:
    if not rev.splits:
        with gr.Blocks() as app:
            gr.Markdown(
                "## No MOT label files found\n\n"
                "Run `autolabel_mot_tracktor.py` first to generate labels.\n\n"
                "Expected layout: `dataset/MOT_labels_3fps/{split}/{scene}/gt/gt.txt`"
            )
        return app

    # ── helpers ────────────────────────────────────────────────────────────

    def _refresh():
        img = rev.load_image()
        if img is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No image", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            return blank, "No image loaded.", rev.stats_text, rev.track_list_html()
        rows = rev.rows_for_current()
        vis = render_frame(img, rows, rev.selected_track_id)
        mod = "  **(unsaved changes)**" if (rev.split, rev.scene) in rev.modified else ""
        fnum = rev.current_frame_num or "?"
        status = (
            f"**{rev.split} / {rev.scene}** — "
            f"Frame {rev.idx + 1}/{len(rev.frames)} (#{fnum}) — "
            f"{len(rows)} box(es)"
            f"{mod}"
        )
        return vis, status, rev.stats_text, rev.track_list_html()

    def _selection_text() -> str:
        tid = rev.selected_track_id
        if tid is None:
            return "_Click a box to select a track._"
        fmin, fmax = rev.track_frame_range(tid)
        count = sum(1 for r in rev.rows if r["id"] == tid)
        next_id = rev.next_free_id()
        return (
            f"**Selected: Track {tid}**  \n"
            f"Appears in frames {fmin}–{fmax} ({count} boxes)  \n"
            f"_Next free ID: {next_id}_"
        )

    # ── initial state ──────────────────────────────────────────────────────

    first_split = rev.splits[0]
    first_scenes = rev.scenes_for(first_split)

    with gr.Blocks(title="MOT Label Reviewer") as app:
        gr.Markdown("# MOT Label Reviewer & Relabeller")

        # ── top navigation bar ────────────────────────────────────────────
        with gr.Row():
            split_dd = gr.Dropdown(rev.splits, value=first_split, label="Split", scale=1)
            scene_dd = gr.Dropdown(
                first_scenes,
                value=first_scenes[0] if first_scenes else None,
                label="Scene",
                scale=3,
            )
            slider = gr.Slider(0, max(len(rev.frames) - 1, 0), step=1, value=0, label="Frame", scale=3)
            prev_btn = gr.Button("◀ Prev", size="sm", scale=0, elem_id="btn_prev")
            next_btn = gr.Button("Next ▶", size="sm", scale=0, elem_id="btn_next")
            save_btn = gr.Button("💾 Save", variant="primary", size="sm", scale=0, elem_id="btn_save")

        # ── main area ─────────────────────────────────────────────────────
        with gr.Row():
            # ---- annotated frame ----
            with gr.Column(scale=3):
                img_out = gr.Image(
                    label="Annotated Frame  (click a box to select)",
                    type="numpy",
                    interactive=False,
                    elem_id="img_out",
                )
                status_md = gr.Markdown("")

            # ---- control panel ----
            with gr.Column(scale=1, min_width=300):
                stats_md = gr.Markdown(rev.stats_text)

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
                    info="Enter the target ID (use an existing ID to merge, a new one to split)",
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
                    del_box_btn  = gr.Button("Delete box",   variant="stop", size="sm")
                    del_track_btn = gr.Button("Delete track", variant="stop", size="sm")

                gr.Markdown("---")
                gr.Markdown("### Tracks in scene")
                track_list = gr.HTML(rev.track_list_html())

                gr.Markdown(
                    "<small>**How to merge**: select the wrong track, set new ID to "
                    "the correct track, scope = All frames, Apply.<br>"
                    "**How to split**: navigate to the split frame, select track, "
                    "enter a fresh ID, scope = From this frame onwards, Apply.</small>"
                )

        # ── event handlers ────────────────────────────────────────────────

        _img_cols = [img_out, status_md, stats_md, track_list]

        def on_split(s):
            rev.split = s
            scenes = rev.scenes_for(s)
            rev.scene = scenes[0] if scenes else None
            rev.idx = 0
            rev.selected_track_id = None
            vis, status, stats, tlist = _refresh()
            return (
                gr.update(choices=scenes, value=rev.scene),
                gr.update(maximum=max(len(rev.frames) - 1, 0), value=0),
                vis, status, stats, tlist,
                _selection_text(),
            )

        def on_scene(s):
            rev.scene = s
            rev.idx = 0
            rev.selected_track_id = None
            vis, status, stats, tlist = _refresh()
            return (
                gr.update(maximum=max(len(rev.frames) - 1, 0), value=0),
                vis, status, stats, tlist,
                _selection_text(),
            )

        def on_slider(i):
            rev.idx = int(i)
            return _refresh()

        def on_prev():
            rev.idx = max(0, rev.idx - 1)
            vis, status, stats, tlist = _refresh()
            return gr.update(value=rev.idx), vis, status, stats, tlist

        def on_next():
            rev.idx = min(len(rev.frames) - 1, rev.idx + 1)
            vis, status, stats, tlist = _refresh()
            return gr.update(value=rev.idx), vis, status, stats, tlist

        def on_save():
            saved = rev.save()
            _, status, stats, tlist = _refresh()
            if saved:
                gr.Info(f"Saved: {', '.join(saved)}")
            else:
                gr.Info("Nothing to save.")
            return status, stats, tlist

        def on_click(evt: gr.SelectData):
            px, py = int(evt.index[0]), int(evt.index[1])
            rows = rev.rows_for_current()
            hit = box_at_point(rows, px, py)
            if hit:
                rev.selected_track_id = hit["id"]
            else:
                rev.selected_track_id = None
                gr.Warning("No bounding box at this location.")
            vis, status, stats, tlist = _refresh()
            return vis, status, stats, tlist, _selection_text()

        def on_apply(new_id, scope):
            if rev.selected_track_id is None:
                gr.Warning("No track selected — click a bounding box first.")
                vis, status, stats, tlist = _refresh()
                return vis, status, stats, tlist, _selection_text()
            if new_id is None:
                gr.Warning("Enter a new Track ID first.")
                vis, status, stats, tlist = _refresh()
                return vis, status, stats, tlist, _selection_text()

            old_id = rev.selected_track_id
            new_id = int(new_id)
            fnum = rev.current_frame_num or 1
            scope_map = {
                "This frame only":           "this_frame",
                "From this frame onwards":   "from_here",
                "All frames":                "all_frames",
            }
            changed = rev.reassign(old_id, new_id, scope_map[scope], fnum)
            if changed:
                rev.selected_track_id = new_id
                gr.Info(f"Reassigned {changed} box(es): Track {old_id} → {new_id} ({scope})")
            else:
                gr.Warning(f"No rows matched the scope for Track {old_id}.")
            vis, status, stats, tlist = _refresh()
            return vis, status, stats, tlist, _selection_text()

        def on_clear_sel():
            rev.selected_track_id = None
            vis, status, stats, tlist = _refresh()
            return vis, status, stats, tlist, _selection_text()

        def on_del_box():
            if rev.selected_track_id is None:
                gr.Warning("No track selected.")
                return _refresh()
            fnum = rev.current_frame_num
            if fnum is None:
                gr.Warning("No frame loaded.")
                return _refresh()
            changed = rev.delete_box(rev.selected_track_id, fnum)
            if changed:
                gr.Info(f"Deleted box for Track {rev.selected_track_id} at frame {fnum}.")
            else:
                gr.Warning(f"Track {rev.selected_track_id} has no box on this frame.")
            return _refresh()

        def on_del_track():
            if rev.selected_track_id is None:
                gr.Warning("No track selected.")
                return _refresh() + (_selection_text(),)
            tid = rev.selected_track_id
            changed = rev.delete_track(tid)
            rev.selected_track_id = None
            if changed:
                gr.Info(f"Deleted {changed} box(es) for Track {tid}.")
            else:
                gr.Warning(f"Track {tid} has no boxes to delete.")
            vis, status, stats, tlist = _refresh()
            return vis, status, stats, tlist, _selection_text()

        # ── wiring ────────────────────────────────────────────────────────

        split_dd.change(
            on_split, [split_dd],
            [scene_dd, slider] + _img_cols + [selected_md],
        )
        scene_dd.change(
            on_scene, [scene_dd],
            [slider] + _img_cols + [selected_md],
        )
        slider.release(on_slider, [slider], _img_cols)
        prev_btn.click(on_prev, [], [slider] + _img_cols)
        next_btn.click(on_next, [], [slider] + _img_cols)
        save_btn.click(on_save, [], [status_md, stats_md, track_list])

        img_out.select(on_click, [], _img_cols + [selected_md])
        apply_btn.click(on_apply, [new_id_input, scope_radio], _img_cols + [selected_md])
        clear_sel_btn.click(on_clear_sel, [], _img_cols + [selected_md])
        del_box_btn.click(on_del_box, [], _img_cols)
        del_track_btn.click(on_del_track, [], _img_cols + [selected_md])

        app.load(lambda: _refresh(), [], _img_cols)

    return app


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradio app for reviewing / relabelling MOT tracking labels"
    )
    parser.add_argument(
        "--mot-dir",
        type=Path,
        default=Path("dataset/MOT_labels_3fps"),
        help="Root directory with {split}/{scene}/gt/gt.txt structure.",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if not args.mot_dir.exists():
        print(f"ERROR: MOT labels directory not found: {args.mot_dir}")
        raise SystemExit(1)

    print(f"Loading MOT labels from: {args.mot_dir}")
    rev = MOTReviewer(args.mot_dir)
    total_scenes = sum(len(v) for v in rev.data.values())
    print(f"Loaded {total_scenes} scene(s) across {len(rev.splits)} split(s).")

    app = build_app(rev)
    app.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css="#img_out {cursor: crosshair;}",
        head=KEYBOARD_HEAD,
    )


if __name__ == "__main__":
    main()
