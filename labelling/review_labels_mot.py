#!/usr/bin/env python3
"""Gradio tool for reviewing and removing BoT-SORT auto-labelled tracks.

Loads MOT labels from `dataset/MOT_labels_sam3_botsort` (or any MOT-format dir).

Also understands clip subsets exported to
`dataset/MOT_labels_sam3_botsort/clips/{split}/{scene}/{clip_name}/` with:

  - `gt/gt.txt`
  - `seqinfo.ini`
  - `frames/` (as referenced by `imDir` in `seqinfo.ini`)

Workflow
--------
  - Navigate frames with the slider or ← / → keys.
  - Click a bounding box on the frame to select that track
    (auto-jumps to the track's first frame so you see it in context).
  - Alternatively, pick a track from the dropdown.
  - Hit "Remove Track" to delete all boxes for the selected track.
  - Use "Bulk Remove" to drop all tracks shorter than N frames at once.
  - Click "Save" (or press Enter) to write changes back to gt.txt.
  - Ctrl+Z / Cmd+Z to undo the last mutation.
  - [ / ] to jump between frames of the selected track.
  - Mark scenes as reviewed to track annotation progress.

Usage
-----
  uv run python labelling/review_botsort_tracks.py

  uv run python labelling/review_botsort_tracks.py \\
      --mot-dir dataset/MOT_labels_sam3_botsort \\
      --port 7862
"""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MAX_UNDO = 30

PALETTE: list[tuple[int, int, int]] = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (128, 0, 0), (170, 255, 195), (0, 0, 128),
]

KEYBOARD_JS = """
<script>
document.addEventListener('keydown', function(e) {
    if (["input","textarea","select"].includes(e.target.tagName.toLowerCase())) return;
    if (e.key === "ArrowLeft")  { e.preventDefault(); document.getElementById("btn_prev").click(); }
    if (e.key === "ArrowRight") { e.preventDefault(); document.getElementById("btn_next").click(); }
    if (e.key === "Enter")      { e.preventDefault(); document.getElementById("btn_save").click(); }
    if (e.key === "[")          { e.preventDefault(); document.getElementById("btn_track_prev").click(); }
    if (e.key === "]")          { e.preventDefault(); document.getElementById("btn_track_next").click(); }
    if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault(); document.getElementById("btn_undo").click();
    }
    if (e.key === "Delete" || e.key === "Backspace") {
        var btn = document.getElementById("btn_remove");
        if (btn && !btn.disabled) { e.preventDefault(); btn.click(); }
    }
});
</script>
"""


def _color(track_id: int) -> tuple[int, int, int]:
    return PALETTE[int(track_id) % len(PALETTE)]


def _hex(track_id: int) -> str:
    r, g, b = _color(track_id)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Rendering ─────────────────────────────────────────────────────────────────


def render_frame(
    img_rgb: np.ndarray,
    rows: list[dict],
    selected_id: int | None,
    min_conf: float = 0.0,
    min_size: int = 0,
) -> np.ndarray:
    vis = img_rgb.copy()
    for row in sorted(rows, key=lambda r: r["id"] == selected_id):
        tid = row["id"]
        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["w"]), int(row["h"])
        is_sel = tid == selected_id
        is_dimmed = row["conf"] < min_conf or (w * h) < min_size

        if is_dimmed:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (130, 130, 130), 1)
            continue

        c = _color(tid)

        if is_sel:
            cv2.rectangle(vis, (x - 4, y - 4), (x + w + 4, y + h + 4), (255, 255, 0), 3)
            thickness = 3
        else:
            thickness = 2

        cv2.rectangle(vis, (x, y), (x + w, y + h), c, thickness)
        label = f"ID {tid}"
        fs = 0.55
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ly = max(y, th + bl + 4)
        cv2.rectangle(vis, (x, ly - th - bl - 4), (x + tw + 6, ly), c, -1)
        cv2.putText(vis, label, (x + 3, ly - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def box_at_point(rows: list[dict], px: int, py: int) -> dict | None:
    best, best_area = None, float("inf")
    for row in rows:
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        if x <= px <= x + w and y <= py <= y + h:
            area = w * h
            if area < best_area:
                best, best_area = row, area
    return best


# ── State ─────────────────────────────────────────────────────────────────────


class TrackCleaner:
    def __init__(self, mot_dir: Path):
        self.mot_dir = mot_dir
        self.data: dict[str, dict[str, dict]] = {}
        self.modified: set[tuple[str, str]] = set()
        self.removed_count = 0
        self.undo_stacks: dict[tuple[str, str], list[list[dict]]] = {}
        self.reviewed: set[tuple[str, str]] = set()
        self._load_all()
        self._load_reviewed()

        self.split: str | None = next(iter(self.data), None)
        self.scene: str | None = (
            next(iter(self.data.get(self.split or "", {})), None)
        )
        self.idx: int = 0
        self.selected_id: int | None = None
        # The specific row the user last clicked (frame, id, x, y match)
        self.selected_box: dict | None = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_all(self):
        """Populate self.data from a MOT-style directory tree.

        Supports two layouts:

        1. Standard scenes
           root/{split}/{scene}/gt/gt.txt

        2. Exported clips (from this tool)
           root/{split}/{scene}/{clip_name}/gt/gt.txt

        In the clips case, each clip becomes its own "scene" with key
        "{scene}/{clip_name}" in the UI dropdown.
        """
        for split_dir in sorted(self.mot_dir.iterdir()):
            if not split_dir.is_dir():
                continue

            scenes: dict[str, dict] = {}

            for first_level in sorted(split_dir.iterdir()):
                if not first_level.is_dir():
                    continue

                # Case 1: standard MOT layout: split/scene/gt/gt.txt
                direct_gt = first_level / "gt" / "gt.txt"
                if direct_gt.exists():
                    seqinfo_file = first_level / "seqinfo.ini"
                    rows = self._parse_gt(direct_gt)
                    seqinfo = (
                        self._parse_seqinfo(seqinfo_file)
                        if seqinfo_file.exists()
                        else {}
                    )
                    frames = self._collect_frames(seqinfo, first_level)
                    scenes[first_level.name] = {
                        "rows": rows,
                        "frames": frames,
                        "gt_file": direct_gt,
                    }
                    continue

                # Case 2: exported clips: split/scene/clip_name/gt/gt.txt
                for clip_dir in sorted(first_level.iterdir()):
                    if not clip_dir.is_dir():
                        continue
                    gt_file = clip_dir / "gt" / "gt.txt"
                    if not gt_file.exists():
                        continue
                    seqinfo_file = clip_dir / "seqinfo.ini"
                    rows = self._parse_gt(gt_file)
                    seqinfo = (
                        self._parse_seqinfo(seqinfo_file)
                        if seqinfo_file.exists()
                        else {}
                    )
                    frames = self._collect_frames(seqinfo, clip_dir)
                    scene_key = f"{first_level.name}/{clip_dir.name}"
                    scenes[scene_key] = {
                        "rows": rows,
                        "frames": frames,
                        "gt_file": gt_file,
                    }

            if scenes:
                self.data[split_dir.name] = scenes

    @staticmethod
    def _parse_gt(gt_file: Path) -> list[dict]:
        rows = []
        for line in gt_file.read_text().splitlines():
            p = line.strip().split(",")
            if len(p) < 7:
                continue
            rows.append({
                "frame": int(p[0]),
                "id":    int(p[1]),
                "x": float(p[2]), "y": float(p[3]),
                "w": float(p[4]), "h": float(p[5]),
                "conf": float(p[6]),
                "rest": ",".join(p[7:]) if len(p) > 7 else "-1,-1,-1",
            })
        return rows

    @staticmethod
    def _parse_seqinfo(f: Path) -> dict:
        cfg = configparser.ConfigParser()
        cfg.read(str(f))
        return dict(cfg["Sequence"]) if "Sequence" in cfg else {}

    @staticmethod
    def _collect_frames(seqinfo: dict, scene_dir: Path) -> list[Path]:
        d = Path(seqinfo["imdir"]) if "imdir" in seqinfo else scene_dir
        if not d.exists():
            return []
        return sorted(p for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

    # ── Reviewed status ────────────────────────────────────────────────────────

    def _reviewed_file(self) -> Path:
        return self.mot_dir / ".reviewed.json"

    def _load_reviewed(self):
        rf = self._reviewed_file()
        if rf.exists():
            try:
                self.reviewed = {tuple(p) for p in json.load(rf)}
            except Exception:
                pass

    def save_reviewed(self):
        with open(self._reviewed_file(), "w") as f:
            json.dump([list(p) for p in sorted(self.reviewed)], f)

    def toggle_reviewed(self) -> bool:
        key = (self.split, self.scene)
        if key in self.reviewed:
            self.reviewed.discard(key)
        else:
            self.reviewed.add(key)
        self.save_reviewed()
        return key in self.reviewed

    def is_reviewed(self) -> bool:
        return (self.split, self.scene) in self.reviewed

    def reviewed_progress(self) -> tuple[int, int]:
        total = sum(len(s) for s in self.data.values())
        return len(self.reviewed), total

    # ── Undo ──────────────────────────────────────────────────────────────────

    def _snapshot(self):
        sd = self._sd
        if sd and self.split and self.scene:
            key = (self.split, self.scene)
            stack = self.undo_stacks.setdefault(key, [])
            stack.append([r.copy() for r in sd["rows"]])
            if len(stack) > MAX_UNDO:
                stack.pop(0)

    def undo(self) -> bool:
        sd = self._sd
        if not sd or not self.split or not self.scene:
            return False
        stack = self.undo_stacks.get((self.split, self.scene))
        if not stack:
            return False
        sd["rows"] = stack.pop()
        self.modified.add((self.split, self.scene))
        return True

    def can_undo(self) -> bool:
        return bool(self.undo_stacks.get((self.split, self.scene)))

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def splits(self) -> list[str]:
        return list(self.data.keys())

    def scenes_for(self, split: str) -> list[str]:
        return list(self.data.get(split, {}).keys())

    @property
    def _sd(self) -> dict | None:
        if self.split and self.scene:
            return self.data.get(self.split, {}).get(self.scene)
        return None

    @property
    def rows(self) -> list[dict]:
        return self._sd["rows"] if self._sd else []

    @property
    def frames(self) -> list[Path]:
        return self._sd["frames"] if self._sd else []

    def _frame_num(self, fp: Path) -> int:
        digits = "".join(c for c in fp.stem if c.isdigit())
        return int(digits) if digits else self.idx + 1

    @property
    def current_frame_num(self) -> int | None:
        fps = self.frames
        if 0 <= self.idx < len(fps):
            return self._frame_num(fps[self.idx])
        return None

    def rows_for_current(self) -> list[dict]:
        fnum = self.current_frame_num
        return [r for r in self.rows if r["frame"] == fnum] if fnum else []

    def load_image(self) -> np.ndarray | None:
        fps = self.frames
        if not (0 <= self.idx < len(fps)):
            return None
        fp = fps[self.idx]
        return np.array(Image.open(fp).convert("RGB")) if fp.exists() else None

    def track_ids(self) -> list[int]:
        return sorted({r["id"] for r in self.rows})

    def track_info(self, tid: int) -> dict:
        rows = [r for r in self.rows if r["id"] == tid]
        frames = [r["frame"] for r in rows]
        confs = [r["conf"] for r in rows]
        return {
            "count": len(rows),
            "first": min(frames) if frames else 0,
            "last": max(frames) if frames else 0,
            "mean_conf": float(np.mean(confs)) if confs else 0.0,
        }

    def track_strip_crops(
        self,
        tid: int,
        n_crops: int = 12,
        pad: int = 12,
    ) -> list[tuple[np.ndarray, str]]:
        """Return evenly-sampled crops of track tid for the filmstrip view."""
        rows = sorted(
            [r for r in self.rows if r["id"] == tid],
            key=lambda r: r["frame"],
        )
        if not rows:
            return []

        # Build frame-number → file-path map once
        fnum_to_path = {self._frame_num(fp): fp for fp in self.frames}

        indices = np.linspace(0, len(rows) - 1, min(n_crops, len(rows)), dtype=int)
        crops: list[tuple[np.ndarray, str]] = []
        for i in indices:
            row = rows[int(i)]
            fp = fnum_to_path.get(row["frame"])
            if fp is None or not fp.exists():
                continue
            img = np.array(Image.open(fp).convert("RGB"))
            h, w = img.shape[:2]
            x1 = max(0, int(row["x"]) - pad)
            y1 = max(0, int(row["y"]) - pad)
            x2 = min(w, int(row["x"] + row["w"]) + pad)
            y2 = min(h, int(row["y"] + row["h"]) + pad)
            crop = img[y1:y2, x1:x2]
            crops.append((crop, f"#{row['frame']}  conf {row['conf']:.2f}"))
        return crops

    def first_frame_idx_for_track(self, tid: int) -> int:
        """Return the frame list index where track tid first appears."""
        rows = [r for r in self.rows if r["id"] == tid]
        if not rows:
            return self.idx
        first_fnum = min(r["frame"] for r in rows)
        for i, fp in enumerate(self.frames):
            if self._frame_num(fp) == first_fnum:
                return i
        return self.idx

    # ── Track-frame navigation ─────────────────────────────────────────────────

    def _fnum_to_idx(self) -> dict[int, int]:
        return {self._frame_num(fp): i for i, fp in enumerate(self.frames)}

    def frames_with_track(self, tid: int) -> list[int]:
        fnum_to_idx = self._fnum_to_idx()
        return sorted({
            fnum_to_idx[r["frame"]]
            for r in self.rows
            if r["id"] == tid and r["frame"] in fnum_to_idx
        })

    def next_frame_with_track(self, tid: int) -> int | None:
        for idx in self.frames_with_track(tid):
            if idx > self.idx:
                return idx
        return None

    def prev_frame_with_track(self, tid: int) -> int | None:
        for idx in reversed(self.frames_with_track(tid)):
            if idx < self.idx:
                return idx
        return None

    # ── Mutations ─────────────────────────────────────────────────────────────

    def remove_box(self, tid: int, frame: int) -> int:
        """Remove a single box (one frame) for the given track. Returns 1 if removed."""
        sd = self._sd
        if not sd:
            return 0
        self._snapshot()
        before = len(sd["rows"])
        sd["rows"] = [r for r in sd["rows"] if not (r["id"] == tid and r["frame"] == frame)]
        n = before - len(sd["rows"])
        if n:
            self.modified.add((self.split, self.scene))
        return n

    def remove_track(self, tid: int) -> int:
        sd = self._sd
        if not sd:
            return 0
        self._snapshot()
        before = len(sd["rows"])
        sd["rows"] = [r for r in sd["rows"] if r["id"] != tid]
        n = before - len(sd["rows"])
        if n:
            self.modified.add((self.split, self.scene))
            self.removed_count += 1
        return n

    def reassign(self, old_id: int, new_id: int, scope: str, from_frame: int) -> int:
        """Reassign track ID in-place.

        scope: "this_frame" | "from_here" | "all_frames"
        Returns number of rows changed.
        """
        sd = self._sd
        if not sd:
            return 0
        self._snapshot()
        changed = 0
        for row in sd["rows"]:
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
        return changed

    def reassign_single_box(self, box: dict, new_id: int) -> bool:
        """Reassign only the one specific box (matched by frame + id + position) to new_id.

        Returns True if a row was changed.
        """
        sd = self._sd
        if not sd:
            return False
        # Among all rows with the same (frame, id), pick the one closest
        # in position to the clicked box — handles the duplicate-ID-same-frame case.
        candidates = [
            r for r in sd["rows"]
            if r["frame"] == box["frame"] and r["id"] == box["id"]
        ]
        if not candidates:
            return False
        target = min(
            candidates,
            key=lambda r: abs(r["x"] - box["x"]) + abs(r["y"] - box["y"]),
        )
        self._snapshot()
        target["id"] = new_id
        self.modified.add((self.split, self.scene))
        return True

    def remove_filtered(self, min_conf: float, min_size: int) -> int:
        """Remove all boxes that fail the conf or size filter. Returns number of boxes removed."""
        sd = self._sd
        if not sd:
            return 0
        self._snapshot()
        before = len(sd["rows"])
        sd["rows"] = [
            r for r in sd["rows"]
            if r["conf"] >= min_conf and int(r["w"]) * int(r["h"]) >= min_size
        ]
        n = before - len(sd["rows"])
        if n:
            self.modified.add((self.split, self.scene))
        return n

    def bulk_remove_short(self, min_len: int) -> tuple[int, int]:
        """Remove all tracks with fewer than min_len frames. Returns (tracks, boxes) removed."""
        tids = self.track_ids()
        tracks_removed = boxes_removed = 0
        # Single snapshot before the loop so the whole bulk op is one undo step
        self._snapshot()
        sd = self._sd
        if not sd:
            return 0, 0
        short_tids = {
            tid for tid in tids if self.track_info(tid)["count"] < min_len
        }
        if short_tids:
            before = len(sd["rows"])
            sd["rows"] = [r for r in sd["rows"] if r["id"] not in short_tids]
            boxes_removed = before - len(sd["rows"])
            tracks_removed = len(short_tids)
            if boxes_removed:
                self.modified.add((self.split, self.scene))
                self.removed_count += tracks_removed
        return tracks_removed, boxes_removed

    def export_subset(
        self,
        start_idx: int,
        end_idx: int,
        output_dir: Path,
    ) -> tuple[str, int, int]:
        """Copy frames [start_idx, end_idx] and their MOT labels to output_dir.

        Output layout mirrors SAM3_video_tracks::

            output_dir/{split}/{scene}/clip_SSSSS_EEEEE/
                frames/        ← copied source frames (original filenames)
                gt/gt.txt      ← MOT rows filtered to this frame range
                seqinfo.ini    ← sequence metadata

        Returns (clip_dir, n_frames_copied, n_label_rows).
        """
        frames = self.frames
        start_idx = max(0, int(start_idx))
        end_idx   = min(len(frames) - 1, int(end_idx))
        if start_idx > end_idx:
            raise ValueError(f"Start index {start_idx} > end index {end_idx}")

        clip_frames = frames[start_idx : end_idx + 1]
        if not clip_frames:
            raise ValueError("No frames in selected range")

        snum  = self._frame_num(clip_frames[0])
        enum_ = self._frame_num(clip_frames[-1])
        clip_name = f"clip_{snum:05d}_{enum_:05d}"
        clip_dir  = output_dir / self.split / self.scene / clip_name

        # ── Copy frames ───────────────────────────────────────────────────────
        frames_dir = clip_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for fp in clip_frames:
            shutil.copy2(fp, frames_dir / fp.name)

        # ── Filter gt.txt ─────────────────────────────────────────────────────
        clip_fnums = {self._frame_num(fp) for fp in clip_frames}
        clip_rows  = [r for r in self.rows if r["frame"] in clip_fnums]
        gt_dir = clip_dir / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{r['frame']},{r['id']},{r['x']:.2f},{r['y']:.2f},"
            f"{r['w']:.2f},{r['h']:.2f},{r['conf']:.4f},{r['rest']}"
            for r in sorted(clip_rows, key=lambda r: (r["frame"], r["id"]))
        ]
        (gt_dir / "gt.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        # ── seqinfo.ini ───────────────────────────────────────────────────────
        first_bgr = cv2.imread(str(clip_frames[0]))
        img_h, img_w = first_bgr.shape[:2] if first_bgr is not None else (0, 0)
        cfg = configparser.ConfigParser()
        cfg["Sequence"] = {
            "name":      clip_name,
            "imDir":     str(frames_dir.resolve()),
            "frameRate": "3",
            "seqLength": str(len(clip_frames)),
            "imWidth":   str(img_w),
            "imHeight":  str(img_h),
            "imExt":     clip_frames[0].suffix,
        }
        with open(clip_dir / "seqinfo.ini", "w") as f:
            cfg.write(f)

        return str(clip_dir), len(clip_frames), len(clip_rows)

    def save(self) -> list[str]:
        saved = []
        for split, scene in list(self.modified):
            sd = self.data[split][scene]
            rows = sorted(sd["rows"], key=lambda r: (r["frame"], r["id"]))
            lines = [
                f"{r['frame']},{r['id']},{r['x']:.2f},{r['y']:.2f},"
                f"{r['w']:.2f},{r['h']:.2f},{r['conf']:.4f},{r['rest']}"
                for r in rows
            ]
            sd["gt_file"].write_text("\n".join(lines) + ("\n" if lines else ""))
            saved.append(f"{split}/{scene}")
        self.modified.clear()
        return saved


# ── UI helpers ────────────────────────────────────────────────────────────────


def _build_track_list_html(cleaner: TrackCleaner) -> str:
    tids = cleaner.track_ids()
    if not tids:
        return "<p style='color:#888;padding:8px'>No tracks remaining.</p>"
    rows_html = []
    for tid in tids:
        info = cleaner.track_info(tid)
        hex_c = _hex(tid)
        is_sel = tid == cleaner.selected_id
        bg = "background:#2a2a00;border:1px solid #ffcc00;" if is_sel else "border:1px solid transparent;"
        rows_html.append(
            f'<tr style="{bg}cursor:default">'
            f'<td style="padding:3px 8px">'
            f'  <span style="display:inline-block;width:11px;height:11px;background:{hex_c};'
            f'border-radius:2px;margin-right:4px;vertical-align:middle"></span>'
            f'  <b>{tid}</b></td>'
            f'<td style="padding:3px 8px;color:#ccc">{info["first"]}–{info["last"]}</td>'
            f'<td style="padding:3px 8px;color:#ccc">{info["count"]}</td>'
            f'<td style="padding:3px 8px;color:#aaa">{info["mean_conf"]:.2f}</td>'
            f'</tr>'
        )
    return (
        '<div style="max-height:420px;overflow-y:auto">'
        '<table style="font-size:0.80em;border-collapse:collapse;width:100%;'
        'color:#eee;background:#1a1a1a;border-radius:6px">'
        '<thead><tr style="position:sticky;top:0;background:#111">'
        '<th style="padding:5px 8px;text-align:left;border-bottom:1px solid #444">Track</th>'
        '<th style="padding:5px 8px;text-align:left;border-bottom:1px solid #444">Frames</th>'
        '<th style="padding:5px 8px;text-align:left;border-bottom:1px solid #444">Boxes</th>'
        '<th style="padding:5px 8px;text-align:left;border-bottom:1px solid #444">Conf</th>'
        '</tr></thead>'
        '<tbody>' + "".join(rows_html) + '</tbody></table></div>'
    )


def _selected_info_md(cleaner: TrackCleaner) -> str:
    tid = cleaner.selected_id
    if tid is None:
        return "_No track selected — click a box on the frame or pick from the dropdown._"
    info = cleaner.track_info(tid)
    hex_c = _hex(tid)
    return (
        f'<div style="background:#1e1e1e;border-left:4px solid {hex_c};'
        f'padding:10px 14px;border-radius:4px;margin:4px 0">'
        f'<b style="font-size:1.1em">Track {tid}</b><br>'
        f'<span style="color:#aaa">Frames: {info["first"]} – {info["last"]} '
        f'&nbsp;·&nbsp; Boxes: {info["count"]} '
        f'&nbsp;·&nbsp; Mean conf: {info["mean_conf"]:.3f}</span>'
        f'</div>'
    )


def _build_all_tracks_strip(cleaner: TrackCleaner) -> list[tuple[np.ndarray, str]]:
    """Build a one-crop-per-track overview strip."""
    crops: list[tuple[np.ndarray, str]] = []
    for tid in cleaner.track_ids():
        tcrops = cleaner.track_strip_crops(tid, n_crops=1)
        if not tcrops:
            continue
        img, caption = tcrops[0]
        crops.append((img, f"ID {tid} · {caption}"))
    return crops


# ── Gradio app ────────────────────────────────────────────────────────────────


def build_app(cleaner: TrackCleaner, clips_dir: Path) -> gr.Blocks:
    if not cleaner.splits:
        with gr.Blocks() as app:
            gr.Markdown(
                "## No MOT label files found\n\n"
                "Run `autolabel_sam3_botsort.py` first.\n\n"
                f"Expected: `{cleaner.mot_dir}/{{split}}/{{scene}}/gt/gt.txt`"
            )
        return app

    first_split = cleaner.splits[0]
    first_scenes = cleaner.scenes_for(first_split)

    # Mutable closure state for filters
    conf_threshold = [0.0]
    min_size_threshold = [0]

    def _refresh():
        img = cleaner.load_image()
        if img is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No image", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)
            return blank, "No image.", _build_track_list_html(cleaner), _selected_info_md(cleaner)

        rows = cleaner.rows_for_current()
        vis = render_frame(img, rows, cleaner.selected_id,
                           min_conf=conf_threshold[0], min_size=min_size_threshold[0])
        n_remaining = len(cleaner.track_ids())
        is_modified = (cleaner.split, cleaner.scene) in cleaner.modified
        mod_badge = ' · <span style="color:#f90">⬤ unsaved</span>' if is_modified else ""
        n_dimmed = sum(
            1 for r in rows
            if r["conf"] < conf_threshold[0] or int(r["w"]) * int(r["h"]) < min_size_threshold[0]
        )
        dim_note = f" · {n_dimmed} dimmed" if n_dimmed else ""
        status = (
            f"**{cleaner.split} / {cleaner.scene}** — "
            f"Frame {cleaner.idx + 1}/{len(cleaner.frames)} "
            f"(#{cleaner.current_frame_num}) — "
            f"{len(rows)} box(es) · {n_remaining} tracks remaining"
            f"{dim_note}"
            f"{mod_badge}"
        )
        return vis, status, _build_track_list_html(cleaner), _selected_info_md(cleaner)

    def _track_dropdown_choices():
        tids = cleaner.track_ids()
        return [str(t) for t in tids]

    def _build_strip():
        if cleaner.selected_id is None:
            return []
        return cleaner.track_strip_crops(cleaner.selected_id)

    def _reviewed_btn_update():
        if cleaner.is_reviewed():
            return gr.update(value="✓ Reviewed", variant="primary")
        return gr.update(value="Mark Reviewed", variant="secondary")

    def _progress_text():
        done, total = cleaner.reviewed_progress()
        return f"**{done}/{total}** reviewed"

    with gr.Blocks(title="BoT-SORT Track Cleaner") as app:
        gr.Markdown("# BoT-SORT Track Cleaner")

        # ── navigation row ────────────────────────────────────────────────────
        with gr.Row():
            split_dd = gr.Dropdown(
                cleaner.splits, value=first_split, label="Split", scale=1
            )
            scene_dd = gr.Dropdown(
                first_scenes,
                value=first_scenes[0] if first_scenes else None,
                label="Scene", scale=4,
            )
            reviewed_btn = gr.Button(
                "✓ Reviewed" if cleaner.is_reviewed() else "Mark Reviewed",
                variant="primary" if cleaner.is_reviewed() else "secondary",
                size="sm", scale=0,
            )
            progress_md = gr.Markdown(_progress_text())
            save_btn = gr.Button(
                "💾 Save", variant="primary", size="sm", scale=0,
                elem_id="btn_save",
            )

        with gr.Row():
            prev_btn = gr.Button("◀ Prev", size="sm", scale=0, elem_id="btn_prev")
            slider = gr.Slider(
                0, max(len(cleaner.frames) - 1, 0), step=1, value=0,
                label="Frame", scale=6,
            )
            next_btn = gr.Button("Next ▶", size="sm", scale=0, elem_id="btn_next")

        # ── track-frame navigation + goto row ─────────────────────────────────
        with gr.Row():
            track_prev_btn = gr.Button(
                "⏮ Track prev", size="sm", scale=1, elem_id="btn_track_prev"
            )
            track_next_btn = gr.Button(
                "Track next ⏭", size="sm", scale=1, elem_id="btn_track_next"
            )
            gr.Markdown("Go to frame #:")
            goto_num = gr.Number(
                value=None, precision=0, minimum=1, label="", scale=1,
                show_label=False,
            )
            goto_btn = gr.Button("Go", size="sm", scale=0)

        # ── main area ─────────────────────────────────────────────────────────
        with gr.Row():
            # left: frame viewer
            with gr.Column(scale=3):
                img_out = gr.Image(
                    label="Frame  (click a box to select track)",
                    type="numpy", interactive=False, elem_id="img_out",
                )
                status_md = gr.Markdown("")
                gr.Markdown("#### Track filmstrip")
                strip_gallery = gr.Gallery(
                    label="",
                    columns=12,
                    rows=1,
                    height=160,
                    object_fit="contain",
                    show_label=False,
                    allow_preview=True,
                )
                gr.Markdown("#### All tracks overview")
                all_strip_gallery = gr.Gallery(
                    label="",
                    columns=8,
                    rows=1,
                    height=140,
                    object_fit="contain",
                    show_label=False,
                    allow_preview=True,
                )

            # right: controls
            with gr.Column(scale=1, min_width=300):

                gr.HTML(
                    "<small style='color:#888'>Keys: ← → frames · "
                    "<b>[ ]</b> track frames · Del remove · Ctrl+Z undo · Enter save</small>"
                )

                # ── track selector ────────────────────────────────────────────
                gr.Markdown("### Select track")
                with gr.Row():
                    track_dd = gr.Dropdown(
                        choices=_track_dropdown_choices(),
                        value=None,
                        label="Track ID",
                        scale=3,
                        info="Or click a box on the frame",
                    )
                    jump_btn = gr.Button("Jump →", size="sm", scale=1)

                selected_html = gr.HTML(_selected_info_md(cleaner))

                # ── remove + undo ─────────────────────────────────────────────
                with gr.Row():
                    remove_box_btn = gr.Button(
                        "✂ Remove Box",
                        variant="secondary",
                        elem_id="btn_remove_box",
                    )
                    remove_btn = gr.Button(
                        "🗑 Remove Track",
                        variant="stop",
                        elem_id="btn_remove",
                    )
                    undo_btn = gr.Button(
                        "↩ Undo",
                        variant="secondary",
                        elem_id="btn_undo",
                    )

                gr.Markdown("---")

                # ── reassign ──────────────────────────────────────────────────
                gr.Markdown("### Reassign Track ID")
                new_id_input = gr.Number(
                    label="New Track ID",
                    precision=0,
                    minimum=1,
                    value=None,
                    info="Use an existing ID to merge, a new ID to split",
                )
                scope_radio = gr.Radio(
                    choices=["This frame only", "From this frame onwards", "All frames"],
                    value="All frames",
                    label="Scope",
                )
                reassign_btn = gr.Button("✏️ Apply Reassign", variant="primary")
                reassign_box_btn = gr.Button(
                    "✏️ Reassign This Box Only",
                    variant="secondary",
                )

                gr.Markdown("### Merge / combine tracks")
                gr.Markdown(
                    "_Select a source track (via click or dropdown), then enter the target"
                    " track ID above and click **Apply Reassign** with scope = 'All frames'"
                    " to merge/combine tracks._"
                )

                gr.Markdown("---")

                # ── confidence / size filter ──────────────────────────────────
                gr.Markdown("### Visibility filters")
                conf_slider = gr.Slider(
                    0.0, 1.0, step=0.01, value=0.0,
                    label="Dim boxes with conf <",
                    info="Boxes below threshold shown dimmed — data unchanged",
                )
                size_slider = gr.Slider(
                    0, 5000, step=50, value=0,
                    label="Dim boxes with area < (px²)",
                    info="w×h below threshold shown dimmed — data unchanged",
                )
                remove_filtered_btn = gr.Button(
                    "🗑 Delete all dimmed boxes", variant="stop", size="sm"
                )

                gr.Markdown("---")

                # ── bulk remove ───────────────────────────────────────────────
                gr.Markdown("### Bulk remove")
                with gr.Row():
                    min_len_input = gr.Number(
                        value=3, minimum=1, precision=0,
                        label="Remove tracks shorter than N frames",
                        scale=3,
                    )
                    bulk_btn = gr.Button("⚡ Apply", variant="secondary", scale=1)
                bulk_status = gr.Markdown("")

                gr.Markdown("---")

                # ── export clip ───────────────────────────────────────────────
                gr.Markdown("### Export clip")
                with gr.Row():
                    start_num = gr.Number(
                        label="Start frame (index)",
                        value=0, precision=0, minimum=0, scale=2,
                        info="Frame index (same as slider)",
                    )
                    end_num = gr.Number(
                        label="End frame (index)",
                        value=max(len(cleaner.frames) - 1, 0),
                        precision=0, minimum=0, scale=2,
                    )
                with gr.Row():
                    set_start_btn = gr.Button("◀ Set start = current", size="sm")
                    set_end_btn   = gr.Button("Set end = current ▶",   size="sm")
                export_btn    = gr.Button("📂 Export Subset", variant="primary")
                export_status = gr.Markdown("")

                gr.Markdown("---")

                # ── track list ────────────────────────────────────────────────
                gr.Markdown("### All tracks")
                track_list = gr.HTML(_build_track_list_html(cleaner))

        # ── output groups ─────────────────────────────────────────────────────
        # _img_outs: pure frame redraw (no strip rebuild)
        _img_outs = [img_out, status_md, track_list, selected_html]
        # _nav_outs: frame redraw + current-track strip (NOT all_strip)
        _nav_outs = [img_out, status_md, track_list, selected_html, strip_gallery]

        # ── event handlers ────────────────────────────────────────────────────

        def on_split(s):
            cleaner.split = s
            scenes = cleaner.scenes_for(s)
            cleaner.scene = scenes[0] if scenes else None
            cleaner.idx = 0
            cleaner.selected_id = None
            cleaner.selected_box = None
            vis, status, tlist, sel = _refresh()
            return (
                gr.update(choices=scenes, value=cleaner.scene),
                gr.update(maximum=max(len(cleaner.frames) - 1, 0), value=0),
                gr.update(choices=_track_dropdown_choices(), value=None),
                vis, status, tlist, sel, [],
                _build_all_tracks_strip(cleaner),
                _reviewed_btn_update(),
                _progress_text(),
            )

        def on_scene(s):
            cleaner.scene = s
            cleaner.idx = 0
            cleaner.selected_id = None
            cleaner.selected_box = None
            vis, status, tlist, sel = _refresh()
            return (
                gr.update(maximum=max(len(cleaner.frames) - 1, 0), value=0),
                gr.update(choices=_track_dropdown_choices(), value=None),
                vis, status, tlist, sel, [],
                _build_all_tracks_strip(cleaner),
                _reviewed_btn_update(),
                _progress_text(),
            )

        def on_slider(i):
            cleaner.idx = int(i)
            vis, status, tlist, sel = _refresh()
            return vis, status, tlist, sel, _build_strip()

        def on_prev():
            cleaner.idx = max(0, cleaner.idx - 1)
            vis, status, tlist, sel = _refresh()
            return gr.update(value=cleaner.idx), vis, status, tlist, sel, _build_strip()

        def on_next():
            cleaner.idx = min(len(cleaner.frames) - 1, cleaner.idx + 1)
            vis, status, tlist, sel = _refresh()
            return gr.update(value=cleaner.idx), vis, status, tlist, sel, _build_strip()

        def on_click(evt: gr.SelectData):
            px, py = int(evt.index[0]), int(evt.index[1])
            hit = box_at_point(cleaner.rows_for_current(), px, py)
            if hit:
                cleaner.selected_id = hit["id"]
                cleaner.selected_box = hit
            else:
                cleaner.selected_id = None
                cleaner.selected_box = None
            vis, status, tlist, sel = _refresh()
            return (
                gr.update(value=cleaner.idx),
                gr.update(value=str(cleaner.selected_id) if cleaner.selected_id is not None else None),
                vis, status, tlist, sel, _build_strip(),
            )

        def on_track_dd(tid_str):
            if tid_str is None:
                cleaner.selected_id = None
            else:
                cleaner.selected_id = int(tid_str)
            cleaner.selected_box = None  # dropdown selection has no specific box
            vis, status, tlist, sel = _refresh()
            return gr.update(value=cleaner.idx), vis, status, tlist, sel, _build_strip()

        def on_jump():
            if cleaner.selected_id is not None:
                cleaner.idx = cleaner.first_frame_idx_for_track(cleaner.selected_id)
            vis, status, tlist, sel = _refresh()
            return gr.update(value=cleaner.idx), vis, status, tlist, sel, _build_strip()

        def on_track_prev():
            if cleaner.selected_id is not None:
                idx = cleaner.prev_frame_with_track(cleaner.selected_id)
                if idx is not None:
                    cleaner.idx = idx
                else:
                    gr.Info("Already at first frame of this track.")
            vis, status, tlist, sel = _refresh()
            return gr.update(value=cleaner.idx), vis, status, tlist, sel, _build_strip()

        def on_track_next():
            if cleaner.selected_id is not None:
                idx = cleaner.next_frame_with_track(cleaner.selected_id)
                if idx is not None:
                    cleaner.idx = idx
                else:
                    gr.Info("Already at last frame of this track.")
            vis, status, tlist, sel = _refresh()
            return gr.update(value=cleaner.idx), vis, status, tlist, sel, _build_strip()

        def on_goto(frame_num):
            if frame_num is not None:
                fnum_to_idx = {cleaner._frame_num(fp): i for i, fp in enumerate(cleaner.frames)}
                idx = fnum_to_idx.get(int(frame_num))
                if idx is not None:
                    cleaner.idx = idx
                else:
                    gr.Warning(f"Frame #{int(frame_num)} not found in this scene.")
            vis, status, tlist, sel = _refresh()
            return gr.update(value=cleaner.idx), vis, status, tlist, sel, _build_strip()

        def on_conf_slider(val):
            conf_threshold[0] = val
            vis, status, tlist, sel = _refresh()
            return vis, status, tlist, sel

        def on_size_slider(val):
            min_size_threshold[0] = int(val)
            vis, status, tlist, sel = _refresh()
            return vis, status, tlist, sel

        def on_remove_filtered():
            mc, ms = conf_threshold[0], min_size_threshold[0]
            if mc == 0.0 and ms == 0:
                gr.Warning("Both filters are at zero — nothing to delete.")
                vis, status, tlist, sel = _refresh()
                return (
                    gr.update(), gr.update(),
                    vis, status, tlist, sel, _build_strip(),
                    _build_all_tracks_strip(cleaner),
                )
            n = cleaner.remove_filtered(mc, ms)
            if n:
                if cleaner.selected_id not in set(cleaner.track_ids()):
                    cleaner.selected_id = None
                gr.Info(f"Deleted {n} dimmed box(es).")
            else:
                gr.Warning("No boxes matched the current filter thresholds.")
            vis, status, tlist, sel = _refresh()
            return (
                gr.update(choices=_track_dropdown_choices(),
                          value=str(cleaner.selected_id) if cleaner.selected_id else None),
                gr.update(),
                vis, status, tlist, sel, _build_strip(),
                _build_all_tracks_strip(cleaner),
            )

        def on_remove_box():
            tid = cleaner.selected_id
            fnum = cleaner.current_frame_num
            if tid is None:
                gr.Warning("No track selected — click a bounding box first.")
                vis, status, tlist, sel = _refresh()
                return (
                    gr.update(),
                    gr.update(),
                    vis, status, tlist, sel, _build_strip(),
                    _build_all_tracks_strip(cleaner),
                )
            if fnum is None:
                gr.Warning("No current frame.")
                vis, status, tlist, sel = _refresh()
                return (
                    gr.update(),
                    gr.update(),
                    vis, status, tlist, sel, _build_strip(),
                    _build_all_tracks_strip(cleaner),
                )
            n = cleaner.remove_box(tid, fnum)
            if n:
                gr.Info(f"Removed box for Track {tid} on frame #{fnum}.")
            else:
                gr.Warning(f"No box for Track {tid} on frame #{fnum}.")
            vis, status, tlist, sel = _refresh()
            return (
                gr.update(choices=_track_dropdown_choices()),
                gr.update(),
                vis, status, tlist, sel, _build_strip(),
                _build_all_tracks_strip(cleaner),
            )

        def on_remove():
            tid = cleaner.selected_id
            if tid is None:
                gr.Warning("No track selected — click a bounding box first.")
                vis, status, tlist, sel = _refresh()
                return (
                    gr.update(),
                    gr.update(),
                    vis, status, tlist, sel, [],
                    _build_all_tracks_strip(cleaner),
                )
            n = cleaner.remove_track(tid)
            cleaner.selected_id = None
            if n:
                gr.Info(f"Removed {n} box(es) for Track {tid}.")
            else:
                gr.Warning(f"Track {tid} had no boxes to remove.")
            vis, status, tlist, sel = _refresh()
            return (
                gr.update(choices=_track_dropdown_choices(), value=None),
                gr.update(),   # slider unchanged
                vis, status, tlist, sel, [],
                _build_all_tracks_strip(cleaner),
            )

        def on_undo():
            ok = cleaner.undo()
            if not ok:
                gr.Warning("Nothing to undo.")
            else:
                if cleaner.selected_id not in set(cleaner.track_ids()):
                    cleaner.selected_id = None
                gr.Info("Undone.")
            vis, status, tlist, sel = _refresh()
            return (
                gr.update(choices=_track_dropdown_choices(),
                          value=str(cleaner.selected_id) if cleaner.selected_id else None),
                gr.update(),   # slider unchanged
                vis, status, tlist, sel, _build_strip(),
                _build_all_tracks_strip(cleaner),
            )

        def on_reassign(new_id, scope):
            tid = cleaner.selected_id
            if tid is None:
                gr.Warning("No track selected — click a bounding box first.")
                vis, status, tlist, sel = _refresh()
                return (
                    gr.update(),
                    gr.update(),
                    vis, status, tlist, sel, [],
                    _build_all_tracks_strip(cleaner),
                )
            if new_id is None:
                gr.Warning("Enter a New Track ID first.")
                vis, status, tlist, sel = _refresh()
                return (
                    gr.update(),
                    gr.update(),
                    vis, status, tlist, sel, [],
                    _build_all_tracks_strip(cleaner),
                )
            new_id = int(new_id)
            fnum = cleaner.current_frame_num or 1
            scope_map = {
                "This frame only":           "this_frame",
                "From this frame onwards":   "from_here",
                "All frames":                "all_frames",
            }
            changed = cleaner.reassign(tid, new_id, scope_map[scope], fnum)
            if changed:
                cleaner.selected_id = new_id
                gr.Info(f"Reassigned {changed} box(es): Track {tid} → {new_id} ({scope})")
            else:
                gr.Warning(f"No rows matched scope for Track {tid}.")
            vis, status, tlist, sel = _refresh()
            return (
                gr.update(choices=_track_dropdown_choices(), value=str(new_id)),
                gr.update(),  # slider unchanged
                vis, status, tlist, sel, _build_strip(),
                _build_all_tracks_strip(cleaner),
            )

        def on_reassign_box(new_id):
            box = cleaner.selected_box
            if box is None:
                gr.Warning("No box selected — click a specific bounding box on the frame first.")
                vis, status, tlist, sel = _refresh()
                return (
                    gr.update(),
                    gr.update(),
                    vis, status, tlist, sel, _build_strip(),
                    _build_all_tracks_strip(cleaner),
                )
            if new_id is None:
                gr.Warning("Enter a New Track ID first.")
                vis, status, tlist, sel = _refresh()
                return (
                    gr.update(),
                    gr.update(),
                    vis, status, tlist, sel, _build_strip(),
                    _build_all_tracks_strip(cleaner),
                )
            new_id = int(new_id)
            old_id = box["id"]
            ok = cleaner.reassign_single_box(box, new_id)
            if ok:
                cleaner.selected_id = new_id
                cleaner.selected_box = None  # box reference is now stale (id changed)
                gr.Info(f"Reassigned box on frame #{box['frame']}: Track {old_id} → {new_id}")
            else:
                gr.Warning(f"Could not find the selected box in current data.")
            vis, status, tlist, sel = _refresh()
            return (
                gr.update(choices=_track_dropdown_choices(), value=str(new_id)),
                gr.update(),
                vis, status, tlist, sel, _build_strip(),
                _build_all_tracks_strip(cleaner),
            )

        def on_bulk(min_len):
            n_tracks, n_boxes = cleaner.bulk_remove_short(int(min_len))
            cleaner.selected_id = None
            msg = (
                f"Removed **{n_tracks} track(s)** ({n_boxes} boxes) "
                f"shorter than {int(min_len)} frames."
                if n_tracks else
                f"No tracks shorter than {int(min_len)} frames found."
            )
            vis, status, tlist, sel = _refresh()
            return (
                gr.update(choices=_track_dropdown_choices(), value=None),
                msg,
                vis, status, tlist, sel, [],
                _build_all_tracks_strip(cleaner),
            )

        def on_reviewed():
            cleaner.toggle_reviewed()
            return _reviewed_btn_update(), _progress_text()

        def on_save():
            saved = cleaner.save()
            if saved:
                gr.Info(f"Saved: {', '.join(saved)}")
            else:
                gr.Info("Nothing to save.")
            _, status, tlist, sel = _refresh()
            return status, tlist, sel

        def on_set_start():
            return gr.update(value=cleaner.idx)

        def on_set_end():
            return gr.update(value=cleaner.idx)

        def on_export(start_idx, end_idx):
            if not cleaner.frames:
                return "No frames loaded."
            try:
                clip_dir, n_frames, n_rows = cleaner.export_subset(
                    int(start_idx), int(end_idx), clips_dir
                )
                return (
                    f"✅ Exported **{n_frames} frames** · **{n_rows} label rows**  \n"
                    f"`{clip_dir}`"
                )
            except Exception as exc:
                gr.Warning(str(exc))
                return f"Error: {exc}"

        # ── wiring ────────────────────────────────────────────────────────────

        split_dd.change(
            on_split, [split_dd],
            [scene_dd, slider, track_dd] + _nav_outs + [all_strip_gallery, reviewed_btn, progress_md],
        )
        scene_dd.change(
            on_scene, [scene_dd],
            [slider, track_dd] + _nav_outs + [all_strip_gallery, reviewed_btn, progress_md],
        )

        # Nav: update slider + _nav_outs (strip_gallery included), NOT all_strip
        slider.release(on_slider, [slider], _nav_outs)
        prev_btn.click(on_prev, [], [slider] + _nav_outs)
        next_btn.click(on_next, [], [slider] + _nav_outs)

        track_prev_btn.click(on_track_prev, [], [slider] + _nav_outs)
        track_next_btn.click(on_track_next, [], [slider] + _nav_outs)
        goto_btn.click(on_goto, [goto_num], [slider] + _nav_outs)

        # Click / track_dd: also skip all_strip
        img_out.select(on_click, [], [slider, track_dd] + _nav_outs)
        track_dd.change(on_track_dd, [track_dd], [slider] + _nav_outs)
        jump_btn.click(on_jump, [], [slider] + _nav_outs)

        # Filter sliders: only redraws image (not strips)
        conf_slider.release(on_conf_slider, [conf_slider], _img_outs)
        size_slider.release(on_size_slider, [size_slider], _img_outs)
        remove_filtered_btn.click(
            on_remove_filtered, [],
            [track_dd, slider] + _nav_outs + [all_strip_gallery],
        )

        # Mutations: full update including all_strip
        remove_box_btn.click(
            on_remove_box, [],
            [track_dd, slider] + _nav_outs + [all_strip_gallery],
        )
        remove_btn.click(
            on_remove, [],
            [track_dd, slider] + _nav_outs + [all_strip_gallery],
        )
        undo_btn.click(
            on_undo, [],
            [track_dd, slider] + _nav_outs + [all_strip_gallery],
        )
        reassign_btn.click(
            on_reassign, [new_id_input, scope_radio],
            [track_dd, slider] + _nav_outs + [all_strip_gallery],
        )
        reassign_box_btn.click(
            on_reassign_box, [new_id_input],
            [track_dd, slider] + _nav_outs + [all_strip_gallery],
        )
        bulk_btn.click(
            on_bulk, [min_len_input],
            [track_dd, bulk_status] + _nav_outs + [all_strip_gallery],
        )

        reviewed_btn.click(on_reviewed, [], [reviewed_btn, progress_md])
        save_btn.click(on_save, [], [status_md, track_list, selected_html])

        set_start_btn.click(on_set_start, [], [start_num])
        set_end_btn.click(on_set_end,   [], [end_num])
        export_btn.click(on_export, [start_num, end_num], [export_status])

        app.load(
            lambda: (*_refresh(), _build_strip(), _build_all_tracks_strip(cleaner)),
            [],
            _img_outs + [strip_gallery, all_strip_gallery],
        )

    return app


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradio tool for reviewing and removing BoT-SORT tracking labels"
    )
    parser.add_argument(
        "--mot-dir",
        type=Path,
        default=Path("dataset/MOT_labels_sam3_botsort"),
    )
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=None,
        help="Directory for exported frame subsets (default: <mot-dir>/clips/).",
    )
    args = parser.parse_args()

    if not args.mot_dir.exists():
        print(f"ERROR: directory not found: {args.mot_dir}")
        raise SystemExit(1)

    clips_dir = args.clips_dir or (args.mot_dir / "clips")

    print(f"Loading tracks from: {args.mot_dir}")
    cleaner = TrackCleaner(args.mot_dir)
    total = sum(len(v) for v in cleaner.data.values())
    print(f"Loaded {total} scene(s) across {len(cleaner.splits)} split(s).")

    app = build_app(cleaner, clips_dir)
    app.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css="#img_out { cursor: crosshair; }",
        head=KEYBOARD_JS,
    )


if __name__ == "__main__":
    main()
