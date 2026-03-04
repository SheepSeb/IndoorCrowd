#!/usr/bin/env python3
"""
MOT benchmark: (YOLOv8n, RT-DETR-L*, SAM3) × (ByteTrack, BoT-SORT, OCSORT, native**)

  *  RT-DETR-L is the fine-tuned model from train/golden_mid/runs/detect/rtdetr-l/
  ** "native" tracker is only valid with sam3: uses SAM3's own built-in track IDs
     without running any external tracker on top.

Evaluates every valid detector+tracker combination on the golden_tracker benchmark
(human-annotated ground truth) and reports:
  MOTA, IDF1, MT%, ML%, ID Switches, FPS

Output
------
  eval/results/benchmark_mot.json   ← full results per combo
  eval/results/benchmark_mot.csv    ← summary table (import into report)

Usage
-----
  uv sync
  uv run python eval/benchmark_mot.py

  # Only specific detectors / trackers
  uv run python eval/benchmark_mot.py --detectors yolov8n --trackers bytetrack ocsort

  # Include SAM3 native tracking
  uv run python eval/benchmark_mot.py --detectors sam3 --trackers native bytetrack

  # Tune detection threshold
  uv run python eval/benchmark_mot.py --conf 0.35 --imgsz 640

Notes
-----
  SAM3 is a video-mode detector: detections for each clip are pre-computed as a
  single video pass (parts of ~55 frames each), then fed frame-by-frame to the
  external tracker. FPS for SAM3 combos includes both detection and tracking time.
  The "native" tracker option skips the external tracker and uses SAM3's own IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

# NumPy 2.0 removed np.asfarray; patch it back for boxmot/OCSORT
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

REPO_ROOT   = Path(__file__).resolve().parents[1]
GT_DIR      = REPO_ROOT / "dataset" / "golden_tracker" / "clips"
RESULTS_DIR = Path(__file__).parent / "results"

DETECTOR_CONFIGS: dict[str, dict] = {
    "yolov8n":  {"type": "ultralytics", "cls": "YOLO",
                 "weights": REPO_ROOT / "train/golden_mid/runs/detect/yolov8n/weights/best.pt"},
    "rtdetr_l": {"type": "ultralytics", "cls": "RTDETR",
                 "weights": REPO_ROOT / "train/golden_mid/runs/detect/rtdetr-l/weights/best.pt"},
    "sam3":     {"type": "sam3", "model": "sam3.pt"},
}

DET_CONF    = 0.30
IMG_SIZE    = 640
FRAME_RATE  = 3     # golden_tracker clips are at 3 fps
SAM3_PART   = 55    # frames per SAM3 video part
TEXT_PROMPT = "person"

# "native" is a pseudo-tracker: only valid with sam3 detector
ALL_TRACKERS = ["bytetrack", "botsort", "ocsort", "native"]


# ── Detectors ──────────────────────────────────────────────────────────────────


def load_detector(det_key: str):
    cfg = DETECTOR_CONFIGS[det_key]
    if cfg["type"] == "ultralytics":
        weights = cfg["weights"]
        if not weights.exists():
            raise FileNotFoundError(
                f"Detector weights not found: {weights}\n"
                "Run train/golden_mid/train_all.py first."
            )
        from ultralytics import RTDETR, YOLO
        cls = RTDETR if cfg["cls"] == "RTDETR" else YOLO
        return cls(str(weights))
    elif cfg["type"] == "sam3":
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
        return SAM3VideoSemanticPredictor(overrides=dict(
            conf=DET_CONF, task="segment", mode="predict",
            model=cfg["model"], imgsz=IMG_SIZE, half=True, verbose=False,
        ))
    raise ValueError(f"Unknown detector type: {cfg['type']}")


def detect(model, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame detection (YOLO / RT-DETR). Returns (xyxy N×4, conf N)."""
    results = model.predict(img_bgr, conf=DET_CONF, imgsz=IMG_SIZE, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32)
    return (
        boxes.xyxy.cpu().numpy().astype(np.float32),
        boxes.conf.cpu().numpy().astype(np.float32),
    )


def _write_video(frames: list[Path], out: Path, fps: int) -> None:
    first  = cv2.imread(str(frames[0]))
    h, w   = first.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        img = cv2.imread(str(f))
        if img is not None:
            writer.write(img)
    writer.release()


def detect_sam3_clip(
    frames: list[Path], predictor
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run SAM3VideoSemanticPredictor on a full clip (split into parts).
    Returns {fnum 1-based: (xyxy N×4, conf N, track_ids N)}.
    """
    per_frame: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    parts  = [frames[i: i + SAM3_PART] for i in range(0, len(frames), SAM3_PART)]
    _empty = (
        np.empty((0, 4), np.float32),
        np.empty(0, np.float32),
        np.empty(0, np.int32),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video = Path(tmpdir) / "_sam3.mp4"
        fnum = 1
        for part_frames in parts:
            _write_video(part_frames, tmp_video, fps=FRAME_RATE)
            predictor.inference_state = {}
            for result in predictor(source=str(tmp_video), text=[TEXT_PROMPT], stream=True):
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    per_frame[fnum] = _empty
                else:
                    tids = (
                        boxes.id.cpu().numpy().astype(np.int32)
                        if boxes.id is not None
                        else np.arange(len(boxes), dtype=np.int32)
                    )
                    per_frame[fnum] = (
                        boxes.xyxy.cpu().numpy().astype(np.float32),
                        boxes.conf.cpu().numpy().astype(np.float32),
                        tids,
                    )
                fnum += 1
    return per_frame


# ── Tracker wrappers ───────────────────────────────────────────────────────────


class _Dets:
    """Minimal ultralytics-compatible detection wrapper for ByteTrack / BoT-SORT."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self._xyxy = xyxy.astype(np.float32)
        self.conf  = torch.from_numpy(conf.astype(np.float32))
        self.cls   = torch.zeros(len(conf), dtype=torch.float32)
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
        w  = xyxy[:, 2] - xyxy[:, 0]
        h  = xyxy[:, 3] - xyxy[:, 1]
        self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx):
        i = idx.numpy() if isinstance(idx, torch.Tensor) else idx
        return _Dets(self._xyxy[i], self.conf.numpy()[i])


_EMPTY_DETS = _Dets(np.empty((0, 4), np.float32), np.empty(0, np.float32))


def _make_bytetrack(fps: int):
    from ultralytics.trackers.byte_tracker import BYTETracker
    args = types.SimpleNamespace(
        track_high_thresh=0.35, track_low_thresh=0.10,
        new_track_thresh=0.40, track_buffer=fps * 2,
        match_thresh=0.80, fuse_score=True,
    )
    return BYTETracker(args, frame_rate=fps)


def _make_botsort(fps: int):
    from ultralytics.trackers.bot_sort import BOTSORT
    args = types.SimpleNamespace(
        track_high_thresh=0.35, track_low_thresh=0.10,
        new_track_thresh=0.40, track_buffer=fps * 2,
        match_thresh=0.80, proximity_thresh=0.50,
        appearance_thresh=0.25, with_reid=False,
        gmc_method="sparseOptFlow", model="auto", fuse_score=True,
    )
    return BOTSORT(args, frame_rate=fps)


def _make_ocsort(fps: int):
    from boxmot import OcSort
    return OcSort(
        det_thresh=0.35,
        max_age=fps * 2,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
    )


TRACKER_MAKERS = {
    "bytetrack": _make_bytetrack,
    "botsort":   _make_botsort,
    "ocsort":    _make_ocsort,
    # "native" is not in TRACKER_MAKERS — handled as a special case
}


def _tracker_update(
    tracker_name: str,
    tracker,
    xyxy: np.ndarray,
    conf: np.ndarray,
    img_bgr: np.ndarray,
) -> dict[int, list[float]]:
    """One external tracker step. Returns {track_id: [x, y, w, h]} (TLWH)."""
    if tracker_name == "ocsort":
        if len(xyxy) > 0:
            dets_np = np.column_stack(
                [xyxy, conf, np.zeros(len(conf), dtype=np.float32)]
            ).astype(np.float32)
        else:
            dets_np = np.empty((0, 6), np.float32)
        rows = tracker.update(dets_np, img_bgr)
        result = {}
        for row in rows:
            row = np.asarray(row).flatten()
            x1, y1, x2, y2, tid = (
                float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
            )
            result[tid] = [x1, y1, x2 - x1, y2 - y1]
        return result
    else:
        # ByteTrack / BoT-SORT — returns (N, 8): x1,y1,x2,y2,tid,conf,cls,idx
        dets = _Dets(xyxy, conf) if len(xyxy) > 0 else _EMPTY_DETS
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        rows = tracker.update(dets, img=gray)
        result = {}
        for row in rows:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            result[int(row[4])] = [x1, y1, x2 - x1, y2 - y1]
        return result


# ── Ground-truth loading ───────────────────────────────────────────────────────


def load_gt(gt_path: Path) -> dict[int, dict[int, list[float]]]:
    """Parse MOT gt.txt → {frame: {tid: [x, y, w, h]}}."""
    gt: dict[int, dict[int, list[float]]] = defaultdict(dict)
    for line in gt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        frame, tid = int(parts[0]), int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        if w > 0 and h > 0:
            gt[frame][tid] = [x, y, w, h]
    return dict(gt)


# ── MOT metrics ────────────────────────────────────────────────────────────────


def _build_acc(gt: dict, hyp: dict) -> object:
    """Build one motmetrics accumulator for a single clip."""
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)
    for fnum in sorted(set(gt.keys()) | set(hyp.keys())):
        gt_frame  = gt.get(fnum,  {})
        hyp_frame = hyp.get(fnum, {})
        gt_ids    = list(gt_frame.keys())
        hyp_ids   = list(hyp_frame.keys())
        gt_boxes  = [gt_frame[tid]  for tid in gt_ids]
        hyp_boxes = [hyp_frame[tid] for tid in hyp_ids]

        if gt_ids and hyp_ids:
            dists = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=0.5)
        else:
            dists = np.empty((len(gt_ids), len(hyp_ids)))

        acc.update(gt_ids, hyp_ids, dists)
    return acc


def _summarize(accs: list, names: list[str]) -> dict:
    """Aggregate motmetrics across clips. Returns flat metrics dict."""
    import motmetrics as mm
    mh = mm.metrics.create()
    df = mh.compute_many(
        accs,
        metrics=["mota", "idf1", "mostly_tracked", "mostly_lost",
                 "num_switches", "num_unique_objects"],
        names=names,
        generate_overall=True,
    )
    ov     = df.loc["OVERALL"]
    n_gt   = int(ov["num_unique_objects"])
    mt_pct = float(ov["mostly_tracked"]) / n_gt if n_gt > 0 else 0.0
    ml_pct = float(ov["mostly_lost"])    / n_gt if n_gt > 0 else 0.0
    return {
        "mota":        round(float(ov["mota"]), 4),
        "idf1":        round(float(ov["idf1"]), 4),
        "mt_pct":      round(mt_pct,            4),
        "ml_pct":      round(ml_pct,            4),
        "id_switches": int(ov["num_switches"]),
        "gt_tracks":   n_gt,
    }


# ── Clip discovery ─────────────────────────────────────────────────────────────


def discover_clips() -> list[Path]:
    """Find all clips across every split under GT_DIR."""
    if not GT_DIR.exists():
        return []
    return sorted(
        clip
        for split_dir in sorted(GT_DIR.iterdir()) if split_dir.is_dir()
        for scene_dir in sorted(split_dir.iterdir()) if scene_dir.is_dir()
        for clip in sorted(scene_dir.iterdir())
        if clip.is_dir()
        and (clip / "gt" / "gt.txt").exists()
        and (clip / "frames").exists()
    )


# ── Per-clip tracking ──────────────────────────────────────────────────────────


def run_clip(
    clip_dir: Path,
    det_key: str,
    detector,
    tracker_name: str,
    fps: int = FRAME_RATE,
) -> tuple:
    """
    Run detector+tracker on every frame of one clip.
    Returns (motmetrics_accumulator, fps_achieved).
    """
    frames = sorted((clip_dir / "frames").glob("frame_*.jpg"))
    if not frames:
        return None, 0.0

    gt  = load_gt(clip_dir / "gt" / "gt.txt")
    hyp: dict[int, dict[int, list[float]]] = {}
    t_total = 0.0

    det_type = DETECTOR_CONFIGS[det_key]["type"]

    def _fnum(path: Path) -> int:
        """Extract absolute frame number from filename, e.g. frame_00166.jpg → 166."""
        return int("".join(filter(str.isdigit, path.stem)))

    if det_type == "sam3":
        # Pre-compute detections for the whole clip as a video pass.
        # detect_sam3_clip returns keys 1..N (relative), so remap to absolute.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        rel_dets = detect_sam3_clip(frames, detector)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_total += time.perf_counter() - t0

        # Remap relative keys (1-based) to absolute frame numbers
        abs_dets = {
            _fnum(frames[rel_idx - 1]): v
            for rel_idx, v in rel_dets.items()
        }

        if tracker_name == "native":
            # Use SAM3's own track IDs directly — no external tracker
            for fnum, (xyxy, _conf, tids) in abs_dets.items():
                hyp[fnum] = {}
                for i, tid in enumerate(tids):
                    x1, y1, x2, y2 = xyxy[i]
                    hyp[fnum][int(tid)] = [
                        float(x1), float(y1), float(x2 - x1), float(y2 - y1)
                    ]
        else:
            tracker = TRACKER_MAKERS[tracker_name](fps)
            _empty3 = (np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int32))
            for frame_path in frames:
                fnum    = _fnum(frame_path)
                img_bgr = cv2.imread(str(frame_path))
                if img_bgr is None:
                    hyp[fnum] = {}
                    continue
                xyxy, conf, _tids = abs_dets.get(fnum, _empty3)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                hyp[fnum] = _tracker_update(tracker_name, tracker, xyxy, conf, img_bgr)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_total += time.perf_counter() - t1
    else:
        if tracker_name == "native":
            raise ValueError(f"'native' tracker is only valid with sam3 detector, got '{det_key}'")
        tracker = TRACKER_MAKERS[tracker_name](fps)
        for frame_path in frames:
            fnum    = _fnum(frame_path)
            img_bgr = cv2.imread(str(frame_path))
            if img_bgr is None:
                hyp[fnum] = {}
                continue

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            xyxy, conf = detect(detector, img_bgr)
            hyp[fnum]  = _tracker_update(tracker_name, tracker, xyxy, conf, img_bgr)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_total += time.perf_counter() - t0

    fps_actual = len(frames) / t_total if t_total > 0 else 0.0
    acc        = _build_acc(gt, hyp)
    return acc, fps_actual


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    global DET_CONF, IMG_SIZE
    parser = argparse.ArgumentParser(
        description="MOT benchmark: YOLOv8n/RT-DETR-L/SAM3 × ByteTrack/BoT-SORT/OCSORT/native"
    )
    parser.add_argument("--detectors", nargs="+",
                        default=list(DETECTOR_CONFIGS.keys()),
                        choices=list(DETECTOR_CONFIGS.keys()))
    parser.add_argument("--trackers",  nargs="+",
                        default=ALL_TRACKERS,
                        choices=ALL_TRACKERS)
    parser.add_argument("--conf",  type=float, default=DET_CONF,
                        help="Detection confidence threshold (default: 0.30).")
    parser.add_argument("--imgsz", type=int,   default=IMG_SIZE,
                        help="Detector input size (default: 640).")
    args = parser.parse_args()

    DET_CONF = args.conf
    IMG_SIZE = args.imgsz

    print(f"Det   : {args.detectors}")
    print(f"Track : {args.trackers}")
    print(f"Conf  : {DET_CONF}   imgsz={IMG_SIZE}")
    print("Note  : 'native' tracker is only evaluated for sam3 detector")

    clips = discover_clips()
    if not clips:
        print(f"\nNo clips found under {GT_DIR}")
        return 1
    print(f"Clips : {len(clips)} (all splits combined)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}   # combo → {overall, by_scene}
    combo_rows:  list[dict] = []

    for det_key in args.detectors:
        print(f"\n{'='*60}\nLoading detector: {det_key}")
        try:
            detector = load_detector(det_key)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            continue

        det_type = DETECTOR_CONFIGS[det_key]["type"]

        for tracker_name in args.trackers:
            if tracker_name == "native" and det_type != "sam3":
                continue

            combo = f"{det_key}+{tracker_name}"
            print(f"\n── {combo}  ({len(clips)} clips)")

            accs:         list        = []
            clip_names:   list[str]   = []
            fps_per_clip: list[float] = []

            scene_accs:  dict[str, list]        = defaultdict(list)
            scene_names: dict[str, list[str]]   = defaultdict(list)
            scene_fps:   dict[str, list[float]] = defaultdict(list)

            for clip_dir in clips:
                label = f"{clip_dir.parent.name}/{clip_dir.name}"
                print(f"  {label} … ", end="", flush=True)

                try:
                    acc, fps_actual = run_clip(clip_dir, det_key, detector, tracker_name)
                except Exception as exc:
                    print(f"ERROR: {exc}")
                    continue

                if acc is None:
                    print("skipped (no frames)")
                    continue

                scene_group = clip_dir.parent.name.split("_recording_")[0]
                accs.append(acc)
                clip_names.append(label)
                fps_per_clip.append(fps_actual)
                scene_accs[scene_group].append(acc)
                scene_names[scene_group].append(label)
                scene_fps[scene_group].append(fps_actual)
                print(f"{fps_actual:.1f} fps")

            if not accs:
                print(f"  [WARN] no clips processed for {combo}")
                continue

            overall = _summarize(accs, clip_names)
            overall["fps"]      = round(float(np.mean(fps_per_clip)), 1)
            overall["detector"] = det_key
            overall["tracker"]  = tracker_name
            overall["scene"]    = "overall"
            overall["clips"]    = len(accs)

            by_scene: dict[str, dict] = {}
            for sg, sg_accs in scene_accs.items():
                sm = _summarize(sg_accs, scene_names[sg])
                sm["fps"] = round(float(np.mean(scene_fps[sg])), 1)
                by_scene[sg] = sm

            all_results[combo] = {"overall": overall, "by_scene": by_scene}
            combo_rows.append(overall)
            for sg, sm in by_scene.items():
                combo_rows.append({**sm, "detector": det_key, "tracker": tracker_name,
                                   "scene": sg, "clips": len(scene_accs[sg])})

            print(
                f"  → MOTA={overall['mota']:.4f}  IDF1={overall['idf1']:.4f}"
                f"  MT={overall['mt_pct']:.1%}  ML={overall['ml_pct']:.1%}"
                f"  IDs={overall['id_switches']}  FPS={overall['fps']:.1f}"
            )
            for sg, sm in sorted(by_scene.items()):
                print(
                    f"     {sg:<20} MOTA={sm['mota']:.4f}  IDF1={sm['idf1']:.4f}"
                    f"  MT={sm['mt_pct']:.1%}  ML={sm['ml_pct']:.1%}"
                    f"  IDs={sm['id_switches']}"
                )

    if not all_results:
        print("\nNo results — check detector weights and dataset paths.")
        return 1

    # ── Save report ────────────────────────────────────────────────────────────
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "dataset":   "golden_tracker (human-annotated, all splits)",
        "det_conf":  DET_CONF,
        "imgsz":     IMG_SIZE,
        "results":   all_results,
    }
    json_path = RESULTS_DIR / "benchmark_mot.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    csv_path = RESULTS_DIR / "benchmark_mot.csv"
    columns  = ["scene", "detector", "tracker",
                "mota", "idf1", "mt_pct", "ml_pct", "id_switches",
                "fps", "gt_tracks", "clips"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(combo_rows)

    # ── Console summary table ──────────────────────────────────────────────────
    W = 88
    print(f"\n{'─'*W}")
    print(f"{'Detector':<12} {'Tracker':<12} {'Scene':<22} {'MOTA':>8} {'IDF1':>8}"
          f" {'MT%':>7} {'ML%':>7} {'IDs':>6} {'FPS':>7}")
    print("─" * W)
    for r in combo_rows:
        is_overall = r.get("scene") == "overall"
        fps_str = f"{r['fps']:>7.1f}" if is_overall else " " * 7
        print(
            f"{r['detector'] if is_overall else '':<12}"
            f" {r['tracker'] if is_overall else '':<12}"
            f" {r.get('scene', 'overall'):<22}"
            f" {r['mota']:>8.4f} {r['idf1']:>8.4f}"
            f" {r['mt_pct']:>7.1%} {r['ml_pct']:>7.1%}"
            f" {r['id_switches']:>6d} {fps_str}"
        )
        if is_overall:
            print()
    print("─" * W)

    print(f"\nJSON → {json_path}")
    print(f"CSV  → {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
