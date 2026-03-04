#!/usr/bin/env python3
"""Multi-object tracking pipeline — Tracktor++ v2 style.

Implements the three components from
"Tracking without bells and whistles" (Bergmann et al., ICCV 2019):

  Tracktor   — detector-driven track propagation + IoU association
  +v1        — Camera Motion Compensation (CMC) via sparse optical flow
  +v2        — Re-identification (ReID) for inactive track recovery

Detector : SAM3 (default) — text-prompted instance segmentation
           YOLOv8 (fallback) — fast bounding-box detector
CMC      : Sparse Lucas-Kanade optical flow → affine warp of track boxes
ReID     : MobileNetV3-small crop features + cosine similarity

Output layout (MOT Challenge format)
-------------------------------------
  dataset/MOT_labels_3fps/
    {split}/
      {scene}/
        gt/
          gt.txt      ← frame,id,x,y,w,h,conf,-1,-1,-1
        seqinfo.ini   ← sequence metadata

Usage
-----
  # SAM3 detector (default, recommended)
  uv run python labelling/autolabel_mot_tracktor.py

  # YOLO detector (faster)
  uv run python labelling/autolabel_mot_tracktor.py \\
      --detector yolo --yolo-model yolov8n.pt --det-conf 0.35

  # List available golden_mid training runs
  uv run python labelling/autolabel_mot_tracktor.py --list-runs

  # Custom golden_mid detector (trained on golden labels)
  uv run python labelling/autolabel_mot_tracktor.py \\
      --detector golden --golden-run yolo11n2 --golden-checkpoint best

  # Full options
  uv run python labelling/autolabel_mot_tracktor.py \\
      --raw-frames-dir dataset/raw_frames_3fps \\
      --output-dir     dataset/MOT_labels_3fps \\
      --detector       sam3 \\
      --sam3-model-id  facebook/sam3 \\
      --det-conf       0.35 \\
      --mask-threshold 0.5 \\
      --iou-threshold  0.5 \\
      --max-lost-age   30 \\
      --n-init         3 \\
      --reid-threshold 0.25

"""

from __future__ import annotations

import argparse
import configparser
import sys
from dataclasses import dataclass, field
from pathlib import Path

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PERSON_CLASS_ID = 0  # YOLO COCO person class index

# ── Appearance model ──────────────────────────────────────────────────────────

_appearance_model: nn.Module | None = None
_transform: "torchvision.transforms.Compose | None" = None  # type: ignore[name-defined]


def _build_appearance_model() -> tuple[nn.Module, object]:
    """Lazy-load MobileNetV3-small as a feature extractor."""
    global _appearance_model, _transform
    if _appearance_model is not None:
        return _appearance_model, _transform

    import torchvision.models as models
    import torchvision.transforms as T

    backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Drop the final classifier; pool5 output is 576-dim
    backbone.classifier = nn.Identity()
    backbone = backbone.to(DEVICE).eval()

    transform = T.Compose([
        T.Resize((128, 64)),   # standard ReID crop size
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    _appearance_model = backbone
    _transform = transform
    return backbone, transform


@torch.no_grad()
def extract_features(crops: list[np.ndarray]) -> np.ndarray:
    """
    Extract L2-normalised appearance features for a batch of BGR crops.

    Returns
    -------
    np.ndarray of shape (N, 576)
    """
    if not crops:
        return np.empty((0, 576), dtype=np.float32)
    model, transform = _build_appearance_model()
    tensors = []
    for crop in crops:
        if crop.size == 0:
            tensors.append(torch.zeros(3, 128, 64))
            continue
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensors.append(transform(pil))
    batch = torch.stack(tensors).to(DEVICE)
    feats = model(batch).cpu().numpy().astype(np.float32)  # (N, 576)
    norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-6)
    return feats / norms


# ── Geometry helpers ──────────────────────────────────────────────────────────


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes in [x1,y1,x2,y2] format.

    Returns
    -------
    np.ndarray of shape (len(boxes_a), len(boxes_b))
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = [boxes_a[:, i] for i in range(4)]
    bx1, by1, bx2, by2 = [boxes_b[:, i] for i in range(4)]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def warp_boxes(boxes: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Apply a 2×3 affine matrix M to boxes in [x1,y1,x2,y2] format.

    Each box is represented by its four corners; the result is the
    axis-aligned bounding box of the warped corners.
    """
    if len(boxes) == 0 or M is None:
        return boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # four corners: TL, TR, BL, BR
    corners = np.stack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y1]),
        np.column_stack([x1, y2]),
        np.column_stack([x2, y2]),
    ], axis=1)  # (N, 4, 2)
    n = len(boxes)
    corners_flat = corners.reshape(-1, 2)  # (4N, 2)
    ones = np.ones((4 * n, 1), dtype=np.float32)
    homogeneous = np.hstack([corners_flat, ones])  # (4N, 3)
    warped = (M @ homogeneous.T).T  # (4N, 2)
    warped = warped.reshape(n, 4, 2)
    wx1 = warped[:, :, 0].min(axis=1)
    wy1 = warped[:, :, 1].min(axis=1)
    wx2 = warped[:, :, 0].max(axis=1)
    wy2 = warped[:, :, 1].max(axis=1)
    return np.column_stack([wx1, wy1, wx2, wy2]).astype(np.float32)


def hungarian_match(
    cost: np.ndarray, threshold: float
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Greedy Hungarian assignment on a cost matrix (higher = better).

    Returns
    -------
    matched_rows, matched_cols, unmatched_rows, unmatched_cols
    """
    if cost.size == 0:
        return [], [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    row_ind, col_ind = linear_sum_assignment(-cost)
    matched_rows, matched_cols = [], []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= threshold:
            matched_rows.append(r)
            matched_cols.append(c)
    all_rows = set(range(cost.shape[0]))
    all_cols = set(range(cost.shape[1]))
    unmatched_rows = sorted(all_rows - set(matched_rows))
    unmatched_cols = sorted(all_cols - set(matched_cols))
    return matched_rows, matched_cols, unmatched_rows, unmatched_cols


# ── Track ─────────────────────────────────────────────────────────────────────


@dataclass
class Track:
    track_id: int
    box: np.ndarray          # [x1, y1, x2, y2]
    score: float
    frame_idx: int           # last frame where this track was updated
    age: int = 1             # total frames since birth (including lost)
    hit_streak: int = 1      # consecutive matched frames
    lost_age: int = 0        # consecutive frames without a match
    state: str = "tentative" # tentative | confirmed | lost
    appearance: np.ndarray | None = None
    history: list = field(default_factory=list)  # list of (frame_idx, box_xywh, score)

    def record(self):
        x1, y1, x2, y2 = self.box
        self.history.append((self.frame_idx, [x1, y1, x2 - x1, y2 - y1], self.score))

    def update(self, box: np.ndarray, score: float, frame_idx: int,
               appearance: np.ndarray | None = None):
        self.box = box
        self.score = score
        self.frame_idx = frame_idx
        self.age += 1
        self.hit_streak += 1
        self.lost_age = 0
        if appearance is not None:
            self.appearance = appearance
        self.record()

    def mark_lost(self):
        self.lost_age += 1
        self.age += 1
        self.hit_streak = 0
        self.state = "lost"

    def mark_confirmed(self, n_init: int):
        if self.state == "tentative" and self.hit_streak >= n_init:
            self.state = "confirmed"

    @property
    def box_xywh(self) -> list[float]:
        x1, y1, x2, y2 = self.box
        return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


# ── Camera Motion Compensation ────────────────────────────────────────────────


def estimate_cmc(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray | None:
    """
    Estimate a 2×3 affine warp from prev_gray to curr_gray using sparse
    Lucas-Kanade optical flow on Shi-Tomasi corner features.

    Returns None if estimation fails (too few correspondences).
    """
    corners = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if corners is None or len(corners) < 10:
        return None

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, corners, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_pts is None or status is None:
        return None

    good_prev = corners[status.ravel() == 1]
    good_next = next_pts[status.ravel() == 1]
    if len(good_prev) < 6:
        return None

    M, inliers = cv2.estimateAffinePartial2D(
        good_prev, good_next,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    return M  # 2×3 or None


# ── Tracker ───────────────────────────────────────────────────────────────────


class TracktorPP:
    """
    Tracktor++ v2 tracker.

    Per-frame pipeline
    ------------------
    1. Detect persons (SAM3 or YOLO).
    2. Apply CMC to warp active track boxes (+v1).
    3. Primary association: IoU-based Hungarian matching (Tracktor core).
    4. Secondary association: ReID cosine similarity for lost tracks (+v2).
    5. Update track states; spawn new tracks for unmatched detections.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        max_lost_age: int = 30,
        n_init: int = 3,
        reid_threshold: float = 0.25,
        reid_max_lost_age: int = 60,
        use_cmc: bool = True,
        use_reid: bool = True,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_age = max_lost_age
        self.n_init = n_init
        self.reid_threshold = reid_threshold
        self.reid_max_lost_age = reid_max_lost_age
        self.use_cmc = use_cmc
        self.use_reid = use_reid

        self._next_id = 1
        self.active: list[Track] = []    # tentative + confirmed
        self.lost: list[Track] = []      # lost but kept for ReID
        self._prev_gray: np.ndarray | None = None

    def reset(self, video_len: int = 1):  # video_len unused, kept for uniform interface
        self._next_id = 1
        self.active = []
        self.lost = []
        self._prev_gray = None

    def _new_track(self, box: np.ndarray, score: float, frame_idx: int,
                   appearance: np.ndarray | None = None) -> Track:
        t = Track(
            track_id=self._next_id,
            box=box,
            score=score,
            frame_idx=frame_idx,
            appearance=appearance,
        )
        t.record()
        self._next_id += 1
        return t

    def step(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        det_boxes: np.ndarray,   # (N, 4) in [x1,y1,x2,y2]
        det_scores: np.ndarray,  # (N,)
    ) -> list[Track]:
        """
        Update tracker with detections for one frame.

        Returns the list of *confirmed* tracks after the update.
        """
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = frame_bgr.shape[:2]

        # ── 1. Camera Motion Compensation ───────────────────────────────────
        M = None
        if self.use_cmc and self._prev_gray is not None and self.active:
            M = estimate_cmc(self._prev_gray, curr_gray)

        # Warp active track boxes to current frame
        if self.active:
            active_boxes = np.array([t.box for t in self.active], dtype=np.float32)
            if M is not None:
                active_boxes = warp_boxes(active_boxes, M)
                # Clip to frame bounds
                active_boxes[:, [0, 2]] = active_boxes[:, [0, 2]].clip(0, w)
                active_boxes[:, [1, 3]] = active_boxes[:, [1, 3]].clip(0, h)
        else:
            active_boxes = np.empty((0, 4), dtype=np.float32)

        # ── 2. Extract appearance features for current detections ────────────
        appearances: list[np.ndarray | None] = [None] * len(det_boxes)
        if self.use_reid and len(det_boxes) > 0:
            crops = []
            for box in det_boxes:
                x1, y1, x2, y2 = (int(v) for v in box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crops.append(frame_bgr[y1:y2, x1:x2])
            feats = extract_features(crops)
            appearances = list(feats)

        # ── 3. Primary association (Tracktor IoU matching) ───────────────────
        iou = iou_matrix(active_boxes, det_boxes) if len(det_boxes) > 0 else np.zeros((len(self.active), 0))
        m_a, m_d, um_a, um_d = hungarian_match(iou, self.iou_threshold)

        # Update matched active tracks
        for ai, di in zip(m_a, m_d):
            self.active[ai].update(det_boxes[di], float(det_scores[di]), frame_idx, appearances[di])
            self.active[ai].mark_confirmed(self.n_init)

        # Mark unmatched active tracks as lost
        for ai in reversed(um_a):   # reversed so we can pop by index
            self.active[ai].mark_lost()

        # Partition active → still_active + newly_lost
        still_active = [t for t in self.active if t.lost_age == 0]
        newly_lost = [t for t in self.active if t.lost_age > 0]
        self.lost.extend(newly_lost)
        self.active = still_active

        # ── 4. ReID: match remaining detections to lost tracks ───────────────
        unmatched_dets = list(um_d)
        if self.use_reid and unmatched_dets and self.lost:
            # Only consider lost tracks that still have appearance features
            reid_lost = [t for t in self.lost if t.appearance is not None]
            if reid_lost and any(appearances[di] is not None for di in unmatched_dets):
                lost_feats = np.array([t.appearance for t in reid_lost], dtype=np.float32)
                det_feats = np.array(
                    [appearances[di] if appearances[di] is not None
                     else np.zeros(576, dtype=np.float32)
                     for di in unmatched_dets],
                    dtype=np.float32,
                )
                # Cosine similarity = dot product (already L2-normalised)
                sim = det_feats @ lost_feats.T   # (n_unmatched_dets, n_lost)
                m_d2, m_l2, um_d2, _ = hungarian_match(sim, self.reid_threshold)

                re_linked: set[int] = set()
                for did2, lid2 in zip(m_d2, m_l2):
                    di = unmatched_dets[did2]
                    track = reid_lost[lid2]
                    track.update(det_boxes[di], float(det_scores[di]), frame_idx, appearances[di])
                    track.state = "confirmed"
                    track.lost_age = 0
                    self.active.append(track)
                    self.lost.remove(track)
                    re_linked.add(did2)

                unmatched_dets = [
                    unmatched_dets[i] for i in range(len(unmatched_dets)) if i not in re_linked
                ]

        # ── 5. Spawn new tracks for remaining unmatched detections ───────────
        for di in unmatched_dets:
            t = self._new_track(det_boxes[di], float(det_scores[di]), frame_idx, appearances[di])
            self.active.append(t)

        # Advance all new tentative tracks
        for t in self.active:
            t.mark_confirmed(self.n_init)

        # ── 6. Age out old lost tracks ───────────────────────────────────────
        self.lost = [
            t for t in self.lost
            if t.lost_age <= self.reid_max_lost_age
        ]
        # Permanently remove active tracks that have been lost too long
        # (this handles tracks that re-entered active from lost but failed again)
        # Prune active tracks inactive for max_lost_age consecutive frames
        still_active2 = []
        for t in self.active:
            if t.lost_age > self.max_lost_age:
                self.lost.append(t)  # give ReID a second chance
            else:
                still_active2.append(t)
        self.active = still_active2

        self._prev_gray = curr_gray

        return [t for t in self.active if t.state == "confirmed"]


# ── Detection ─────────────────────────────────────────────────────────────────

# Lazy-loaded SAM3 singleton
_sam3: dict = {"model": None, "processor": None, "model_id": None}


def _load_sam3(model_id: str):
    """Lazy-load SAM3 model and processor (cached after first call)."""
    if _sam3["model"] is None or _sam3["model_id"] != model_id:
        from transformers import Sam3Model, Sam3Processor
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN")
        print(f"Loading SAM3 from {model_id} …")
        _sam3["processor"] = Sam3Processor.from_pretrained(model_id, token=hf_token)
        _sam3["model"] = (
            Sam3Model.from_pretrained(model_id, token=hf_token)
            .to(DEVICE)
            .eval()
        )
        _sam3["model_id"] = model_id
    return _sam3["model"], _sam3["processor"]


@torch.no_grad()
def detect_persons_sam3(
    frame_bgr: np.ndarray,
    model_id: str,
    threshold: float,
    mask_threshold: float,
    text_prompt: str = "person",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect persons in a BGR frame using SAM3 text-prompted segmentation.

    Returns
    -------
    boxes  : np.ndarray (N, 4)  — [x1, y1, x2, y2]
    scores : np.ndarray (N,)
    """
    model, processor = _load_sam3(model_id)
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(
        images=[pil_img], text=[text_prompt], return_tensors="pt"
    ).to(DEVICE)
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    boxes_list = results.get("boxes", [])
    scores_list = results.get("scores", [])
    if len(boxes_list) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    boxes = np.array([b.cpu().tolist() for b in boxes_list], dtype=np.float32)
    scores = np.array([float(s) for s in scores_list], dtype=np.float32)
    return boxes, scores


def detect_persons_yolo(
    model,
    frame_bgr: np.ndarray,
    conf: float,
    imgsz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run YOLO person detection on a BGR frame.

    Returns
    -------
    boxes  : np.ndarray (N, 4)  — [x1, y1, x2, y2]
    scores : np.ndarray (N,)
    """
    res = model(
        frame_bgr,
        device=DEVICE,
        verbose=False,
        conf=conf,
        classes=[PERSON_CLASS_ID],
        imgsz=imgsz,
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = res.boxes.conf.cpu().numpy().astype(np.float32)
    return boxes, scores


# ── Golden-run loader ─────────────────────────────────────────────────────────

GOLDEN_RUNS_DIR = Path("train/golden_mid/runs/detect")


def list_golden_runs(runs_dir: Path) -> None:
    """Print a table of available runs and their checkpoints, then return."""
    import yaml as _yaml
    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        return
    print(f"Available runs in {runs_dir}:\n")
    print(f"  {'Run':<20}  {'Model':<12}  {'Epochs':>6}  {'imgsz':>5}  Checkpoints")
    print(f"  {'-'*20}  {'-'*12}  {'-'*6}  {'-'*5}  -----------")
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        weights_dir = run_dir / "weights"
        ckpts = sorted(p.name for p in weights_dir.glob("*.pt")) if weights_dir.exists() else []
        model_name, epochs, imgsz = "?", "?", "?"
        args_yaml = run_dir / "args.yaml"
        if args_yaml.exists():
            with open(args_yaml) as f:
                a = _yaml.safe_load(f)
            model_name = Path(a.get("model", "?")).stem
            epochs = str(a.get("epochs", "?"))
            imgsz = str(a.get("imgsz", "?"))
        print(f"  {run_dir.name:<20}  {model_name:<12}  {epochs:>6}  {imgsz:>5}  {', '.join(ckpts) or '(none)'}")


def load_golden_run(
    runs_dir: Path,
    run_name: str,
    checkpoint: str,
) -> tuple:
    """
    Resolve and load a YOLO model from a golden_mid training run.

    Parameters
    ----------
    runs_dir   : root of detect runs (e.g. train/golden_mid/runs/detect)
    run_name   : subdirectory name   (e.g. 'yolo11n2')
    checkpoint : 'best', 'last', or a path relative to the run dir / absolute

    Returns
    -------
    (model, imgsz)  where imgsz is read from the run's args.yaml
    """
    import yaml as _yaml
    from ultralytics import YOLO as _YOLO

    run_dir = runs_dir / run_name
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run '{run_name}' not found in {runs_dir}. "
            f"Run with --list-runs to see available runs."
        )

    if checkpoint in ("best", "last"):
        ckpt_path = run_dir / "weights" / f"{checkpoint}.pt"
    else:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = run_dir / checkpoint

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Read training imgsz so inference matches training resolution
    imgsz = 640
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        with open(args_yaml) as f:
            run_args = _yaml.safe_load(f)
        imgsz = int(run_args.get("imgsz", 640))

    model = _YOLO(str(ckpt_path))
    return model, imgsz


# ── Scene processing ──────────────────────────────────────────────────────────


def collect_frames(scene_dir: Path) -> list[Path]:
    frames = sorted(
        p for p in scene_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return frames


def frame_number(frame_path: Path) -> int:
    """Extract the 1-based integer index from a frame filename."""
    stem = frame_path.stem          # e.g. "frame_00042"
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else 0


def run_scene(
    scene_dir: Path,
    output_dir: Path,
    detector,
    tracker: TracktorPP,
) -> dict:
    """
    Track persons through all frames of one scene.

    Parameters
    ----------
    detector : callable
        ``detector(frame_bgr) -> (boxes_xyxy, scores)`` where
        boxes_xyxy is np.ndarray (N, 4) and scores is np.ndarray (N,).

    Writes MOT-format output and returns summary statistics.
    """
    frames = collect_frames(scene_dir)
    if not frames:
        return {"scene": scene_dir.name, "frames": 0, "tracks": 0, "rows": 0}

    # Prepare output
    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / "gt.txt"

    # Get image dimensions from first frame
    first = cv2.imread(str(frames[0]))
    if first is None:
        return {"scene": scene_dir.name, "frames": 0, "tracks": 0, "rows": 0}
    img_h, img_w = first.shape[:2]

    # Write seqinfo.ini
    seq_cfg = configparser.ConfigParser()
    seq_cfg["Sequence"] = {
        "name": scene_dir.name,
        "imDir": str(scene_dir.resolve()),
        "frameRate": "3",
        "seqLength": str(len(frames)),
        "imWidth": str(img_w),
        "imHeight": str(img_h),
        "imExt": frames[0].suffix,
    }
    with open(output_dir / "seqinfo.ini", "w") as f:
        seq_cfg.write(f)

    tracker.reset(video_len=len(frames))
    mot_rows: list[str] = []
    all_track_ids: set[int] = set()

    for frame_path in frames:
        fnum = frame_number(frame_path)
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        det_boxes, det_scores = detector(frame_bgr)
        confirmed = tracker.step(frame_bgr, fnum, det_boxes, det_scores)

        for t in confirmed:
            x, y, w, h = t.box_xywh
            # MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
            mot_rows.append(
                f"{fnum},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{t.score:.4f},-1,-1,-1"
            )
            all_track_ids.add(t.track_id)

    gt_path.write_text("\n".join(mot_rows) + ("\n" if mot_rows else ""))
    return {
        "scene": scene_dir.name,
        "frames": len(frames),
        "tracks": len(all_track_ids),
        "rows": len(mot_rows),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tracktor++ v2 multi-object tracking pipeline for person tracking."
    )
    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        default=Path("dataset/raw_frames_3fps"),
        help="Root directory with {split}/{scene}/frame_XXXXX.jpg structure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/MOT_labels_3fps"),
        help="Output root for MOT-format labels.",
    )
    # Detector choice
    parser.add_argument(
        "--detector",
        choices=["sam3", "yolo", "golden"],
        default="sam3",
        help=(
            "Person detector.  "
            "'sam3' (default): text-prompted SAM3 segmentation.  "
            "'yolo': generic YOLOv8/11 model.  "
            "'golden': custom YOLO trained on train/golden_mid (use with "
            "--golden-run and --golden-checkpoint)."
        ),
    )
    # SAM3 args
    parser.add_argument(
        "--sam3-model-id",
        type=str,
        default="facebook/sam3",
        help="HuggingFace model ID for SAM3 (used when --detector sam3).",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="person",
        help="Text prompt for SAM3 detection (default: 'person').",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="SAM3 mask binarisation threshold.",
    )
    # YOLO args
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights (used when --detector yolo). Auto-downloaded if absent.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help=(
            "YOLO inference image size. "
            "Defaults to the imgsz recorded in the run's args.yaml for --detector golden, "
            "or 1280 for --detector yolo."
        ),
    )
    # Golden-run args
    parser.add_argument(
        "--golden-runs-dir",
        type=Path,
        default=GOLDEN_RUNS_DIR,
        help=f"Root directory of golden_mid training runs (default: {GOLDEN_RUNS_DIR}).",
    )
    parser.add_argument(
        "--golden-run",
        type=str,
        default=None,
        help="Name of the training run to use (e.g. 'yolo11n2'). Required for --detector golden.",
    )
    parser.add_argument(
        "--golden-checkpoint",
        type=str,
        default="best",
        help="Checkpoint to load: 'best' (default), 'last', or a path relative to the run dir.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List available golden_mid runs and exit (use with --golden-runs-dir to change root).",
    )
    # Shared detection threshold
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold (SAM3 instance threshold or YOLO conf).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to process (default: all found).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional: restrict to specific scene names.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        metavar="PATTERN",
        help="Glob patterns for scene names to skip (e.g. 'acs_s1_*' 'ie_*').",
    )
    # Tracking args
    parser.add_argument("--iou-threshold",  type=float, default=0.5,
                        help="[Tracktor] Min IoU for primary track-detection association.")
    parser.add_argument("--max-lost-age",   type=int,   default=30,
                        help="[Tracktor] Frames a track survives without a match before removal.")
    parser.add_argument("--n-init",         type=int,   default=3,
                        help="[Tracktor] Consecutive matches before a track is confirmed.")
    # ReID (Tracktor only)
    parser.add_argument("--reid-threshold", type=float, default=0.25,
                        help="[Tracktor] Min cosine similarity for ReID re-linkage.")
    parser.add_argument("--reid-max-lost",  type=int,   default=60,
                        help="[Tracktor] Maximum frames a lost track is kept in the ReID gallery.")
    parser.add_argument("--no-cmc",  action="store_true", help="[Tracktor] Disable Camera Motion Compensation.")
    parser.add_argument("--no-reid", action="store_true", help="[Tracktor] Disable ReID.")
    parser.add_argument("--overwrite", action="store_true", help="Re-process already labelled scenes.")
    args = parser.parse_args()

    # --list-runs: print available golden runs and exit
    if args.list_runs:
        list_golden_runs(args.golden_runs_dir)
        return 0

    if not args.raw_frames_dir.exists():
        print(f"ERROR: raw frames directory not found: {args.raw_frames_dir}", file=sys.stderr)
        return 1

    # Discover splits / scenes
    splits = sorted(d for d in args.raw_frames_dir.iterdir() if d.is_dir())
    if args.splits:
        splits = [s for s in splits if s.name in set(args.splits)]
    if not splits:
        print("No splits found.", file=sys.stderr)
        return 1

    # Build detector callable
    if args.detector == "sam3":
        print(f"Detector: SAM3 ({args.sam3_model_id}), prompt='{args.text_prompt}', "
              f"threshold={args.det_conf}, mask_threshold={args.mask_threshold}")
        _load_sam3(args.sam3_model_id)  # warm up now so the first frame isn't slow
        def detector(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return detect_persons_sam3(
                frame_bgr,
                model_id=args.sam3_model_id,
                threshold=args.det_conf,
                mask_threshold=args.mask_threshold,
                text_prompt=args.text_prompt,
            )

    elif args.detector == "golden":
        if not args.golden_run:
            print(
                "ERROR: --golden-run is required when --detector golden.\n"
                "Use --list-runs to see available runs.",
                file=sys.stderr,
            )
            return 1
        golden_model, auto_imgsz = load_golden_run(
            args.golden_runs_dir, args.golden_run, args.golden_checkpoint
        )
        imgsz = args.imgsz or auto_imgsz
        print(
            f"Detector: Golden run '{args.golden_run}' "
            f"checkpoint='{args.golden_checkpoint}', "
            f"conf={args.det_conf}, imgsz={imgsz}"
        )
        def detector(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return detect_persons_yolo(golden_model, frame_bgr, conf=args.det_conf, imgsz=imgsz)

    else:  # yolo
        from ultralytics import YOLO
        imgsz = args.imgsz or 1280
        print(f"Detector: YOLO ({args.yolo_model}), conf={args.det_conf}, imgsz={imgsz}")
        yolo = YOLO(args.yolo_model)
        def detector(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return detect_persons_yolo(yolo, frame_bgr, conf=args.det_conf, imgsz=imgsz)

    # Build tracker
    components = [f"Detector: {args.detector.upper()}", "Tracktor++ (IoU association)"]
    if not args.no_cmc:
        components.append("+v1 CMC (optical flow)")
    if not args.no_reid:
        components.append("+v2 ReID (MobileNetV3 features)")
        print("Loading appearance model (MobileNetV3-small) …")
        _build_appearance_model()   # warm up
    tracker = TracktorPP(
        iou_threshold=args.iou_threshold,
        max_lost_age=args.max_lost_age,
        n_init=args.n_init,
        reid_threshold=args.reid_threshold,
        reid_max_lost_age=args.reid_max_lost,
        use_cmc=not args.no_cmc,
        use_reid=not args.no_reid,
    )

    print("Active components:", " | ".join(components))

    total_scenes = total_tracks = total_rows = 0

    for split_dir in splits:
        scenes = sorted(d for d in split_dir.iterdir() if d.is_dir())
        if args.scenes:
            scenes = [s for s in scenes if s.name in set(args.scenes)]
        if args.exclude:
            import fnmatch
            scenes = [
                s for s in scenes
                if not any(fnmatch.fnmatch(s.name, pat) for pat in args.exclude)
            ]

        print(f"\n── {split_dir.name} ({len(scenes)} scenes) ──")

        for scene_dir in tqdm(scenes, desc=split_dir.name, unit="scene"):
            out_scene = args.output_dir / split_dir.name / scene_dir.name
            gt_file = out_scene / "gt" / "gt.txt"
            if gt_file.exists() and not args.overwrite:
                tqdm.write(f"  skip (exists): {scene_dir.name}")
                continue

            stats = run_scene(
                scene_dir=scene_dir,
                output_dir=out_scene,
                detector=detector,
                tracker=tracker,
            )
            tqdm.write(
                f"  {stats['scene']}: "
                f"{stats['frames']} frames | "
                f"{stats['tracks']} tracks | "
                f"{stats['rows']} annotations"
            )
            total_scenes += 1
            total_tracks += stats["tracks"]
            total_rows += stats["rows"]

    print(f"\n{'─'*60}")
    print(f"Done.  Scenes: {total_scenes}  |  Unique tracks: {total_tracks}  |  Annotations: {total_rows}")
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
