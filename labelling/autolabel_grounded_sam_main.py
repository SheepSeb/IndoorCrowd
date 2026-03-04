#!/usr/bin/env python3
"""GroundedSAM (Grounding DINO + SAM) → COCO JSON for dataset/main.

Produces per-recording COCO files with bbox + RLE or polygon segmentation.
Wall-clock timing is saved to timing.json in the output root.

Output layout
-------------
  dataset/main/labels_grounded_sam/
    {scene}/
      {recording}.json
    timing.json

Usage
-----
  uv run python labelling/autolabel_grounded_sam_main.py

  # Specific scenes
  uv run python labelling/autolabel_grounded_sam_main.py --scenes acs_ec acs_eg

  # Tune thresholds
  uv run python labelling/autolabel_grounded_sam_main.py \\
      --box-threshold 0.30 --text-threshold 0.20 --overwrite
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import stat
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm


# ── compatibility helpers (same as autolabel_grounded_edgesam_3fps.py) ─────────


def _major_version(version: str) -> int:
    digits = []
    for ch in version:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else 0


def ensure_transformers_compatible() -> tuple[bool, str]:
    try:
        import transformers
    except Exception:
        return False, "transformers is not installed."
    version = transformers.__version__
    if _major_version(version) >= 5:
        return (
            False,
            (
                f"Incompatible transformers version: {version}. "
                "autodistill grounded sam wrappers need transformers<5.\n"
                "Fix: uv pip install \"transformers<5\""
            ),
        )
    return True, version


def ensure_pip_command() -> str:
    pip_on_path = shutil.which("pip")
    if pip_on_path:
        return pip_on_path
    shim_dir = Path.cwd() / ".tmp_bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    pip_shim = shim_dir / "pip"
    pip_shim.write_text(
        "#!/usr/bin/env sh\n"
        f"exec {sys.executable} -m pip \"$@\"\n"
    )
    pip_shim.chmod(pip_shim.stat().st_mode | stat.S_IEXEC)
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    return str(pip_shim)


# ── COCO helpers ───────────────────────────────────────────────────────────────


def collect_frames(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def xyxy_to_xywh(box: np.ndarray, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), float(width - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def mask_to_rle(binary: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_polygons(binary: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [
        contour.flatten().astype(float).tolist()
        for contour in contours if contour.shape[0] >= 3
    ]


# ── per-recording processing ───────────────────────────────────────────────────


def run_recording(
    model,
    recording_dir: Path,
    out_json: Path,
    raw_frames_dir: Path,
    seg_format: str,
) -> dict:
    frames = collect_frames(recording_dir)
    if not frames:
        return {"frames": 0, "annotations": 0, "seconds": 0.0}

    coco_images: list[dict] = []
    coco_anns: list[dict] = []
    ann_id = 1

    t0 = time.perf_counter()

    for image_id, img_path in enumerate(
        tqdm(frames, desc=f"  {recording_dir.name}", leave=False), 1
    ):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        height, width = img.shape[:2]

        coco_images.append({
            "id": image_id,
            "file_name": str(img_path.relative_to(raw_frames_dir)),
            "width": width,
            "height": height,
        })

        detections = model.predict(str(img_path))
        xyxy = np.asarray(getattr(detections, "xyxy", []))
        masks = getattr(detections, "mask", None)
        confidences = getattr(detections, "confidence", None)
        class_ids = getattr(detections, "class_id", None)

        if xyxy.size == 0:
            continue

        for det_idx, box in enumerate(xyxy):
            bbox = xyxy_to_xywh(box, width, height)
            score = (
                float(confidences[det_idx])
                if confidences is not None and det_idx < len(confidences)
                else 1.0
            )
            cat_id = 1
            if class_ids is not None and det_idx < len(class_ids):
                cat_id = int(class_ids[det_idx]) + 1

            has_mask = masks is not None and det_idx < len(masks)
            if has_mask:
                binary = np.asarray(masks[det_idx]).astype(np.uint8)
                if binary.shape[:2] != (height, width):
                    continue
                area = float(binary.sum())
                if seg_format == "polygon":
                    segmentation = mask_to_polygons(binary)
                    if not segmentation:
                        continue
                else:
                    segmentation = mask_to_rle(binary)
            else:
                x, y, w, h = bbox
                area = w * h
                segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]

            coco_anns.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [round(v, 2) for v in bbox],
                "area": round(area, 2),
                "segmentation": segmentation,
                "score": round(score, 4),
                "iscrowd": 0,
            })
            ann_id += 1

    elapsed = time.perf_counter() - t0

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "images": coco_images,
                "annotations": coco_anns,
                "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
            },
            f,
        )

    return {"frames": len(frames), "annotations": len(coco_anns), "seconds": elapsed}


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GroundedSAM (Grounding DINO + SAM) → COCO JSON for dataset/main"
    )
    parser.add_argument("--raw-frames-dir", type=Path,
                        default=Path("dataset/main/raw_frames_5fps"))
    parser.add_argument("--output-dir",     type=Path,
                        default=Path("dataset/main/labels_grounded_sam"))
    parser.add_argument("--ontology", nargs="+", default=["person=person"],
                        help="Caption=class pairs, e.g. person=person.")
    parser.add_argument("--segmentation-format", choices=["rle", "polygon"], default="rle")
    parser.add_argument("--box-threshold",  type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--scenes",  nargs="+", default=None,
                        help="Restrict to specific scene dirs (e.g. acs_ec acs_eg).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.raw_frames_dir.exists():
        print(f"ERROR: {args.raw_frames_dir} not found", file=sys.stderr)
        return 1

    ok, transformers_info = ensure_transformers_compatible()
    if not ok:
        print(transformers_info, file=sys.stderr)
        return 1

    # Parse ontology
    ontology_pairs: OrderedDict[str, str] = OrderedDict()
    for pair in args.ontology:
        if "=" not in pair:
            print(f"Invalid ontology pair '{pair}' (expected caption=class)", file=sys.stderr)
            return 1
        caption, cls = pair.split("=", 1)
        ontology_pairs[caption.strip()] = cls.strip()

    try:
        from autodistill.detection import CaptionOntology
        from autodistill_grounded_sam import GroundedSAM
    except ImportError:
        print(
            "Missing dependency: autodistill-grounded-sam.\n"
            "Install with: pip install autodistill-grounded-sam",
            file=sys.stderr,
        )
        return 1

    ensure_pip_command()

    t_load = time.perf_counter()
    print("Loading GroundedSAM ...")
    model = GroundedSAM(
        ontology=CaptionOntology(dict(ontology_pairs)),
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    model_load_s = time.perf_counter() - t_load

    print(f"Model       : GroundedSAM  (load: {model_load_s:.1f}s)")
    print(f"Ontology    : {dict(ontology_pairs)}")
    print(f"Seg format  : {args.segmentation_format}")
    print(f"Input       : {args.raw_frames_dir}")
    print(f"Output      : {args.output_dir}")

    scene_dirs = sorted(d for d in args.raw_frames_dir.iterdir() if d.is_dir())
    if args.scenes:
        scene_dirs = [s for s in scene_dirs if s.name in set(args.scenes)]
    if not scene_dirs:
        print("No scene directories found.", file=sys.stderr)
        return 1

    timing_path = args.output_dir / "timing.json"
    if timing_path.exists() and not args.overwrite:
        with open(timing_path) as f:
            timing = json.load(f)
    else:
        timing = {
            "method": "grounded_sam",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_load_seconds": round(model_load_s, 2),
            "total_seconds": 0.0,
            "scenes": {},
        }

    t_total = time.perf_counter()

    for scene_dir in scene_dirs:
        recordings = sorted(d for d in scene_dir.iterdir() if d.is_dir())
        print(f"\n── {scene_dir.name} ({len(recordings)} recordings) ──")

        for rec_dir in tqdm(recordings, desc=scene_dir.name, unit="rec"):
            out_json = args.output_dir / scene_dir.name / f"{rec_dir.name}.json"
            key = f"{scene_dir.name}/{rec_dir.name}"

            if out_json.exists() and not args.overwrite:
                tqdm.write(f"  skip (exists): {rec_dir.name}")
                continue

            tqdm.write(f"  {rec_dir.name}")
            stats = run_recording(
                model, rec_dir, out_json, args.raw_frames_dir, args.segmentation_format
            )

            timing["scenes"][key] = {
                "frames": stats["frames"],
                "annotations": stats["annotations"],
                "seconds": round(stats["seconds"], 2),
                "fps_processed": (
                    round(stats["frames"] / stats["seconds"], 2)
                    if stats["seconds"] > 0 else 0.0
                ),
            }
            tqdm.write(
                f"  → {stats['frames']} frames | {stats['annotations']} anns"
                f" | {stats['seconds']:.1f}s"
            )

    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\nDone in {timing['total_seconds']:.1f}s")
    print(f"Timing: {timing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
