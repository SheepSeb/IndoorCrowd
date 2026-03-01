"""Auto-label 3fps frames with GroundedSAM (Autodistill).

Produces per-scene COCO files with both object detection (bbox) and
instance segmentation (RLE or polygon).

Default layout:
    input:  dataset/raw_frames_3fps/{split}/{scene}/*.jpg
    output: dataset/labels_grounding_sam_3fps/{split}/{scene}.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm


def _major_version(version: str) -> int:
    digits = []
    for ch in version:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else 0


def ensure_transformers_compatible() -> tuple[bool, str]:
    """GroundedSAM currently depends on a transformers<5 API."""
    try:
        import transformers
    except Exception:
        return False, "transformers is not installed."

    version = transformers.__version__
    if _major_version(version) >= 5:
        return (
            False,
            (
                f"Incompatible transformers version detected: {version}. "
                "autodistill-grounded-sam currently needs transformers<5.\n"
                "Fix with:\n"
                "  uv pip install \"transformers<5\"\n"
                "or run:\n"
                "  uv sync"
            ),
        )
    return True, version


def parse_ontology_pairs(pairs: list[str]) -> OrderedDict[str, str]:
    """Parse CLI ontology values in the form caption=class."""
    if not pairs:
        raise ValueError("At least one --ontology mapping is required.")

    mapping: OrderedDict[str, str] = OrderedDict()
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"Invalid ontology pair '{pair}'. Expected format: caption=class"
            )
        caption, class_name = pair.split("=", 1)
        caption = caption.strip()
        class_name = class_name.strip()
        if not caption or not class_name:
            raise ValueError(
                f"Invalid ontology pair '{pair}'. Caption and class must be non-empty."
            )
        mapping[caption] = class_name
    return mapping


def discover_scenes(raw_frames_dir: Path, split: str) -> list[Path]:
    split_dir = raw_frames_dir / split
    if not split_dir.exists():
        return []
    return sorted(path for path in split_dir.iterdir() if path.is_dir())


def list_image_paths(scene_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(scene_dir.glob(pattern))
    return sorted(image_paths)


def xyxy_to_xywh(box: np.ndarray, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), float(width - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1, y1, w, h]


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_polygons(binary_mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: list[list[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        polygons.append(contour.flatten().astype(float).tolist())
    return polygons


def process_scene(
    model,
    scene_dir: Path,
    raw_frames_dir: Path,
    class_names: list[str],
    segmentation_format: str,
) -> dict | None:
    image_paths = list_image_paths(scene_dir)
    if not image_paths:
        return None

    class_name_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}
    categories = [
        {"id": category_id, "name": name, "supercategory": "object"}
        for name, category_id in class_name_to_id.items()
    ]

    coco: dict = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    ann_id = 1
    for image_id, img_path in enumerate(tqdm(image_paths, desc=f"  {scene_dir.name}"), 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        height, width = img.shape[:2]

        rel_path = img_path.relative_to(raw_frames_dir)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": str(rel_path),
                "width": width,
                "height": height,
            }
        )

        detections = model.predict(str(img_path))
        xyxy = np.asarray(getattr(detections, "xyxy", []))
        masks = getattr(detections, "mask", None)
        confidences = getattr(detections, "confidence", None)
        class_ids = getattr(detections, "class_id", None)

        if xyxy.size == 0:
            continue

        for det_idx, box in enumerate(xyxy):
            bbox = xyxy_to_xywh(box, width=width, height=height)
            score = (
                float(confidences[det_idx])
                if confidences is not None and det_idx < len(confidences)
                else 1.0
            )

            # GroundedSAM returns ontology class indexes in class_id.
            class_name = class_names[0]
            if class_ids is not None and det_idx < len(class_ids):
                class_index = int(class_ids[det_idx])
                if 0 <= class_index < len(class_names):
                    class_name = class_names[class_index]
            category_id = class_name_to_id[class_name]

            segmentation: dict | list[list[float]]
            area: float

            has_mask = masks is not None and det_idx < len(masks)
            if has_mask:
                binary_mask = np.asarray(masks[det_idx]).astype(np.uint8)
                if binary_mask.shape[:2] != (height, width):
                    # Safety fallback for unexpected mask dimensions.
                    continue
                area = float(binary_mask.sum())
                if segmentation_format == "polygon":
                    segmentation = mask_to_polygons(binary_mask)
                    if not segmentation:
                        continue
                else:
                    segmentation = mask_to_rle(binary_mask)
            else:
                # Fallback if mask is absent: encode bbox as polygon.
                x, y, w, h = bbox
                area = float(w * h)
                segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [round(v, 2) for v in bbox],
                    "area": round(area, 2),
                    "segmentation": segmentation,
                    "score": round(score, 4),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return coco


def process_split(
    model,
    raw_frames_dir: Path,
    output_dir: Path,
    split: str,
    class_names: list[str],
    segmentation_format: str,
    overwrite: bool,
) -> None:
    scenes = discover_scenes(raw_frames_dir, split)
    if not scenes:
        print(f"No scenes found for split '{split}', skipping.")
        return

    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_anns = 0
    print(f"\n{'=' * 60}")
    print(f"Split '{split}': {len(scenes)} scene(s)")
    print(f"{'=' * 60}")

    for scene_index, scene_dir in enumerate(scenes, 1):
        out_path = split_output_dir / f"{scene_dir.name}.json"
        scene_images = list_image_paths(scene_dir)

        if out_path.exists() and not overwrite:
            with open(out_path) as f:
                existing = json.load(f)
            if len(existing.get("images", [])) == len(scene_images):
                print(
                    f"[{scene_index}/{len(scenes)}] {scene_dir.name}: already done "
                    "(use --overwrite to regenerate) — skipping"
                )
                total_images += len(existing.get("images", []))
                total_anns += len(existing.get("annotations", []))
                continue

        print(f"[{scene_index}/{len(scenes)}] {scene_dir.name}: {len(scene_images)} frames")

        coco = process_scene(
            model=model,
            scene_dir=scene_dir,
            raw_frames_dir=raw_frames_dir,
            class_names=class_names,
            segmentation_format=segmentation_format,
        )
        if coco is None:
            print("  -> no images, skipped")
            continue

        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)

        num_images = len(coco.get("images", []))
        num_anns = len(coco.get("annotations", []))
        total_images += num_images
        total_anns += num_anns
        print(f"  -> {num_images} images, {num_anns} annotations => {out_path}")

    print(f"\nSplit '{split}' done: {total_images} images, {total_anns} annotations total")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-label frames using GroundedSAM (Autodistill) into COCO JSON."
    )
    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        default=Path("dataset/raw_frames_3fps"),
        help="Input raw frames directory (default: dataset/raw_frames_3fps)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/labels_grounding_sam_3fps"),
        help="Output COCO labels directory (default: dataset/labels_grounding_sam_3fps)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "challange"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--ontology",
        nargs="+",
        default=["person=person"],
        help="Caption ontology pairs in caption=class format. Example: person=person.",
    )
    parser.add_argument(
        "--segmentation-format",
        choices=["rle", "polygon"],
        default="rle",
        help="COCO segmentation format in output JSON.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.35,
        help="GroundedSAM box threshold.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="GroundedSAM text threshold.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing scene JSON files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.raw_frames_dir.exists():
        print(f"Input directory does not exist: {args.raw_frames_dir}")
        return 1

    ok, transformers_info = ensure_transformers_compatible()
    if not ok:
        print(transformers_info)
        return 1

    try:
        ontology_pairs = parse_ontology_pairs(args.ontology)
    except ValueError as exc:
        print(f"Ontology error: {exc}")
        return 1

    try:
        from autodistill.detection import CaptionOntology
        from autodistill_grounded_sam import GroundedSAM
    except ImportError:
        print(
            "Missing dependency: autodistill-grounded-sam.\n"
            "Install with: pip install autodistill-grounded-sam"
        )
        return 1

    class_names = list(OrderedDict.fromkeys(ontology_pairs.values()))
    print("Loading GroundedSAM ...")
    print(f"transformers: {transformers_info}")
    model = GroundedSAM(
        ontology=CaptionOntology(dict(ontology_pairs)),
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    print(f"Ontology: {dict(ontology_pairs)}")
    print(f"Segmentation format: {args.segmentation_format}")
    print(f"Input:  {args.raw_frames_dir}")
    print(f"Output: {args.output_dir}")

    t0 = time.time()
    for split in args.splits:
        process_split(
            model=model,
            raw_frames_dir=args.raw_frames_dir,
            output_dir=args.output_dir,
            split=split,
            class_names=class_names,
            segmentation_format=args.segmentation_format,
            overwrite=args.overwrite,
        )

    print(f"\nAll done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
