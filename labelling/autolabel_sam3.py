"""Auto-label frames using SAM3 (Segment Anything Model 3) with text-prompted
segmentation, producing COCO-format annotations with bounding boxes and
instance segmentation masks.

Processes each scene (recording folder) independently, showing per-scene
progress and saving annotations immediately after each scene finishes.

Output layout:
    dataset/labels/{split}/{scene_name}.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

load_dotenv()


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert a binary mask to COCO RLE format via pycocotools."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_polygons(binary_mask: np.ndarray) -> list[list[float]]:
    """Convert a binary mask to COCO polygon format via OpenCV contours."""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        polygons.append(contour.flatten().tolist())
    return polygons


def xyxy_to_xywh(box: list[float]) -> list[float]:
    """Convert [x1, y1, x2, y2] to COCO [x, y, w, h]."""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def discover_scenes(raw_frames_dir: Path, split: str) -> list[Path]:
    """Return sorted list of scene directories inside a split folder."""
    split_dir = raw_frames_dir / split
    if not split_dir.exists():
        return []
    return sorted(p for p in split_dir.iterdir() if p.is_dir())


def process_scene(
    model: Sam3Model,
    processor: Sam3Processor,
    scene_dir: Path,
    raw_frames_dir: Path,
    text_prompt: str,
    batch_size: int,
    threshold: float,
    mask_threshold: float,
    seg_format: str,
    device: str,
) -> dict:
    """Process all frames in a single scene and return a COCO dict."""
    image_paths = sorted(scene_dir.glob("*.jpg"))
    if not image_paths:
        return None

    coco: dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }
    ann_id = 1

    for batch_start in tqdm(
        range(0, len(image_paths), batch_size),
        desc=f"  {scene_dir.name}",
        unit="batch",
        leave=False,
    ):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
        text_prompts = [text_prompt] * len(pil_images)

        inputs = processor(
            images=pil_images, text=text_prompts, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

        for idx, (img_path, pil_img, result) in enumerate(
            zip(batch_paths, pil_images, results)
        ):
            image_id = batch_start + idx + 1
            w, h = pil_img.size
            rel_path = img_path.relative_to(raw_frames_dir)

            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": str(rel_path),
                    "width": w,
                    "height": h,
                }
            )

            masks = result.get("masks", [])
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])

            for mask_t, box_t, score_t in zip(masks, boxes, scores):
                mask_np = mask_t.cpu().numpy().astype(np.uint8)
                bbox = xyxy_to_xywh(box_t.cpu().tolist())
                area = float(mask_np.sum())

                if seg_format == "rle":
                    segmentation = mask_to_rle(mask_np)
                else:
                    segmentation = mask_to_polygons(mask_np)
                    if not segmentation:
                        continue

                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [round(v, 2) for v in bbox],
                        "area": round(area, 2),
                        "segmentation": segmentation,
                        "score": round(float(score_t), 4),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    return coco


def process_split(
    model: Sam3Model,
    processor: Sam3Processor,
    raw_frames_dir: Path,
    output_dir: Path,
    split: str,
    text_prompt: str,
    batch_size: int,
    threshold: float,
    mask_threshold: float,
    seg_format: str,
    device: str,
) -> None:
    scenes = discover_scenes(raw_frames_dir, split)
    if not scenes:
        print(f"No scenes found for split '{split}', skipping.")
        return

    split_output = output_dir / split
    split_output.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_anns = 0

    print(f"\n{'=' * 60}")
    print(f"Split '{split}': {len(scenes)} scene(s)")
    print(f"{'=' * 60}")

    for i, scene_dir in enumerate(scenes, 1):
        out_path = split_output / f"{scene_dir.name}.json"
        n_frames = len(list(scene_dir.glob("*.jpg")))

        if out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)
            n_existing = len(existing.get("images", []))
            if n_existing == n_frames and n_frames > 0:
                n_ann = len(existing.get("annotations", []))
                print(f"[{i}/{len(scenes)}] {scene_dir.name}: "
                      f"already done ({n_existing} imgs, {n_ann} anns) — skipping")
                total_images += n_existing
                total_anns += n_ann
                continue

        print(f"[{i}/{len(scenes)}] {scene_dir.name}: {n_frames} frames")

        coco = process_scene(
            model=model,
            processor=processor,
            scene_dir=scene_dir,
            raw_frames_dir=raw_frames_dir,
            text_prompt=text_prompt,
            batch_size=batch_size,
            threshold=threshold,
            mask_threshold=mask_threshold,
            seg_format=seg_format,
            device=device,
        )

        if coco is None:
            print("  -> no images, skipped")
            continue

        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)

        n_img = len(coco["images"])
        n_ann = len(coco["annotations"])
        total_images += n_img
        total_anns += n_ann
        print(f"  -> {n_img} images, {n_ann} annotations => {out_path}")

    print(f"\nSplit '{split}' done: {total_images} images, {total_anns} annotations total")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-label frames with SAM3 text-prompted segmentation (COCO format)"
    )
    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        default=Path("dataset/raw_frames"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/labels"),
    )
    parser.add_argument("--text-prompt", type=str, default="person")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "challange"],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--segmentation-format",
        choices=["rle", "polygon"],
        default="rle",
    )
    parser.add_argument("--model-id", type=str, default="facebook/sam3")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading SAM3 from {args.model_id} ...")

    hf_token = os.environ.get("HF_TOKEN")
    model = Sam3Model.from_pretrained(args.model_id, token=hf_token).to(device)
    processor = Sam3Processor.from_pretrained(args.model_id, token=hf_token)
    model.eval()

    print(f"Text prompt: '{args.text_prompt}'")
    print(f"Segmentation format: {args.segmentation_format}")
    print(f"Confidence threshold: {args.threshold} | Mask threshold: {args.mask_threshold}")

    t0 = time.time()
    for split in args.splits:
        process_split(
            model=model,
            processor=processor,
            raw_frames_dir=args.raw_frames_dir,
            output_dir=args.output_dir,
            split=split,
            text_prompt=args.text_prompt,
            batch_size=args.batch_size,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            seg_format=args.segmentation_format,
            device=device,
        )

    print(f"\nAll done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
