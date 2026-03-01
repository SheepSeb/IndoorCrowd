#!/usr/bin/env python3
"""Compare SAM3, GroundedSAM and Efficient GroundedSAM COCO labels."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from pycocotools import mask as mask_utils

SAM3 = "sam3"
GROUNDED = "grounded_sam"
EFFICIENT = "efficient_grounded_sam"


@dataclass
class ThreeWaySceneSummary:
    split: str
    scene: str
    common_images_all: int
    sam3_images: int
    grounded_sam_images: int
    efficient_grounded_sam_images: int
    sam3_annotations: int
    grounded_sam_annotations: int
    efficient_grounded_sam_annotations: int


@dataclass
class ImagePairStats:
    image_file: str
    model_a_count: int
    model_b_count: int
    matched: int
    model_a_covered: float
    model_b_covered: float
    mean_bbox_iou: float | None
    mean_mask_iou: float | None


@dataclass
class PairwiseSceneStats:
    split: str
    scene: str
    model_a: str
    model_b: str
    model_a_images: int
    model_b_images: int
    common_images: int
    model_a_annotations: int
    model_b_annotations: int
    matched_annotations: int
    model_a_coverage: float
    model_b_coverage: float
    mean_abs_count_delta: float
    mean_bbox_iou: float | None
    mean_mask_iou: float | None
    image_stats: list[ImagePairStats]


def list_scene_files(labels_dir: Path) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    if not labels_dir.exists():
        return out
    for split_dir in sorted(labels_dir.iterdir()):
        if not split_dir.is_dir() or split_dir.name == "session_logs":
            continue
        scenes = {p.stem: p for p in sorted(split_dir.glob("*.json"))}
        if scenes:
            out[split_dir.name] = scenes
    return out


def decode_mask(segmentation: dict | list, h: int, w: int):
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return mask_utils.decode(rle)
    rles = mask_utils.frPyObjects(segmentation, h, w)
    return mask_utils.decode(mask_utils.merge(rles))


def bbox_iou_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = (aw * ah) + (bw * bh) - inter_area
    return 0.0 if union <= 0 else inter_area / union


def greedy_match_by_bbox_iou(
    anns_a: list[dict], anns_b: list[dict], iou_threshold: float
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for i, ann_a in enumerate(anns_a):
        for j, ann_b in enumerate(anns_b):
            iou = bbox_iou_xywh(ann_a["bbox"], ann_b["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, i, j))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, iou))
    return matches


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def safe_ratio(num: int, den: int) -> float:
    return 0.0 if den <= 0 else float(num) / float(den)


def fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def load_coco_by_filename(coco_path: Path) -> tuple[dict[str, tuple[int, int]], dict[str, list[dict]]]:
    with open(coco_path) as f:
        coco = json.load(f)

    image_info_by_id = {
        img["id"]: (img["file_name"], int(img["height"]), int(img["width"]))
        for img in coco.get("images", [])
    }
    dims_by_filename: dict[str, tuple[int, int]] = {}
    anns_by_filename: dict[str, list[dict]] = {}
    for image_id, (file_name, h, w) in image_info_by_id.items():
        dims_by_filename[file_name] = (h, w)
        anns_by_filename[file_name] = []
    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id not in image_info_by_id:
            continue
        file_name, _, _ = image_info_by_id[image_id]
        anns_by_filename[file_name].append(ann)
    return dims_by_filename, anns_by_filename


def compare_pair_scene(
    split: str,
    scene: str,
    model_a: str,
    model_b: str,
    json_a: Path,
    json_b: Path,
    iou_threshold: float,
) -> PairwiseSceneStats:
    dims_a, anns_a_by_file = load_coco_by_filename(json_a)
    dims_b, anns_b_by_file = load_coco_by_filename(json_b)

    files_a = set(anns_a_by_file.keys())
    files_b = set(anns_b_by_file.keys())
    common_files = sorted(files_a & files_b)

    anns_a_total = 0
    anns_b_total = 0
    matched_total = 0
    bbox_ious_all: list[float] = []
    mask_ious_all: list[float] = []
    abs_count_delta: list[float] = []
    image_stats: list[ImagePairStats] = []

    for file_name in common_files:
        anns_a = anns_a_by_file[file_name]
        anns_b = anns_b_by_file[file_name]
        anns_a_total += len(anns_a)
        anns_b_total += len(anns_b)
        abs_count_delta.append(abs(len(anns_a) - len(anns_b)))

        matches = greedy_match_by_bbox_iou(anns_a, anns_b, iou_threshold=iou_threshold)
        matched_total += len(matches)
        image_bbox_ious = [m[2] for m in matches]
        bbox_ious_all.extend(image_bbox_ious)

        h, w = dims_a.get(file_name, dims_b[file_name])
        image_mask_ious: list[float] = []
        for i, j, _ in matches:
            seg_a = anns_a[i].get("segmentation")
            seg_b = anns_b[j].get("segmentation")
            if seg_a is None or seg_b is None:
                continue
            mask_a = decode_mask(seg_a, h=h, w=w)
            mask_b = decode_mask(seg_b, h=h, w=w)
            inter = ((mask_a > 0) & (mask_b > 0)).sum()
            union = ((mask_a > 0) | (mask_b > 0)).sum()
            if union > 0:
                iou = float(inter) / float(union)
                image_mask_ious.append(iou)
                mask_ious_all.append(iou)

        image_stats.append(
            ImagePairStats(
                image_file=file_name,
                model_a_count=len(anns_a),
                model_b_count=len(anns_b),
                matched=len(matches),
                model_a_covered=safe_ratio(len(matches), len(anns_a)),
                model_b_covered=safe_ratio(len(matches), len(anns_b)),
                mean_bbox_iou=mean(image_bbox_ious),
                mean_mask_iou=mean(image_mask_ious),
            )
        )

    return PairwiseSceneStats(
        split=split,
        scene=scene,
        model_a=model_a,
        model_b=model_b,
        model_a_images=len(files_a),
        model_b_images=len(files_b),
        common_images=len(common_files),
        model_a_annotations=anns_a_total,
        model_b_annotations=anns_b_total,
        matched_annotations=matched_total,
        model_a_coverage=safe_ratio(matched_total, anns_a_total),
        model_b_coverage=safe_ratio(matched_total, anns_b_total),
        mean_abs_count_delta=float(statistics.fmean(abs_count_delta)) if abs_count_delta else 0.0,
        mean_bbox_iou=mean(bbox_ious_all),
        mean_mask_iou=mean(mask_ious_all),
        image_stats=image_stats,
    )


def scene_summary_three_way(
    split: str,
    scene: str,
    sam3_json: Path,
    grounded_json: Path,
    efficient_json: Path,
) -> ThreeWaySceneSummary:
    _, sam3_by_file = load_coco_by_filename(sam3_json)
    _, grounded_by_file = load_coco_by_filename(grounded_json)
    _, efficient_by_file = load_coco_by_filename(efficient_json)
    common_images_all = len(
        set(sam3_by_file.keys())
        & set(grounded_by_file.keys())
        & set(efficient_by_file.keys())
    )
    return ThreeWaySceneSummary(
        split=split,
        scene=scene,
        common_images_all=common_images_all,
        sam3_images=len(sam3_by_file),
        grounded_sam_images=len(grounded_by_file),
        efficient_grounded_sam_images=len(efficient_by_file),
        sam3_annotations=sum(len(v) for v in sam3_by_file.values()),
        grounded_sam_annotations=sum(len(v) for v in grounded_by_file.values()),
        efficient_grounded_sam_annotations=sum(len(v) for v in efficient_by_file.values()),
    )


def print_three_way_table(summaries: list[ThreeWaySceneSummary]) -> None:
    print("\nThree-way scene coverage")
    print("-" * 115)
    print(
        f"{'split/scene':45s} {'imgs(common/all)':>16s} {'anns SAM3':>12s} "
        f"{'anns G-SAM':>12s} {'anns E-GSAM':>13s}"
    )
    print("-" * 115)
    for s in summaries:
        all_imgs = f"{s.sam3_images},{s.grounded_sam_images},{s.efficient_grounded_sam_images}"
        print(
            f"{f'{s.split}/{s.scene}'[:45]:45s} "
            f"{f'{s.common_images_all}/{all_imgs}':>16s} "
            f"{s.sam3_annotations:12d} "
            f"{s.grounded_sam_annotations:12d} "
            f"{s.efficient_grounded_sam_annotations:13d}"
        )
    print("-" * 115)
    print(
        f"scenes: {len(summaries)} | "
        f"total common images: {sum(s.common_images_all for s in summaries)}"
    )


def print_pair_report(title: str, stats: list[PairwiseSceneStats]) -> None:
    if not stats:
        print(f"\n{title}\nNo overlapping scenes.")
        return

    print(f"\n{title}")
    print("-" * 114)
    print(
        f"{'split/scene':45s} {'imgs':>9s} {'anns(A/B)':>14s} {'match':>7s} "
        f"{'cov_A':>7s} {'cov_B':>7s} {'bboxIoU':>8s} {'maskIoU':>8s} {'|Δcount|':>9s}"
    )
    print("-" * 114)
    for s in stats:
        scene_name = f"{s.split}/{s.scene}"
        print(
            f"{scene_name[:45]:45s} "
            f"{f'{s.common_images}/{s.model_a_images},{s.model_b_images}':>9s} "
            f"{f'{s.model_a_annotations}/{s.model_b_annotations}':>14s} "
            f"{s.matched_annotations:7d} "
            f"{s.model_a_coverage:7.3f} "
            f"{s.model_b_coverage:7.3f} "
            f"{fmt_float(s.mean_bbox_iou):>8s} "
            f"{fmt_float(s.mean_mask_iou):>8s} "
            f"{s.mean_abs_count_delta:9.3f}"
        )
    print("-" * 114)

    total_common_images = sum(s.common_images for s in stats)
    total_a = sum(s.model_a_annotations for s in stats)
    total_b = sum(s.model_b_annotations for s in stats)
    total_matched = sum(s.matched_annotations for s in stats)
    print("OVERALL")
    print(f"- scenes compared: {len(stats)}")
    print(f"- common images:   {total_common_images}")
    print(f"- annotations:     A={total_a} B={total_b}")
    print(f"- matched pairs:   {total_matched}")
    print(f"- A coverage:      {safe_ratio(total_matched, total_a):.3f}")
    print(f"- B coverage:      {safe_ratio(total_matched, total_b):.3f}")
    print(f"- mean bbox IoU:   {fmt_float(mean([v.mean_bbox_iou for v in stats if v.mean_bbox_iou is not None]))}")
    print(f"- mean mask IoU:   {fmt_float(mean([v.mean_mask_iou for v in stats if v.mean_mask_iou is not None]))}")


def write_json(
    output_path: Path,
    summaries: list[ThreeWaySceneSummary],
    pair_results: dict[str, list[PairwiseSceneStats]],
    model_dirs: dict[str, str],
    iou_threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "bbox_iou_threshold": iou_threshold,
            "model_dirs": model_dirs,
        },
        "three_way_summaries": [asdict(s) for s in summaries],
        "pairwise": {key: [asdict(item) for item in value] for key, value in pair_results.items()},
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare SAM3, GroundedSAM and Efficient GroundedSAM labels scene-by-scene."
    )
    parser.add_argument("--sam3-dir", type=Path, default=Path("dataset/labels_3fps"))
    parser.add_argument("--grounded-dir", type=Path, default=Path("dataset/labels_grounding_sam_3fps"))
    parser.add_argument(
        "--efficient-dir",
        type=Path,
        default=Path("dataset/labels_efficient_grounded_sam_3fps"),
    )
    parser.add_argument("--splits", nargs="+", default=None, help="Optional split filter")
    parser.add_argument("--bbox-iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("analysis/reports/sam3_grounded_efficient_comparison.json"),
    )
    args = parser.parse_args()

    for key, path in {
        SAM3: args.sam3_dir,
        GROUNDED: args.grounded_dir,
        EFFICIENT: args.efficient_dir,
    }.items():
        if not path.exists():
            print(f"{key} labels directory not found: {path}", file=sys.stderr)
            return 1

    if not (0.0 <= args.bbox_iou_threshold <= 1.0):
        print("--bbox-iou-threshold must be in [0, 1].", file=sys.stderr)
        return 1

    scene_maps = {
        SAM3: list_scene_files(args.sam3_dir),
        GROUNDED: list_scene_files(args.grounded_dir),
        EFFICIENT: list_scene_files(args.efficient_dir),
    }

    splits = sorted(
        set(scene_maps[SAM3].keys())
        & set(scene_maps[GROUNDED].keys())
        & set(scene_maps[EFFICIENT].keys())
    )
    if args.splits:
        selected = set(args.splits)
        splits = [s for s in splits if s in selected]

    summaries: list[ThreeWaySceneSummary] = []
    pair_results: dict[str, list[PairwiseSceneStats]] = {
        f"{SAM3}_vs_{GROUNDED}": [],
        f"{SAM3}_vs_{EFFICIENT}": [],
        f"{GROUNDED}_vs_{EFFICIENT}": [],
    }

    for split in splits:
        common_scenes = sorted(
            set(scene_maps[SAM3][split].keys())
            & set(scene_maps[GROUNDED][split].keys())
            & set(scene_maps[EFFICIENT][split].keys())
        )
        for scene in common_scenes:
            sam3_json = scene_maps[SAM3][split][scene]
            grounded_json = scene_maps[GROUNDED][split][scene]
            efficient_json = scene_maps[EFFICIENT][split][scene]

            summaries.append(
                scene_summary_three_way(
                    split=split,
                    scene=scene,
                    sam3_json=sam3_json,
                    grounded_json=grounded_json,
                    efficient_json=efficient_json,
                )
            )
            pair_results[f"{SAM3}_vs_{GROUNDED}"].append(
                compare_pair_scene(
                    split=split,
                    scene=scene,
                    model_a=SAM3,
                    model_b=GROUNDED,
                    json_a=sam3_json,
                    json_b=grounded_json,
                    iou_threshold=args.bbox_iou_threshold,
                )
            )
            pair_results[f"{SAM3}_vs_{EFFICIENT}"].append(
                compare_pair_scene(
                    split=split,
                    scene=scene,
                    model_a=SAM3,
                    model_b=EFFICIENT,
                    json_a=sam3_json,
                    json_b=efficient_json,
                    iou_threshold=args.bbox_iou_threshold,
                )
            )
            pair_results[f"{GROUNDED}_vs_{EFFICIENT}"].append(
                compare_pair_scene(
                    split=split,
                    scene=scene,
                    model_a=GROUNDED,
                    model_b=EFFICIENT,
                    json_a=grounded_json,
                    json_b=efficient_json,
                    iou_threshold=args.bbox_iou_threshold,
                )
            )

    if not summaries:
        print("No overlapping scenes found across all three models.")
        return 0

    print_three_way_table(summaries)
    print_pair_report("Pairwise: SAM3 vs GroundedSAM", pair_results[f"{SAM3}_vs_{GROUNDED}"])
    print_pair_report("Pairwise: SAM3 vs Efficient GroundedSAM", pair_results[f"{SAM3}_vs_{EFFICIENT}"])
    print_pair_report(
        "Pairwise: GroundedSAM vs Efficient GroundedSAM",
        pair_results[f"{GROUNDED}_vs_{EFFICIENT}"],
    )

    write_json(
        output_path=args.output_json,
        summaries=summaries,
        pair_results=pair_results,
        model_dirs={
            SAM3: str(args.sam3_dir),
            GROUNDED: str(args.grounded_dir),
            EFFICIENT: str(args.efficient_dir),
        },
        iou_threshold=args.bbox_iou_threshold,
    )
    print(f"\nDetailed JSON report saved to: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
