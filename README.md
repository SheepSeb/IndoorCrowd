# Yet Another Person Tracker

A machine learning pipeline for automatically annotating people in video data using multiple state-of-the-art segmentation models. The project extracts frames, auto-labels them with several models, compares results, and trains a YOLO-based person detector.

## Pipeline Overview

```
Raw Video
    ↓
Frame Extraction (sample_frames.py)
    ↓
Auto-labeling (SAM3 / GroundedSAM / GroundedEdgeSAM)
    ↓
Comparison & Review (compare_sam3_vs_groundedsam.py / review_annotations.py)
    ↓
Golden Dataset Selection (build_golden_mid_frames.py)
    ↓
YOLO Training (train/yolo26.py)
```

## Project Structure

```
.
├── labelling/
│   ├── sample_frames.py                      # Extract frames from video at fixed FPS
│   ├── autolabel_sam3.py                     # Auto-label with SAM3
│   ├── autolabel_grounding_sam_3fps.py       # Auto-label with GroundedSAM
│   ├── autolabel_grounded_edgesam_3fps.py    # Auto-label with GroundedEdgeSAM
│   ├── build_golden_mid_frames.py            # Select best frames for golden dataset
│   ├── review_annotations.py                 # Gradio UI to review/edit annotations
│   └── segment_with_fastsam.py              # Gradio UI for interactive segmentation
├── analysis/
│   ├── sample_rate.py                        # Estimate frame counts at different FPS
│   ├── compare_sam3_vs_groundedsam.py        # Compare annotation quality across models
│   └── reports/                             # Generated comparison JSON reports
├── train/
│   └── yolo26.py                            # YOLO training script
├── dataset/                                 # Frames, labels, and golden datasets
├── FastSAM-s.pt                             # FastSAM small model weights
├── FastSAM-x.pt                             # FastSAM extra-large model weights
└── pyproject.toml
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

A HuggingFace token is required for downloading SAM3. Create a `.env` file:

```
HF_TOKEN=your_token_here
```

PyTorch is sourced from CUDA 12.8 wheels. Adjust `pyproject.toml` if using a different CUDA version.

## Usage

### 1. Extract Frames

```bash
python labelling/sample_frames.py --fps 3 \
    --input dataset/raw_video \
    --output dataset/raw_frames_3fps
```

### 2. Auto-Label

**SAM3** (text-prompted):
```bash
python labelling/autolabel_sam3.py \
    --raw-frames-dir dataset/raw_frames_3fps \
    --output-dir dataset/labels_3fps \
    --text-prompt "person"
```

**GroundedSAM** (language-grounded):
```bash
python labelling/autolabel_grounding_sam_3fps.py \
    --raw-frames-dir dataset/raw_frames_3fps \
    --output-dir dataset/labels_grounding_sam_3fps
```

**GroundedEdgeSAM** (efficient variant):
```bash
python labelling/autolabel_grounded_edgesam_3fps.py \
    --raw-frames-dir dataset/raw_frames_3fps \
    --output-dir dataset/labels_efficient_grounded_sam_3fps
```

All auto-labelers output COCO-format JSON and support `--segmentation-format rle` or `polygon`.

### 3. Compare Model Outputs

```bash
python analysis/compare_sam3_vs_groundedsam.py
```

Generates reports in `analysis/reports/` with per-scene metrics including annotation counts, bounding box IoU, and mask IoU.

### 4. Review & Correct Annotations

Interactive Gradio UI for reviewing and editing annotations:

```bash
python labelling/review_annotations.py --annotation-source sam3_3fps
```

Available sources: `sam3_3fps`, `grounded_sam_3fps`, `efficient_grounded_sam_3fps`.

For manual segmentation from scratch:

```bash
python labelling/segment_with_fastsam.py \
    --images-dir dataset/raw_frames_3fps \
    --output-json dataset/my_labels.json
```

### 5. Build Golden Dataset

Select up to N frames, prioritizing scene centers:

```bash
python labelling/build_golden_mid_frames.py \
    --input dataset/raw_frames_3fps \
    --output dataset/golden_frames_3fps_mid \
    --max-frames 600
```

Use `--dry-run` to preview selection without copying files.

### 6. Train YOLO

```bash
python train/yolo26.py \
    --dataset dataset/labels_3fps_golden_mid \
    --epochs 100 \
    --batch-size 16
```

## Models

| Model | Size | Purpose |
|---|---|---|
| SAM3 | Downloaded via HuggingFace | Text-prompted segmentation |
| GroundedSAM | Bundled via autodistill | Language-grounded detection |
| GroundedEdgeSAM | Bundled via autodistill | Efficient language-grounded detection |
| FastSAM-s | 24 MB (included) | Interactive segmentation (fast) |
| FastSAM-x | 145 MB (included) | Interactive segmentation (accurate) |

## Dataset Format

All annotations use the [COCO format](https://cocodataset.org/#format-data) with one JSON file per scene. Masks are stored as either RLE or polygon segmentations.

```
dataset/
├── raw_frames_3fps/          # Extracted frames organized by scene
├── labels_3fps/              # SAM3 annotations
├── labels_grounding_sam_3fps/
├── labels_efficient_grounded_sam_3fps/
├── golden_frames_3fps_mid/   # Selected golden frames
├── labels_3fps_golden_mid/
├── labels_grounding_sam_3fps_golden_mid/
└── labels_efficient_grounded_sam_3fps_golden_mid/
```
