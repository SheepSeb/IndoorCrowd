"""
Train ultralytics YOLO / RT-DETR models on the full SAM3-annotated dataset.

Run prepare_data.py first to generate yolo_dataset/.

Examples
--------
# segmentation (primary use-case — SAM3 provides high-quality masks)
uv run python train/sam3/train.py --task segment --model yolo11n
uv run python train/sam3/train.py --task segment --model yolo11n yolo11s yolo11x

# detection
uv run python train/sam3/train.py --task detect --model yolo11n yolo26n rtdetr-l

# multiple models in sequence
uv run python train/sam3/train.py --task segment --model yolo11n yolo11s yolo26n
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO, RTDETR

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
DATA_DIR  = HERE / "yolo_dataset"

# Models that only support detection (no -seg variant)
DETECT_ONLY_PREFIXES = ("rtdetr",)

TASK_SUFFIX = {"detect": "", "segment": "-seg"}


def _resolve_weights(base_name: str, suffix: str) -> str:
    """Return an absolute path if the .pt lives in the repo root, else the bare name."""
    filename = f"{base_name}{suffix}.pt"
    local = REPO_ROOT / filename
    return str(local) if local.exists() else filename


def _model_class(base_name: str):
    if base_name.startswith("rtdetr"):
        return RTDETR
    return YOLO


def train_model(task: str, model_name: str, epochs: int, batch: int, imgsz: int) -> None:
    # RT-DETR and similar models only support detection
    if task == "segment" and any(model_name.startswith(p) for p in DETECT_ONLY_PREFIXES):
        print(f"[skip] {model_name} does not support segmentation — skipping.")
        return

    suffix  = TASK_SUFFIX[task]
    weights = _resolve_weights(model_name, suffix)
    data_yaml = DATA_DIR / task / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found — run prepare_data.py first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Task:    {task}")
    print(f"Model:   {weights}")
    print(f"Data:    {data_yaml}")
    print(f"Device:  {device}")
    print(f"{'='*60}\n")

    ModelCls = _model_class(model_name)
    model = ModelCls(weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(HERE / "runs" / task),
        name=model_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO/RT-DETR on SAM3 dataset")
    parser.add_argument(
        "--task",
        choices=["detect", "segment"],
        default="segment",
        help="detect or segment (default: segment)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["yolo11n"],
        help=(
            "Model base name(s), e.g. yolo11n yolo11s yolo26n rtdetr-l "
            "(default: yolo11n)"
        ),
    )
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--imgsz",   type=int, default=640)
    args = parser.parse_args()

    for model_name in args.model:
        train_model(args.task, model_name, args.epochs, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
