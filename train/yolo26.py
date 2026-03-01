from ultralytics import YOLO
import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset/labels")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO("yolo26.pt")
    model.train(
        data=args.dataset,
        epochs=args.epochs,
        batch=args.batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
