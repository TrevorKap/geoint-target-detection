r"""
train_dota.py — train a YOLO-OBB oriented-object detector on tiled DOTAv1.

Overhead imagery has no canonical "up", so we enable full rotational and flip
augmentation for orientation invariance — a core requirement for satellite
targets. Defaults are tuned for an 8 GB GPU (RTX 3070 Ti Laptop).

Run (after ml/src/prepare_dota.py):
    .\.venv\Scripts\python.exe ml\src\train_dota.py --epochs 100 --model yolo11s-obb.pt

Resume an interrupted run:
    .\.venv\Scripts\python.exe ml\src\train_dota.py --resume runs/obb/dota_obb/weights/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "configs" / "dota_v1_split.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO-OBB on tiled DOTAv1")
    p.add_argument("--model", default="yolo11s-obb.pt", help="base model / weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8, help="-1 for auto-batch")
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="dota_obb")
    p.add_argument("--resume", default=None, help="path to last.pt to resume")
    p.add_argument(
        "--scratch",
        action="store_true",
        help="train from random init (no pretrained) to show the full learning "
        "curve; builds from the architecture YAML and writes to dota_obb_scratch",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    # From scratch: build from the architecture YAML (random weights) so the
    # mAP curve rises from ~0 and the checkpoint montage shows real learning.
    model = YOLO("yolo11s-obb.yaml") if args.scratch else YOLO(args.model)
    name = args.name if args.name != "dota_obb" or not args.scratch else "dota_obb_scratch"
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        pretrained=not args.scratch,
        project=str(ROOT / "runs"),
        name=name,
        exist_ok=True,
        # ── overhead-view augmentation (orientation invariance) ──
        degrees=180.0,   # full-range rotation
        fliplr=0.5,
        flipud=0.5,      # vertical flips are valid top-down
        scale=0.5,
        mosaic=1.0,
        close_mosaic=10,
        # ── logging / checkpoints ──
        patience=30,
        save_period=10,
        plots=True,
    )


if __name__ == "__main__":
    main()
