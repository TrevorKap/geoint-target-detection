r"""
eval_checkpoints.py — visual "more epochs = better detection" comparison.

Runs several training checkpoints (epoch0.pt, epoch10.pt, …, best.pt) over the
SAME scene and composes a side-by-side montage with oriented-box overlays, so you
can see detections get denser and tighter as training progresses. This is the
headline portfolio/LinkedIn visual.

By default it runs on CPU so it never competes with an in-progress training run
for GPU memory. Add --revalidate to also compute each checkpoint's val mAP
(slower; use --device 0 once training is done).

    .\.venv\Scripts\python.exe ml\src\eval_checkpoints.py                # auto scene
    .\.venv\Scripts\python.exe ml\src\eval_checkpoints.py --sample path\to\tile.png
    .\.venv\Scripts\python.exe ml\src\eval_checkpoints.py --revalidate --device 0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = ROOT / "runs" / "dota_obb" / "weights"
VAL_IMAGES = ROOT / "data" / "processed" / "dota_split" / "images" / "val"
VAL_LABELS = ROOT / "data" / "processed" / "dota_split" / "labels" / "val"
DATA_YAML = ROOT / "ml" / "configs" / "dota_v1_split.yaml"
OUT = ROOT / "ml" / "outputs" / "checkpoint_comparison"

# Milestone order for the montage (only those present are used).
DEFAULT_CKPTS = ["epoch0.pt", "epoch10.pt", "epoch20.pt", "epoch30.pt",
                 "epoch40.pt", "epoch50.pt", "epoch90.pt", "best.pt"]


def _epoch_sort_key(name: str) -> tuple[int, str]:
    m = re.search(r"epoch(\d+)", name)
    return (int(m.group(1)) if m else 10_000, name)


def pick_busiest_val_tile() -> Path:
    """Val image whose label file has the most objects — a lively demo scene."""
    best, best_n = None, -1
    for txt in VAL_LABELS.glob("*.txt"):
        n = sum(1 for _ in txt.open())
        if n > best_n:
            img = next(VAL_IMAGES.glob(txt.stem + ".*"), None)
            if img is not None:
                best, best_n = img, n
    if best is None:
        raise SystemExit(f"No val tiles found under {VAL_IMAGES}")
    return best


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def label_panel(rgb: np.ndarray, title: str, subtitle: str) -> Image.Image:
    """Annotated prediction image with a dark caption bar underneath."""
    img = Image.fromarray(rgb)
    w, h = img.size
    bar = 46
    canvas = Image.new("RGB", (w, h + bar), (14, 20, 28))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((10, h + 6), title, fill=(255, 176, 0), font=_font(20))
    d.text((10, h + 26), subtitle, fill=(159, 178, 198), font=_font(14))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="checkpoint filenames under --weights-dir")
    ap.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR),
                    help="directory containing the checkpoint .pt files")
    ap.add_argument("--sample", default=None, help="scene image; default = busiest val tile")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--revalidate", action="store_true", help="also compute val mAP (slow)")
    args = ap.parse_args()
    weights_dir = Path(args.weights_dir)

    from ultralytics import YOLO  # lazy import

    names = args.checkpoints or DEFAULT_CKPTS
    ckpts = sorted(
        [weights_dir / n for n in names if (weights_dir / n).exists()],
        key=lambda p: _epoch_sort_key(p.name),
    )
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {weights_dir}")

    sample = Path(args.sample) if args.sample else pick_busiest_val_tile()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Scene: {sample.name}")
    print(f"Checkpoints: {[c.name for c in ckpts]}")

    panels: list[Image.Image] = []
    for c in ckpts:
        model = YOLO(str(c))
        res = model.predict(str(sample), conf=args.conf, imgsz=1024,
                            device=args.device, verbose=False)[0]
        ndet = 0 if res.obb is None else len(res.obb)
        subtitle = f"{ndet} detections"
        if args.revalidate:
            m = model.val(data=str(DATA_YAML), split="val", imgsz=1024,
                          device=args.device, verbose=False)
            subtitle += f"  ·  mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}"
        rgb = res.plot()[..., ::-1]  # BGR -> RGB
        panels.append(label_panel(rgb, c.stem, subtitle))
        print(f"  {c.name}: {ndet} detections")

    # horizontal montage
    h = min(p.height for p in panels)
    scaled = [p.resize((round(p.width * h / p.height), h)) for p in panels]
    gap = 8
    total_w = sum(p.width for p in scaled) + gap * (len(scaled) - 1)
    montage = Image.new("RGB", (total_w, h), (7, 11, 16))
    x = 0
    for p in scaled:
        montage.paste(p, (x, 0))
        x += p.width + gap
    out_path = OUT / "checkpoint_montage.png"
    montage.save(out_path)
    print(f"\nWrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
