r"""
train_landcover.py — impervious-surface (buildings + pavement) segmentation,
trained identically on Sentinel-2 and Landsat imagery over the same San Diego
AOI. This is the actual "same model, all else equal, different sensor"
comparison: one small U-Net architecture, same hyperparameters/epochs/loss,
run once per sensor at each sensor's own native resolution.

    .\.venv\Scripts\python.exe ml\src\train_landcover.py --sensor sentinel2
    .\.venv\Scripts\python.exe ml\src\train_landcover.py --sensor landsat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
LC = ROOT / "data" / "raw" / "landcover"
RUNS = ROOT / "runs" / "landcover"

PATCH = 128
STRIDE = 64
VAL_FRACTION = 0.2  # rightmost slice of the AOI, held out spatially


# ── Data ─────────────────────────────────────────────────────────────────────
class TileDataset(Dataset):
    def __init__(self, image: np.ndarray, label: np.ndarray, split: str):
        # image: (4, H, W) float32 reflectance; label: (H, W) uint8 {0,1}
        h, w = label.shape
        xs = list(range(0, w - PATCH + 1, STRIDE))
        ys = list(range(0, h - PATCH + 1, STRIDE))
        if not xs or not ys:
            raise ValueError(f"image {w}x{h} is smaller than PATCH={PATCH}")
        # Index-based split (not pixel-fraction) so narrow grids (e.g. Landsat's
        # ~8 tile columns) still get a non-empty, spatially-separate val slice.
        val_x_from = max(1, round(len(xs) * (1 - VAL_FRACTION)))
        val_xs = set(xs[val_x_from:]) or {xs[-1]}
        self.coords = [
            (y, x) for y in ys for x in xs
            if (split == "val") == (x in val_xs)
        ]
        self.image, self.label = image, label

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        y, x = self.coords[i]
        img = self.image[:, y:y + PATCH, x:x + PATCH]
        lbl = self.label[y:y + PATCH, x:x + PATCH]
        return torch.from_numpy(img.copy()), torch.from_numpy(lbl.copy()).float()


def load_sensor(sensor: str) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(LC / sensor / "aoi_san_diego.tif") as ds:
        img = ds.read().astype("float32")
    with rasterio.open(LC / sensor / "label_san_diego.tif") as ds:
        lbl = ds.read(1)
    return img, lbl


# ── Model: small U-Net, 4-band in -> 1-band impervious-surface probability ──
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.b = ConvBlock(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def iou(logits, target, thresh=0.5, eps=1e-6):
    pred = (torch.sigmoid(logits) > thresh).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", required=True, choices=["sentinel2", "landsat"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    image, label = load_sensor(args.sensor)
    train_ds = TileDataset(image, label, "train")
    val_ds = TileDataset(image, label, "val")
    print(f"[{args.sensor}] train tiles: {len(train_ds)} | val tiles: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = SmallUNet(in_ch=image.shape[0]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = RUNS / args.sensor
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            opt.zero_grad()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(1, len(train_ds))

        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                logits = model(x)
                val_iou += iou(logits, y).item() * x.size(0)
        val_iou /= max(1, len(val_ds))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_iou": val_iou})
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), out_dir / "best.pt")
        torch.save(model.state_dict(), out_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | "
                  f"val IoU {val_iou:.4f} | best {best_iou:.4f}")

    print(f"\n[{args.sensor}] done. best val IoU = {best_iou:.4f} -> {out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
