r"""
plot_metrics.py — publication-quality training charts for the LinkedIn writeup.

Reads the live Ultralytics results.csv and renders "accuracy vs epochs" graphs on
the app's dark tactical theme. Re-run any time (during or after training) to get
current curves.

    .\.venv\Scripts\python.exe ml\src\plot_metrics.py
    # options: --results <path> --out <dir> --total-epochs 100 --close-mosaic 10

Colours use the validated colourblind-safe dark palette (Okabe-Ito-adjacent),
verified with the dataviz palette validator (CVD ΔE >= 8, contrast >= 3:1).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

# ── validated dark palette ───────────────────────────────────────────────────
SURFACE = "#1a1a19"
INK = "#ffffff"
INK_DIM = "#c3c2b7"
GRID = "#2e2e2c"
AXIS = "#52514e"
BLUE, ORANGE, AQUA, YELLOW = "#3987e5", "#d95926", "#199e70", "#c98500"


def _style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", pad=10, loc="left")
    ax.set_xlabel(xlabel, color=INK_DIM, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_DIM, fontsize=10)
    ax.tick_params(colors=INK_DIM, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(AXIS)


def _mosaic_marker(ax, total_epochs: int, close_mosaic: int, xmax: float) -> None:
    """Mark where mosaic augmentation turns off (metrics jump after this)."""
    x = total_epochs - close_mosaic
    if x <= xmax:
        ax.axvline(x, color=INK_DIM, linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(x, ax.get_ylim()[1], "  mosaic off", color=INK_DIM, fontsize=8,
                va="top", ha="left")


def _end_label(ax, xs, ys, color: str) -> None:
    """Annotate the latest value in ink (identity stays with the coloured line)."""
    if xs:
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, 0), color=INK, fontsize=9, va="center")


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def col(rows, key):
    return [float(r[key]) for r in rows]


def plot_accuracy(rows, out: Path, total, close_mosaic) -> Path:
    ep = col(rows, "epoch")
    m50 = col(rows, "metrics/mAP50(B)")
    m5095 = col(rows, "metrics/mAP50-95(B)")

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    ax.plot(ep, m50, color=BLUE, linewidth=2.4, marker="o", markersize=4, label="mAP@50")
    ax.plot(ep, m5095, color=ORANGE, linewidth=2.4, marker="o", markersize=4,
            label="mAP@50-95")
    _style_ax(ax, "Detection accuracy improves with training",
              "Epoch", "Validation mAP")
    ax.set_xlim(left=min(ep), right=max(total, max(ep)))
    ax.set_ylim(0, max(0.9, max(m50) * 1.15))
    _mosaic_marker(ax, total, close_mosaic, max(ep))
    _end_label(ax, ep, m50, BLUE)
    _end_label(ax, ep, m5095, ORANGE)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK)
    fig.text(0.012, 0.02, "DOTAv1 · YOLO11s-OBB · RTX 3070 Ti", color=INK_DIM,
             fontsize=8)
    fig.tight_layout()
    p = out / "accuracy_vs_epochs.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return p


def plot_dashboard(rows, out: Path, total, close_mosaic) -> Path:
    ep = col(rows, "epoch")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=SURFACE)

    ax = axes[0, 0]
    ax.plot(ep, col(rows, "metrics/mAP50(B)"), color=BLUE, lw=2.2, label="mAP@50")
    ax.plot(ep, col(rows, "metrics/mAP50-95(B)"), color=ORANGE, lw=2.2,
            label="mAP@50-95")
    _style_ax(ax, "Accuracy (mAP) vs epoch", "Epoch", "mAP")
    _mosaic_marker(ax, total, close_mosaic, max(ep))
    ax.legend(loc="lower right", frameon=False, labelcolor=INK, fontsize=9)

    ax = axes[0, 1]
    ax.plot(ep, col(rows, "metrics/precision(B)"), color=BLUE, lw=2.2,
            label="Precision")
    ax.plot(ep, col(rows, "metrics/recall(B)"), color=ORANGE, lw=2.2, label="Recall")
    _style_ax(ax, "Precision & recall vs epoch", "Epoch", "Score")
    ax.legend(loc="lower right", frameon=False, labelcolor=INK, fontsize=9)

    ax = axes[1, 0]
    for key, c, lab in [("train/box_loss", BLUE, "box"), ("train/cls_loss", ORANGE, "cls"),
                        ("train/dfl_loss", AQUA, "dfl"), ("train/angle_loss", YELLOW, "angle")]:
        ax.plot(ep, col(rows, key), color=c, lw=2.0, label=lab)
    _style_ax(ax, "Training loss vs epoch", "Epoch", "Loss")
    ax.legend(loc="upper right", frameon=False, labelcolor=INK, fontsize=9, ncol=2)

    ax = axes[1, 1]
    for key, c, lab in [("val/box_loss", BLUE, "box"), ("val/cls_loss", ORANGE, "cls"),
                        ("val/dfl_loss", AQUA, "dfl"), ("val/angle_loss", YELLOW, "angle")]:
        ax.plot(ep, col(rows, key), color=c, lw=2.0, label=lab)
    _style_ax(ax, "Validation loss vs epoch", "Epoch", "Loss")
    ax.legend(loc="upper right", frameon=False, labelcolor=INK, fontsize=9, ncol=2)

    fig.suptitle("Tactical GEOINT Analyzer — YOLO-OBB training on DOTAv1",
                 color=INK, fontsize=15, fontweight="bold", x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = out / "training_dashboard.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "runs/dota_obb/results.csv"))
    ap.add_argument("--out", default=str(ROOT / "ml/outputs/charts"))
    ap.add_argument("--total-epochs", type=int, default=100)
    ap.add_argument("--close-mosaic", type=int, default=10)
    args = ap.parse_args()

    rows = load_rows(Path(args.results))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    p1 = plot_accuracy(rows, out, args.total_epochs, args.close_mosaic)
    p2 = plot_dashboard(rows, out, args.total_epochs, args.close_mosaic)
    last = rows[-1]
    print(f"epochs plotted: {len(rows)} (through epoch {last['epoch']})")
    print(f"latest mAP@50: {last['metrics/mAP50(B)']} | "
          f"mAP@50-95: {last['metrics/mAP50-95(B)']}")
    print(f"wrote:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
