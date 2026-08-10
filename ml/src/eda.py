r"""
eda.py — exploratory data audit for the Tactical GEOINT Analyzer datasets.

Summarises class / role distributions and per-tile object density for:
  • RarePlanes (tiled COCO annotations)   -> aircraft roles, instances/tile
  • DOTAv1     (YOLO-OBB .txt labels)      -> 15-class object counts

No heavy deps: stdlib + matplotlib. Run after the env is set up:

    .\.venv\Scripts\python.exe ml\src\eda.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RP = ROOT / "data" / "raw" / "rareplanes"
DOTA = ROOT / "data" / "raw" / "dota" / "DOTAv1"
OUT = ROOT / "ml" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Ultralytics DOTAv1 class order.
DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]


def bar(counter: Counter, title: str, fname: str, *, top: int | None = None) -> None:
    items = counter.most_common(top)
    if not items:
        print(f"  (no data for {title})")
        return
    labels, values = zip(*items)
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(labels)), values, color="#ffb000")
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    path = OUT / fname
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  chart -> {path.relative_to(ROOT)}")


def audit_rareplanes_coco(coco_path: Path, split: str) -> None:
    if not coco_path.exists():
        print(f"[RarePlanes {split}] MISSING: {coco_path}")
        return
    data = json.loads(coco_path.read_text())
    anns = data.get("annotations", [])
    imgs = data.get("images", [])
    roles = Counter(str(a.get("role", "unknown")) for a in anns)
    per_img = Counter(a["image_id"] for a in anns)

    n_imgs = len(imgs) if imgs else len(set(per_img))
    tiles_with = len(per_img)
    print(f"\n[RarePlanes {split}]")
    print(f"  aircraft instances : {len(anns):,}")
    print(f"  tiles (images)     : {n_imgs:,}")
    print(f"  tiles w/ aircraft  : {tiles_with:,}")
    if per_img:
        dens = list(per_img.values())
        print(f"  aircraft/tile      : mean {sum(dens)/len(dens):.2f}, max {max(dens)}")
    print(f"  distinct roles     : {len(roles)}")
    for role, n in roles.most_common():
        print(f"    - {role:<40} {n:>6,}")
    bar(roles, f"RarePlanes {split}: aircraft role distribution",
        f"rareplanes_{split}_roles.png")


def audit_dota_yolo(labels_dir: Path, split: str) -> None:
    if not labels_dir.exists():
        print(f"[DOTAv1 {split}] MISSING: {labels_dir}")
        return
    cls_counter: Counter = Counter()
    per_img = []
    txts = list(labels_dir.glob("*.txt"))
    for t in txts:
        n = 0
        for line in t.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            idx = int(float(parts[0]))
            name = DOTA_CLASSES[idx] if 0 <= idx < len(DOTA_CLASSES) else f"cls_{idx}"
            cls_counter[name] += 1
            n += 1
        per_img.append(n)

    total = sum(cls_counter.values())
    print(f"\n[DOTAv1 {split}]")
    print(f"  label files        : {len(txts):,}")
    print(f"  total OBB instances: {total:,}")
    if per_img:
        print(f"  objects/image      : mean {sum(per_img)/len(per_img):.2f}, "
              f"max {max(per_img)}")
    for name, n in cls_counter.most_common():
        print(f"    - {name:<20} {n:>7,}")
    bar(cls_counter, f"DOTAv1 {split}: class distribution",
        f"dota_{split}_classes.png")


def main() -> None:
    print("=" * 68)
    print("GEOINT DATASET AUDIT")
    print("=" * 68)

    audit_rareplanes_coco(RP / "RarePlanes_Train_Coco_Annotations_tiled.json", "train")
    audit_rareplanes_coco(RP / "RarePlanes_Test_Coco_Annotations_tiled.json", "test")

    audit_dota_yolo(DOTA / "labels" / "train", "train")
    audit_dota_yolo(DOTA / "labels" / "val", "val")

    print("\nDone. Charts written to ml/outputs/.")


if __name__ == "__main__":
    main()
