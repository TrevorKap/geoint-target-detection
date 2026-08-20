r"""
eval_per_class.py — per-DOTA-class AP50 for the default (from-scratch) model,
cached to JSON so the backend's /api/per-class-metrics endpoint doesn't have to
re-run a multi-minute validation pass on every request.

Run standalone (needs the GPU free -- stop the backend first if it's serving
on GEOINT_DEVICE=0, or this will contend for VRAM and slow to a crawl):
    .\.venv\Scripts\python.exe ml\src\eval_per_class.py
Output: runs/dota_obb_scratch/per_class_ap50.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "runs" / "dota_obb_scratch" / "weights" / "best.pt"
DATA_YAML = ROOT / "ml" / "configs" / "dota_v1_split.yaml"
OUT = ROOT / "runs" / "dota_obb_scratch" / "per_class_ap50.json"


def main() -> None:
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    metrics = model.val(data=str(DATA_YAML), split="val", device="0", verbose=False)
    names = model.names
    ap50 = metrics.box.ap50

    result = {
        "model_id": "dota-obb-scratch",
        "overall_map50": float(metrics.box.map50),
        "classes": [{"name": names[i], "ap50": float(a)} for i, a in enumerate(ap50)],
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUT}")
    for c in sorted(result["classes"], key=lambda c: -c["ap50"]):
        print(f"  {c['name']:20s} AP50={c['ap50']:.3f}")


if __name__ == "__main__":
    main()
