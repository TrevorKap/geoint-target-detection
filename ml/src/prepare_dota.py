r"""
prepare_dota.py — tile DOTAv1 into fixed-size windows for OBB training.

DOTA images are large and highly variable (median ~1900 px, up to 6500 px), so
training on them whole starves small objects. This crops them into overlapping
1024x1024 windows — the same sliding-window strategy used at inference time —
using Ultralytics' DOTA splitter, which correctly re-projects oriented boxes and
drops/clips boxes at tile borders.

Input  : data/raw/dota/DOTAv1/{images,labels}/{train,val}   (normalized YOLO-OBB)
Output : data/processed/dota_split/{images,labels}/{train,val}

Run:
    .\.venv\Scripts\python.exe ml\src\prepare_dota.py
"""

from __future__ import annotations

from pathlib import Path

from ultralytics.data.split_dota import split_trainval

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "dota" / "DOTAv1"
DST = ROOT / "data" / "processed" / "dota_split"

CROP_SIZE = 1024
GAP = 200  # ~20% overlap between adjacent tiles


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    print(f"Splitting DOTAv1 -> {DST}")
    print(f"  crop_size={CROP_SIZE}, gap={GAP} (overlap {GAP/CROP_SIZE:.0%})")
    split_trainval(
        data_root=str(SRC),
        save_dir=str(DST),
        crop_size=CROP_SIZE,
        gap=GAP,
        rates=(1.0,),  # single scale; add (0.5, 1.0, 1.5) for multiscale later
    )
    for split in ("train", "val"):
        n = len(list((DST / "images" / split).glob("*")))
        print(f"  {split}: {n} tiles")
    print("Done.")


if __name__ == "__main__":
    main()
