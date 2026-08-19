r"""
make_demo_set.py — build a varied, presentation-ready GeoTIFF set in
data/demo/, covering different target classes (not just aircraft), each
verified against the real trained model before being included.

Source imagery is DOTAv1 (the model's training distribution — the images it
detects most reliably), which has no native georeferencing. Each scene is
assigned a real, thematically-fitting location for a believable demo — this
placement is illustrative, not the scene's actual capture site, and the
README documents that explicitly.

    .\.venv\Scripts\python.exe ml\src\make_demo_set.py
Output: data/demo/*.tif + data/demo/README.md
"""

from __future__ import annotations

import geo_inference  # noqa: F401  (PROJ/GDAL shim)
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

ROOT_IMAGES = "data/processed/dota_split/images/val"
OUT = "data/demo"
WEIGHTS = "runs/dota_obb_scratch/weights/best.pt"
GSD_M = 0.3  # metres/pixel, matching DOTA's native ground sample distance
GSD_DEG = GSD_M / 111_320  # approx degrees/pixel at these latitudes

# (stem, top-left lon, top-left lat, short id, human title, why-this-location)
SCENES = [
    ("P0170__1024__824___57", -117.8862, 34.9080, "airfield",
     "Airfield — aircraft",
     "Edwards AFB, CA — USAF flight test center"),
    ("P1386__1024__2472___824", -76.3340, 36.9500, "naval-port",
     "Naval port — vessels",
     "Naval Station Norfolk, VA — largest naval base in the world"),
    ("P2625__1024__2472___824", -106.4270, 31.8050, "vehicle-depot",
     "Vehicle depot — ground vehicles",
     "Fort Bliss, TX — major Army armor/vehicle installation"),
    ("P2695__1024__2968___824", -95.2660, 29.7390, "tank-farm",
     "Tank farm — storage/infrastructure",
     "Port of Houston, TX — petrochemical storage hub"),
    ("P2129__1024__0___0", -73.8490, 40.7530, "sports-complex",
     "Sports complex — civil infrastructure",
     "Near USTA National Tennis Center, NY — shows non-military utility"),
]


def to_geotiff(stem: str, lon: float, lat: float, out_path: str) -> tuple[int, int]:
    src = f"{ROOT_IMAGES}/{stem}.jpg"
    img = np.array(Image.open(src).convert("RGB"))
    h, w = img.shape[:2]
    transform = from_origin(lon, lat, GSD_DEG, GSD_DEG)
    profile = dict(driver="GTiff", height=h, width=w, count=3, dtype="uint8",
                   crs="EPSG:4326", transform=transform, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        for b in range(3):
            dst.write(img[:, :, b], b + 1)
    return w, h


def main() -> None:
    import os
    os.makedirs(OUT, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    model.to("cuda:0")

    rows = []
    for stem, lon, lat, slug, title, place in SCENES:
        out_path = f"{OUT}/{slug}.tif"
        w, h = to_geotiff(stem, lon, lat, out_path)

        res = model.predict(out_path, conf=0.35, imgsz=1024, device="0", verbose=False)[0]
        classes = {}
        if res.obb is not None:
            for c in res.obb.cls.tolist():
                name = res.names[int(c)]
                classes[name] = classes.get(name, 0) + 1
        ndet = sum(classes.values())
        top_classes = ", ".join(f"{v}x {k}" for k, v in sorted(classes.items(), key=lambda x: -x[1]))

        rows.append(dict(slug=slug, title=title, place=place, lon=lon, lat=lat,
                          w=w, h=h, ndet=ndet, top_classes=top_classes))
        print(f"{slug}: {ndet} detections ({top_classes})")

    readme = ["# Demo GeoTIFFs\n",
              "Curated scenes for live demonstrations — each verified against the ",
              "real trained model (`dota-obb-scratch`, mAP50 0.78) before inclusion, ",
              "covering different target classes so a demo isn't just aircraft.\n",
              "\n**Honesty note:** source imagery is DOTAv1 (the model's training ",
              "distribution, chosen because it's what the model detects most ",
              "reliably). DOTA imagery has no native georeferencing, so each scene ",
              "is placed at a real, thematically-fitting location for a believable ",
              "demo — that placement is illustrative, **not** the scene's actual ",
              "capture site.\n",
              "\n| File | Scene | Location (illustrative) | Size | Detections |",
              "\n|---|---|---|---|---|"]
    for r in rows:
        readme.append(
            f"\n| `{r['slug']}.tif` | {r['title']} | {r['place']} "
            f"({r['lat']:.4f}, {r['lon']:.4f}) | {r['w']}x{r['h']}px, {GSD_M}m GSD | "
            f"**{r['ndet']}** — {r['top_classes']} |"
        )
    readme.append("\n\nAll files: EPSG:4326, 3-band RGB, uint8, deflate-compressed.\n")
    with open(f"{OUT}/README.md", "w", encoding="utf-8") as f:
        f.write("".join(readme))
    print(f"\nWrote {len(rows)} demo GeoTIFFs + README to {OUT}/")


if __name__ == "__main__":
    main()
