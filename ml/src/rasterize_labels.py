r"""
rasterize_labels.py — combine building footprints + pavement into a binary
"impervious surface" mask, rasterized separately onto EACH sensor's own native
grid (step 5 of the pipeline). This is what makes the Sentinel-2 vs Landsat
comparison fair: same real-world label, projected at each sensor's true
resolution rather than resampled to a shared grid.

    .\.venv\Scripts\python.exe ml\src\rasterize_labels.py
Output: data/raw/landcover/{sentinel2,landsat}/label_san_diego.tif
"""

from __future__ import annotations

import json
from pathlib import Path

import geo_inference  # noqa: F401  (PROJ/GDAL shim)
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
LC = ROOT / "data" / "raw" / "landcover"
BUILDINGS = LC / "building_footprints_san_diego.geojson"
PAVEMENT = LC / "pavement_san_diego.geojson"


def load_geoms(path: Path) -> list:
    fc = json.loads(path.read_text())
    geoms = []
    for f in fc["features"]:
        g = shape(f["geometry"])
        if g.is_valid and not g.is_empty:
            geoms.append(g)
    return geoms


def rasterize_for(image_path: Path, geoms_4326: list, out_path: Path) -> None:
    with rasterio.open(image_path) as ds:
        crs, transform, shape_hw = ds.crs, ds.transform, (ds.height, ds.width)
        profile = ds.profile.copy()

    # Reproject each geometry from EPSG:4326 into the image's own CRS.
    reprojected = [
        shape(transform_geom("EPSG:4326", crs, g.__geo_interface__))
        for g in geoms_4326
    ]
    mask = rasterize(
        [(g, 1) for g in reprojected if g.is_valid and not g.is_empty],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    profile.update(driver="GTiff", count=1, dtype="uint8", compress="deflate", nodata=None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)
    pct = 100 * mask.mean()
    print(f"  {out_path.name}: {mask.shape}, {pct:.1f}% impervious")


def main() -> None:
    print("Loading labels …")
    geoms = load_geoms(BUILDINGS) + load_geoms(PAVEMENT)
    print(f"  {len(geoms)} total geometries (buildings + pavement)")

    for sensor in ("sentinel2", "landsat"):
        img = LC / sensor / "aoi_san_diego.tif"
        out = LC / sensor / "label_san_diego.tif"
        print(f"\nRasterizing for {sensor} …")
        rasterize_for(img, geoms, out)


if __name__ == "__main__":
    main()
