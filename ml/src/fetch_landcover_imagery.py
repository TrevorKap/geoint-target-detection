r"""
fetch_landcover_imagery.py — download AOI-clipped Sentinel-2 & Landsat imagery
for the built-up/impervious-surface comparison (step 2 of the pipeline).

Pulls the same San Diego AOI at each sensor's NATIVE resolution (Sentinel-2
10 m, Landsat 30 m) — the honest real-world comparison, not resampled to a
common grid. Surface-reflectance DN is rescaled to each sensor's documented
scale/offset so both are in comparable ~0-1 reflectance units.

    .\.venv\Scripts\python.exe ml\src\fetch_landcover_imagery.py
Output: data/raw/landcover/{sentinel2,landsat}/aoi_san_diego.tif  (4-band: R,G,B,NIR)
"""

from __future__ import annotations

from pathlib import Path

import geo_inference  # noqa: F401  (applies the PROJ/GDAL shim on import)
import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "landcover"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = [-117.25, 32.65, -117.05, 32.85]  # lon/lat, matches find_landcover_scenes.py
DATE_RANGE = "2024-05-01/2024-10-31"

# Fixed scene ids from find_landcover_scenes.py, for a reproducible pair.
S2_ID = "S2B_MSIL2A_20240909T182919_R027_T11SMS_20240909T224820"
LS_ID = "LC09_L2SP_040037_20241019_02_T1"


def fetch_item(catalog: Client, collection: str, item_id: str):
    return catalog.get_collection(collection).get_item(item_id)


def clipped_band(href: str, bbox_lonlat: list[float]) -> tuple[np.ndarray, dict]:
    with rasterio.open(href) as ds:
        west, south, east, north = transform_bounds("EPSG:4326", ds.crs, *bbox_lonlat)
        window = from_bounds(west, south, east, north, ds.transform)
        data = ds.read(1, window=window)
        transform = ds.window_transform(window)
        profile = ds.profile.copy()
        profile.update(height=data.shape[0], width=data.shape[1], transform=transform)
        return data, profile


def write_rgbn(bands: dict[str, np.ndarray], profile: dict, scale, offset, out: Path) -> None:
    order = ["red", "green", "blue", "nir"]
    stack = np.stack([bands[b] for b in order]).astype(np.float32)
    reflectance = np.clip(stack * scale + offset, 0, 1.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile.update(driver="GTiff", count=4, dtype="float32", compress="deflate")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(reflectance)
        dst.descriptions = tuple(order)


def main() -> None:
    catalog = Client.open(STAC_URL)

    print("Fetching Sentinel-2 …")
    s2_item = pc.sign(fetch_item(catalog, "sentinel-2-l2a", S2_ID))
    s2_bands = {}
    for name, key in [("red", "B04"), ("green", "B03"), ("blue", "B02"), ("nir", "B08")]:
        data, profile = clipped_band(s2_item.assets[key].href, BBOX)
        s2_bands[name] = data
        print(f"  {key} ({name}): {data.shape}")
    write_rgbn(s2_bands, profile, scale=1 / 10000, offset=0.0,
              out=OUT / "sentinel2" / "aoi_san_diego.tif")

    print("\nFetching Landsat …")
    ls_item = pc.sign(fetch_item(catalog, "landsat-c2-l2", LS_ID))
    ls_bands = {}
    for name, key in [("red", "red"), ("green", "green"), ("blue", "blue"), ("nir", "nir08")]:
        data, profile = clipped_band(ls_item.assets[key].href, BBOX)
        ls_bands[name] = data
        print(f"  {key} ({name}): {data.shape}")
    # Landsat Collection 2 Level-2 surface reflectance scale/offset (USGS docs).
    write_rgbn(ls_bands, profile, scale=0.0000275, offset=-0.2,
              out=OUT / "landsat" / "aoi_san_diego.tif")

    print(f"\nWrote:\n  {OUT / 'sentinel2' / 'aoi_san_diego.tif'}\n  {OUT / 'landsat' / 'aoi_san_diego.tif'}")


if __name__ == "__main__":
    main()
