r"""
find_landcover_scenes.py — locate matching Sentinel-2 / Landsat scenes over an
AOI for the built-up/impervious-surface comparison (step 1 of the pipeline).

Queries Microsoft Planetary Computer's public STAC catalog for the lowest-cloud
Sentinel-2 L2A and Landsat Collection-2 L2 scenes over San Diego, in the same
season (for comparable vegetation/lighting), and confirms real asset access by
opening one signed band with rasterio.

    .\.venv\Scripts\python.exe ml\src\find_landcover_scenes.py
"""

from __future__ import annotations

import planetary_computer as pc
import rasterio
from pystac_client import Client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# San Diego: urban core + suburbs + bay, matching the app's default map view.
BBOX = [-117.25, 32.65, -117.05, 32.85]
DATE_RANGE = "2024-05-01/2024-10-31"  # dry season, low cloud likelihood


def best_scene(catalog: Client, collection: str, query: dict | None = None):
    search = catalog.search(
        collections=[collection],
        bbox=BBOX,
        datetime=DATE_RANGE,
        query=query or {},
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
        max_items=5,
    )
    items = list(search.items())
    return items[0] if items else None


def main() -> None:
    catalog = Client.open(STAC_URL)

    s2 = best_scene(catalog, "sentinel-2-l2a", {"eo:cloud_cover": {"lt": 10}})
    ls = best_scene(catalog, "landsat-c2-l2", {"eo:cloud_cover": {"lt": 10}})

    for name, item in [("Sentinel-2 L2A", s2), ("Landsat C2 L2", ls)]:
        if item is None:
            print(f"{name}: NO MATCH")
            continue
        signed = pc.sign(item)
        cloud = item.properties.get("eo:cloud_cover")
        date = item.properties.get("datetime")
        print(f"\n{name}: {item.id}")
        print(f"  date: {date} | cloud: {cloud}%")
        print(f"  assets: {list(signed.assets.keys())[:12]}")

        # confirm real access: open one band's signed COG
        band_key = next(
            (k for k in signed.assets if k.lower() in ("visual", "b04", "red", "swir22")),
            list(signed.assets.keys())[0],
        )
        href = signed.assets[band_key].href
        with rasterio.open(href) as ds:
            print(f"  opened '{band_key}': {ds.width}x{ds.height}, {ds.count} band(s), "
                  f"crs={ds.crs}, dtype={ds.dtypes[0]}")


if __name__ == "__main__":
    main()
