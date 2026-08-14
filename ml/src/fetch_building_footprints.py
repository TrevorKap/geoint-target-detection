r"""
fetch_building_footprints.py — download Microsoft Global ML Building Footprints
for the San Diego AOI (step 3 of the pipeline: labels for the impervious-surface
segmentation comparison).

The dataset is tiled by Bing quadkey; this finds which US quadkey tiles
intersect our AOI, downloads just those, and clips to the exact bbox.

    .\.venv\Scripts\python.exe ml\src\fetch_building_footprints.py
Output: data/raw/landcover/building_footprints_san_diego.geojson
"""

from __future__ import annotations

import csv
import gzip
import json
from io import BytesIO, TextIOWrapper
from pathlib import Path

import requests
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "landcover" / "building_footprints_san_diego.geojson"
INDEX_URL = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
BBOX = (-117.25, 32.65, -117.05, 32.85)  # matches fetch_landcover_imagery.py


def quadkey_to_bbox(qk: str) -> tuple[float, float, float, float]:
    """Bing quadkey -> (west, south, east, north) in lon/lat."""
    import math

    tile_x = tile_y = 0
    level = len(qk)
    for i in range(level):
        bit = level - i
        mask = 1 << (bit - 1)
        d = qk[i]
        if d == "1":
            tile_x |= mask
        elif d == "2":
            tile_y |= mask
        elif d == "3":
            tile_x |= mask
            tile_y |= mask
    n = 2**level
    west = tile_x / n * 360.0 - 180.0
    east = (tile_x + 1) / n * 360.0 - 180.0

    def y_to_lat(y):
        yy = 0.5 - y / n
        return 90.0 - 360.0 * math.atan(math.exp(-yy * 2 * math.pi)) / math.pi

    north = y_to_lat(tile_y)
    south = y_to_lat(tile_y + 1)
    return west, south, east, north


def bbox_intersects(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def main() -> None:
    print("Loading dataset index …")
    idx = requests.get(INDEX_URL, timeout=60).text
    rows = list(csv.DictReader(idx.splitlines()))
    us_rows = [r for r in rows if r["Location"] == "UnitedStates"]
    print(f"  {len(us_rows)} US quadkey tiles total")

    matches = []
    for r in us_rows:
        qb = quadkey_to_bbox(r["QuadKey"])
        if bbox_intersects(qb, BBOX):
            matches.append(r)
    print(f"  {len(matches)} tile(s) intersect the San Diego AOI")
    for m in matches:
        print(f"    quadkey={m['QuadKey']} size={m['Size']}")

    aoi = box(*BBOX)
    features = []
    for m in matches:
        print(f"  downloading {m['QuadKey']} …")
        raw = requests.get(m["Url"], timeout=120).content
        with gzip.open(BytesIO(raw)) as gz:
            for line in TextIOWrapper(gz, encoding="utf-8"):
                line = line.strip().rstrip(",")
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                geom = shape(rec["geometry"])
                if geom.intersects(aoi):
                    features.append({
                        "type": "Feature",
                        "properties": rec.get("properties", {}),
                        "geometry": rec["geometry"],
                    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print(f"\nWrote {len(features)} building footprints -> {OUT}")


if __name__ == "__main__":
    main()
