r"""
fetch_osm_pavement.py — OSM roads + parking for the San Diego AOI, combined with
building footprints into the "impervious surface" label (step 4 of the pipeline).

Roads are buffered by an approximate real-world width (in metres, via a local
UTM reprojection) so they read as area, not lines.

    .\.venv\Scripts\python.exe ml\src\fetch_osm_pavement.py
Output: data/raw/landcover/pavement_san_diego.geojson  (polygons, EPSG:4326)
"""

from __future__ import annotations

import json
from pathlib import Path

import geo_inference  # noqa: F401  (PROJ/GDAL shim)
import requests
from pyproj import Transformer
from shapely.geometry import LineString, Polygon, mapping, shape
from shapely.ops import transform as shp_transform, unary_union

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "landcover" / "pavement_san_diego.geojson"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
BBOX = (-117.25, 32.65, -117.05, 32.85)  # west, south, east, north
UTM_CRS = "EPSG:32611"  # matches the downloaded imagery

# Approximate paved width in metres by highway class.
WIDTH_BY_HIGHWAY = {
    "motorway": 18, "trunk": 15, "primary": 12, "secondary": 10,
    "tertiary": 9, "residential": 7, "service": 5, "unclassified": 6,
    "footway": 3, "pedestrian": 4, "cycleway": 3, "track": 4,
}


def main() -> None:
    south, west, north, east = BBOX[1], BBOX[0], BBOX[3], BBOX[2]
    query = f"""
    [out:json][timeout:180];
    (
      way["highway"]({south},{west},{north},{east});
      way["amenity"="parking"]({south},{west},{north},{east});
      way["landuse"="parking"]({south},{west},{north},{east});
    );
    out geom;
    """
    print("Querying Overpass …")
    headers = {
        "User-Agent": "TacticalGEOINTAnalyzer/0.1 (portfolio project; contact: n/a)",
        "Accept": "application/json, */*;q=0.5",
    }
    resp = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=200)
    resp.raise_for_status()
    elements = resp.json()["elements"]
    print(f"  {len(elements)} OSM ways")

    to_utm = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True).transform
    to_wgs = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True).transform

    polys = []
    n_roads = n_parking = 0
    for el in elements:
        coords = [(g["lon"], g["lat"]) for g in el.get("geometry", [])]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        line_utm = shp_transform(to_utm, LineString(coords))
        if tags.get("amenity") == "parking" or tags.get("landuse") == "parking":
            if len(coords) >= 4:
                poly = Polygon(line_utm)
                if poly.is_valid and poly.area > 0:
                    polys.append(poly)
                    n_parking += 1
                    continue
        width = WIDTH_BY_HIGHWAY.get(tags.get("highway"), 6)
        polys.append(line_utm.buffer(width / 2))
        n_roads += 1

    print(f"  roads buffered: {n_roads} | parking polygons: {n_parking}")
    merged = unary_union(polys)
    merged_wgs = shp_transform(to_wgs, merged)
    geoms = list(merged_wgs.geoms) if merged_wgs.geom_type == "MultiPolygon" else [merged_wgs]

    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(g)} for g in geoms],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fc))
    print(f"\nWrote {len(geoms)} pavement polygon(s) -> {OUT}")


if __name__ == "__main__":
    main()
