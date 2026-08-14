r"""
make_samples.py — build "completely new" test GeoTIFFs for the app.

Picks RarePlanes tiles with the most aircraft and rewrites them as proper
GeoTIFFs (embedding their REAL georeferencing, which the PNG+sidecar tiles lose
on upload). RarePlanes is a different dataset/sensor (WorldView-3) than the DOTA
training data, so these are a genuine generalization test — and they land at
their true coordinates on the map.

    .\.venv\Scripts\python.exe ml\src\make_samples.py
Output: data/samples/*.tif
"""

from __future__ import annotations

import json
from pathlib import Path

import geo_inference  # noqa: F401  (applies the PROJ/GDAL shim on import)
import rasterio

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "data" / "raw" / "rareplanes" / "test"
OUT = ROOT / "data" / "samples"
N_SAMPLES = 5


def aircraft_count(geojson: Path) -> int:
    try:
        return len(json.loads(geojson.read_text()).get("features", []))
    except Exception:
        return 0


def main() -> None:
    geojsons = list(TEST.rglob("*.geojson"))
    if not geojsons:
        raise SystemExit(f"No RarePlanes tiled GeoJSON found under {TEST}")

    ranked = sorted(geojsons, key=aircraft_count, reverse=True)
    OUT.mkdir(parents=True, exist_ok=True)

    made = 0
    print(f"Scanned {len(geojsons)} tiles; writing top {N_SAMPLES} by aircraft count.\n")
    for gj in ranked:
        if made >= N_SAMPLES:
            break
        png = next(TEST.rglob(gj.stem + ".png"), None)
        if png is None:
            continue
        n = aircraft_count(gj)
        with rasterio.open(png) as ds:
            if ds.crs is None:
                continue  # need real georeferencing
            profile = ds.profile.copy()
            profile.update(driver="GTiff", compress="deflate", count=min(3, ds.count))
            data = ds.read(indexes=list(range(1, min(3, ds.count) + 1)))
            w, s, e, n_ = rasterio.warp.transform_bounds(
                ds.crs, "EPSG:4326", *ds.bounds
            )
        out = OUT / f"sample_{made + 1}_{gj.stem}.tif"
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data)
        made += 1
        print(f"  {out.name}")
        print(f"     {ds.width}x{ds.height}px · ~{n} aircraft · center "
              f"{(s + n_) / 2:.4f}, {(w + e) / 2:.4f}  ({ds.crs})")

    print(f"\nWrote {made} GeoTIFF sample(s) to {OUT}")


if __name__ == "__main__":
    main()
