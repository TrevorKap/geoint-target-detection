r"""
geo_inference.py — geospatial sliding-window inference for the Tactical GEOINT
Analyzer.

Turns a (possibly huge, possibly multi-band) GeoTIFF into GIS-ready detections:

    GeoTIFF ─▶ overlapping 1024px tiles (windowed reads, no full-raster load)
            ─▶ YOLO-OBB inference per tile
            ─▶ global pixel-space oriented polygons
            ─▶ stitched polygon-IoU NMS across tile seams (dedupe)
            ─▶ pixel ─▶ EPSG:4326 (lon/lat) via the raster's affine + CRS
            ─▶ AnalysisResult dict matching the front-end contract (types.ts)

Runnable standalone for testing:
    .\.venv\Scripts\python.exe ml\src\geo_inference.py <raster.tif> [--weights best.pt]
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import math
import os
import time
import uuid
from pathlib import Path

# ── PROJ/GDAL data shim ──────────────────────────────────────────────────────
# Some machines have a global PROJ_LIB/GDAL_DATA pointing at another install
# (e.g. PostgreSQL/PostGIS) with an incompatible proj.db, which breaks every CRS
# transform. Redirect to rasterio's bundled data *before* importing rasterio.
def _fix_proj_env() -> None:
    spec = importlib.util.find_spec("rasterio")
    if not spec or not spec.submodule_search_locations:
        return
    pkg = Path(spec.submodule_search_locations[0])
    proj = pkg / "proj_data"
    if (proj / "proj.db").exists():
        os.environ["PROJ_LIB"] = str(proj)
        os.environ["PROJ_DATA"] = str(proj)
    gdal = pkg / "gdal_data"
    if gdal.is_dir():
        os.environ["GDAL_DATA"] = str(gdal)


_fix_proj_env()

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from shapely.geometry import Polygon

# ── DOTAv1 classes (training order) ──────────────────────────────────────────
DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]

# DOTA class -> front-end TargetClass (see src/types.ts). Anything not
# militarily distinct is bucketed to "building" so the map/legend stay coherent.
CLASS_MAP = {
    "plane": "aircraft",
    "helicopter": "aircraft",
    "ship": "vessel",
    "large-vehicle": "vehicle",
    "small-vehicle": "vehicle",
    "storage-tank": "storage_tank",
    "harbor": "building",
    "bridge": "building",
    "roundabout": "building",
    "ground-track-field": "building",
    "soccer-ball-field": "building",
    "baseball-diamond": "building",
    "basketball-court": "building",
    "tennis-court": "building",
    "swimming-pool": "building",
}
DEFAULT_TARGET = "building"


# ── raster helpers ───────────────────────────────────────────────────────────
def _estimate_gsd_m(ds: rasterio.DatasetReader) -> float | None:
    """Ground sample distance in metres/pixel, best-effort from the transform."""
    if ds.crs is None:
        return None
    px = abs(ds.transform.a)
    if ds.crs.is_projected:  # units already metres
        return float(px)
    # geographic CRS (degrees) -> approximate metres at the raster's centre lat
    lat = (ds.bounds.bottom + ds.bounds.top) / 2.0
    return float(px * 111_320.0 * math.cos(math.radians(lat)))


def read_metadata(path: str | Path) -> dict:
    """RasterMetadata-shaped dict (matches src/types.ts)."""
    p = Path(path)
    with rasterio.open(p) as ds:
        bounds_wgs = None
        if ds.crs is not None:
            w, s, e, n = rasterio.warp.transform_bounds(
                ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21
            )
            bounds_wgs = [w, s, e, n]
        return {
            "filename": p.name,
            "sizeBytes": p.stat().st_size,
            "width": ds.width,
            "height": ds.height,
            "gsdMeters": _estimate_gsd_m(ds),
            "crs": str(ds.crs) if ds.crs else None,
            "acquired": None,  # populated from tags/EXIF in a later pass
            "bounds": bounds_wgs,
        }


def _iter_windows(width: int, height: int, tile: int, overlap: float):
    """Yield (x, y, w, h) tile windows with fractional overlap."""
    step = max(1, int(tile * (1.0 - overlap)))
    for y in range(0, height, step):
        for x in range(0, width, step):
            w = min(tile, width - x)
            h = min(tile, height - y)
            if w > 0 and h > 0:
                yield x, y, w, h
            if x + tile >= width:
                break
        if y + tile >= height:
            break


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """(bands, h, w) any dtype -> (h, w, 3) uint8 via 2–98% percentile stretch."""
    bands = arr.shape[0]
    if bands == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = arr[:3].astype(np.float32)
    out = np.empty_like(arr)
    for b in range(3):
        band = arr[b]
        lo, hi = np.percentile(band, (2, 98))
        if hi <= lo:
            hi = lo + 1.0
        out[b] = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255)
    return np.transpose(out, (1, 2, 0)).astype(np.uint8)


def build_overlay(
    ds: rasterio.DatasetReader, max_size: int = 1600, visualization: str = "rgb"
) -> dict | None:
    """Downsampled PNG of the raster + its EPSG:4326 corners, for map display.

    `visualization`:
      - "rgb": natural-colour, bands 1-3.
      - "ir":  false-colour NIR composite (NIR, Red, Green = bands 4,1,2), which
               makes vegetation pop red — but only if the raster actually has a
               4th (near-infrared) band. Otherwise it falls back to RGB and
               reports irApplied=False.

    Returns {image: data-URI, coordinates: [TL,TR,BR,BL] lon/lat, visualization,
    irApplied}. None if the raster isn't georeferenced (can't be placed on a map).
    """
    if ds.crs is None:
        return None
    scale = min(1.0, max_size / max(ds.width, ds.height))
    out_w = max(1, round(ds.width * scale))
    out_h = max(1, round(ds.height * scale))

    ir_applied = False
    if visualization == "ir" and ds.count >= 4:
        band_idx = [4, 1, 2]  # NIR, Red, Green false-colour composite
        ir_applied = True
    else:
        band_idx = list(range(1, min(3, ds.count) + 1))

    arr = ds.read(indexes=band_idx, out_shape=(len(band_idx), out_h, out_w))
    png = Image.fromarray(_to_uint8_rgb(arr), "RGB")
    buf = io.BytesIO()
    png.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("ascii")

    # raster corners in pixel space (upper-left of each): TL, TR, BR, BL
    cols = [0, ds.width, ds.width, 0]
    rows = [0, 0, ds.height, ds.height]
    xs, ys = rasterio.transform.xy(ds.transform, rows, cols, offset="ul")
    lons, lats = warp_transform(ds.crs, "EPSG:4326", xs, ys)
    coordinates = [[float(lo), float(la)] for lo, la in zip(lons, lats)]
    return {
        "image": f"data:image/png;base64,{data}",
        "coordinates": coordinates,
        "visualization": visualization,
        "irApplied": ir_applied,
    }


def overlay_from_path(path: str | Path, visualization: str = "rgb") -> dict | None:
    """Open a raster and build just its display overlay (for the /api/overlay seam)."""
    with rasterio.open(path) as ds:
        return build_overlay(ds, visualization=visualization)


# ── inference ────────────────────────────────────────────────────────────────
def load_model(weights: str | Path, device: str = "cpu"):
    """Lazy import so the module is importable without ultralytics/torch loaded."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.to(device)
    return model


def _infer_tile(model, img: np.ndarray, conf: float, iou: float, imgsz: int):
    """Return list of (poly(4,2) local px, cls_idx, confidence) for one tile."""
    res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    out = []
    if getattr(res, "obb", None) is not None and res.obb is not None and len(res.obb):
        polys = res.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
        clss = res.obb.cls.cpu().numpy().astype(int)
        confs = res.obb.conf.cpu().numpy()
        for poly, c, cf in zip(polys, clss, confs):
            out.append((poly, int(c), float(cf)))
    return out


def _polygon_nms(dets: list[dict], iou_thresh: float) -> list[dict]:
    """Greedy, class-aware NMS using true oriented-polygon IoU (shapely)."""
    order = sorted(range(len(dets)), key=lambda i: dets[i]["conf"], reverse=True)
    polys = []
    for d in dets:
        poly = Polygon(d["poly"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        polys.append(poly)

    suppressed = [False] * len(dets)
    keep = []
    for oi, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(dets[i])
        for j in order[oi + 1:]:
            if suppressed[j] or dets[i]["cls"] != dets[j]["cls"]:
                continue
            inter = polys[i].intersection(polys[j]).area
            if inter <= 0:
                continue
            union = polys[i].area + polys[j].area - inter
            if union > 0 and inter / union > iou_thresh:
                suppressed[j] = True
    return keep


def run_pipeline(
    path: str | Path,
    model,
    *,
    confidence: float = 0.25,
    iou_nms: float = 0.45,
    classes: set[str] | None = None,
    tile: int = 1024,
    overlap: float = 0.2,
    imgsz: int = 1024,
    visualization: str = "rgb",
) -> dict:
    """Full GeoTIFF -> AnalysisResult dict. `classes` filters by TargetClass."""
    t0 = time.perf_counter()
    meta = read_metadata(path)

    raw: list[dict] = []
    with rasterio.open(path) as ds:
        meta["overlay"] = build_overlay(ds, visualization=visualization)
        band_idx = list(range(1, min(3, ds.count) + 1))
        for x, y, w, h in _iter_windows(ds.width, ds.height, tile, overlap):
            arr = ds.read(indexes=band_idx, window=Window(x, y, w, h))
            img = _to_uint8_rgb(arr)
            for poly, cls, cf in _infer_tile(model, img, confidence, iou_nms, imgsz):
                raw.append({"poly": poly + np.array([x, y]), "cls": cls, "conf": cf})

        kept = _polygon_nms(raw, iou_nms)

        # project each polygon's pixel corners to EPSG:4326
        gsd = meta["gsdMeters"]
        detections = []
        for d in kept:
            dota_name = DOTA_CLASSES[d["cls"]] if d["cls"] < len(DOTA_CLASSES) else "?"
            target = CLASS_MAP.get(dota_name, DEFAULT_TARGET)
            if classes and target not in classes:
                continue

            cols = d["poly"][:, 0].tolist()
            rows = d["poly"][:, 1].tolist()
            if ds.crs is not None:
                xs, ys = rasterio.transform.xy(ds.transform, rows, cols)
                lons, lats = warp_transform(ds.crs, "EPSG:4326", xs, ys)
                ring = [[float(lo), float(la)] for lo, la in zip(lons, lats)]
            else:
                # no georeferencing: fall back to pixel coords (frontend can't map)
                ring = [[float(c), float(r)] for c, r in zip(cols, rows)]
            ring.append(ring[0])  # close the ring

            area = None
            if gsd:
                area = float(Polygon(d["poly"]).area * gsd * gsd)

            detections.append({
                "id": str(uuid.uuid4()),
                "targetClass": target,
                "confidence": round(float(d["conf"]), 4),
                "polygon": ring,
                "areaSqMeters": area,
                "attributes": {"dota_class": dota_name},
            })

    return {
        "raster": meta,
        "detections": detections,
        "inferenceMs": int((time.perf_counter() - t0) * 1000),
    }


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Geospatial YOLO-OBB inference")
    ap.add_argument("raster")
    ap.add_argument("--weights", default="runs/dota_obb/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = load_model(args.weights, device=args.device)
    result = run_pipeline(args.raster, model, confidence=args.conf, iou_nms=args.iou)
    print(f"{len(result['detections'])} detections in {result['inferenceMs']} ms")
    print(json.dumps(result["raster"], indent=2))


if __name__ == "__main__":
    _cli()
