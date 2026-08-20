r"""
FastAPI backend for the Tactical GEOINT Analyzer.

Exposes the geospatial inference pipeline at POST /api/infer — the exact contract
the front-end's src/services/inference.ts seam is written against.

Run (from project root, with the venv):
    .\.venv\Scripts\python.exe -m uvicorn ml.backend.app:app --reload --port 8000

The model loads lazily on first request. Device defaults to CPU so this never
competes with a training run for GPU VRAM; set GEOINT_DEVICE=0 once training is
done to serve from the GPU.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# make ml/src importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

import geo_inference as gi  # noqa: E402

DEVICE = os.environ.get("GEOINT_DEVICE", "cpu")

# ── Model registry ───────────────────────────────────────────────────────────
# Each selectable model, tagged by satellite / algorithm / training data so the
# UI can label it. Only entries whose weights exist on disk are offered. Adding a
# new model (e.g. a Sentinel-2 land-cover net) is just another dict here.
MODELS: list[dict] = [
    {
        "id": "dota-obb-scratch",
        "satellite": "High-res aerial",
        "algorithm": "YOLO11s-OBB",
        "training_data": "DOTAv1 (trained from scratch, 100 epochs)",
        "weights": ROOT / "runs" / "dota_obb_scratch" / "weights" / "best.pt",
    },
    {
        "id": "dota-obb-finetuned",
        "satellite": "High-res aerial",
        "algorithm": "YOLO11s-OBB",
        "training_data": "DOTAv1 (fine-tuned, partial run)",
        "weights": ROOT / "runs" / "dota_obb" / "weights" / "best.pt",
    },
    {
        "id": "dota-obb-pretrained",
        "satellite": "High-res aerial",
        "algorithm": "YOLO11s-OBB",
        "training_data": "DOTAv1 (stock pretrained)",
        "weights": ROOT / "yolo11s-obb.pt",
    },
]

# Training runs with a real per-epoch history to chart (Analytics tab). The
# broken batch=8 VRAM-thrashing attempt isn't a model anyone selects, so
# surfacing it would be confusing, not informative.
TRAINING_RUNS: list[dict] = [
    {
        "id": "dota-obb-scratch",
        "label": "DOTAv1 (trained from scratch, 100 epochs)",
        "csv": ROOT / "runs" / "dota_obb_scratch" / "results.csv",
    },
    {
        "id": "dota-obb-finetuned",
        "label": "DOTAv1 (fine-tuned, partial run)",
        "csv": ROOT / "runs" / "dota_obb" / "results.csv",
    },
]

# The third registered model (dota-obb-pretrained) is Ultralytics' own
# official yolo11s-obb.pt release -- never trained in this project, so there's
# no local results.csv / per-epoch curve for it. Its mAP50 below is the
# officially published benchmark (confirmed via Ultralytics' docs, mirrored on
# the Ultralytics/YOLO11 Hugging Face model card): 79.5 mAP50 at imgsz 1024.
# That number is measured on DOTA's held-out *test* split (labels withheld,
# scored via DOTA's own submission server) -- a different protocol from the
# *val*-split numbers the other two runs report here, and Ultralytics doesn't
# publicly document the epoch count used to produce it, so it's shown as a
# single reference point, not a fabricated per-epoch trajectory.
PRETRAINED_REFERENCE = {
    "id": "dota-obb-pretrained",
    "label": "DOTAv1 (stock pretrained, official Ultralytics release)",
    "map50": 0.795,
    "note": (
        "Officially published benchmark (mAP50 @ DOTA test split, imgsz 1024), "
        "not a locally-trained run -- Ultralytics doesn't publish a per-epoch "
        "curve or epoch count for this release, and the test split (vs. val) "
        "makes it not a strictly apples-to-apples comparison."
    ),
}

app = FastAPI(title="Tactical GEOINT Analyzer — Inference API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict[str, object] = {}  # model id -> loaded YOLO


def available_models() -> list[dict]:
    return [m for m in MODELS if Path(m["weights"]).exists()]


def _get_model(model_id: str | None = None):
    avail = available_models()
    if not avail:
        raise HTTPException(status_code=503, detail="No model weights found on disk.")
    by_id = {m["id"]: m for m in avail}
    chosen = by_id.get(model_id or "", avail[0])
    if chosen["id"] not in _cache:
        _cache[chosen["id"]] = gi.load_model(chosen["weights"], device=DEVICE)
    return _cache[chosen["id"]]


@app.get("/api/models")
def list_models() -> dict:
    avail = available_models()
    return {
        "models": [
            {k: m[k] for k in ("id", "satellite", "algorithm", "training_data")}
            for m in avail
        ],
        "default": avail[0]["id"] if avail else None,
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": DEVICE,
        "models_available": [m["id"] for m in available_models()],
        "models_loaded": list(_cache.keys()),
    }


def _read_training_csv(path: Path) -> list[dict]:
    """Ultralytics results.csv -> [{epoch, map50, map50_95, precision, recall}]."""
    rows: list[dict] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "epoch": int(r["epoch"]),
                "map50": float(r["metrics/mAP50(B)"]),
                "map50_95": float(r["metrics/mAP50-95(B)"]),
                "precision": float(r["metrics/precision(B)"]),
                "recall": float(r["metrics/recall(B)"]),
            })
    return rows


@app.get("/api/training-metrics")
def training_metrics() -> dict:
    """Per-epoch accuracy history for each trained model, for the Analytics tab."""
    runs = []
    for run in TRAINING_RUNS:
        if run["csv"].exists():
            runs.append({
                "id": run["id"],
                "label": run["label"],
                "epochs": _read_training_csv(run["csv"]),
            })
    return {"runs": runs, "reference": PRETRAINED_REFERENCE}


@app.get("/api/per-class-metrics")
def per_class_metrics() -> dict:
    """Per-DOTA-class AP50 for the default (from-scratch) model, for the
    Analytics tab's bar chart. Reads a cached JSON (see
    ml/src/eval_per_class.py) rather than re-running validation on every
    request -- a full val pass takes minutes."""
    path = ROOT / "runs" / "dota_obb_scratch" / "per_class_ap50.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Per-class metrics not yet computed. Run ml/src/eval_per_class.py.",
        )
    import json
    data = json.loads(path.read_text())
    for c in data["classes"]:
        c["target_class"] = gi.CLASS_MAP.get(c["name"], gi.DEFAULT_TARGET)
    return data


@app.post("/api/infer")
async def infer(
    raster: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou_nms: float = Form(0.45),
    classes: str = Form(""),  # comma-separated TargetClass values; empty = all
    visualization: str = Form("rgb"),  # "rgb" | "ir" for the display overlay
    model: str = Form(""),  # model id from /api/models; empty = default
) -> dict:
    """Accept a GeoTIFF + detector settings, return an AnalysisResult."""
    detector = _get_model(model)
    class_set = {c.strip() for c in classes.split(",") if c.strip()} or None

    suffix = Path(raster.filename or "upload.tif").suffix or ".tif"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await raster.read())
        tmp.flush()
        tmp.close()
        result = gi.run_pipeline(
            tmp.name,
            detector,
            confidence=confidence,
            iou_nms=iou_nms,
            classes=class_set,
            visualization=visualization,
        )
        # preserve the client-supplied filename in the response metadata
        result["raster"]["filename"] = raster.filename or result["raster"]["filename"]
        return result
    except HTTPException:
        raise
    except Exception as exc:  # surface pipeline errors as 500s with context
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/api/overlay")
async def overlay(
    raster: UploadFile = File(...),
    visualization: str = Form("rgb"),
) -> dict:
    """Regenerate just the display overlay for a raster (used by the RGB/IR toggle
    so switching bands doesn't require re-running detection)."""
    suffix = Path(raster.filename or "upload.tif").suffix or ".tif"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await raster.read())
        tmp.flush()
        tmp.close()
        return {"overlay": gi.overlay_from_path(tmp.name, visualization=visualization)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Overlay build failed: {exc}") from exc
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/api/snapshot")
async def snapshot(
    image: UploadFile = File(...),
    west: float = Form(...),
    south: float = Form(...),
    east: float = Form(...),
    north: float = Form(...),
) -> Response:
    """Turn a captured map-canvas PNG + its current view bounds into a real
    GeoTIFF, so a region panned/zoomed to on the live basemap can be fed
    straight into detection without a separate real-imagery file."""
    try:
        png_bytes = await image.read()
        tif_bytes = gi.snapshot_to_geotiff(png_bytes, (west, south, east, north))
        return Response(
            content=tif_bytes,
            media_type="image/tiff",
            headers={"Content-Disposition": 'attachment; filename="map-snapshot.tif"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Snapshot capture failed: {exc}") from exc


@app.post("/api/export/shapefile")
def export_shapefile(featurecollection: dict = Body(...)) -> Response:
    """Convert a detections GeoJSON FeatureCollection into a zipped ESRI Shapefile.

    Returns application/zip containing detections.shp/.shx/.dbf/.prj (EPSG:4326).
    Independent of the model, so it works whether or not weights are loaded.
    """
    import io
    import shutil
    import tempfile
    import zipfile

    import geopandas as gpd

    features = featurecollection.get("features", [])
    if not features:
        raise HTTPException(status_code=400, detail="No detections to export.")

    gdf = gpd.GeoDataFrame.from_features(features)
    # Shapefile DBF field names are capped at 10 chars; keep a controlled set.
    keep = [c for c in ("class", "confidence", "area_sqm", "dota_class") if c in gdf.columns]
    gdf = gdf[keep + ["geometry"]]
    gdf.set_crs("EPSG:4326", inplace=True, allow_override=True)

    tmp = tempfile.mkdtemp()
    try:
        gdf.to_file(Path(tmp) / "detections.shp", driver="ESRI Shapefile")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(os.listdir(tmp)):
                zf.write(os.path.join(tmp, f), f)
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="detections_shapefile.zip"'
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Shapefile export failed: {exc}") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
