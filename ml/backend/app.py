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

import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# make ml/src importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

import geo_inference as gi  # noqa: E402

WEIGHTS = Path(os.environ.get("GEOINT_WEIGHTS", ROOT / "runs/dota_obb/weights/best.pt"))
DEVICE = os.environ.get("GEOINT_DEVICE", "cpu")

app = FastAPI(title="Tactical GEOINT Analyzer — Inference API", version="0.1.0")

# Vite dev server origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None  # lazy singleton


def _get_model():
    global _model
    if _model is None:
        if not WEIGHTS.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model weights not found at {WEIGHTS}. Training may still "
                       f"be running; set GEOINT_WEIGHTS to a valid .pt file.",
            )
        _model = gi.load_model(WEIGHTS, device=DEVICE)
    return _model


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "weights": str(WEIGHTS),
        "weights_present": WEIGHTS.exists(),
        "device": DEVICE,
        "model_loaded": _model is not None,
    }


@app.post("/api/infer")
async def infer(
    raster: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou_nms: float = Form(0.45),
    classes: str = Form(""),  # comma-separated TargetClass values; empty = all
) -> dict:
    """Accept a GeoTIFF + detector settings, return an AnalysisResult."""
    model = _get_model()
    class_set = {c.strip() for c in classes.split(",") if c.strip()} or None

    suffix = Path(raster.filename or "upload.tif").suffix or ".tif"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await raster.read())
        tmp.flush()
        tmp.close()
        result = gi.run_pipeline(
            tmp.name,
            model,
            confidence=confidence,
            iou_nms=iou_nms,
            classes=class_set,
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
