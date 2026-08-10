# ML / DL Pipeline

Model training + inference for the Tactical GEOINT Analyzer.

## Environment setup (Windows, one-time)

```powershell
powershell -ExecutionPolicy Bypass -File ml\setup_env.ps1
```

This creates `.venv`, installs **CUDA PyTorch (cu124)** for the RTX 3070 Ti, then
the rest of [requirements.txt](requirements.txt), and verifies `torch.cuda`.

Activate it later with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Layout

```
ml/
├─ requirements.txt   ML deps (torch installed separately, see setup_env.ps1)
├─ setup_env.ps1      venv + CUDA torch + deps installer
├─ configs/           dataset YAMLs + training configs
└─ src/               data-prep, training, inference scripts
```

## Inference backend (FastAPI)

The geospatial pipeline ([src/geo_inference.py](src/geo_inference.py)) does
sliding-window tiling over a GeoTIFF, YOLO-OBB inference per tile, stitched
polygon-IoU NMS across tile seams, and pixel→EPSG:4326 projection — returning an
`AnalysisResult` that matches the front-end contract in `src/types.ts`.

The API ([backend/app.py](backend/app.py)) serves it at `POST /api/infer`, the
exact seam the front-end's `src/services/inference.ts` targets.

Run it (from the project root, venv active or via the venv python):

```powershell
.\.venv\Scripts\python.exe -m uvicorn ml.backend.app:app --reload --port 8000
```

- `GET  /api/health` — weights path, device, load state
- `POST /api/infer`  — multipart `raster` + form `confidence`, `iou_nms`, `classes`

Env overrides: `GEOINT_WEIGHTS` (default `runs/dota_obb/weights/best.pt`),
`GEOINT_DEVICE` (default `cpu`; set to `0` to serve from the GPU after training).
The model loads lazily on the first request.

> A PROJ/GDAL shim in `geo_inference.py` redirects to rasterio's bundled proj.db,
> working around a system-wide `PROJ_LIB` pointing at an incompatible PostGIS
> install.

## Roadmap

1. **EDA + label conversion** — parse RarePlanes tiled GeoJSON/COCO, chart the
   class/role distribution, convert to YOLO format.
2. **Smoke test** — train YOLO-OBB on DOTA8 to validate the loop end-to-end.
3. **Train** — RarePlanes aircraft (YOLO-Seg) + DOTA (YOLO-OBB) on GPU.
4. **Geospatial inference** — sliding-window tiling over full GeoTIFFs, stitched
   NMS, pixel→EPSG:4326 projection, GeoJSON output.
5. **Serve** — FastAPI + ONNX Runtime behind `/api/infer` (the front-end's
   `src/services/inference.ts` seam).
```

