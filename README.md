# Tactical GEOINT Analyzer

An end-to-end **Geospatial AI** web app for overhead target segmentation — parse
satellite rasters, segment fine-grained targets (aircraft, maritime vessels,
ground vehicles, structures), and export GIS-compliant vectors for C4ISR
pipelines.

> **Status:** Front-end framework complete. ML/DL detection backend is the next
> phase — see [ML Integration Seam](#ml-integration-seam).

## Stack

- **React 19 + TypeScript + Vite 8** — component-based tactical dashboard SPA
- **MapLibre GL JS** — interactive satellite map canvas (Esri World Imagery
  basemap), no API key or account required

## Getting started

```bash
npm install
npm run dev        # http://localhost:5173
```

## Architecture

```
src/
├─ App.tsx                  Top-level layout + app state (settings, raster, result)
├─ types.ts                 Domain contract: Detection, RasterMetadata, AnalysisResult…
├─ config.ts                Mapbox token detection, map defaults, per-class metadata
├─ components/
│  ├─ Header.tsx            Title bar + live pipeline status (idle/loaded/analyzing)
│  ├─ ControlPanel.tsx      GeoTIFF dropzone, target selection, confidence/IoU sliders,
│  │                        RGB / False-Color IR toggle, run/clear actions
│  ├─ MapCanvas.tsx         Mapbox GL map, detection polygon layers, LAT/LON/GSD readout
│  └─ AnalyticalSummary.tsx Target counts, class breakdown, footprint, GeoJSON export
├─ services/
│  └─ inference.ts          ⇦ ML INTEGRATION SEAM (see below)
└─ utils/
   └─ geojson.ts            AnalysisResult → spec-compliant GeoJSON + download
```

The UI is driven by a single `DetectorSettings` object and one `AnalysisResult`.
The confidence slider and target-class toggles filter detections **live** on the
map and in the summary without re-running inference.

## ML Integration Seam

Everything the model backend needs to fulfil lives in **one file**:
`src/services/inference.ts`.

```ts
runInference(file, raster, settings) => Promise<AnalysisResult>
```

Currently returns an empty result (no mock data) so the UI is fully exercised
against a not-yet-connected backend. The planned contract:

```
POST /api/infer   (multipart: raster, confidence, iou_nms, classes[])
  ↳ AnalysisResult   detections already projected to EPSG:4326 lon/lat
```

## Roadmap (ML/DL phase)

- [ ] Sliding-window tiling of large GeoTIFFs (512×512, ~20% overlap) with
      stitched NMS across tile boundaries (`Rasterio`).
- [ ] Model: YOLO-Seg / U-Net (+ EfficientNet backbone); YOLO-OBB for oriented
      boxes on ships & runways.
- [ ] 360° rotational + radiometric augmentation for overhead robustness.
- [ ] Multi-spectral indices (NDVI / NDWI) as extra input channels.
- [ ] Pixel → geographic projection via affine transform (GDAL/Rasterio).
- [ ] FastAPI + ONNX Runtime inference service behind `/api/infer`.
- [ ] Datasets: RarePlanes, xView, SpaceNet, DOTA.
```

