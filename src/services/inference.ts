import type { AnalysisResult, DetectorSettings, RasterMetadata } from '../types';

/**
 * ─────────────────────────────────────────────────────────────────────────
 *  ML / DL INTEGRATION POINT
 * ─────────────────────────────────────────────────────────────────────────
 *  This is the single seam between the front-end and the detection backend.
 *  When we build the model pipeline, this function will POST the raster (or a
 *  server-side reference to it) plus the detector settings to a FastAPI /
 *  ONNX-Runtime endpoint and return the projected detections.
 *
 *  Expected backend contract (to be implemented later):
 *    POST /api/infer
 *      body: multipart { raster: File, confidence, iou_nms, classes[] }
 *      resp: AnalysisResult (detections already in EPSG:4326 lon/lat)
 *
 *  For now it returns an empty result so the UI stays fully functional
 *  against a not-yet-connected backend (no mock detections).
 */
export async function runInference(
  _file: File,
  raster: RasterMetadata,
  _settings: DetectorSettings,
): Promise<AnalysisResult> {
  // TODO(ml): replace with a real fetch() to the inference service.
  await new Promise((r) => setTimeout(r, 400)); // simulate round-trip latency
  return { raster, detections: [], inferenceMs: 0 };
}

/**
 * Extract raster metadata that is resolvable purely client-side. Full GeoTIFF
 * tag parsing (GSD, CRS, bounds via affine transform) will move server-side or
 * into a geotiff.js worker during the ML phase.
 */
export function extractMetadata(file: File): RasterMetadata {
  return {
    filename: file.name,
    sizeBytes: file.size,
    crs: undefined,
    gsdMeters: undefined,
    bounds: undefined,
  };
}
