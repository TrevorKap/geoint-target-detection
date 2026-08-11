import type { AnalysisResult, DetectorSettings, RasterMetadata } from '../types';
import { toGeoJSON } from '../utils/geojson';

/**
 * ─────────────────────────────────────────────────────────────────────────
 *  ML / DL INTEGRATION POINT
 * ─────────────────────────────────────────────────────────────────────────
 *  The single seam between the front-end and the detection backend. Posts the
 *  uploaded raster + detector settings to the FastAPI inference service and
 *  returns detections already projected to EPSG:4326 (lon/lat).
 *
 *    POST {API_BASE}/api/infer
 *      body: multipart { raster: File, confidence, iou_nms, classes }
 *      resp: AnalysisResult
 *
 *  Backend: ml/backend/app.py  (run: uvicorn ml.backend.app:app --port 8000)
 */

/** Inference API base URL; override with VITE_API_URL in .env. */
const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

/** Error carrying an HTTP status for the UI to distinguish "backend down" etc. */
export class InferenceError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = 'InferenceError';
    this.status = status;
  }
}

export async function runInference(
  file: File,
  raster: RasterMetadata,
  settings: DetectorSettings,
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append('raster', file, file.name);
  form.append('confidence', String(settings.confidence));
  form.append('iou_nms', String(settings.iouNms));
  form.append('classes', Array.from(settings.enabledClasses).join(','));
  form.append('visualization', settings.visualization);

  let resp: Response;
  try {
    resp = await fetch(`${API_BASE}/api/infer`, { method: 'POST', body: form });
  } catch {
    throw new InferenceError(
      `Could not reach the inference backend at ${API_BASE}. Is it running? ` +
        `(uvicorn ml.backend.app:app --port 8000)`,
    );
  }

  if (!resp.ok) {
    // FastAPI errors come back as { detail: "..." }
    let detail = `Inference failed (HTTP ${resp.status}).`;
    try {
      const body = (await resp.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body; keep the generic message */
    }
    throw new InferenceError(detail, resp.status);
  }

  const result = (await resp.json()) as AnalysisResult;
  // Prefer the client-known filename/size if the backend omitted them.
  result.raster = { ...raster, ...result.raster };
  return result;
}

/**
 * Extract raster metadata resolvable purely client-side, for immediate staging
 * in the UI before inference runs. The backend returns authoritative metadata
 * (CRS, GSD, bounds) in its AnalysisResult.
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

/**
 * Rebuild just the display overlay for a different band composite (RGB vs
 * false-colour IR), so the toggle doesn't require re-running detection.
 * Returns the new overlay, or null if the raster isn't georeferenced.
 */
export async function regenerateOverlay(
  file: File,
  visualization: 'rgb' | 'ir',
): Promise<RasterMetadata['overlay'] | null> {
  const form = new FormData();
  form.append('raster', file, file.name);
  form.append('visualization', visualization);
  let resp: Response;
  try {
    resp = await fetch(`${API_BASE}/api/overlay`, { method: 'POST', body: form });
  } catch {
    throw new InferenceError(`Could not reach the backend at ${API_BASE}.`);
  }
  if (!resp.ok) throw new InferenceError(`Overlay build failed (HTTP ${resp.status}).`);
  const body = (await resp.json()) as { overlay: RasterMetadata['overlay'] | null };
  return body.overlay;
}

/**
 * Export the (already-filtered) detections as a zipped ESRI Shapefile via the
 * backend, and trigger a browser download. Requires the backend to be running.
 */
export async function exportShapefile(result: AnalysisResult): Promise<void> {
  const fc = toGeoJSON(result);
  let resp: Response;
  try {
    resp = await fetch(`${API_BASE}/api/export/shapefile`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(fc),
    });
  } catch {
    throw new InferenceError(
      `Could not reach the backend at ${API_BASE} to build the Shapefile.`,
    );
  }
  if (!resp.ok) {
    let detail = `Shapefile export failed (HTTP ${resp.status}).`;
    try {
      const body = (await resp.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      /* keep generic message */
    }
    throw new InferenceError(detail, resp.status);
  }

  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const base = result.raster.filename.replace(/\.[^.]+$/, '') || 'detections';
  a.href = url;
  a.download = `${base}_shapefile.zip`;
  a.click();
  URL.revokeObjectURL(url);
}

/** Liveness probe for the backend; used to show connection status in the UI. */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const resp = await fetch(`${API_BASE}/api/health`);
    return resp.ok;
  } catch {
    return false;
  }
}
