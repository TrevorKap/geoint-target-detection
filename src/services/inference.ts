import type { AnalysisResult, DetectorSettings, RasterMetadata } from '../types';

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

/** Liveness probe for the backend; used to show connection status in the UI. */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const resp = await fetch(`${API_BASE}/api/health`);
    return resp.ok;
  } catch {
    return false;
  }
}
