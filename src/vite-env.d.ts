/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Mapbox public access token (`pk.…`). Optional; Esri fallback used if absent. */
  readonly VITE_MAPBOX_TOKEN?: string;
  /** Inference backend base URL. Defaults to http://localhost:8000. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
