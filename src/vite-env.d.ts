/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Mapbox public access token (`pk.…`). Optional; Esri fallback used if absent. */
  readonly VITE_MAPBOX_TOKEN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
