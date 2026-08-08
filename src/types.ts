// Shared domain types for the Tactical GEOINT Analyzer.
// These mirror the contract the ML/DL backend will fulfil later, so the
// front-end can be wired against real inference output with minimal changes.

/** Fine-grained target classes the detector is expected to emit. */
export type TargetClass =
  | 'aircraft'
  | 'vessel'
  | 'vehicle'
  | 'building'
  | 'storage_tank'
  | 'helipad';

/** A single model detection, already projected into geographic space. */
export interface Detection {
  id: string;
  targetClass: TargetClass;
  /** Model confidence in [0, 1]. */
  confidence: number;
  /**
   * Detection geometry as a GeoJSON Polygon ring in [lon, lat] order
   * (EPSG:4326). Instance masks are simplified to their exterior polygon
   * for display; oriented bounding boxes are a 5-point closed ring.
   */
  polygon: [number, number][];
  /** Estimated real-world footprint in square metres. */
  areaSqMeters?: number;
  /** Free-form fine-grained attributes (e.g. wingspan_m, role, engine_count). */
  attributes?: Record<string, string | number>;
}

/** Sensor / raster metadata parsed from an uploaded GeoTIFF. */
export interface RasterMetadata {
  filename: string;
  sizeBytes: number;
  /** Width / height in pixels, if resolvable client-side. */
  width?: number;
  height?: number;
  /** Ground Sample Distance in metres/pixel. */
  gsdMeters?: number;
  /** Coordinate reference system, e.g. "EPSG:32611". */
  crs?: string;
  /** Acquisition timestamp, ISO-8601. */
  acquired?: string;
  /** Geographic bounds [west, south, east, north] for map fit. */
  bounds?: [number, number, number, number];
}

/** The full result of an inference run over one raster. */
export interface AnalysisResult {
  raster: RasterMetadata;
  detections: Detection[];
  /** Milliseconds of wall-clock inference time. */
  inferenceMs?: number;
}

/** Detector control parameters bound to the UI sliders / toggles. */
export interface DetectorSettings {
  confidence: number; // [0, 1]
  iouNms: number; // [0, 1]
  enabledClasses: Set<TargetClass>;
  visualization: 'rgb' | 'ir';
}

/** Live cursor position readout for the map footer. */
export interface CursorPosition {
  lat: number;
  lon: number;
}
