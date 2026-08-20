// Shared domain types for the Tactical GEOINT Analyzer.
// These mirror the contract the ML/DL backend will fulfil later, so the
// front-end can be wired against real inference output with minimal changes.

/**
 * Fine-grained target classes the detector is expected to emit. Kept in sync
 * with the DOTA-class -> TargetClass mapping in ml/src/geo_inference.py
 * (CLASS_MAP / DEFAULT_TARGET) -- only classes something can actually map to
 * belong here, so the UI never offers a filter that can never match anything.
 */
export type TargetClass =
  | 'aircraft'
  | 'vessel'
  | 'vehicle'
  | 'building'
  | 'storage_tank';

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
  /**
   * A downsampled, georeferenced preview of the raster for display on the map.
   * `image` is a PNG data URI; `coordinates` are the four corners as [lon, lat]
   * in Mapbox image-source order: top-left, top-right, bottom-right, bottom-left.
   */
  overlay?: {
    image: string;
    coordinates: [
      [number, number],
      [number, number],
      [number, number],
      [number, number],
    ];
    /** Which composite this overlay was rendered with. */
    visualization?: 'rgb' | 'ir';
    /** True only if a real NIR band was used for the false-colour composite. */
    irApplied?: boolean;
  };
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
  /** Opacity of the uploaded-raster overlay on the map, [0, 1]. */
  overlayOpacity: number;
  /** Selected detection model id (from GET /api/models). */
  modelId: string;
}

/** A selectable detection model, tagged by sensor / algorithm / training data. */
export interface ModelInfo {
  id: string;
  satellite: string;
  algorithm: string;
  training_data: string;
}

/** One epoch's metrics from a training run (ultralytics results.csv). */
export interface TrainingEpoch {
  epoch: number;
  map50: number;
  map50_95: number;
  precision: number;
  recall: number;
}

/** A full training run's per-epoch history, for the Analytics tab. */
export interface TrainingRun {
  id: string;
  label: string;
  epochs: TrainingEpoch[];
}

/** A single-point accuracy reference (e.g. an officially published pretrained
 * benchmark with no local per-epoch history to chart). */
export interface TrainingReference {
  id: string;
  label: string;
  map50: number;
  note: string;
}

/** One DOTA class's AP50, plus which app TargetClass it maps to. */
export interface PerClassMetric {
  name: string;
  ap50: number;
  target_class: TargetClass;
}

/** Live cursor position readout for the map footer. */
export interface CursorPosition {
  lat: number;
  lon: number;
}
