import type { FeatureCollection } from 'geojson';
import type { AnalysisResult, Detection } from '../types';

/** Build a spec-compliant GeoJSON FeatureCollection from an analysis result. */
export function toGeoJSON(result: AnalysisResult): FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: result.detections.map((d: Detection) => ({
      type: 'Feature',
      id: d.id,
      properties: {
        class: d.targetClass,
        confidence: Number(d.confidence.toFixed(4)),
        area_sqm: d.areaSqMeters ?? null,
        ...d.attributes,
      },
      geometry: { type: 'Polygon', coordinates: [d.polygon] },
    })),
    // Non-standard but widely-read metadata block for downstream GIS tools.
    ...({
      metadata: {
        source: result.raster.filename,
        crs: result.raster.crs ?? 'EPSG:4326',
        gsd_m: result.raster.gsdMeters ?? null,
        acquired: result.raster.acquired ?? null,
        detection_count: result.detections.length,
      },
    } as object),
  };
}

/** Trigger a browser download of the analysis as a `.geojson` file. */
export function downloadGeoJSON(result: AnalysisResult): void {
  const blob = new Blob([JSON.stringify(toGeoJSON(result), null, 2)], {
    type: 'application/geo+json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const base = result.raster.filename.replace(/\.[^.]+$/, '') || 'detections';
  a.href = url;
  a.download = `${base}_detections.geojson`;
  a.click();
  URL.revokeObjectURL(url);
}
