import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import type { Feature, FeatureCollection } from 'geojson';
import type { Detection, DetectorSettings, RasterMetadata } from '../types';
import {
  DEFAULT_CENTER,
  DEFAULT_ZOOM,
  HAS_MAPBOX_TOKEN,
  MAPBOX_TOKEN,
  TARGET_META,
} from '../config';

// Mapbox GL requires *some* token to initialise. When the operator has not
// supplied one we set a sentinel and swap in a token-free Esri raster style.
mapboxgl.accessToken = HAS_MAPBOX_TOKEN ? MAPBOX_TOKEN : 'no-token';

const ESRI_FALLBACK_STYLE: mapboxgl.StyleSpecification = {
  version: 8,
  sources: {
    'esri-imagery': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      attribution: 'Esri, Maxar, Earthstar Geographics, and the GIS User Community',
    },
  },
  layers: [{ id: 'esri-imagery', type: 'raster', source: 'esri-imagery' }],
};

const DETECTIONS_SOURCE = 'detections';
const FILL_LAYER = 'detections-fill';
const LINE_LAYER = 'detections-line';

interface MapCanvasProps {
  detections: Detection[];
  settings: DetectorSettings;
  raster: RasterMetadata | null;
}

/** Convert visible detections into a GeoJSON FeatureCollection for Mapbox. */
function toFeatureCollection(
  detections: Detection[],
  settings: DetectorSettings,
): FeatureCollection {
  const features: Feature[] = detections
    .filter(
      (d) =>
        d.confidence >= settings.confidence &&
        settings.enabledClasses.has(d.targetClass),
    )
    .map((d) => ({
      type: 'Feature',
      id: d.id,
      properties: {
        targetClass: d.targetClass,
        confidence: d.confidence,
        color: TARGET_META[d.targetClass].color,
      },
      geometry: { type: 'Polygon', coordinates: [d.polygon] },
    }));
  return { type: 'FeatureCollection', features };
}

export default function MapCanvas({ detections, settings, raster }: MapCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const [ready, setReady] = useState(false);
  const [cursor, setCursor] = useState<{ lat: number; lon: number } | null>(null);

  // Initialise the map once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: HAS_MAPBOX_TOKEN
        ? 'mapbox://styles/mapbox/satellite-streets-v12'
        : ESRI_FALLBACK_STYLE,
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      attributionControl: true,
    });
    mapRef.current = map;

    map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), 'top-right');
    map.addControl(new mapboxgl.ScaleControl({ unit: 'metric' }), 'bottom-left');

    map.on('mousemove', (e) => setCursor({ lat: e.lngLat.lat, lon: e.lngLat.lng }));
    map.on('mouseout', () => setCursor(null));

    map.on('load', () => {
      map.addSource(DETECTIONS_SOURCE, {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      map.addLayer({
        id: FILL_LAYER,
        type: 'fill',
        source: DETECTIONS_SOURCE,
        paint: { 'fill-color': ['get', 'color'], 'fill-opacity': 0.25 },
      });
      map.addLayer({
        id: LINE_LAYER,
        type: 'line',
        source: DETECTIONS_SOURCE,
        paint: { 'line-color': ['get', 'color'], 'line-width': 2 },
      });
      setReady(true);
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Push detection updates into the map source whenever inputs change.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const src = map.getSource(DETECTIONS_SOURCE) as mapboxgl.GeoJSONSource | undefined;
    src?.setData(toFeatureCollection(detections, settings));
  }, [detections, settings, ready]);

  // Fit the map to a newly-uploaded raster's footprint.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !raster?.bounds) return;
    const [w, s, e, n] = raster.bounds;
    map.fitBounds([[w, s], [e, n]], { padding: 48, duration: 800 });
  }, [raster, ready]);

  return (
    <div className="map-canvas">
      <div ref={containerRef} className="map-canvas__viewport" />

      {!HAS_MAPBOX_TOKEN && (
        <div className="map-canvas__badge" title="Using Esri World Imagery fallback">
          NO MAPBOX TOKEN · ESRI FALLBACK
        </div>
      )}

      <div className="map-canvas__readout">
        <span>
          LAT {cursor ? cursor.lat.toFixed(5) : '———'} · LON{' '}
          {cursor ? cursor.lon.toFixed(5) : '———'}
        </span>
        <span>GSD {raster?.gsdMeters ? `${raster.gsdMeters.toFixed(2)} m` : '—'}</span>
        <span>{raster?.crs ?? 'EPSG:4326'}</span>
      </div>
    </div>
  );
}
