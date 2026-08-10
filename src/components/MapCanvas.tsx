import { useEffect, useMemo, useRef, useState } from 'react';
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
  TARGET_ORDER,
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
const OVERLAY_SOURCE = 'raster-overlay';
const OVERLAY_LAYER = 'raster-overlay-layer';

interface MapCanvasProps {
  detections: Detection[];
  settings: DetectorSettings;
  raster: RasterMetadata | null;
  analyzing: boolean;
}

/** Detections passing the current confidence + class filters. */
function visibleDetections(detections: Detection[], settings: DetectorSettings) {
  return detections.filter(
    (d) =>
      d.confidence >= settings.confidence &&
      settings.enabledClasses.has(d.targetClass),
  );
}

/** Convert visible detections into a GeoJSON FeatureCollection for Mapbox. */
function toFeatureCollection(visible: Detection[]): FeatureCollection {
  const features: Feature[] = visible.map((d) => ({
    type: 'Feature',
    id: d.id,
    properties: {
      label: TARGET_META[d.targetClass].label,
      glyph: TARGET_META[d.targetClass].glyph,
      confidence: d.confidence,
      color: TARGET_META[d.targetClass].color,
      area: d.areaSqMeters ?? null,
      dota: (d.attributes?.dota_class as string) ?? null,
    },
    geometry: { type: 'Polygon', coordinates: [d.polygon] },
  }));
  return { type: 'FeatureCollection', features };
}

export default function MapCanvas({
  detections,
  settings,
  raster,
  analyzing,
}: MapCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const popupRef = useRef<mapboxgl.Popup | null>(null);
  const hoveredRef = useRef<string | number | null>(null);
  const [ready, setReady] = useState(false);
  const [cursor, setCursor] = useState<{ lat: number; lon: number } | null>(null);

  const visible = useMemo(
    () => visibleDetections(detections, settings),
    [detections, settings],
  );

  // Classes present in the current view, in stable order — drives the legend.
  const legendClasses = useMemo(() => {
    const present = new Set(visible.map((d) => d.targetClass));
    return TARGET_ORDER.filter((c) => present.has(c));
  }, [visible]);

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
      // Keep the WebGL buffer so the map can be screenshotted / exported as an
      // image (used for portfolio captures); negligible perf cost at this scale.
      preserveDrawingBuffer: true,
    });
    mapRef.current = map;
    if (import.meta.env.DEV) {
      (window as unknown as { __geoMap?: mapboxgl.Map }).__geoMap = map;
    }

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
        paint: {
          'fill-color': ['get', 'color'],
          'fill-opacity': [
            'case',
            ['boolean', ['feature-state', 'hover'], false],
            0.45,
            0.22,
          ],
        },
      });
      map.addLayer({
        id: LINE_LAYER,
        type: 'line',
        source: DETECTIONS_SOURCE,
        paint: {
          'line-color': ['get', 'color'],
          'line-width': [
            'case',
            ['boolean', ['feature-state', 'hover'], false],
            3.5,
            1.8,
          ],
        },
      });
      setReady(true);
    });

    // Interactivity: pointer cursor + hover highlight over detections.
    const setHover = (id: string | number | null) => {
      if (hoveredRef.current !== null) {
        map.setFeatureState(
          { source: DETECTIONS_SOURCE, id: hoveredRef.current },
          { hover: false },
        );
      }
      hoveredRef.current = id;
      if (id !== null) {
        map.setFeatureState({ source: DETECTIONS_SOURCE, id }, { hover: true });
      }
    };

    map.on('mousemove', FILL_LAYER, (e) => {
      map.getCanvas().style.cursor = 'pointer';
      const f = e.features?.[0];
      if (f && f.id != null) setHover(f.id);
    });
    map.on('mouseleave', FILL_LAYER, () => {
      map.getCanvas().style.cursor = '';
      setHover(null);
    });

    // Click a detection -> detail popup.
    map.on('click', FILL_LAYER, (e) => {
      const f = e.features?.[0];
      if (!f) return;
      const p = f.properties as {
        label: string;
        glyph: string;
        confidence: number;
        color: string;
        area: number | null;
        dota: string | null;
      };
      const conf = `${(p.confidence * 100).toFixed(1)}%`;
      const area = p.area != null ? `${Math.round(p.area).toLocaleString()} m²` : '—';
      const html = `
        <div class="det-popup">
          <div class="det-popup__head" style="color:${p.color}">
            <span>${p.glyph}</span><span>${p.label}</span>
          </div>
          <dl class="det-popup__grid">
            <dt>Confidence</dt><dd>${conf}</dd>
            <dt>Footprint</dt><dd>${area}</dd>
            ${p.dota ? `<dt>Source class</dt><dd>${p.dota}</dd>` : ''}
          </dl>
        </div>`;
      popupRef.current?.remove();
      popupRef.current = new mapboxgl.Popup({
        closeButton: true,
        className: 'geoint-popup',
        maxWidth: '260px',
      })
        .setLngLat(e.lngLat)
        .setHTML(html)
        .addTo(map);
    });

    return () => {
      popupRef.current?.remove();
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Push detection updates into the map source whenever inputs change.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const src = map.getSource(DETECTIONS_SOURCE) as mapboxgl.GeoJSONSource | undefined;
    src?.setData(toFeatureCollection(visible));
  }, [visible, ready]);

  // Fit the map to a newly-uploaded raster's footprint.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !raster?.bounds) return;
    const [w, s, e, n] = raster.bounds;
    map.fitBounds([[w, s], [e, n]], { padding: 48, duration: 800 });
  }, [raster, ready]);

  // Render the uploaded raster as a georeferenced image overlay, under the
  // detection layers so boxes stay on top. Rebuilt whenever the raster changes.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    if (map.getLayer(OVERLAY_LAYER)) map.removeLayer(OVERLAY_LAYER);
    if (map.getSource(OVERLAY_SOURCE)) map.removeSource(OVERLAY_SOURCE);

    const overlay = raster?.overlay;
    if (!overlay) return;
    map.addSource(OVERLAY_SOURCE, {
      type: 'image',
      url: overlay.image,
      coordinates: overlay.coordinates,
    });
    map.addLayer(
      {
        id: OVERLAY_LAYER,
        type: 'raster',
        source: OVERLAY_SOURCE,
        paint: { 'raster-opacity': 1, 'raster-fade-duration': 0 },
      },
      map.getLayer(FILL_LAYER) ? FILL_LAYER : undefined,
    );
  }, [raster, ready]);

  return (
    <div className="map-canvas">
      <div ref={containerRef} className="map-canvas__viewport" />

      {!HAS_MAPBOX_TOKEN && (
        <div className="map-canvas__badge" title="Using Esri World Imagery fallback">
          NO MAPBOX TOKEN · ESRI FALLBACK
        </div>
      )}

      {analyzing && (
        <div className="map-canvas__scanning" role="status" aria-live="polite">
          <div className="scanline" />
          <span className="map-canvas__scanning-label">RUNNING INFERENCE…</span>
        </div>
      )}

      {legendClasses.length > 0 && (
        <div className="map-legend" aria-label="Target legend">
          <span className="map-legend__title">TARGETS</span>
          {legendClasses.map((c) => {
            const meta = TARGET_META[c];
            const n = visible.filter((d) => d.targetClass === c).length;
            return (
              <div key={c} className="map-legend__row">
                <span
                  className="map-legend__swatch"
                  style={{ background: meta.color }}
                />
                <span className="map-legend__label">{meta.label}</span>
                <span className="map-legend__count">{n}</span>
              </div>
            );
          })}
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
