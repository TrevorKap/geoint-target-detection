import { useEffect, useMemo, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { Feature, FeatureCollection } from 'geojson';
import type { Detection, DetectorSettings, RasterMetadata } from '../types';
import { DEFAULT_CENTER, DEFAULT_ZOOM, TARGET_META, TARGET_ORDER } from '../config';

// Renderer: MapLibre GL, the community fork of mapbox-gl v1 built for
// token-free, self-hosted use -- avoids the CORS-blocked telemetry noise
// mapbox-gl v2+ throws with no access token.
//
// The true root cause of the long black-map saga wasn't the renderer or the
// tile provider at all: index.css's .map-canvas__viewport rule (position:
// absolute; inset:0, meant to stretch the map to fill its parent) was losing
// a same-specificity CSS cascade fight against maplibre-gl.css's own
// .maplibregl-map { position: relative } base rule, so the container
// resolved to 0 height and every tile source rendered into an invisible box.
// Fixed via a higher-specificity selector in index.css. Esri World Imagery
// (real satellite/aerial photography) was never actually broken.
const SATELLITE_STYLE: maplibregl.StyleSpecification = {
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
  focusRequest: { id: string; seq: number } | null;
  onSnapshot?: (png: Blob, bounds: [number, number, number, number]) => void;
}

/** Detections passing the current confidence + class filters. */
function visibleDetections(detections: Detection[], settings: DetectorSettings) {
  return detections.filter(
    (d) =>
      d.confidence >= settings.confidence &&
      settings.enabledClasses.has(d.targetClass),
  );
}

/** Convert visible detections into a GeoJSON FeatureCollection for the map. */
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

/** Manual WebGL probe — MapLibre GL v6 dropped the built-in `supported()` check. */
function isWebGLSupported(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return !!(
      window.WebGLRenderingContext &&
      (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    );
  } catch {
    return false;
  }
}

/** Shared detail-popup markup for both map clicks and list focus. */
function detailPopupHTML(o: {
  label: string;
  glyph: string;
  confidence: number;
  color: string;
  area: number | null;
  dota: string | null;
}): string {
  const conf = `${(o.confidence * 100).toFixed(1)}%`;
  const area = o.area != null ? `${Math.round(o.area).toLocaleString()} m²` : '—';
  return `
    <div class="det-popup">
      <div class="det-popup__head" style="color:${o.color}">
        <span>${o.glyph}</span><span>${o.label}</span>
      </div>
      <dl class="det-popup__grid">
        <dt>Confidence</dt><dd>${conf}</dd>
        <dt>Footprint</dt><dd>${area}</dd>
        ${o.dota ? `<dt>Source class</dt><dd>${o.dota}</dd>` : ''}
      </dl>
    </div>`;
}

/**
 * Parse "lat, lon" / "lat lon" (optionally with a trailing ° symbol) into
 * [lon, lat] for map.flyTo. Returns null if the string isn't a coordinate
 * pair, so callers can fall back to place-name geocoding.
 */
function parseCoordinates(input: string): [number, number] | null {
  const m = input.trim().match(/^(-?\d+(?:\.\d+)?)\s*°?\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*°?$/);
  if (!m) return null;
  const lat = Number(m[1]);
  const lon = Number(m[2]);
  if (Number.isNaN(lat) || Number.isNaN(lon)) return null;
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;
  return [lon, lat];
}

/** Free, token-free place-name lookup via OpenStreetMap's Nominatim API. */
async function geocodePlace(query: string): Promise<[number, number] | null> {
  const url = `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(query)}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Geocoding failed (HTTP ${resp.status}).`);
  const results = (await resp.json()) as { lat: string; lon: string }[];
  if (results.length === 0) return null;
  return [Number(results[0].lon), Number(results[0].lat)];
}

export default function MapCanvas({
  detections,
  settings,
  raster,
  analyzing,
  focusRequest,
  onSnapshot,
}: MapCanvasProps) {
  // Latest detections, read by the focus effect without re-subscribing to them.
  const detectionsRef = useRef(detections);
  detectionsRef.current = detections;
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const hoveredRef = useRef<string | number | null>(null);
  const [ready, setReady] = useState(false);
  const [cursor, setCursor] = useState<{ lat: number; lon: number } | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [locationQuery, setLocationQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  // Surfaced instead of letting the map fail silently (a black canvas with no
  // explanation) -- WebGL/style/tile failures previously vanished entirely.
  const [mapError, setMapError] = useState<string | null>(null);

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

    // Fails silently otherwise: if WebGL is unavailable (disabled hardware
    // acceleration, an old/locked-down browser, a restrictive VM), the map
    // constructor either throws or produces a canvas that never paints --
    // with no indication why. Catch it up front instead.
    if (!isWebGLSupported()) {
      setMapError(
        'Your browser does not support WebGL, which the map requires. Try a ' +
          'different browser, or enable hardware acceleration in browser settings.',
      );
      return;
    }

    let map: maplibregl.Map;
    try {
      map = new maplibregl.Map({
        container: containerRef.current,
        style: SATELLITE_STYLE,
        center: DEFAULT_CENTER,
        zoom: DEFAULT_ZOOM,
        // Keep the WebGL buffer so the map can be screenshotted / exported as an
        // image (used for portfolio captures); negligible perf cost at this scale.
        canvasContextAttributes: { preserveDrawingBuffer: true },
      });
    } catch (e) {
      setMapError(`Map failed to initialize: ${e instanceof Error ? e.message : e}`);
      return;
    }
    mapRef.current = map;
    if (import.meta.env.DEV) {
      (window as unknown as { __geoMap?: maplibregl.Map }).__geoMap = map;
    }

    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
    map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');

    map.on('mousemove', (e) => setCursor({ lat: e.lngLat.lat, lon: e.lngLat.lng }));
    map.on('mouseout', () => setCursor(null));

    // Style/tile/WebGL failures surface as an 'error' event that nothing was
    // listening for previously -- the exact cause of a permanently black map
    // with zero feedback. Surface it.
    map.on('error', (e) => {
      const msg = e?.error?.message || 'Unknown map error';
      console.error('[MapCanvas] map error:', e.error);
      setMapError(`Map error: ${msg}`);
    });

    // If the style/tiles never finish loading (network/firewall/ad-blocker
    // blocking the tile host, etc.), 'load' never fires and the map just sits
    // there blank forever with no signal. Time it out instead.
    const loadTimeout = window.setTimeout(() => {
      if (!mapRef.current) return;
      setMapError((prev) =>
        prev ??
          'Map tiles are taking unusually long to load. This is usually a ' +
            'network/firewall/ad-blocker issue blocking the basemap tile server.',
      );
    }, 12000);

    map.on('load', () => {
      window.clearTimeout(loadTimeout);
      setMapError(null);
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

    // Belt-and-suspenders: sample real pixels once rendering has settled so a
    // WebGL canvas that "loads" but paints nothing (GPU/driver/compositing
    // issue) doesn't look identical to a genuinely working map.
    let pixelCheckDone = false;
    map.once('idle', () => {
      if (pixelCheckDone) return;
      pixelCheckDone = true;
      try {
        const glCanvas = map.getCanvas();
        const probe = document.createElement('canvas');
        probe.width = glCanvas.width;
        probe.height = glCanvas.height;
        const ctx2d = probe.getContext('2d');
        if (!ctx2d) return;
        ctx2d.drawImage(glCanvas, 0, 0);
        const { data } = ctx2d.getImageData(0, 0, probe.width, probe.height);
        let nonBlack = 0;
        for (let i = 0; i < data.length; i += 4 * 997) {
          if (data[i] > 8 || data[i + 1] > 8 || data[i + 2] > 8) nonBlack++;
        }
        if (nonBlack === 0) {
          setMapError(
            'The map loaded with no errors, but the display is rendering ' +
              'completely black -- this points to a GPU/graphics driver or ' +
              'remote-desktop compositing issue in this browser environment, ' +
              'not a data or network problem. Try a different browser, check ' +
              'that hardware acceleration is enabled, or test outside any ' +
              'remote-desktop/VM session.',
          );
        }
      } catch {
        // Canvas read can fail under some security contexts; not worth
        // surfacing a second, less-specific error over the ones above.
      }
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
      const html = detailPopupHTML(p);
      popupRef.current?.remove();
      popupRef.current = new maplibregl.Popup({
        closeButton: true,
        className: 'geoint-popup',
        maxWidth: '260px',
      })
        .setLngLat(e.lngLat)
        .setHTML(html)
        .addTo(map);
    });

    return () => {
      window.clearTimeout(loadTimeout);
      popupRef.current?.remove();
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Push detection updates into the map source whenever inputs change.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const src = map.getSource(DETECTIONS_SOURCE) as maplibregl.GeoJSONSource | undefined;
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
        paint: {
          'raster-opacity': settings.overlayOpacity,
          'raster-fade-duration': 0,
        },
      },
      map.getLayer(FILL_LAYER) ? FILL_LAYER : undefined,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raster, ready]);

  // Live-update overlay opacity without rebuilding the image source.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !map.getLayer(OVERLAY_LAYER)) return;
    map.setPaintProperty(OVERLAY_LAYER, 'raster-opacity', settings.overlayOpacity);
  }, [settings.overlayOpacity, ready]);

  // Focus a detection selected from the list: fly to it, highlight, and popup.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !focusRequest) return;
    const det = detectionsRef.current.find((d) => d.id === focusRequest.id);
    if (!det) return;

    const lons = det.polygon.map((p) => p[0]);
    const lats = det.polygon.map((p) => p[1]);
    const sw: [number, number] = [Math.min(...lons), Math.min(...lats)];
    const ne: [number, number] = [Math.max(...lons), Math.max(...lats)];
    // Keep padding well under the container's smallest dimension so fitBounds
    // never silently no-ops on a short viewport.
    map.fitBounds([sw, ne], { padding: 60, maxZoom: 19, duration: 700 });

    // move the hover highlight to the focused feature
    if (hoveredRef.current !== null) {
      map.setFeatureState(
        { source: DETECTIONS_SOURCE, id: hoveredRef.current },
        { hover: false },
      );
    }
    map.setFeatureState({ source: DETECTIONS_SOURCE, id: det.id }, { hover: true });
    hoveredRef.current = det.id;

    const meta = TARGET_META[det.targetClass];
    const html = detailPopupHTML({
      label: meta.label,
      glyph: meta.glyph,
      confidence: det.confidence,
      color: meta.color,
      area: det.areaSqMeters ?? null,
      dota: (det.attributes?.dota_class as string) ?? null,
    });
    popupRef.current?.remove();
    popupRef.current = new maplibregl.Popup({
      closeButton: true,
      className: 'geoint-popup',
      maxWidth: '260px',
    })
      .setLngLat([(sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2])
      .setHTML(html)
      .addTo(map);
  }, [focusRequest, ready]);

  // Jump the map to a typed coordinate pair or place name. Coordinates are
  // parsed locally; anything else goes through Nominatim (free, no API key).
  const handleLocationSearch = async () => {
    const map = mapRef.current;
    const query = locationQuery.trim();
    if (!map || !query || searching) return;
    setSearching(true);
    setSearchError(null);
    try {
      const coords = parseCoordinates(query) ?? (await geocodePlace(query));
      if (!coords) {
        setSearchError(`No results for "${query}".`);
        return;
      }
      map.flyTo({ center: coords, zoom: 16, duration: 1500 });
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Search failed.');
    } finally {
      setSearching(false);
    }
  };

  // Capture the current view (whatever the basemap is showing right now) as
  // a real georeferenced GeoTIFF, so a panned/zoomed-to region can be run
  // through detection without needing a separate real-imagery file.
  const handleCapture = () => {
    const map = mapRef.current;
    if (!map || capturing) return;
    setCapturing(true);

    const grab = () => {
      const b = map.getBounds();
      const bounds: [number, number, number, number] = [
        b.getWest(),
        b.getSouth(),
        b.getEast(),
        b.getNorth(),
      ];
      map.getCanvas().toBlob((blob) => {
        setCapturing(false);
        if (blob) onSnapshot?.(blob, bounds);
      }, 'image/png');
    };

    // A just-panned/zoomed view can still have tiles mid-load; capturing
    // immediately grabs whatever's rendered so far, including black
    // not-yet-loaded tile placeholders. Wait for the map to fully settle.
    if (map.areTilesLoaded()) {
      grab();
    } else {
      map.once('idle', grab);
    }
  };

  return (
    <div className="map-canvas">
      <div ref={containerRef} className="map-canvas__viewport" />

      <div className="map-canvas__badge" title="Esri World Imagery via MapLibre GL — no API key required">
        ESRI WORLD IMAGERY · NO API KEY
      </div>

      {onSnapshot && (
        <button
          type="button"
          className="map-canvas__capture"
          onClick={handleCapture}
          disabled={!ready || capturing || analyzing}
          title="Capture the current map view as a GeoTIFF for detection"
        >
          {capturing ? 'CAPTURING…' : '📸 CAPTURE VIEW'}
        </button>
      )}

      <form
        className="map-canvas__search"
        onSubmit={(e) => {
          e.preventDefault();
          handleLocationSearch();
        }}
      >
        <input
          type="text"
          className="map-canvas__search-input"
          placeholder="Coordinates (34.9, -117.9) or place name…"
          value={locationQuery}
          onChange={(e) => {
            setLocationQuery(e.target.value);
            setSearchError(null);
          }}
          disabled={!ready}
        />
        <button
          type="submit"
          className="map-canvas__search-btn"
          disabled={!ready || searching || !locationQuery.trim()}
        >
          {searching ? '…' : '🔍'}
        </button>
      </form>
      {searchError && <div className="map-canvas__search-error">{searchError}</div>}

      {mapError && (
        <div className="map-canvas__fault" role="alert">
          <strong>MAP FAILED TO LOAD</strong>
          <p>{mapError}</p>
          <p className="map-canvas__fault-hint">
            Check the browser console (F12) for more detail, or try disabling
            ad-blockers/privacy extensions for this page.
          </p>
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
