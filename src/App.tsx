import { useEffect, useState } from 'react';
import Header from './components/Header';
import ControlPanel from './components/ControlPanel';
import MapCanvas from './components/MapCanvas';
import AnalyticalSummary from './components/AnalyticalSummary';
import AnalyticsPanel from './components/AnalyticsPanel';
import type {
  AnalysisResult,
  DetectorSettings,
  ModelInfo,
  RasterMetadata,
} from './types';
import { TARGET_ORDER } from './config';
import {
  extractMetadata,
  runInference,
  regenerateOverlay,
  fetchModels,
  captureSnapshot,
  InferenceError,
} from './services/inference';

const DEFAULT_SETTINGS: DetectorSettings = {
  confidence: 0.75,
  iouNms: 0.45,
  enabledClasses: new Set(TARGET_ORDER),
  visualization: 'rgb',
  overlayOpacity: 1,
  modelId: '',
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'detection' | 'analytics'>('detection');
  const [settings, setSettings] = useState<DetectorSettings>(DEFAULT_SETTINGS);
  const [raster, setRaster] = useState<RasterMetadata | null>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A monotonically-bumped request so re-selecting the same target re-focuses it.
  const [focusRequest, setFocusRequest] = useState<{ id: string; seq: number } | null>(
    null,
  );

  const handleFocusDetection = (id: string) =>
    setFocusRequest((prev) => ({ id, seq: (prev?.seq ?? 0) + 1 }));

  // Note shown when False-Color IR is requested but the raster has no NIR band.
  const [vizNote, setVizNote] = useState<string | null>(null);
  // Note shown when a successfully-analyzed raster has no embedded location
  // data, so there's nothing to place on the map -- the #1 cause of "I
  // uploaded a file and don't see anything."
  const [noOverlayNote, setNoOverlayNote] = useState<string | null>(null);

  // Available detection models (from the backend), and the current selection.
  const [models, setModels] = useState<ModelInfo[]>([]);
  useEffect(() => {
    fetchModels().then(({ models: list, default: def }) => {
      setModels(list);
      if (def) setSettings((prev) => ({ ...prev, modelId: prev.modelId || def }));
    });
  }, []);

  // Re-render the map overlay when the RGB/IR toggle changes (no re-inference).
  useEffect(() => {
    if (!pendingFile || !result) return;
    let cancelled = false;
    regenerateOverlay(pendingFile, settings.visualization)
      .then((overlay) => {
        if (cancelled) return;
        setRaster((prev) => (prev ? { ...prev, overlay: overlay ?? undefined } : prev));
        setVizNote(
          settings.visualization === 'ir' && overlay && overlay.irApplied === false
            ? 'No near-infrared band in this raster — showing RGB.'
            : null,
        );
      })
      .catch(() => {
        if (!cancelled) setVizNote('Could not rebuild the overlay (backend offline?).');
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.visualization]);

  const status: 'idle' | 'loaded' | 'analyzing' = analyzing
    ? 'analyzing'
    : raster
      ? 'loaded'
      : 'idle';

  const handleSettingsChange = (patch: Partial<DetectorSettings>) =>
    setSettings((prev) => ({ ...prev, ...patch }));

  const handleFileSelected = (file: File) => {
    setPendingFile(file);
    setRaster(extractMetadata(file));
    setResult(null);
    setError(null);
    setNoOverlayNote(null);
  };

  // Capture the live basemap view as a real GeoTIFF and stage it exactly like
  // a normal upload, so it can be run through detection immediately.
  const handleSnapshot = async (png: Blob, bounds: [number, number, number, number]) => {
    setError(null);
    try {
      const file = await captureSnapshot(png, bounds);
      handleFileSelected(file);
    } catch (err) {
      const msg =
        err instanceof InferenceError ? err.message : 'Could not capture the map view.';
      setError(msg);
    }
  };

  const handleRunAnalysis = async () => {
    if (!pendingFile || !raster) return;
    setAnalyzing(true);
    setError(null);
    try {
      const res = await runInference(pendingFile, raster, settings);
      setResult(res);
      // Adopt the backend's authoritative metadata (CRS, GSD, bounds) so the
      // map can fit to the raster footprint.
      setRaster(res.raster);
      setNoOverlayNote(
        res.raster.overlay
          ? null
          : `${res.raster.filename} has no embedded location data (not a ` +
              `georeferenced GeoTIFF), so it can't be shown on the map and ` +
              `detections may not be positioned correctly. Try one of the ` +
              `sample GeoTIFFs in data/samples/.`,
      );
      setVizNote(
        settings.visualization === 'ir' && res.raster.overlay?.irApplied === false
          ? 'No near-infrared band in this raster — showing RGB.'
          : null,
      );
    } catch (err) {
      const msg =
        err instanceof InferenceError
          ? err.message
          : 'Unexpected error running inference.';
      setError(msg);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleClear = () => {
    setPendingFile(null);
    setRaster(null);
    setResult(null);
    setError(null);
    setFocusRequest(null);
    setVizNote(null);
    setNoOverlayNote(null);
    setSettings((prev) => ({ ...DEFAULT_SETTINGS, modelId: prev.modelId }));
  };

  return (
    <div className="app">
      <Header status={status} />
      <nav className="app-tabs">
        <button
          type="button"
          className={`app-tabs__tab ${activeTab === 'detection' ? 'app-tabs__tab--on' : ''}`}
          onClick={() => setActiveTab('detection')}
        >
          🎯 Detection
        </button>
        <button
          type="button"
          className={`app-tabs__tab ${activeTab === 'analytics' ? 'app-tabs__tab--on' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          📈 Model Analytics
        </button>
      </nav>
      {activeTab === 'analytics' ? (
        <AnalyticsPanel />
      ) : (
        <div className="app__body">
          <ControlPanel
            settings={settings}
            onSettingsChange={handleSettingsChange}
            models={models}
            raster={raster}
            onFileSelected={handleFileSelected}
            onRunAnalysis={handleRunAnalysis}
            onClear={handleClear}
            analyzing={analyzing}
            vizNote={vizNote}
          />
          <main className="app__main">
            {error && (
              <div className="app__error" role="alert">
                <span className="app__error-tag">INFERENCE ERROR</span>
                <span>{error}</span>
                <button
                  type="button"
                  className="app__error-close"
                  onClick={() => setError(null)}
                  aria-label="Dismiss error"
                >
                  ✕
                </button>
              </div>
            )}
            {noOverlayNote && (
              <div className="app__notice" role="status">
                <span className="app__notice-tag">NO LOCATION DATA</span>
                <span>{noOverlayNote}</span>
                <button
                  type="button"
                  className="app__error-close"
                  onClick={() => setNoOverlayNote(null)}
                  aria-label="Dismiss notice"
                >
                  ✕
                </button>
              </div>
            )}
            <MapCanvas
              detections={result?.detections ?? []}
              settings={settings}
              raster={raster}
              analyzing={analyzing}
              focusRequest={focusRequest}
              onSnapshot={handleSnapshot}
            />
            <AnalyticalSummary
              result={result}
              settings={settings}
              onFocus={handleFocusDetection}
            />
          </main>
        </div>
      )}
    </div>
  );
}
