import { useState } from 'react';
import Header from './components/Header';
import ControlPanel from './components/ControlPanel';
import MapCanvas from './components/MapCanvas';
import AnalyticalSummary from './components/AnalyticalSummary';
import type { AnalysisResult, DetectorSettings, RasterMetadata } from './types';
import { TARGET_ORDER } from './config';
import { extractMetadata, runInference, InferenceError } from './services/inference';

const DEFAULT_SETTINGS: DetectorSettings = {
  confidence: 0.75,
  iouNms: 0.45,
  enabledClasses: new Set(TARGET_ORDER),
  visualization: 'rgb',
};

export default function App() {
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
    setSettings(DEFAULT_SETTINGS);
  };

  return (
    <div className="app">
      <Header status={status} />
      <div className="app__body">
        <ControlPanel
          settings={settings}
          onSettingsChange={handleSettingsChange}
          raster={raster}
          onFileSelected={handleFileSelected}
          onRunAnalysis={handleRunAnalysis}
          onClear={handleClear}
          analyzing={analyzing}
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
          <MapCanvas
            detections={result?.detections ?? []}
            settings={settings}
            raster={raster}
            analyzing={analyzing}
            focusRequest={focusRequest}
          />
          <AnalyticalSummary
            result={result}
            settings={settings}
            onFocus={handleFocusDetection}
          />
        </main>
      </div>
    </div>
  );
}
