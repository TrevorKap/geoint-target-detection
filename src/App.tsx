import { useState } from 'react';
import Header from './components/Header';
import ControlPanel from './components/ControlPanel';
import MapCanvas from './components/MapCanvas';
import AnalyticalSummary from './components/AnalyticalSummary';
import type { AnalysisResult, DetectorSettings, RasterMetadata } from './types';
import { TARGET_ORDER } from './config';
import { extractMetadata, runInference } from './services/inference';

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
  };

  const handleRunAnalysis = async () => {
    if (!pendingFile || !raster) return;
    setAnalyzing(true);
    try {
      const res = await runInference(pendingFile, raster, settings);
      setResult(res);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleClear = () => {
    setPendingFile(null);
    setRaster(null);
    setResult(null);
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
          <MapCanvas
            detections={result?.detections ?? []}
            settings={settings}
            raster={raster}
          />
          <AnalyticalSummary result={result} settings={settings} />
        </main>
      </div>
    </div>
  );
}
