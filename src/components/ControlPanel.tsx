import { useRef, useState } from 'react';
import type { DetectorSettings, RasterMetadata, TargetClass } from '../types';
import { TARGET_META, TARGET_ORDER } from '../config';

interface ControlPanelProps {
  settings: DetectorSettings;
  onSettingsChange: (patch: Partial<DetectorSettings>) => void;
  raster: RasterMetadata | null;
  onFileSelected: (file: File) => void;
  onRunAnalysis: () => void;
  onClear: () => void;
  analyzing: boolean;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(1)} ${units[i]}`;
}

export default function ControlPanel({
  settings,
  onSettingsChange,
  raster,
  onFileSelected,
  onRunAnalysis,
  onClear,
  analyzing,
}: ControlPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = (files: FileList | null) => {
    if (files && files.length > 0) onFileSelected(files[0]);
  };

  const toggleClass = (cls: TargetClass) => {
    const next = new Set(settings.enabledClasses);
    if (next.has(cls)) next.delete(cls);
    else next.add(cls);
    onSettingsChange({ enabledClasses: next });
  };

  return (
    <aside className="control-panel">
      <section className="panel-section">
        <h2 className="panel-section__title">⚙ Controls</h2>

        <label className="panel-label">Upload GeoTIFF / Select Area</label>
        <div
          className={`dropzone ${dragOver ? 'dropzone--active' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".tif,.tiff,.geotiff,image/tiff"
            hidden
            onChange={(e) => handleFiles(e.target.files)}
          />
          {raster ? (
            <div className="dropzone__file">
              <span className="dropzone__filename">{raster.filename}</span>
              <span className="dropzone__meta">{formatBytes(raster.sizeBytes)}</span>
            </div>
          ) : (
            <span className="dropzone__hint">Drop file here…</span>
          )}
        </div>
      </section>

      <section className="panel-section">
        <label className="panel-label">Target Selection</label>
        <div className="target-grid">
          {TARGET_ORDER.map((cls) => {
            const meta = TARGET_META[cls];
            const on = settings.enabledClasses.has(cls);
            return (
              <button
                key={cls}
                type="button"
                className={`target-chip ${on ? 'target-chip--on' : ''}`}
                style={on ? { borderColor: meta.color, color: meta.color } : undefined}
                onClick={() => toggleClass(cls)}
              >
                <span className="target-chip__glyph">{meta.glyph}</span>
                {meta.label}
              </button>
            );
          })}
        </div>
      </section>

      <section className="panel-section">
        <label className="panel-label">Model Thresholds</label>

        <div className="slider-row">
          <div className="slider-row__head">
            <span>Confidence</span>
            <span className="slider-row__value">{settings.confidence.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={settings.confidence}
            onChange={(e) => onSettingsChange({ confidence: Number(e.target.value) })}
          />
        </div>

        <div className="slider-row">
          <div className="slider-row__head">
            <span>IoU · NMS</span>
            <span className="slider-row__value">{settings.iouNms.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={settings.iouNms}
            onChange={(e) => onSettingsChange({ iouNms: Number(e.target.value) })}
          />
        </div>
      </section>

      <section className="panel-section">
        <label className="panel-label">Visualization Layer</label>
        <div className="segmented">
          <button
            type="button"
            className={`segmented__opt ${settings.visualization === 'rgb' ? 'segmented__opt--on' : ''}`}
            onClick={() => onSettingsChange({ visualization: 'rgb' })}
          >
            RGB
          </button>
          <button
            type="button"
            className={`segmented__opt ${settings.visualization === 'ir' ? 'segmented__opt--on' : ''}`}
            onClick={() => onSettingsChange({ visualization: 'ir' })}
          >
            False-Color IR
          </button>
        </div>
      </section>

      <section className="panel-section panel-section--actions">
        <button
          type="button"
          className="btn btn--primary"
          disabled={!raster || analyzing}
          onClick={onRunAnalysis}
        >
          {analyzing ? 'Running Inference…' : 'Run Detection'}
        </button>
        <button
          type="button"
          className="btn btn--ghost"
          disabled={!raster && settings.enabledClasses.size === TARGET_ORDER.length}
          onClick={onClear}
        >
          Clear
        </button>
      </section>
    </aside>
  );
}
