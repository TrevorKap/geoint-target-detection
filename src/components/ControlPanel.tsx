import { useRef, useState } from 'react';
import type {
  DetectorSettings,
  ModelInfo,
  RasterMetadata,
  TargetClass,
} from '../types';
import { TARGET_META, TARGET_ORDER } from '../config';

interface ControlPanelProps {
  settings: DetectorSettings;
  onSettingsChange: (patch: Partial<DetectorSettings>) => void;
  models: ModelInfo[];
  raster: RasterMetadata | null;
  onFileSelected: (file: File) => void;
  onRunAnalysis: () => void;
  onClear: () => void;
  analyzing: boolean;
  vizNote: string | null;
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
  models,
  raster,
  onFileSelected,
  onRunAnalysis,
  onClear,
  analyzing,
  vizNote,
}: ControlPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [rejectReason, setRejectReason] = useState<string | null>(null);

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    // Drag-and-drop bypasses the <input accept> filter entirely, so validate
    // here too -- otherwise a plain photo gets staged with no explanation for
    // why nothing ever shows up on the map.
    if (!/\.(tif|tiff|geotiff)$/i.test(file.name)) {
      setRejectReason(
        `"${file.name}" isn't a GeoTIFF (.tif/.tiff). A regular photo has no ` +
          `location data, so it can't be placed on the map.`,
      );
      return;
    }
    setRejectReason(null);
    onFileSelected(file);
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
        {rejectReason && <p className="dropzone__reject">{rejectReason}</p>}
      </section>

      {models.length > 0 && (
        <section className="panel-section">
          <label className="panel-label">Detection Model</label>
          <select
            className="model-select"
            value={settings.modelId}
            onChange={(e) => onSettingsChange({ modelId: e.target.value })}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.satellite} · {m.algorithm} · {m.training_data}
              </option>
            ))}
          </select>
        </section>
      )}

      <section className="panel-section">
        <div className="panel-label-row">
          <label className="panel-label">Target Selection</label>
          <button
            type="button"
            className="panel-label-action"
            onClick={() => onSettingsChange({ enabledClasses: new Set(TARGET_ORDER) })}
            disabled={settings.enabledClasses.size === TARGET_ORDER.length}
          >
            Reset
          </button>
        </div>
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

        {vizNote && <p className="panel-note">{vizNote}</p>}

        {raster?.overlay && (
          <div className="slider-row">
            <div className="slider-row__head">
              <span>Raster Overlay Opacity</span>
              <span className="slider-row__value">
                {settings.overlayOpacity.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={settings.overlayOpacity}
              onChange={(e) =>
                onSettingsChange({ overlayOpacity: Number(e.target.value) })
              }
            />
          </div>
        )}
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
