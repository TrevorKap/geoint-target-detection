import type { AnalysisResult, DetectorSettings } from '../types';
import { TARGET_META, TARGET_ORDER } from '../config';
import { downloadGeoJSON } from '../utils/geojson';

interface AnalyticalSummaryProps {
  result: AnalysisResult | null;
  settings: DetectorSettings;
}

export default function AnalyticalSummary({ result, settings }: AnalyticalSummaryProps) {
  // Detections currently passing the confidence + class filters.
  const visible =
    result?.detections.filter(
      (d) =>
        d.confidence >= settings.confidence &&
        settings.enabledClasses.has(d.targetClass),
    ) ?? [];

  const counts = TARGET_ORDER.map((cls) => ({
    cls,
    n: visible.filter((d) => d.targetClass === cls).length,
  })).filter((c) => c.n > 0);

  const totalArea = visible.reduce((sum, d) => sum + (d.areaSqMeters ?? 0), 0);

  return (
    <section className="summary">
      <h2 className="summary__title">📊 Analytical Summary</h2>

      {!result ? (
        <div className="summary__empty">
          <p>No analysis run yet.</p>
          <p className="summary__empty-hint">
            Stage a GeoTIFF and run detection to populate target counts, class
            breakdown, and vector export.
          </p>
        </div>
      ) : (
        <>
          <div className="summary__stats">
            <div className="stat">
              <span className="stat__value">{visible.length}</span>
              <span className="stat__label">Targets Identified</span>
            </div>
            <div className="stat">
              <span className="stat__value">
                {totalArea > 0 ? `${Math.round(totalArea).toLocaleString()}` : '—'}
              </span>
              <span className="stat__label">Footprint m²</span>
            </div>
            <div className="stat">
              <span className="stat__value">
                {result.inferenceMs ? `${result.inferenceMs} ms` : '—'}
              </span>
              <span className="stat__label">Inference</span>
            </div>
          </div>

          <div className="summary__breakdown">
            {counts.length === 0 ? (
              <p className="summary__empty-hint">
                No targets above current thresholds.
              </p>
            ) : (
              counts.map(({ cls, n }) => {
                const meta = TARGET_META[cls];
                return (
                  <div key={cls} className="breakdown-row">
                    <span
                      className="breakdown-row__swatch"
                      style={{ background: meta.color }}
                    />
                    <span className="breakdown-row__label">{meta.label}</span>
                    <span className="breakdown-row__count">{n}</span>
                  </div>
                );
              })
            )}
          </div>

          <button
            type="button"
            className="btn btn--primary summary__export"
            disabled={visible.length === 0}
            onClick={() => downloadGeoJSON({ ...result, detections: visible })}
          >
            ⬇ Download GeoJSON
          </button>
        </>
      )}
    </section>
  );
}
