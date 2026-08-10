import { useMemo } from 'react';
import type { AnalysisResult, DetectorSettings } from '../types';
import { TARGET_META, TARGET_ORDER } from '../config';
import { downloadGeoJSON } from '../utils/geojson';

interface AnalyticalSummaryProps {
  result: AnalysisResult | null;
  settings: DetectorSettings;
  /** Focus (fly + popup) a detection on the map by id. */
  onFocus: (id: string) => void;
}

export default function AnalyticalSummary({
  result,
  settings,
  onFocus,
}: AnalyticalSummaryProps) {
  // Detections currently passing the confidence + class filters, highest first.
  const visible = useMemo(
    () =>
      (result?.detections ?? [])
        .filter(
          (d) =>
            d.confidence >= settings.confidence &&
            settings.enabledClasses.has(d.targetClass),
        )
        .sort((a, b) => b.confidence - a.confidence),
    [result, settings],
  );

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
        <div className="summary__body">
          <div className="summary__left">
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
              {counts.map(({ cls, n }) => {
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
              })}
            </div>

            <button
              type="button"
              className="btn btn--primary summary__export"
              disabled={visible.length === 0}
              onClick={() => downloadGeoJSON({ ...result, detections: visible })}
            >
              ⬇ Download GeoJSON
            </button>
          </div>

          <div className="target-list">
            <div className="target-list__head">
              <span>TARGET LIST</span>
              <span className="target-list__hint">click to locate</span>
            </div>
            {visible.length === 0 ? (
              <p className="summary__empty-hint">No targets above current thresholds.</p>
            ) : (
              <ul className="target-list__items">
                {visible.map((d, i) => {
                  const meta = TARGET_META[d.targetClass];
                  return (
                    <li key={d.id}>
                      <button
                        type="button"
                        className="target-list__row"
                        onClick={() => onFocus(d.id)}
                      >
                        <span className="target-list__idx">
                          {String(i + 1).padStart(2, '0')}
                        </span>
                        <span
                          className="target-list__glyph"
                          style={{ color: meta.color }}
                        >
                          {meta.glyph}
                        </span>
                        <span className="target-list__label">{meta.label}</span>
                        <span className="target-list__conf">
                          {(d.confidence * 100).toFixed(0)}%
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
