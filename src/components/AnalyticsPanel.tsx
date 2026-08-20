import { useEffect, useMemo, useRef, useState } from 'react';
import type { TrainingRun } from '../types';
import { fetchTrainingMetrics } from '../services/inference';

// Validated against this app's actual dark panel surface (#0e141b) with the
// dataviz skill's palette checker: lightness band, chroma floor, CVD
// separation (protan/deutan/tritan), normal-vision floor, and contrast all
// pass. The app's raw UI accent colors (amber #ffb000, cyan #22d3ee) are too
// light for chart marks specifically -- they read fine as buttons/badges
// against near-black, but fail the dark-mode chart lightness band (L 0.48-0.67).
const SERIES_COLORS: Record<string, string> = {
  'dota-obb-scratch': '#c98500',
  'dota-obb-finetuned': '#3987e5',
};

const CHART_W = 760;
const CHART_H = 360;
const MARGIN = { top: 16, right: 16, bottom: 36, left: 44 };
const PLOT_W = CHART_W - MARGIN.left - MARGIN.right;
const PLOT_H = CHART_H - MARGIN.top - MARGIN.bottom;

function formatPct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

export default function AnalyticsPanel() {
  const [runs, setRuns] = useState<TrainingRun[] | null>(null);
  const [showTable, setShowTable] = useState(false);
  const [hoverEpoch, setHoverEpoch] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    fetchTrainingMetrics().then(setRuns);
  }, []);

  const maxEpoch = useMemo(
    () => Math.max(1, ...(runs ?? []).map((r) => r.epochs.length)),
    [runs],
  );

  const xScale = (epoch: number) => MARGIN.left + ((epoch - 1) / (maxEpoch - 1)) * PLOT_W;
  const yScale = (v: number) => MARGIN.top + (1 - v) * PLOT_H;

  const handleMouseMove: React.MouseEventHandler<SVGSVGElement> = (e) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * CHART_W;
    const epoch = Math.round(1 + ((x - MARGIN.left) / PLOT_W) * (maxEpoch - 1));
    setHoverEpoch(Math.min(maxEpoch, Math.max(1, epoch)));
  };

  const gridY = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
  const gridXStep = maxEpoch <= 30 ? 5 : 20;
  const gridX = Array.from(
    { length: Math.floor(maxEpoch / gridXStep) + 1 },
    (_, i) => i * gridXStep,
  ).filter((v) => v >= 1);

  return (
    <div className="analytics">
      <div className="analytics__header">
        <div>
          <h2 className="analytics__title">📈 Model Accuracy Over Training</h2>
          <p className="analytics__subtitle">
            mAP50 per epoch, from the real ultralytics training logs for each model.
          </p>
        </div>
        {runs && runs.length > 0 && (
          <button
            type="button"
            className="analytics__table-toggle"
            onClick={() => setShowTable((v) => !v)}
          >
            {showTable ? 'Hide' : 'View'} data table
          </button>
        )}
      </div>

      {runs === null && <p className="analytics__empty">Loading training history…</p>}
      {runs !== null && runs.length === 0 && (
        <p className="analytics__empty">
          No training history available (backend offline, or no results.csv found for
          any registered model).
        </p>
      )}

      {runs !== null && runs.length > 0 && (
        <>
          <div className="analytics__legend">
            {runs.map((r) => (
              <div key={r.id} className="analytics__legend-item">
                <span
                  className="analytics__legend-swatch"
                  style={{ background: SERIES_COLORS[r.id] ?? '#888' }}
                />
                <span>{r.label}</span>
              </div>
            ))}
          </div>

          <svg
            ref={svgRef}
            className="analytics__chart"
            viewBox={`0 0 ${CHART_W} ${CHART_H}`}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setHoverEpoch(null)}
          >
            {/* Recessive horizontal gridlines + Y axis labels */}
            {gridY.map((v) => (
              <g key={v}>
                <line
                  x1={MARGIN.left}
                  x2={CHART_W - MARGIN.right}
                  y1={yScale(v)}
                  y2={yScale(v)}
                  className="analytics__gridline"
                />
                <text x={MARGIN.left - 8} y={yScale(v)} className="analytics__axis-label" textAnchor="end" dominantBaseline="middle">
                  {formatPct(v)}
                </text>
              </g>
            ))}

            {/* X axis labels */}
            {gridX.map((epoch) => (
              <text
                key={epoch}
                x={xScale(Math.max(1, epoch))}
                y={CHART_H - MARGIN.bottom + 18}
                className="analytics__axis-label"
                textAnchor="middle"
              >
                {epoch}
              </text>
            ))}
            <text
              x={MARGIN.left + PLOT_W / 2}
              y={CHART_H - 4}
              className="analytics__axis-title"
              textAnchor="middle"
            >
              EPOCH
            </text>

            {/* Hover crosshair */}
            {hoverEpoch !== null && (
              <line
                x1={xScale(hoverEpoch)}
                x2={xScale(hoverEpoch)}
                y1={MARGIN.top}
                y2={CHART_H - MARGIN.bottom}
                className="analytics__crosshair"
              />
            )}

            {/* Series lines */}
            {runs.map((r) => {
              const color = SERIES_COLORS[r.id] ?? '#888';
              const d = r.epochs
                .map((e, i) => `${i === 0 ? 'M' : 'L'}${xScale(e.epoch)},${yScale(e.map50)}`)
                .join(' ');
              return (
                <path
                  key={r.id}
                  d={d}
                  fill="none"
                  stroke={color}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              );
            })}

            {/* Hover markers */}
            {hoverEpoch !== null &&
              runs.map((r) => {
                const point = r.epochs.find((e) => e.epoch === hoverEpoch);
                if (!point) return null;
                const color = SERIES_COLORS[r.id] ?? '#888';
                return (
                  <circle
                    key={r.id}
                    cx={xScale(point.epoch)}
                    cy={yScale(point.map50)}
                    r={4.5}
                    fill={color}
                    stroke="var(--panel-2)"
                    strokeWidth={2}
                  />
                );
              })}
          </svg>

          {hoverEpoch !== null && (
            <div className="analytics__tooltip">
              <div className="analytics__tooltip-epoch">EPOCH {hoverEpoch}</div>
              {runs.map((r) => {
                const point = r.epochs.find((e) => e.epoch === hoverEpoch);
                if (!point) return null;
                return (
                  <div key={r.id} className="analytics__tooltip-row">
                    <span
                      className="analytics__legend-swatch"
                      style={{ background: SERIES_COLORS[r.id] ?? '#888' }}
                    />
                    <span className="analytics__tooltip-label">{r.label}</span>
                    <span className="analytics__tooltip-value">{formatPct(point.map50)}</span>
                  </div>
                );
              })}
            </div>
          )}

          {showTable && (
            <div className="analytics__table-wrap">
              {runs.map((r) => (
                <div key={r.id} className="analytics__table-block">
                  <h3 className="analytics__table-title">{r.label}</h3>
                  <table className="analytics__table">
                    <thead>
                      <tr>
                        <th>Epoch</th>
                        <th>mAP50</th>
                        <th>mAP50-95</th>
                        <th>Precision</th>
                        <th>Recall</th>
                      </tr>
                    </thead>
                    <tbody>
                      {r.epochs.map((e) => (
                        <tr key={e.epoch}>
                          <td>{e.epoch}</td>
                          <td>{formatPct(e.map50)}</td>
                          <td>{formatPct(e.map50_95)}</td>
                          <td>{formatPct(e.precision)}</td>
                          <td>{formatPct(e.recall)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
