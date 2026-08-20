import { useEffect, useMemo, useRef, useState } from 'react';
import type { PerClassMetric, TrainingReference, TrainingRun } from '../types';
import { fetchPerClassMetrics, fetchTrainingMetrics } from '../services/inference';
import { TARGET_META } from '../config';

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
const REFERENCE_COLOR = '#8a8f98';
const BAR_COLOR = '#c98500';

const CHART_W = 760;
const CHART_H = 360;
const MARGIN = { top: 16, right: 16, bottom: 36, left: 44 };
const PLOT_W = CHART_W - MARGIN.left - MARGIN.right;
const PLOT_H = CHART_H - MARGIN.top - MARGIN.bottom;

const BAR_MARGIN = { top: 8, right: 46, bottom: 28, left: 168 };
const BAR_ROW_H = 24;
const BAR_CHART_W = 760;

function formatPct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

export default function AnalyticsPanel() {
  const [runs, setRuns] = useState<TrainingRun[] | null>(null);
  const [reference, setReference] = useState<TrainingReference | null>(null);
  const [perClass, setPerClass] = useState<PerClassMetric[] | null>(null);
  const [showTable, setShowTable] = useState(false);
  const [hoverEpoch, setHoverEpoch] = useState<number | null>(null);
  const [hoverClass, setHoverClass] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    fetchTrainingMetrics().then(({ runs, reference }) => {
      setRuns(runs);
      setReference(reference);
    });
    fetchPerClassMetrics().then(setPerClass);
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

  const sortedClasses = useMemo(
    () => (perClass ?? []).slice().sort((a, b) => b.ap50 - a.ap50),
    [perClass],
  );
  const barPlotW = BAR_CHART_W - BAR_MARGIN.left - BAR_MARGIN.right;
  const barChartH = BAR_MARGIN.top + BAR_MARGIN.bottom + sortedClasses.length * BAR_ROW_H;
  const barX = (v: number) => BAR_MARGIN.left + v * barPlotW;
  const barY = (i: number) => BAR_MARGIN.top + i * BAR_ROW_H;

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
            {reference && (
              <div className="analytics__legend-item" title={reference.note}>
                <span
                  className="analytics__legend-swatch analytics__legend-swatch--dashed"
                  style={{ borderColor: REFERENCE_COLOR }}
                />
                <span>{reference.label} (reference, ⓘ)</span>
              </div>
            )}
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

            {/* Pretrained reference: a single published number, not a
                per-epoch curve -- drawn as a dashed horizontal line so it
                reads as "external benchmark," not fabricated training data. */}
            {reference && (
              <line
                x1={MARGIN.left}
                x2={CHART_W - MARGIN.right}
                y1={yScale(reference.map50)}
                y2={yScale(reference.map50)}
                stroke={REFERENCE_COLOR}
                strokeWidth={1.5}
                strokeDasharray="5 4"
              />
            )}

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

      <div className="analytics__section-divider" />

      <div className="analytics__header">
        <div>
          <h2 className="analytics__title">🎯 Per-Class Accuracy</h2>
          <p className="analytics__subtitle">
            AP50 per DOTA class (default model, from-scratch run) — icons show which app
            target-class filter each one falls under.
          </p>
        </div>
      </div>

      {perClass === null && <p className="analytics__empty">Loading per-class metrics…</p>}
      {perClass !== null && perClass.length === 0 && (
        <p className="analytics__empty">
          Per-class metrics not available yet — run{' '}
          <code>ml/src/eval_per_class.py</code> to compute them.
        </p>
      )}

      {perClass !== null && perClass.length > 0 && (
        <>
          <div className="analytics__legend">
            {Object.entries(TARGET_META).map(([cls, meta]) => (
              <div key={cls} className="analytics__legend-item">
                <span>{meta.glyph}</span>
                <span>{meta.label}</span>
              </div>
            ))}
          </div>

          <svg
            className="analytics__chart"
            viewBox={`0 0 ${BAR_CHART_W} ${barChartH}`}
            onMouseLeave={() => setHoverClass(null)}
          >
            {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
              <g key={v}>
                <line
                  x1={barX(v)}
                  x2={barX(v)}
                  y1={BAR_MARGIN.top}
                  y2={barChartH - BAR_MARGIN.bottom}
                  className="analytics__gridline"
                />
                <text
                  x={barX(v)}
                  y={barChartH - BAR_MARGIN.bottom + 16}
                  className="analytics__axis-label"
                  textAnchor="middle"
                >
                  {formatPct(v)}
                </text>
              </g>
            ))}

            {sortedClasses.map((c, i) => {
              const meta = TARGET_META[c.target_class];
              const y = barY(i);
              const barH = BAR_ROW_H - 8;
              return (
                <g
                  key={c.name}
                  onMouseEnter={() => setHoverClass(c.name)}
                  style={{ cursor: 'pointer' }}
                >
                  <rect
                    x={BAR_MARGIN.left - 168}
                    y={y}
                    width={168}
                    height={BAR_ROW_H}
                    fill="transparent"
                  />
                  <text
                    x={BAR_MARGIN.left - 10}
                    y={y + BAR_ROW_H / 2}
                    className="analytics__axis-label"
                    textAnchor="end"
                    dominantBaseline="middle"
                  >
                    {meta.glyph} {c.name}
                  </text>
                  <rect
                    x={BAR_MARGIN.left}
                    y={y + 4}
                    width={Math.max(2, barX(c.ap50) - BAR_MARGIN.left)}
                    height={barH}
                    rx={2}
                    fill={BAR_COLOR}
                    opacity={hoverClass === null || hoverClass === c.name ? 1 : 0.45}
                  />
                  <text
                    x={barX(c.ap50) + 6}
                    y={y + BAR_ROW_H / 2}
                    className="analytics__axis-label"
                    dominantBaseline="middle"
                  >
                    {formatPct(c.ap50)}
                  </text>
                </g>
              );
            })}
          </svg>
        </>
      )}
    </div>
  );
}
