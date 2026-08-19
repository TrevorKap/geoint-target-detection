import type { TargetClass } from './types';

/** Default map view: San Diego / Coronado NAS — a dense military flightline. */
export const DEFAULT_CENTER: [number, number] = [-117.19, 32.7];
export const DEFAULT_ZOOM = 13;

/** Per-class display metadata: label, overlay colour, and glyph. */
export const TARGET_META: Record<
  TargetClass,
  { label: string; color: string; glyph: string }
> = {
  aircraft: { label: 'Aircraft', color: '#22d3ee', glyph: '✈' },
  vessel: { label: 'Vessel', color: '#ffb000', glyph: '🚢' },
  vehicle: { label: 'Ground Vehicle', color: '#4ade80', glyph: '🚚' },
  building: { label: 'Structure', color: '#f472b6', glyph: '▢' },
  storage_tank: { label: 'Storage Tank', color: '#a78bfa', glyph: '◍' },
};

/** Stable ordering for target-selection UI and summary breakdowns. */
export const TARGET_ORDER: TargetClass[] = [
  'aircraft',
  'vessel',
  'vehicle',
  'building',
  'storage_tank',
];
