import type { TargetClass } from './types';

/**
 * Mapbox access token, supplied via `.env` as `VITE_MAPBOX_TOKEN`.
 * If absent, MapCanvas transparently falls back to a token-free Esri World
 * Imagery raster basemap so the app remains fully usable without an account.
 */
export const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN ?? '';

/** Whether a real Mapbox token (public key `pk.…`) is configured. */
export const HAS_MAPBOX_TOKEN = MAPBOX_TOKEN.startsWith('pk.');

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
  helipad: { label: 'Helipad', color: '#fb7185', glyph: '✛' },
};

/** Stable ordering for target-selection UI and summary breakdowns. */
export const TARGET_ORDER: TargetClass[] = [
  'aircraft',
  'vessel',
  'vehicle',
  'building',
  'storage_tank',
  'helipad',
];
