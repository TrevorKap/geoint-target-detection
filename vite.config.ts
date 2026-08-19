import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // MapLibre GL bundles its tile-processing logic as a web worker; Vite's
  // dependency optimizer can't pre-bundle that worker script correctly, so
  // exclude the package from optimization entirely.
  optimizeDeps: {
    exclude: ['maplibre-gl'],
  },
})
