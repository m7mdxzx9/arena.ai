import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev server: binds 0.0.0.0 and proxies /api to the FastAPI backend (port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: true,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
