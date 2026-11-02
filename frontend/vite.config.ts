import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => ({
  base: mode === 'streamlit' ? './' : '/',
  build: { outDir: mode === 'streamlit' ? 'streamlit_build' : 'dist' },
  plugins: [react()],
  server: {
    port: 5173, strictPort: true,
    proxy: { '/api': { target: process.env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000', changeOrigin: true, rewrite: path => path.replace(/^\/api/, '') } },
  },
}))
