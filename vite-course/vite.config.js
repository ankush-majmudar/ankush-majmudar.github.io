import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { qrcode } from 'vite-plugin-qrcode'
import svgr from "vite-plugin-svgr";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), qrcode(), svgr()],
  server: {
    port: 3000
  },
  build: {
    outDir: "out"
  }
})
  