import { fileURLToPath } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  // The server's .env (one directory up). Read here, in Node, and never
  // exposed to the browser: only VITE_* variables reach client code.
  const env = loadEnv(mode, fileURLToPath(new URL('..', import.meta.url)), '')

  return {
    plugins: [react()],
    server: {
      // Localhost by default. The proxy below attaches the server's write
      // token, so anyone who can open this page can change robots, and a LAN
      // bind would hand that to the whole lab network. Set DASHBOARD_HOST=0.0.0.0
      // in the server's .env to serve it to other machines on purpose.
      host: env.DASHBOARD_HOST || '127.0.0.1',
      port: 3000,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:5000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
          headers: env.API_TOKEN ? { 'X-API-Token': env.API_TOKEN } : {},
        },
      },
    },
  }
})
