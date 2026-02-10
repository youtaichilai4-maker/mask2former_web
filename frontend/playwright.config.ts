import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  retries: 0,
  workers: 1,
  timeout: 120000,
  use: {
    baseURL: 'http://127.0.0.1:14000',
    trace: 'on-first-retry',
  },
  webServer: [
    {
      command:
        'cd ../backend && source .venv/bin/activate && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uvicorn app.main:app --host 127.0.0.1 --port 18000',
      url: 'http://127.0.0.1:18000/health',
      reuseExistingServer: true,
      timeout: 120000,
    },
    {
      command:
        'NEXT_PUBLIC_API_BASE=http://127.0.0.1:18000 npm run dev -- --hostname 127.0.0.1 --port 14000',
      url: 'http://127.0.0.1:14000',
      reuseExistingServer: true,
      timeout: 120000,
    },
  ],
});
