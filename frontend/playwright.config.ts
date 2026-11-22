import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 60000,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:8000',
    channel: 'chrome',
    trace: 'on-first-retry',
    headless: true,
  },
  projects: [
    {
      name: 'Google Chrome',
      use: {
        channel: 'chrome',
        viewport: { width: 1280, height: 800 },
      },
    },
  ],
});
