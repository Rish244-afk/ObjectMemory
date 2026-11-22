import { test, expect } from '@playwright/test';

// ---------------------------------------------------------------------------
// Unit Tests: Crop Coordinate Clamping & Metric Formatting
// ---------------------------------------------------------------------------
test.describe('Phase 13 Unit Tests: Clamping & Formatting Logic', () => {
  function clampBoundingBox(
    bbox: { x_min: number; y_min: number; x_max: number; y_max: number },
    naturalWidth: number,
    naturalHeight: number
  ) {
    const clampX1 = Math.max(0, Math.min(naturalWidth, Math.round(bbox.x_min)));
    const clampY1 = Math.max(0, Math.min(naturalHeight, Math.round(bbox.y_min)));
    const clampX2 = Math.max(0, Math.min(naturalWidth, Math.round(bbox.x_max)));
    const clampY2 = Math.max(0, Math.min(naturalHeight, Math.round(bbox.y_max)));
    return {
      x_min: clampX1,
      y_min: clampY1,
      x_max: clampX2,
      y_max: clampY2,
      width: clampX2 - clampX1,
      height: clampY2 - clampY1,
    };
  }

  test('Correctly clamps negative and overflowing bounding box coordinates to image boundaries', () => {
    const naturalWidth = 1280;
    const naturalHeight = 720;

    // Out of bounds on all sides
    const clamped1 = clampBoundingBox(
      { x_min: -45.5, y_min: -100, x_max: 1500.2, y_max: 900.8 },
      naturalWidth,
      naturalHeight
    );
    expect(clamped1.x_min).toBe(0);
    expect(clamped1.y_min).toBe(0);
    expect(clamped1.x_max).toBe(1280);
    expect(clamped1.y_max).toBe(720);
    expect(clamped1.width).toBe(1280);
    expect(clamped1.height).toBe(720);

    // Completely within bounds
    const clamped2 = clampBoundingBox(
      { x_min: 100, y_min: 150, x_max: 300, y_max: 450 },
      naturalWidth,
      naturalHeight
    );
    expect(clamped2.x_min).toBe(100);
    expect(clamped2.y_min).toBe(150);
    expect(clamped2.x_max).toBe(300);
    expect(clamped2.y_max).toBe(450);
    expect(clamped2.width).toBe(200);
    expect(clamped2.height).toBe(300);

    // Partially clipping right/bottom edge
    const clamped3 = clampBoundingBox(
      { x_min: 1200, y_min: 700, x_max: 1350, y_max: 750 },
      naturalWidth,
      naturalHeight
    );
    expect(clamped3.x_min).toBe(1200);
    expect(clamped3.y_min).toBe(700);
    expect(clamped3.x_max).toBe(1280);
    expect(clamped3.y_max).toBe(720);
    expect(clamped3.width).toBe(80);
    expect(clamped3.height).toBe(20);
  });

  test('Evidence ledger metric formatting never substitutes 0 for missing values', () => {
    function formatDisplacement(d: number | null | undefined): string {
      if (d === null || d === undefined) return 'Not available';
      return `${d.toFixed(1)} px`;
    }

    expect(formatDisplacement(undefined)).toBe('Not available');
    expect(formatDisplacement(null)).toBe('Not available');
    expect(formatDisplacement(0)).toBe('0.0 px');
    expect(formatDisplacement(14.82)).toBe('14.8 px');
  });
});

// ---------------------------------------------------------------------------
// End-to-End Test: Synchronized Comparative Analysis & Visual Crop Gallery
// ---------------------------------------------------------------------------
test.describe('Phase 13 End-to-End: Comparative Scene Analysis & Evidence Audit', () => {
  test('Synchronized dual-canvas zoom/pan, wipe slider, crop gallery, and evidence ledger', async ({
    page,
  }) => {
    // Navigate to studio workbench demo workspace
    await page.goto('/');
    await expect(page).toHaveTitle(/ObjectMemory/i);

    // Find and open Studio Workbench Demo
    const demoCard = page.locator('div.group', { hasText: 'Studio Workbench Demo' });
    await expect(demoCard).toBeVisible({ timeout: 10000 });
    await demoCard.click();

    // 1. Verify Scene Comparison Tab
    await page.click('button:has-text("Scene Comparison")');
    await expect(page.locator('h1:has-text("Temporal Scene Comparison")')).toBeVisible();
    await expect(page.locator('text=Comparative Entity Changes')).toBeVisible();

    // Check Reference (T1) and Query (T2) theme tags
    await expect(page.locator('span:has-text("REFERENCE")').first()).toBeVisible();
    await expect(page.locator('span:has-text("QUERY")').first()).toBeVisible();

    // Check Synchronized Pan/Zoom toggle
    const syncBtn = page.locator('button:has-text("Pan/Zoom: Synced")');
    await expect(syncBtn).toBeVisible();

    // Test Zoom In with synchronization
    const zoomInBtn = page.locator('button[title="Zoom in"]').first();
    await zoomInBtn.click();

    // Verify zoom level updated to 125%
    await expect(page.locator('span:has-text("125%")').first()).toBeVisible();

    // Toggle sync lock to Independent
    await syncBtn.click();
    await expect(page.locator('button:has-text("Pan/Zoom: Independent")')).toBeVisible();

    // Reset zoom
    const resetZoomBtn = page.locator('button[title="Reset zoom to 100%"]').first();
    if (await resetZoomBtn.isVisible()) {
      await resetZoomBtn.click();
    }

    // Toggle to Wipe Slider mode
    await page.click('button:has-text("Wipe Slider")');
    await expect(page.locator('text=Interactive Wipe Comparison')).toBeVisible();
    await expect(page.locator('div[role="slider"]')).toBeVisible();

    // Toggle back to Side-by-Side mode
    await page.click('button:has-text("Side-by-Side")');
    await expect(page.locator('text=Comparative Entity Changes')).toBeVisible();

    // Verify 2D Canonical Plane Disclaimer is visible
    await expect(
      page.locator('text=Displacement vectors depict 2D centroid shifts on the canonical workbench plane')
    ).toBeVisible();

    // 2. Verify Tracked Entities & Visual Crop Gallery
    await page.click('button:has-text("Entities & Beliefs")');
    await expect(page.locator('h1:has-text("Tracked Entities")')).toBeVisible();

    // Select first entity
    await page.locator('.cursor-pointer:has-text("ent_")').first().click();
    await expect(page.locator('text=Canonical Reference Coordinates:')).toBeVisible();

    // Verify Observation History & Visual Crop Gallery
    await expect(page.locator('text=Observation History & Visual Crop Gallery')).toBeVisible();
    await expect(
      page.locator('text=Visual crops are dynamic extracts from raw camera frames')
    ).toBeVisible();
    await expect(page.locator('canvas').first()).toBeVisible();

    // 3. Verify Change Timeline with Evidence Ledger
    await page.click('button:has-text("Change Timeline")');
    await expect(page.locator('h1:has-text("Change Event Audit Log")')).toBeVisible();
    await expect(page.locator('text=Chronological Change Events')).toBeVisible();

    // Verify scientific metric columns exist
    await expect(page.locator('text=Displacement (d)').first()).toBeVisible();
    await expect(page.locator('text=Threshold (T_mov)').first()).toBeVisible();
    await expect(page.locator('text=Planar Reg / Inliers').first()).toBeVisible();
    await expect(page.locator('text=Boundary Clip').first()).toBeVisible();

    // Verify Evidence JSON Payload expansion
    const jsonBtn = page.locator('button:has-text("Evidence JSON Payload")').first();
    if (await jsonBtn.isVisible()) {
      await jsonBtn.click();
      await expect(page.locator('text=Raw Supporting Evidence Payload')).toBeVisible();
    }
  });
});
