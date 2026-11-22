import { test, expect } from '@playwright/test';
import path from 'path';

const V1_PATH = path.resolve('../data/real_workbench/visits/visit_01.png');
const V2_PATH = path.resolve('../data/real_workbench/visits/visit_02.png');

test.describe('ObjectMemory Studio End-to-End Workflow', () => {
  test('Complete physical workspace lifecycle, image ingest, comparison, and isolation', async ({ page }) => {
    const runId = Date.now().toString().slice(-6);
    const labSlug = `e2e_lab_${runId}`;
    const isoSlug = `e2e_iso_${runId}`;

    // 1. Load Application
    await page.goto('/');
    await expect(page).toHaveTitle(/ObjectMemory/i);

    // Verify backend connectivity badge
    await expect(page.locator('text=Backend Connected')).toBeVisible({ timeout: 10000 });

    // 2. Create Workspace
    await page.click('button:has-text("Create Workspace")');
    await page.fill('input[placeholder="e.g. lab_workbench_01"]', labSlug);
    await page.fill('input[placeholder="e.g. Electronics Assembly Bench"]', `E2E Testing Lab ${runId}`);
    await page.fill('textarea', 'Automated end-to-end verification workspace');
    await page.click('button[type="submit"]:has-text("Create Workspace")');

    // Should navigate into workspace overview
    await expect(page.locator(`h1:has-text("E2E Testing Lab ${runId}")`)).toBeVisible({ timeout: 8000 });
    await expect(page.locator('text=Establish Baseline Reference Visit')).toBeVisible();

    // 3. Upload Baseline Visit 1
    await page.setInputFiles('input[type="file"]', V1_PATH);

    // Verify preview appears
    await expect(page.locator('img[alt="Selected preview"]')).toBeVisible();

    // Submit analysis
    await page.click('button:has-text("Analyze Scene & Ingest Observation")');

    // Wait for analysis result banner
    await expect(page.locator('text=Observation Successfully Analyzed: visit_01')).toBeVisible({ timeout: 20000 });

    // 4. Upload Follow-Up Visit 2
    await page.setInputFiles('input[type="file"]', V2_PATH);

    await page.click('button:has-text("Analyze Scene & Ingest Observation")');
    await expect(page.locator('text=Observation Successfully Analyzed: visit_02')).toBeVisible({ timeout: 20000 });

    // Verify session table shows 2 visits
    await expect(page.locator('tbody tr')).toHaveCount(2);

    // 5. Navigate to Scene Comparison
    await page.click('button:has-text("Scene Comparison")');
    await expect(page.locator('h1:has-text("Temporal Scene Comparison")')).toBeVisible();
    await expect(page.locator('text=Comparative Entity Changes')).toBeVisible();

    // 5b. Navigate to Spatial Memory Explorer
    await page.click('button:has-text("Spatial Memory")');
    await expect(page.locator('h1:has-text("Spatial Memory Explorer")')).toBeVisible();
    await expect(page.locator('text=Canonical Space')).toBeVisible();
    // Test Canonical 2D Workbench Plane Toggle
    await page.click('button:has-text("Canonical 2D Workbench Plane")');
    await expect(page.locator('button:has-text("Angled Inspection Plane")')).toBeVisible();

    // 6. Navigate to Entities & Beliefs
    await page.click('button:has-text("Entities & Beliefs")');
    await expect(page.locator('h1:has-text("Tracked Entities")')).toBeVisible();
    // Select first entity
    await page.locator('.cursor-pointer:has-text("ent_")').first().click();
    await expect(page.locator('text=Canonical Reference Coordinates:')).toBeVisible();

    // 7. Navigate to Change Timeline
    await page.click('button:has-text("Change Timeline")');
    await expect(page.locator('h1:has-text("Change Event Audit Log")')).toBeVisible();

    // 8. Navigate to Diagnostics
    await page.click('button:has-text("Diagnostics")');
    await expect(page.locator('text=Real Object Detector Engine')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Planar Registration Health' })).toBeVisible();

    // 8b. Navigate to Evaluation Lab
    await page.click('button:has-text("Evaluation Lab")');
    await expect(page.locator('h1:has-text("Evaluation Lab & Benchmark Harness")')).toBeVisible();
    await expect(page.locator('text=DATASET PROVENANCE NOTICE')).toBeVisible();
    // Execute benchmark run
    await page.click('button:has-text("Execute Benchmark Run")');
    await expect(page.locator('text=Change Precision')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=Ground Truth vs Prediction Audit Matrix')).toBeVisible();

    // 9. Test Browser Refresh Persistence
    await page.reload();
    await expect(page.locator(`text=E2E Testing Lab ${runId}`).first()).toBeVisible({ timeout: 10000 });

    // 10. Test Workspace Isolation
    await page.click('text=ObjectMemory'); // Navigate back to Dashboard
    await expect(page.locator('h1:has-text("Physical Workspaces")')).toBeVisible();

    // Create second workspace
    await page.click('button:has-text("Create Workspace")');
    await page.fill('input[placeholder="e.g. lab_workbench_01"]', isoSlug);
    await page.fill('input[placeholder="e.g. Electronics Assembly Bench"]', `Isolated Clean ${runId}`);
    await page.click('button[type="submit"]:has-text("Create Workspace")');

    // Verify isolated workspace has 0 visits and clean slate
    await expect(page.locator(`h1:has-text("Isolated Clean ${runId}")`)).toBeVisible();
    await expect(page.locator('text=No observations recorded yet')).toBeVisible();

    // 11. Destructive Cleanup
    await page.click('text=ObjectMemory'); // Return to Dashboard
    await expect(page.locator('h1:has-text("Physical Workspaces")')).toBeVisible();

    // Delete isolated workspace
    const isolatedCard = page.locator('div.group', { hasText: `Isolated Clean ${runId}` });
    await isolatedCard.locator('button[title*="Delete"]').click();
    await isolatedCard.locator('button:has-text("Confirm?")').click();
    await expect(isolatedCard).not.toBeVisible({ timeout: 10000 });

    // Delete lab workspace
    const labCard = page.locator('div.group', { hasText: `E2E Testing Lab ${runId}` });
    await labCard.locator('button[title*="Delete"]').click();
    await labCard.locator('button:has-text("Confirm?")').click();
    await expect(labCard).not.toBeVisible({ timeout: 10000 });
  });

  test('Open-vocabulary detection with YOLO-World and dynamic CLIP category prompts', async ({ page }) => {
    const runId = Date.now().toString().slice(-6);
    const wsSlug = `e2e_world_${runId}`;

    await page.goto('/');
    await expect(page).toHaveTitle(/ObjectMemory/i);

    // Create Workspace
    await page.click('button:has-text("Create Workspace")');
    await page.fill('input[placeholder="e.g. lab_workbench_01"]', wsSlug);
    await page.fill('input[placeholder="e.g. Electronics Assembly Bench"]', `YOLO-World Lab ${runId}`);
    await page.click('button[type="submit"]:has-text("Create Workspace")');

    await expect(page.locator(`h1:has-text("YOLO-World Lab ${runId}")`)).toBeVisible({ timeout: 8000 });

    // Select YOLO-World detector
    await page.selectOption('select', 'yolo-world');

    // Verify open-vocabulary prompts field appears
    await expect(page.locator('text=Open-Vocabulary Category Prompts')).toBeVisible();

    // Fill in custom prompts
    await page.fill('input[placeholder*="notebook, drawing compass"]', 'notebook, book, bottle, cup, scissors');

    // Upload Image
    await page.setInputFiles('input[type="file"]', V1_PATH);
    await page.click('button:has-text("Analyze Scene & Ingest Observation")');

    await expect(page.locator('text=Observation Successfully Analyzed: visit_01')).toBeVisible({ timeout: 25000 });

    // Verify detections table shows visit
    await expect(page.locator('tbody tr')).toHaveCount(1);

    // Cleanup
    await page.click('text=ObjectMemory');
    const wsCard = page.locator('div.group', { hasText: `YOLO-World Lab ${runId}` });
    await wsCard.locator('button[title*="Delete"]').click();
    await wsCard.locator('button:has-text("Confirm?")').click();
    await expect(wsCard).not.toBeVisible({ timeout: 10000 });
  });
});
