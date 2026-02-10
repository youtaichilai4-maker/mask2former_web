import { expect, test } from '@playwright/test';

test('gallery image selection triggers real inference and shows analysis', async ({ page }) => {
  test.setTimeout(180000);

  await page.goto('/');

  const card1 = page.getByRole('button', { name: 'ade20k_val_0001' });
  const card2 = page.getByRole('button', { name: 'ade20k_val_0002' });
  const card3 = page.getByRole('button', { name: 'ade20k_val_0003' });

  await expect(card1).toBeVisible({ timeout: 30000 });
  await expect(card2).toBeVisible({ timeout: 30000 });
  await expect(card3).toBeVisible({ timeout: 30000 });

  await card2.click();
  await page.getByRole('button', { name: '選択画像で推論' }).click();

  await expect(page.locator('img[alt="入力画像"]')).toBeVisible({ timeout: 120000 });
  await expect(page.locator('img[alt="予測オーバーレイ"]')).toBeVisible({ timeout: 120000 });

  // Ensure analysis content rendered from real inference response.
  const topClassesPanel = page.getByRole('heading', { name: '上位クラス信頼度（Top-K）' });
  await expect(topClassesPanel).toBeVisible();
  const metricValues = page.locator('section article strong');
  await expect(metricValues.first()).toBeVisible();
  expect(await metricValues.count()).toBeGreaterThan(1);

  // Ensure class mask list is populated and interactive.
  const maskButtons = page.locator('button', { has: page.locator('img[alt$=" mask"]') });
  const count = await maskButtons.count();
  expect(count).toBeGreaterThan(0);
  await maskButtons.first().click();
  await expect(page.locator('img[alt="選択マスク"]')).toBeVisible();
});
