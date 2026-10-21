// Offline generated-document QA. Requires Playwright and installed Microsoft Edge.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const assert = require('node:assert/strict');
(async () => {
  const root = path.resolve(__dirname, '../../..');
  const out = path.resolve(process.argv[2]);
  fs.mkdirSync(out); // Preserve earlier evidence; require a fresh directory.
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  const completed = Number(process.argv[3] || 4);
  page.on('pageerror', error => errors.push(String(error)));
  try {
    await page.goto(pathToFileURL(path.join(root, 'docs/web/gui-v2-vulkan-plan-20261021.html')).href);
    assert.match(await page.locator('#progress').innerText(), /P2：已完成/);
    assert.equal(await page.locator('[data-phase]').count(), 7);
    assert.equal(await page.locator('[data-state="已完成"]').count(), completed);
    for (let i = 0; i < 7; ++i) {
      await page.locator(`[data-phase="${i}"]`).click();
      assert.match(await page.locator('#phase-index').innerText(), new RegExp(`P${i} · ${i < completed ? '已完成' : '待实施'}`));
      assert.ok((await page.locator('#phase-gate').innerText()).length > 30);
    }
    await page.locator('#theme-toggle').click();
    assert.equal(await page.locator('html').getAttribute('data-theme'), 'dark');
    await page.locator('#narrow-toggle').click();
    assert.equal(await page.locator('#layout-mock').getAttribute('data-width'), 'narrow');
    await page.locator('a[href="#full-plan"]').click();
    assert.equal(await page.locator('#full-details').getAttribute('open'), '');
    assert.match(await page.locator('.full-body').innerText(), /P0–P[2-6] 已完成/);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({path: path.join(out, 'desktop.png')});
    await page.setViewportSize({width: 390, height: 844});
    await page.evaluate(() => window.scrollTo(0, 0));
    const widths = await page.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
    assert.ok(widths[0] <= widths[1] + 1, widths);
    await page.screenshot({path: path.join(out, 'mobile.png')});
    assert.deepEqual(errors, []);
    const result = {passed: true, phases: 7, completed, consoleErrors: errors, mobileWidths: widths};
    fs.writeFileSync(path.join(out, 'result.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
