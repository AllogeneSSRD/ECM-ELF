// Offline gallery checks; never starts the GUI or a graphics backend.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { createHash } = require('node:crypto');
const assert = require('node:assert/strict');
(async () => {
  const root = path.resolve(__dirname, '../../..');
  const out = path.resolve(process.argv[2]);
  fs.mkdirSync(out); // Require fresh evidence, retain previous runs.
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
  const errors = [], network = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('request', request => { if (/^https?:/.test(request.url())) network.push(request.url()); });
  try {
    await page.goto(pathToFileURL(path.join(root, 'docs/web/gui-v2-icons.html')).href);
    const source = fs.readFileSync(path.join(root, 'src/ecm/application/gui_v2/ui/icons.h'));
    const names = source.toString('utf8').match(/enum class Icon\s*\{([^}]+)\}/)[1].split(',').map(s => s.trim());
    const data = await page.locator('#icon-data').evaluate(e => JSON.parse(e.textContent));
    assert.deepEqual(data.map(i => i.name), names);
    assert.equal(names.length, 18);
    assert.equal(data.find(i => i.name === 'None').svg, '');
    assert.ok(data.filter(i => i.name !== 'None').every(i => i.svg.length));
    assert.equal(await page.locator('#source-sha').innerText(), createHash('sha256').update(source).digest('hex'));
    for (const name of names) {
      const button = page.locator(`[data-icon="${name}"]`);
      await button.click();
      assert.equal(await button.getAttribute('aria-pressed'), 'true');
      assert.equal(await page.locator('#identifier').innerText(), 'Icon::' + name);
      assert.equal(await page.locator('#code').inputValue(), 'Icon::' + name);
      assert.equal(await page.locator('#sizes .sample').count(), 5);
      assert.equal(await page.locator('#preview svg').count(), name === 'None' ? 0 : 1);
    }
    await page.locator('[data-group="操作"]').click();
    assert.deepEqual(await page.locator('.tile').evaluateAll(es => es.map(e => e.dataset.icon)), ['Play', 'Stop', 'Calculate']);
    await page.locator('[data-group="全部"]').click();
    await page.locator('#search').fill('paLETte');
    assert.equal(await page.locator('.tile').count(), 1);
    await page.locator('#search').fill('概率');
    assert.equal(await page.locator('.tile').getAttribute('data-icon'), 'Chart');
    await page.locator('#search').fill('no-such-icon');
    assert.equal(await page.locator('.tile').count(), 0);
    assert.equal(await page.locator('#empty').isVisible(), true);
    await page.locator('#search').fill('');
    await page.locator('[data-icon="Play"]').focus();
    await page.keyboard.press('Enter');
    assert.equal(await page.locator('#code').inputValue(), 'Icon::Play');
    // file:// or browser policy can block clipboard: selected text remains usable.
    await page.evaluate(() => {
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
      document.execCommand = () => false;
    });
    await page.locator('#copy').click();
    assert.match(await page.locator('#status').innerText(), /Ctrl\+C/);
    assert.equal(await page.locator('#code').evaluate(e => e.value.slice(e.selectionStart, e.selectionEnd)), 'Icon::Play');
    await page.locator('[data-icon="Play"]').click();
    await page.screenshot({ path: path.join(out, 'desktop-dark.png'), fullPage: true });
    await page.locator('#theme').click();
    assert.equal(await page.locator('html').getAttribute('data-theme'), 'light');
    await page.screenshot({ path: path.join(out, 'desktop-light.png'), fullPage: true });
    const widths = [];
    for (const width of [720, 390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      widths.push(await page.evaluate(() => [innerWidth, document.documentElement.scrollWidth]));
      assert.ok(widths.at(-1)[1] <= width, JSON.stringify(widths));
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('[data-icon="Pet"]').click();
    assert.equal(await page.locator('#identifier').innerText(), 'Icon::Pet');
    await page.screenshot({ path: path.join(out, 'mobile.png'), fullPage: true });
    assert.deepEqual(errors, []);
    assert.deepEqual(network, []);
    const result = { passed: true, icons: names.length - 1, none: true, selectedAll: true,
      keyboard: true, clipboardFallback: true, themes: 2, widths, errors, network };
    fs.writeFileSync(path.join(out, 'result.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
