const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const {pathToFileURL} = require('node:url');
const path = require('node:path');
(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1792,height:1736},deviceScaleFactor:1});
    await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
    await page.locator('main').screenshot({path:path.resolve(process.argv[3]),omitBackground:true});
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
