const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const {pathToFileURL} = require('node:url');
const path = require('node:path');
const fs = require('node:fs');
const crypto = require('node:crypto');
(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1792,height:1736},deviceScaleFactor:1});
    await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
    const output=path.resolve(process.argv[3]);
    if(path.extname(output).toLowerCase()!=='.png')throw Error('Atlas output must be PNG');
    await page.locator('main').screenshot({path:output,omitBackground:true});
    const metadata=JSON.parse(fs.readFileSync(path.join(path.dirname(path.resolve(process.argv[2])),'metadata.json'),'utf8'));
    metadata.png_sha256=crypto.createHash('sha256').update(fs.readFileSync(output)).digest('hex');
    fs.writeFileSync(output.replace(/\.png$/i,'.json'),JSON.stringify(metadata,null,2)+'\n');
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
