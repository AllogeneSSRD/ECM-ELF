const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
(async()=>{
  const root=path.resolve(__dirname,'../../..'),out=path.resolve(process.argv[2]);fs.mkdirSync(out);
  const browser=await chromium.launch({channel:'msedge',headless:true});const errors=[];
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1060}});page.on('pageerror',e=>errors.push(String(e)));
    await page.goto(pathToFileURL(path.join(root,'docs/web/ely-skins-20261022.html')).href);
    assert.equal(await page.locator('.card').count(),3);assert.equal(await page.locator('[data-mood]').count(),7);
    assert.equal(await page.locator('button[data-eyes]').count(),2);
    assert.equal(await page.locator('body').getAttribute('data-eyes'),'diamond');
    const atlasUrls=new Set();
    for(const mode of ['original','diamond']){
      await page.locator(`button[data-eyes="${mode}"]`).click();
      assert.equal(await page.locator(`button[data-eyes="${mode}"]`).getAttribute('aria-pressed'),'true');
      assert.match(await page.locator('#variant-note').innerText(),mode==='diamond'?/仅 Web 演示，未接入程序/:/程序中已登记/);
      for(const card of await page.locator('.card').all()){
        const sprite=card.locator('.sprite'),url=await sprite.getAttribute('data-'+mode);
        assert.equal(await sprite.evaluate(n=>n.style.backgroundImage),`url("${url}")`);
        assert.equal(await card.locator('.eye-detail').evaluate(n=>n.style.backgroundImage),`url("${url}")`);
        assert.equal(await card.locator('.atlas-link').getAttribute('href'),url);atlasUrls.add(new URL(url,page.url()).href);
      }
      for(let i=0;i<7;i++){await page.locator(`[data-mood="${i}"]`).click();assert.equal(await page.locator(`[data-mood="${i}"]`).getAttribute('aria-pressed'),'true')}
    }
    assert.equal(atlasUrls.size,6);
    for(const url of atlasUrls)assert.deepEqual(await page.evaluate(src=>new Promise(resolve=>{const image=new Image();image.onload=()=>resolve([image.naturalWidth,image.naturalHeight]);image.onerror=()=>resolve([]);image.src=src}),url),[1792,1736]);
    await page.locator('#play').click();assert.equal(await page.locator('#play').getAttribute('aria-pressed'),'false');
    for(let i=0;i<7;i++){await page.locator(`[data-mood="${i}"]`).click();assert.equal(await page.locator(`[data-mood="${i}"]`).getAttribute('aria-pressed'),'true');
      for(const s of await page.locator('.sprite').evaluateAll(nodes=>nodes.map(n=>({y:n.style.backgroundPositionY,h:n.style.height}))))assert.equal(parseFloat(s.y),i?-i*parseFloat(s.h):0);}
    await page.locator('[data-mood="0"]').click();await page.locator('.gallery').screenshot({path:path.join(out,'three-skins.png')});
    await page.locator('.eye-detail').first().screenshot({path:path.join(out,'diamond-eye-detail.png')});
    await page.locator('button[data-eyes="original"]').click();await page.locator('.gallery').screenshot({path:path.join(out,'original-skins.png')});
    await page.locator('button[data-eyes="diamond"]').click();
    await page.screenshot({path:path.join(out,'desktop.png')});
    await page.locator('[data-mood="5"]').click();await page.locator('#theme').click();await page.locator('#bg').click();await page.locator('#scale').selectOption('1');
    assert.equal(await page.locator('.sprite').first().evaluate(n=>n.style.width),'112px');
    await page.screenshot({path:path.join(out,'light-sleep-native-size.png')});
    await page.setViewportSize({width:390,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
    for(const url of await page.locator('.sprite').evaluateAll(nodes=>nodes.map(n=>getComputedStyle(n).backgroundImage.slice(5,-2))))assert.deepEqual(await page.evaluate(src=>new Promise(resolve=>{const image=new Image();image.onload=()=>resolve([image.naturalWidth,image.naturalHeight]);image.onerror=()=>resolve([]);image.src=src}),url),[1792,1736]);
    assert.deepEqual(errors,[]);const result={passed:true,skins:3,eye_versions:2,atlas_files:atlasUrls.size,moods:7,web_only_notice:true,eye_detail:true,play_pause:true,scales:true,light_dark:true,checkerboard:true,mobile_width:390,consoleErrors:errors};fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
