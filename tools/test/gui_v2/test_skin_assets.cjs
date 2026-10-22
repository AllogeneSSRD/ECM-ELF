// Inspect all generated SVG poses and the real PNG/metadata contract.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {pathToFileURL}=require('node:url');
(async()=>{
  const root=path.resolve(__dirname,'../../..'),input=path.resolve(process.argv[2]),out=path.resolve(process.argv[3]);fs.mkdirSync(out);
  const manifest=JSON.parse(fs.readFileSync(path.join(root,'assets/gui_v2/skins/manifest.json'),'utf8'));
  const browser=await chromium.launch({channel:'msedge',headless:true});const hashes=[],reports=[];
  try{
    for(const skin of manifest.skins){
      const folder=path.join(input,skin.id),png=fs.readFileSync(path.join(folder,skin.atlas)),meta=JSON.parse(fs.readFileSync(path.join(folder,skin.atlas.replace('.png','.json')),'utf8'));
      const hash=crypto.createHash('sha256').update(png).digest('hex');assert.equal(meta.png_sha256,hash);hashes.push(hash);
      assert.equal(png.readUInt32BE(16),1792);assert.equal(png.readUInt32BE(20),1736);assert.equal(png[25],6);assert.equal(meta.english,skin.english);
      const page=await browser.newPage({viewport:{width:1792,height:1736}});await page.goto(pathToFileURL(path.join(folder,'atlas.html')).href);
      const cells=page.locator('main>svg');assert.equal(await cells.count(),56);
      const ids=await page.locator('[id]').evaluateAll(nodes=>nodes.map(n=>n.id));assert.equal(new Set(ids).size,ids.length);
      assert.ok(await cells.evaluateAll(cells=>cells.every(svg=>[...svg.querySelectorAll('[fill]')].every(n=>{
        const id=n.getAttribute('fill').match(/^url\(#(.+)\)$/)?.[1];return !id||document.getElementById(id)?.ownerSVGElement===svg;
      }))),'Gradient definitions must belong to their own pose');
      for(let frame=0;frame<8;frame++){
        const cell=cells.nth(40+frame);const bounds=await cell.evaluate(svg=>[...svg.querySelectorAll('[data-part="z1"],[data-part="z2"]')].map(n=>{const a=n.getBoundingClientRect(),b=svg.getBoundingClientRect();return {x:a.x-b.x,y:a.y-b.y,right:a.right-b.x,bottom:a.bottom-b.y,w:b.width,h:b.height};}));
        assert.equal(bounds.length,2);for(const b of bounds)assert.ok(b.x>=0&&b.y>=0&&b.right<=b.w&&b.bottom<=b.h,JSON.stringify(b));
        const ornament=await cells.nth(frame).evaluate(svg=>[...svg.querySelectorAll('[data-part$="feather-mask"],[data-part$="white-crown"],[data-part$="beret"],[data-part$="rose-cluster"]')].map(n=>{const a=n.getBoundingClientRect(),b=svg.getBoundingClientRect();return {x:a.x-b.x,y:a.y-b.y,right:a.right-b.x,bottom:a.bottom-b.y,w:b.width,h:b.height};}));
        assert.ok(ornament.length);for(const b of ornament)assert.ok(b.x>=0&&b.y>=0&&b.right<=b.w&&b.bottom<=b.h,JSON.stringify(b));
      }
      for(const [mood,index] of [['calm',0],['doze',40]])await cells.nth(index).screenshot({path:path.join(out,`${skin.id}-${mood}.png`),omitBackground:true});
      reports.push({skin:skin.id,english:skin.english,frames:56,unique_ids:ids.length,sleep_marks_complete:true,accessories_complete:true,sha256:hash});await page.close();
    }
    assert.equal(new Set(hashes).size,3);const report={passed:true,skins:reports};fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
