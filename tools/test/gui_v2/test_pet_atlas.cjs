// Validate actual inline SVG mask ownership and complete sleeping marks.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
(async()=>{
  const html=path.resolve(process.argv[2]),out=path.resolve(process.argv[3]);fs.mkdirSync(out);
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try {
    const atlas=await browser.newPage({viewport:{width:1792,height:1736},deviceScaleFactor:1});await atlas.goto(pathToFileURL(html).href);
    const ids=await atlas.locator('[id]').evaluateAll(nodes=>nodes.map(n=>n.id));assert.equal(new Set(ids).size,ids.length,'SVG clip IDs must be unique across atlas cells');
    for(let frame=0;frame<8;frame++){
      const svg=atlas.locator('main > svg').nth(40+frame);
      assert.ok(await svg.evaluate(root=>[...root.querySelectorAll('[clip-path]')].every(n=>{const id=n.getAttribute('clip-path').match(/^url\(#(.+)\)$/)?.[1];return id&&document.getElementById(id)?.ownerSVGElement===root;})),'Clip masks must belong to this frame');
      const marks=await svg.locator('[data-part="z1"],[data-part="z2"]').evaluateAll(nodes=>nodes.map(n=>{const a=n.getBoundingClientRect(),b=n.ownerSVGElement.getBoundingClientRect();return a.left>=b.left&&a.top>=b.top&&a.right<=b.right&&a.bottom<=b.bottom;}));
      assert.equal(marks.length,2);assert.ok(marks.every(Boolean),'Both upstream sleep marks must fit inside the cell');
      await svg.screenshot({omitBackground:true,path:path.join(out,`doze-${frame}.png`)});
    }
    const result={passed:true,unique_clip_ids:ids.length,owning_svg_mask_frames:8,complete_upstream_sleep_marks_per_frame:2,limits:['Cross-document screenshot byte equality is not a valid mask test: browser rasterization can differ by small RGB rounding. Native adjacent-tile contamination is tested separately.']};fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
