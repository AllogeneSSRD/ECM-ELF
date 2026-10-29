(function () {
  'use strict';
  const M=window.EcmStage1Math,byId=id=>document.getElementById(id);
  const ns='http://www.w3.org/2000/svg',reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  let model=M.trace(5),index=0,running=false,timer=0;
  const widths={p:420,q:420};
  const point=P=>P===null?'O':`(${P[0]}, ${P[1]})`;
  const powers=n=>M.factorPowers(n).map(({prime,exponent})=>`${prime}${exponent>1?'^'+exponent:''}`).join(' × ');
  function element(tag,attrs,parent,text) {
    const node=document.createElementNS(ns,tag);
    Object.entries(attrs).forEach(([key,value])=>node.setAttribute(key,String(value)));
    if(text!==undefined)node.textContent=text;
    parent.appendChild(node);return node;
  }
  function drawField(key,f,R0,R1) {
    const svg=byId('field-'+key),g=byId('plot-'+key),W=widths[key],H=W;
    svg.setAttribute('viewBox',`0 0 ${W} ${H}`);g.replaceChildren();
    const left=40,top=46,side=W-90,bottom=top+side;
    const xy=P=>[left+Number(P[0])/Number(f.p-1n)*side,bottom-Number(P[1])/Number(f.p-1n)*side];
    const ticks=[0,.25,.5,.75,1].map(v=>Math.round(v*Number(f.p-1n)));
    for(const value of ticks){
      const x=left+value/Number(f.p-1n)*side,y=bottom-value/Number(f.p-1n)*side;
      element('line',{x1:x,x2:x,y1:top,y2:bottom,stroke:'#233d4c','stroke-width':1},g);
      element('line',{x1:left,x2:left+side,y1:y,y2:y,stroke:'#233d4c','stroke-width':1},g);
      element('text',{x,y:bottom+19,'text-anchor':'middle',class:'tick'},g,value);
      element('text',{x:left-9,y:y+4,'text-anchor':'end',class:'tick'},g,value);
    }
    element('text',{x:left+side/2,y:H-3,'text-anchor':'middle',class:'axis-title'},g,`x mod ${f.p}`);
    element('text',{x:4,y:top-13,class:'axis-title'},g,`y mod ${f.p}`);
    element('rect',{x:left,y:top,width:side,height:side,fill:'none',stroke:'#426274','stroke-width':1},g);
    for(const P of f.points){
      const [x,y]=xy(P),n=element('circle',{cx:x,cy:y,r:2.6,fill:'#608995',opacity:.8},g);
      element('title',{},n,`曲线点 ${point(P)}，模 ${f.p}`);
    }
    const trail=model.states.slice(Math.max(0,index-4),index);
    trail.forEach((s,i)=>{const P=key==='p'?s.p0:s.q0;if(P!==null){const [x,y]=xy(P);element('circle',{cx:x,cy:y,r:4.5,fill:'#77ddbe',opacity:.15+i*.1},g);}});
    const ox=W-83,oy=22;
    element('rect',{x:W-126,y:7,width:119,height:32,rx:5,fill:'#142b38',stroke:'#426274'},g);
    element('text',{x:W-116,y:28},g,'O');
    function marker(P,color,kind,label) {
      const positions=P===null?[[ox+(label==='R1'?35:0),oy]]:[xy(P)];
      if(P!==null && P[1]!==0n)positions.push(xy([P[0],M.mod(-P[1],f.p)]));
      positions.forEach(([x,y],i)=>{
        const attrs={fill:i?'none':color,stroke:color,'stroke-width':2,opacity:i?.7:1};
        const n=kind==='circle'?element('circle',{cx:x,cy:y,r:6,...attrs},g):element('rect',{x:x-5,y:y-5,width:10,height:10,...attrs},g);
        element('title',{},n,`${label}${i?' 的负点':''} = ${point(i?[P[0],M.mod(-P[1],f.p)]:P)}`);
        if(P===null)element('text',{x,y:oy+3,'text-anchor':'middle',fill:'#091119','font-size':9},g,label==='R0'?'0':'1');
      });
    }
    marker(R0,'#77ddbe','circle','R0');marker(R1,'#85bff4','square','R1');
    byId('desc-'+key).textContent=`模 ${f.p} 有 ${f.points.length} 个仿射点和 O。当前 R0=${point(R0)}，R1=${point(R1)}；空心标记表示其负点。`;
  }
  function orderProof(id,f) {
    const target=byId(id),divides=model.s%f.order===0n;
    target.replaceChildren();
    const label=document.createElement('span');label.className=divides?'covered':'missing';
    label.textContent=divides?'点阶被标量完整覆盖':'点阶尚未被标量完整覆盖';
    target.append(document.createTextNode(`mod ${f.p}：ord(P) = ${f.order} = ${powers(f.order)}；`),label);
  }
  function draw() {
    const s=model.states[index],final=index===model.states.length-1;
    byId('scalar-equation').textContent=`s = lcm(1,…,${model.B1}) = ${powers(model.s)} = ${model.s}；二进制 ${model.bits}`;
    orderProof('order-p',M.pField);orderProof('order-q',M.qField);
    byId('point-p0').textContent=`R0 = [${s.k}]P = ${point(s.p0)}`;
    byId('point-p1').textContent=`R1 = [${s.k+1n}]P = ${point(s.p1)}`;
    byId('point-q0').textContent=`R0 = [${s.k}]P = ${point(s.q0)}`;
    byId('point-q1').textContent=`R1 = [${s.k+1n}]P = ${point(s.q1)}`;
    drawField('p',M.pField,s.p0,s.p1);drawField('q',M.qField,s.q0,s.q1);
    byId('prefix-equation').textContent=index===0?'初态：R0 = O，R1 = P；始终保持 R1 − R0 = P'
      :`消费位 b = ${s.bit}：k = 2 × ${s.previousK} + ${s.bit} = ${s.k}；R1 − R0 = P`;
    byId('double-operation').textContent=index===0?'等待首位':s.bit==='0'?`[${s.previousK}]P → [${2n*s.previousK}]P → R0`:`[${s.previousK+1n}]P → [${2n*(s.previousK+1n)}]P → R1`;
    byId('add-operation').textContent=index===0?'固定差点为 P':`[${s.previousK}]P + [${s.previousK+1n}]P → [${2n*s.previousK+1n}]P → ${s.bit==='0'?'R1':'R0'}`;
    byId('operation-detail').textContent=index===0?'初始点 P 的 x-only 坐标为 (177:156)，仿射 x=943 mod 1139。':`两条右侧结果均使用更新前的双点；xADD 额外使用差点 P，不能从两个 x 坐标直接做一般加法。`;
    byId('xz0').textContent=`R0 (X:Z) = (${s.R0[0]}:${s.R0[1]}) mod 1139`;
    byId('xz1').textContent=`R1 (X:Z) = (${s.R1[0]}:${s.R1[1]}) mod 1139`;
    byId('crt-z').textContent=`R0 的 Z 约化：${M.mod(s.R0[1],17n)} mod 17；${M.mod(s.R0[1],67n)} mod 67`;
    byId('outcome').className='math-outcome'+(final?' '+model.outcome:'');
    byId('gcd-result').textContent=final?`gcd(${s.R0[1]}, 1139) = ${model.d}`:'完成 [s]P 后计算 gcd(Z, N)';
    byId('outcome-explanation').textContent=!final?'此处显示数学状态，不估计发现因子的概率；自动演算在末点停留后重播。'
      :model.outcome==='factor'?`模 17 时 [${model.s}]P=O，模 67 时仍是有限点。Z 只被 17 整除，因此暴露因子 17。`
      :model.outcome==='saturated'?`两个点阶都整除 ${model.s}，两个域都归零；末点 Z=0，gcd=N，不能把 1139 当成非平凡因子。`
      :`两个点阶都不整除 ${model.s}，两边末点都非 O；gcd=1，本次 Stage1 未检出因子。`;
    [...byId('bits').children].forEach((button,i)=>{button.setAttribute('aria-pressed',String(i===index));button.classList.toggle('past',i<index);});
    byId('previous').disabled=index===0;byId('next').disabled=final;
    byId('play').textContent=running?'暂停演算':'自动演算';byId('play').setAttribute('aria-pressed',String(running));
  }
  function halt(){running=false;clearTimeout(timer);timer=0;}
  function schedule(){
    clearTimeout(timer);timer=0;
    if(running&&!document.hidden)timer=setTimeout(()=>{index=index===model.states.length-1?0:index+1;draw();schedule();},index===model.states.length-1?4200:1500);
  }
  function buildBits(){
    byId('bits').replaceChildren();
    ['初态',...model.bits].forEach((bit,i)=>{const button=document.createElement('button');button.type='button';button.textContent=bit;button.setAttribute('aria-label',i?`标量第 ${i} 位，值 ${bit}`:'ladder 初态');button.addEventListener('click',()=>{halt();index=i;draw();});byId('bits').appendChild(button);});
  }
  byId('play').addEventListener('click',()=>{running=!running;if(running&&index===model.states.length-1)index=0;draw();schedule();});
  byId('previous').addEventListener('click',()=>{halt();index=Math.max(0,index-1);draw();});
  byId('next').addEventListener('click',()=>{halt();index=Math.min(model.states.length-1,index+1);draw();});
  byId('reset').addEventListener('click',()=>{halt();index=0;draw();});
  byId('bound').addEventListener('change',event=>{model=M.trace(Number(event.target.value));index=0;buildBits();draw();schedule();});
  document.addEventListener('visibilitychange',()=>{clearTimeout(timer);timer=0;if(!document.hidden)schedule();});
  reduced.addEventListener('change',()=>{if(reduced.matches){halt();draw();}});
  const observer=new ResizeObserver(entries=>{for(const entry of entries){if(entry.contentRect.width>0)widths[entry.target.id==='field-p'?'p':'q']=entry.contentRect.width;}draw();});
  buildBits();draw();observer.observe(byId('field-p'));observer.observe(byId('field-q'));
  if(!reduced.matches){running=true;draw();schedule();}
})();
