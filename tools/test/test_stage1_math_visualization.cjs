'use strict';
const assert=require('node:assert/strict');
const M=require('../../docs/web/gpu-stage1-math.js');
let checked=0;
function checkXZ(projective,affine,f) {
  const X=M.mod(projective[0],f.p),Z=M.mod(projective[1],f.p);
  assert.ok(X!==0n||Z!==0n,'Projective pair must not vanish');
  if(affine===null){assert.equal(Z,0n);return;}
  assert.notEqual(Z,0n);
  assert.equal(M.mod(X*M.inverse(Z,f.p),f.p),affine[0]);
  assert.equal(M.mod(f.B*affine[1]**2n-affine[0]**3n-f.A*affine[0]**2n-affine[0],f.p),0n);
}
for(const f of [M.pField,M.qField]){
  assert.equal(M.mod(f.B*(f.A*f.A-4n),f.p)===0n,false);
  assert.equal(BigInt(f.points.length+1)%f.order,0n);
  assert.equal(M.multiply(f.P,f.order,f),null);
  for(let k=1n;k<f.order;k++)assert.notEqual(M.multiply(f.P,k,f),null);
}
for(let bound=2;bound<=18;bound++){
  const t=M.trace(bound);
  assert.equal(t.states.at(-1).k,t.s);
  for(let k=1n;k<=BigInt(bound);k++)assert.equal(t.s%k,0n);
  for(const state of t.states){
    for(const [f,P,Q] of [[M.pField,state.p0,state.p1],[M.qField,state.q0,state.q1]]){
      checkXZ(state.R0,P,f);checkXZ(state.R1,Q,f);
      const minusP=P===null?null:[P[0],M.mod(-P[1],f.p)];
      assert.deepEqual(M.add(Q,minusP,f),f.P,'Ladder difference remains P');checked++;
    }
  }
  const expected=(t.s%M.pField.order===0n?17n:1n)*(t.s%M.qField.order===0n?67n:1n);
  assert.equal(t.d,expected,'CRT infinity classification matches final GCD');
}
for(const [bound,s,d,xz] of [[3,6n,1n,[935n,909n]],[5,60n,17n,[356n,187n]],[9,2520n,1139n,[528n,0n]]]){
  const t=M.trace(bound);assert.equal(t.s,s);assert.equal(t.d,d);assert.deepEqual(t.states.at(-1).R0,xz);
}
console.log(`PASS: ${checked} field-state checks; B1=2..18; affine/XZ agreement, ladder invariant, point orders, and all three GCD outcomes.`);
