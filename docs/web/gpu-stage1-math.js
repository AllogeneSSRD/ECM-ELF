/* Exact arithmetic for the small, explicitly labelled ECM teaching example. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.EcmStage1Math = api;
})(typeof window !== 'undefined' ? window : this, function () {
  'use strict';
  const mod = (a, n) => (a % n + n) % n;
  function gcd(a, b) { while (b) [a, b] = [b, a % b]; return a < 0n ? -a : a; }
  function inverse(a, n) {
    let r = n, s = mod(a, n), x = 0n, y = 1n;
    while (s) { const q = r / s; [r, s] = [s, r - q * s]; [x, y] = [y, x - q * y]; }
    if (r !== 1n) throw new Error('Noninvertible denominator');
    return mod(x, n);
  }
  function construct(n, sigma) {
    const u = sigma * sigma - 5n, v = 4n * sigma;
    const X = mod(u ** 3n, n), Z = mod(v ** 3n, n);
    const a24 = mod((v-u) ** 3n * (3n*u+v) * inverse(16n*u**3n*v,n),n);
    const A = mod(4n*a24-2n,n), x = mod(X*inverse(Z,n),n);
    // Choosing y(P)=1 fixes an equivalent representative of the Montgomery twist.
    const B = mod(x**3n+A*x*x+x,n);
    if (gcd(B*(A*A-4n),n)!==1n) throw new Error('Singular teaching curve');
    return { n, sigma, A, B, a24, P: [x,1n], base: [X,Z] };
  }
  function field(curve, p) {
    const f = {p,A:mod(curve.A,p),B:mod(curve.B,p),P:curve.P.map(x=>mod(x,p))};
    f.points = [];
    for (let x=0n;x<p;x++) for (let y=0n;y<p;y++)
      if (mod(f.B*y*y-x**3n-f.A*x*x-x,p)===0n) f.points.push([x,y]);
    f.order = 1n;
    for (let Q=f.P;Q!==null;f.order++) Q=add(Q,f.P,f);
    return f;
  }
  function add(P,Q,f) {
    if(P===null)return Q;
    if(Q===null)return P;
    const [x,y]=P,[u,v]=Q,p=f.p;
    if(x===u && mod(y+v,p)===0n)return null;
    const slope=x===u ? mod((3n*x*x+2n*f.A*x+1n)*inverse(2n*f.B*y,p),p)
      : mod((v-y)*inverse(u-x,p),p);
    const xx=mod(f.B*slope*slope-f.A-x-u,p);
    return [xx,mod(slope*(x-xx)-y,p)];
  }
  function multiply(P,k,f) {
    let R=null,Q=P;
    while(k>0n){if(k&1n)R=add(R,Q,f);Q=add(Q,Q,f);k>>=1n;}
    return R;
  }
  function doubleXZ(P,c) {
    const [X,Z]=P,n=c.n;
    const aa=mod((X+Z)**2n,n),bb=mod((X-Z)**2n,n),e=mod(aa-bb,n);
    return [mod(aa*bb,n),mod(e*(bb+c.a24*e),n)];
  }
  function addXZ(P,Q,D,c) {
    const [X,Z]=P,[U,V]=Q,[dX,dZ]=D,n=c.n;
    const a=mod((X+Z)*(U-V),n),b=mod((X-Z)*(U+V),n);
    return [mod(dZ*(a+b)**2n,n),mod(dX*(a-b)**2n,n)];
  }
  function scalar(B1) {
    let s=1n;
    for(let i=2n;i<=BigInt(B1);i++)s=s/gcd(s,i)*i;
    return s;
  }
  function factorPowers(v) {
    const result=[];
    for(let p=2n;p*p<=v;p++)if(v%p===0n){let exponent=0;while(v%p===0n){v/=p;exponent++;}result.push({prime:p,exponent});}
    if(v>1n)result.push({prime:v,exponent:1});
    return result;
  }
  const curve=construct(1139n,6n),pField=field(curve,17n),qField=field(curve,67n);
  function trace(B1) {
    const s=scalar(B1),bits=s.toString(2),states=[];
    let k=0n,R0=[1n,0n],R1=curve.base;
    const snapshot=(bit,previousK)=>({k,R0,R1,bit,previousK,
      p0:multiply(pField.P,k,pField),p1:multiply(pField.P,k+1n,pField),
      q0:multiply(qField.P,k,qField),q1:multiply(qField.P,k+1n,qField)});
    states.push(snapshot(null,0n));
    for(const b of bits){
      const sum=addXZ(R0,R1,curve.base,curve),previousK=k;
      if(b==='0'){R0=doubleXZ(R0,curve);R1=sum;}
      else {R0=sum;R1=doubleXZ(R1,curve);}
      k=2n*k+BigInt(b);states.push(snapshot(b,previousK));
    }
    const d=gcd(R0[1],curve.n);
    return {B1,s,bits,states,d,outcome:d===1n?'none':d===curve.n?'saturated':'factor'};
  }
  return {mod,gcd,inverse,construct,field,add,multiply,doubleXZ,addXZ,scalar,factorPowers,curve,pField,qField,trace};
});
