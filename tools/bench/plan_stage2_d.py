"""Offline D ranking with the measured phase model; integrated GPU plans remain authoritative."""
import argparse
import json
import math
from pathlib import Path
from functools import lru_cache
from calibrate_stage2_d import tree,inverse,unit,shape
from fit_stage2_d import predict

PRIMES=(2,3,5,7,11,13,17,19,23,29,31,37,41,43,47)

def candidates(limit):
    stack=[(0,1,1)]
    while stack:
        pos,d,tot=stack.pop()
        if pos==len(PRIMES):
            if tot//2:yield d,tot//2
            continue
        prime=PRIMES[pos];v=d;t=tot
        stack.append((pos+1,v,t))
        first=True
        while v<=limit//prime:
            v*=prime;t*=prime-1 if first else prime;first=False
            stack.append((pos+1,v,t))

@lru_cache(None)
def static(p,bits):
    n=shape(p+1,bits)[0];top=(1<<((p-1).bit_length()-1)) if p>1 else 1
    nt=shape(top+1,bits)[0]
    # Existing conservative arena filter, plus actual owner budget. This is an
    # estimate; runtime allocation/eviction/headroom still decides the path.
    nominal=shape(p//2+1,bits)[0]
    arena=(3*n+2*p+1)+2*(3*nominal+2*(p//2+1)-1)
    return n,nt,arena*8,8*((bits+63)//64)*(9*p+8)+48

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--b2',type=int,required=True)
    p.add_argument('--bits',type=int,default=4423);p.add_argument('--arena-mb',type=int,default=6300)
    p.add_argument('--fold-mb',type=int,default=640);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();fit=json.loads(a.fit.read_text());rates=fit['rates'];profile=fit.get('feature_profile',0);rows=[];count=0
    for d,pb in candidates(200000000):
        count+=1
        owner=8*((a.bits+63)//64)*(9*pb+8)+48
        if owner>a.fold_mb*(1<<20):continue
        try:n,nt,arena,owner=static(pb,a.bits)
        except ValueError:continue
        if arena>a.arena_mb*(1<<20):continue
        i=a.b2//d+2;g=(i+pb-1)//pb;q,r=divmod(i,pb)
        # G=1 takes a different direct-remainder path, absent from this fit.
        if g<2:continue
        f=dict(D=d,P=pb,I=i,G=g,baby=pb*max(1,math.log2(d)-2),affine=pb,ftree=tree(pb,a.bits,profile),
               gtrees=q*tree(pb,a.bits,profile)+(tree(r,a.bits,profile) if r else 0),fold=(g-1)*unit(pb+1,a.bits,profile),
               descent=tree(pb,a.bits,profile),inverse=inverse(pb+1,a.bits,profile),giant=i*(6+22*math.log2(a.b2)/64),accum=pb)
        pred=predict(f,rates)
        rows.append(dict(features=f,prediction=pred,arena_bytes=arena,owner_bytes=owner,fold_ntt=n,tree_ntt=nt))
    rows.sort(key=lambda r:r['prediction']['full'])
    result={'candidates':count,'admitted':len(rows),'bits':a.bits,'B2':a.b2,'feature_profile':profile,'arena_mb':a.arena_mb,'fold_mb':a.fold_mb,'top':rows[:20]}
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'candidates':count,'admitted':len(rows),'top':[(r['features']['D'],r['features']['P'],r['prediction']['full']) for r in rows[:6]]}))

if __name__=='__main__':main()
