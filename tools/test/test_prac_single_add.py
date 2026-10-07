#!/usr/bin/env python3
"""Check single-site PRAC point roles against baseline traces and a ladder oracle."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stat'))
from ecm_prac_plan import P95_RATIOS, prac_counts, primes
from test_cuda_prac_windows import add, double, multiply
from test_prac_disjoint_add import add_disjoint


def double_compact(point,a24,n):
    # Follow the candidate's AA/BB -> X, E, a24*E, BB+a24*E, Z schedule.
    # Values are ordinary normalized integers here, not native Montgomery limbs.
    x,z=point;a=(x+z)**2%n;b=(x-z)**2%n
    ox=a*b%n;a=(a-b)%n;oz=a24*a%n;b=(b+oz)%n;oz=a*b%n
    return ox,oz


def chain(p,r,seed,a24,n,single,compact=False,disjoint=False):
    dbl_point=double_compact if compact else double
    add_point=add_disjoint if disjoint else add
    a=seed;b=dbl_point(seed,a24,n);c=seed
    e=p-r;d=r-e;trace=[];rules=Counter();dbl=1;dadd=0
    while True:
        finish=d==e
        if not finish and d<e:d,e=e,d;a,b=b,a
        rule=0 if finish or 100*d<=296*e else 1 if d%2==e%2 else 2 if d%2==0 else 3
        rules['final' if finish else str(rule)]+=1
        dadd+=1
        if single:
            if rule==3:a,b=b,a
            if rule in (2,3):b,c=c,b
            t=add_point(a,b,c,n)
            if finish:a=t;break
            if rule==0:c,b=b,t;d-=e
            else:
                a=dbl_point(a,a24,n);dbl+=1
                if rule==1:b=t;d=(d-e)//2
                else:
                    if rule==2:b=c;d//=2
                    else:b,a=a,c;e//=2
                    c=t
        else:
            if finish:a=add(b,a,c,n);break
            if rule==0:c=add(a,b,c,n);b,c=c,b;d-=e
            else:
                if rule==1:b,c=c,b
                if rule==3:a,b=b,a
                c=add(a,c,b,n);a=dbl_point(a,a24,n);dbl+=1
                if rule==1:b,c=c,b;d=(d-e)//2
                elif rule==2:d//=2
                else:a,b=b,a;e//=2
        trace.append((d,e,a,b,c))
    assert d==e==1
    return a,trace,(dbl,dadd),rules


def cached_pairs(path,count=32):
    with path.open('rb') as f:
        h=struct.unpack('<6Q4I',f.read(64));pairs=[]
        for first in (0,(h[3]-count)//2,h[3]-count):
            f.seek(64+16*first)
            pairs += [(p,d) for p,d,_,_ in (struct.unpack('<4I',f.read(16)) for _ in range(count)) if p!=2]
        return pairs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,default=Path('build_cuda_cmake/prac'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();pairs=set()
    for p in primes(10000):
        if p==2:continue
        for ratio in P95_RATIOS[:7]:
            r=int(p*ratio+0.5)
            if p//2<r<p:pairs.add((p,r))
    for b1 in (10000000,260000000):pairs.update(cached_pairs(args.cache/f'prac_v1_b{b1}_t1_s7.bin'))
    n=2**127-1;coverage=Counter();steps=0
    for sigma in (26,4611686018427511360):
        u=(sigma*sigma-5)%n;v=4*sigma%n
        seed=pow(u,3,n),pow(v,3,n)
        a24=pow(v-u,3,n)*(3*u+v)*pow(16*pow(u,3,n)*v%n,-1,n)%n
        for p,r in sorted(pairs):
            baseline=chain(p,r,seed,a24,n,False);single=chain(p,r,seed,a24,n,True)
            combined=chain(p,r,seed,a24,n,True,True)
            distinct=chain(p,r,seed,a24,n,True,True,True)
            assert baseline==single==combined==distinct,(sigma,p,r,'point roles/projective bytes/counts differ')
            assert single[2]==prac_counts(p,r)
            oracle=multiply(sigma,p,n)
            assert single[0][0]*oracle[1]%n==oracle[0]*single[0][1]%n
            coverage.update(single[3]);steps+=len(single[1])+1
    assert all(coverage[k]>0 for k in ('0','1','2','3','final'))
    source=Path('kernels/cuda/cgbn_stage1_prac_single_add.cuh')
    report=dict(prime_d_pairs=len(pairs),sigmas=2,chains=2*len(pairs),point_role_steps=steps,
        models=['baseline','single-add','single-compact-v1','single-compact-disjoint'],model_chain_evaluations=8*len(pairs),
        rule_coverage=dict(coverage),modulus_bits=127,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        arithmetic_source_sha256=hashlib.sha256(Path('kernels/cuda/cgbn_stage1_prac_kernel.cuh').read_bytes()).hexdigest(),
        scope='CPU point-role model; exact baseline intermediate/final XZ and independent ladder; not native GPU verification',passed=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
