#!/usr/bin/env python3
"""Model normalized Montgomery xADD and the private output-disjoint contract."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def add_disjoint(p, q, difference, n):
    # Ordinary integer model of the two-temporary schedule; outputs are fresh.
    t=(p[0]+p[1])%n;u=(q[0]-q[1])%n;u=u*t%n
    t=(p[0]-p[1])%n;ox=(q[0]+q[1])%n;ox=ox*t%n
    t=(u+ox)%n;u=(u-ox)%n;t=t*t%n;u=u*u%n
    ox=difference[1]*t%n;u=difference[0]*u%n
    return ox,u


class Montgomery:
    def __init__(self, n, bits):
        self.n=n;self.bits=bits;self.R=1<<bits;self.mask=self.R-1
        self.np=-pow(n,-1,self.R)&self.mask;self.counts=Counter()

    def encode(self, value):
        return value*self.R%self.n

    def add(self, a, b):
        assert 0<=a<self.n and 0<=b<self.n
        value=a+b
        if value>=self.n:value-=self.n;self.counts['add_corrections']+=1
        return value

    def sub(self, a, b):
        assert 0<=a<self.n and 0<=b<self.n
        value=a-b
        if value<0:value+=self.n;self.counts['sub_corrections']+=1
        return value

    def raw(self, a, b):
        assert 0<=a<self.n and 0<=b<self.n
        product=a*b;m=product*self.np&self.mask
        numerator=product+m*self.n
        assert numerator&self.mask==0
        value=numerator>>self.bits
        assert 0<=value<2*self.n
        return value

    def mul(self, a, b, square=False):
        self.counts['sqr' if square else 'mul']+=1
        value=self.raw(a,b)
        if value>=self.n:value-=self.n;self.counts['REDC_corrections']+=1
        return value

    def square(self, a):
        return self.mul(a,a,True)


def run(env, coordinates, outputs, disjoint):
    # Indices model actual storage aliases. The modulus is never an output.
    b=list(coordinates)+[0,0];ox,oz=outputs;env.counts.clear()
    t=env.add(b[0],b[1]);u=env.sub(b[2],b[3]);u=env.mul(u,t)
    t=env.sub(b[0],b[1])
    if disjoint:
        b[ox]=env.add(b[2],b[3]);b[ox]=env.mul(b[ox],t)
        t=env.add(u,b[ox]);u=env.sub(u,b[ox])
        t=env.square(t);u=env.square(u)
        b[ox]=env.mul(b[5],t);u=env.mul(b[4],u);b[oz]=u
    else:
        v=env.add(b[2],b[3]);v=env.mul(v,t)
        t=env.add(u,v);u=env.sub(u,v);t=env.square(t);u=env.square(u)
        v=env.mul(b[5],t);u=env.mul(b[4],u);b[ox]=v;b[oz]=u
    assert env.counts['mul']==4 and env.counts['sqr']==2
    assert 0<=b[ox]<env.n and 0<=b[oz]<env.n
    return (b[ox],b[oz]),dict(env.counts)


def run_output_reuse(env, coordinates, outputs):
    # Model each early output write independently of the old local-u schedule.
    # Only (6,7) satisfies the private contract; aliased calls are negative controls.
    b=list(coordinates)+[0,0];ox,oz=outputs;env.counts.clear()
    t=env.add(b[0],b[1]);b[oz]=env.sub(b[2],b[3]);b[oz]=env.mul(b[oz],t)
    t=env.sub(b[0],b[1]);b[ox]=env.add(b[2],b[3]);b[ox]=env.mul(b[ox],t)
    t=env.add(b[oz],b[ox]);b[oz]=env.sub(b[oz],b[ox])
    t=env.square(t);b[oz]=env.square(b[oz])
    b[ox]=env.mul(b[5],t);b[oz]=env.mul(b[4],b[oz])
    assert env.counts['mul']==4 and env.counts['sqr']==2
    assert 0<=b[ox]<env.n and 0<=b[oz]<env.n
    return (b[ox],b[oz]),dict(env.counts)


def reference(values, n):
    x1,z1,x2,z2,xd,zd=values
    u=(x1+z1)*(x2-z2)%n;v=(x1-z1)*(x2+z2)%n
    return zd*(u+v)**2%n,xd*(u-v)**2%n


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rng=random.Random(20261007)
    moduli=[(101,16),(1000036000099,64),(2**127-1,160),
            (2**2203-1,2560),(2**4423-1,4608),(2**8191-1,9216)]
    aliases=[(6,7),(0,1),(2,3),(4,5),(0,3),(4,1)]
    totals=Counter();results=[]
    for n,bits in moduli:
        env=Montgomery(n,bits);counts=Counter();cases=0;negative_differences=0;reuse_negative=0
        edges=[(0,)*6,(1,)*6,(n-1,)*6,(n-1,0,0,n-1,1,1),
               (0,n-1,n-1,0,n-1,n-1),tuple(x%n for x in (2,3,5,7,11,13))]
        vectors=edges+[tuple(rng.randrange(n) for _ in range(6)) for _ in range(128)]
        for values in vectors:
            coordinates=tuple(env.encode(x) for x in values)
            expected=tuple(env.encode(x) for x in reference(values,n))
            actual,c=run(env,coordinates,(6,7),True)
            assert actual==expected,(n.bit_length(),values,'disjoint Montgomery outputs')
            reused,reuse_counts=run_output_reuse(env,coordinates,(6,7))
            assert reused==actual and reuse_counts==c,(n.bit_length(),values,'output-Z schedule/counts')
            counts.update(c);cases+=1
            for pair in aliases:
                baseline,_=run(env,coordinates,pair,False)
                assert baseline==expected,(n.bit_length(),pair,'public alias-safe baseline')
            bad,_=run(env,coordinates,(4,5),True)
            negative_differences+=bad!=expected
            bad,_=run_output_reuse(env,coordinates,(4,5))
            reuse_negative+=bad!=expected
        assert negative_differences>0,'Aliased difference must demonstrate contract violation'
        assert reuse_negative>0,'Output-Z reuse must reject the difference-alias contract'
        # Production-like Mersenne headroom can give raw REDC >= N even with R >> N.
        witness=env.encode(n-1);raw=env.raw(witness,witness)
        if n.bit_length()>=127:
            assert raw>=n and raw-n==env.encode(1)
        totals.update(counts)
        results.append(dict(modulus_bits=n.bit_length(),container_bits=bits,cases=cases,
            public_alias_pairs=len(aliases),invalid_difference_alias_mismatches=negative_differences,
            output_reuse_cases=cases,output_reuse_invalid_difference_alias_mismatches=reuse_negative,
            minus_one_raw_REDC_exceeds_N=raw>=n,counts=dict(counts),passed=True))
    source=Path('kernels/cuda/cgbn_stage1_prac_single_add.cuh')
    report=dict(passed=True,cases=sum(r['cases'] for r in results),
        output_reuse_cases=sum(r['output_reuse_cases'] for r in results),results=results,
        counts=dict(totals),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Independent normalized-integer Montgomery/alias model; not native GPU proof')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
