#!/usr/bin/env python3
"""Independent word REDC and fixed-M4423 layout witnesses, not native GPU proof."""
import argparse
import hashlib
import json
from pathlib import Path
import random

MASK=(1<<32)-1


def redc_words(a,b,n,bits,np0):
    value=a*b
    for _ in range(bits//32):
        q=((value&MASK)*np0)&MASK
        value=(value+q*n)>>32
    return value-n if value>=n else value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rng=random.Random(20261007);cases=0;results=[]
    for n,bits in [(2**32-1,64),(2**127-1,160),(2**2203-1,2560),
                   (2**4423-1,4608),(2**4423-1-(7<<32),4608),(2**8191-1,9216)]:
        np0=-pow(n,-1,1<<32)&MASK
        assert n&MASK==MASK and np0==1
        inverse=pow(1<<bits,-1,n)
        vectors=[(0,0),(1,1),(n-1,n-1),(n-1,1),(1,n-1)]
        vectors += [(rng.randrange(n),rng.randrange(n)) for _ in range(64)]
        for x,y in vectors:
            actual=redc_words(x,y,n,bits,1)
            assert actual==x*y*inverse%n and 0<=actual<n
        cases+=len(vectors)
        results.append(dict(modulus_bits=n.bit_length(),container_bits=bits,cases=len(vectors),np0=np0,pure_mersenne=(n+1).bit_count()==1))
    n=2**4423-1
    words=[]
    for lane in range(16):
        for limb in range(9):
            words.append(MASK if limb<3 or lane!=15 else 0x7f if limb==3 else 0)
    assert len(words)==144 and sum(v<<(32*i) for i,v in enumerate(words))==n
    assert words==[(n>>(32*i))&MASK for i in range(144)]
    bad_n=n-2;bad_np0=-pow(bad_n,-1,1<<32)&MASK
    assert bad_np0!=1 and bad_n.bit_length()==4423
    vectors=[(rng.randrange(bad_n),rng.randrange(bad_n)) for _ in range(64)]
    inverse=pow(1<<4608,-1,bad_n)
    wrong=sum(redc_words(x,y,bad_n,4608,1)!=x*y*inverse%bad_n for x,y in vectors)
    assert wrong>0
    same_width=n-(7<<32)
    assert same_width.bit_length()==4423 and same_width&MASK==MASK and (same_width+1).bit_count()!=1
    report=dict(passed=True,word_REDC_cases=cases,results=results,M4423_words_reconstructed=144,
        invalid_np0_control_cases=64,invalid_np0_mismatches=wrong,exact_M4423_guard_rejects_same_width_non_mersenne=True,
        sources={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in map(Path,['kernels/cuda/cgbn_stage1_prac_constants.cu','kernels/cuda/cgbn_stage1_prac_host.cuh'])},
        scope='Independent radix-2^32 REDC/host integer shape model; no native CUDA performance or full-chain claim')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
