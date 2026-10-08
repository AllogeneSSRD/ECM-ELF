"""Create an isolated fixed-arithmetic NTT probe from exact current sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mask',type=int,choices=(0,1,2,3),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise ValueError('use a fresh output directory')
    files=['tools/test/ntt_outer_v_probe.cu','tools/test/ntt_coop_outer_probe.cu',
           'tools/bench/ntt_poly_probe.cu','tools/bench/ntt_coop_outer.cuh',
           'tools/bench/ntt_goldilocks_reduce.cuh','tools/bench/ntt_goldilocks_ptx.cuh',
           'tools/bench/ntt_carry_partial.cuh','tools/bench/ntt_goldilocks_addsub.cuh',
           'tools/build/build_ntt_outer_v_probe.ps1']
    original={n:sha(ROOT/n) for n in files};edits={}
    for name in files:
        src=ROOT/name;raw=src.read_bytes();changes=[]
        if name=='tools/bench/ntt_poly_probe.cu':
            changes=[(b'#include "ntt_goldilocks_ptx.cuh"',b'#include "ntt_goldilocks_ptx.cuh"\n#include "ntt_goldilocks_addsub.cuh"'),
                (b'    return gl_sub(a, b);',b'#if (NTT_GL_ADD_SUB_MASK & 1)\n    return gl_sub_canonical_ptx(a,b);\n#else\n    return gl_sub(a, b);\n#endif'),
                (b'    return gl_add(a, b);',b'#if (NTT_GL_ADD_SUB_MASK & 2)\n    return gl_add_canonical_ptx(a,b);\n#else\n    return gl_add(a, b);\n#endif')]
        if name=='tools/test/ntt_outer_v_probe.cu':
            changes=[(b'for(int mask:{0,1,3,2,1,2,0,3,2,3,1,0,3,0,2,1})',b'for(int mask:{0,0,0,0,0,0,0,0})'),
                (b'    int device=argc>1 ? std::atoi(argv[1]) : 1;CK(cudaSetDevice(device));',
                 b'    int device=argc>1 ? std::atoi(argv[1]) : 1;CK(cudaSetDevice(device));\n    std::printf("ntt_addsub_mask: value=%d\\n",NTT_GL_ADD_SUB_MASK);')]
        if name=='tools/build/build_ntt_outer_v_probe.ps1':
            changes=[(b"'tools/bench/ntt_carry_partial.cuh',",b"'tools/bench/ntt_carry_partial.cuh','tools/bench/ntt_goldilocks_addsub.cuh',"),
                (b'-DNTT_GL_FIXED_MODE=3',f'-DNTT_GL_ADD_SUB_MASK={a.mask} -DNTT_GL_FIXED_MODE=3'.encode())]
        before=raw
        for old,new in changes:
            if raw.count(old)!=1:raise ValueError('ambiguous source edit: '+name)
            raw=raw.replace(old,new)
        restored=raw
        for old,new in reversed(changes):restored=restored.replace(new,old)
        if restored!=before:raise ValueError('nonreversible edit')
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        if changes:edits[name]=[[x.decode(),y.decode()] for x,y in changes]
    support=['third_party/gmp-zen3/dist/include/gmp.h','third_party/gmp-zen3/dist/lib/gmp.lib',
             'third_party/gmp-zen3/dist/lib/gmp.dll.lib','third_party/gmp-zen3/dist/bin/gmp-10.dll']
    for name in support:
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    receipt=dict(complete=True,mask=a.mask,original_sources=original,generated_sources={n:sha(out/n) for n in files},
                 edits=edits,support={n:sha(out/n) for n in support},generator_sha256=sha(__file__),
                 scope='Isolated compile-time aliases only; benchmark V fixed0. Dense gate still covers all V choices. Original runtime and production sources untouched.')
    (out/'generation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(out)


if __name__=='__main__':main()
