"""Generate an isolated Stage2 project with host-only profiler API ranges.

Copy the exact frozen production closure; insert only range calls around real
tile/cooperative launches. The original workspace and released exe are untouched.
The resulting binary needs a GPU SASS/resource comparison and complete curve
validation before its reports can qualify the original production kernels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    exe, out = a.exe.resolve(), a.output.resolve()
    frozen = json.loads((exe.parent/'frozen_sources_manifest.json').read_text())
    if sha(exe) != frozen['binary_sha256']:
        raise ValueError('frozen binary changed')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('use a fresh project directory')
    for name, want in frozen['sources'].items():
        src = exe.parent/'sources'/name
        if sha(src) != want:
            raise ValueError('frozen dependency changed: ' + name)
        dest = out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    helper = ROOT/'tools/bench/stage2_ntt_profile_range.cuh'
    shutil.copyfile(helper,out/'src/cuda/stage2/ntt_profile_range.cuh')
    runtime = out/'src/cuda/stage2/ntt_runtime.cuh'
    raw = runtime.read_bytes()
    replacements = [
        (b'#include <cuda_runtime.h>',b'#include <cuda_runtime.h>\n#include "ntt_profile_range.cuh"'),
        (b'        if(c.warp_tail && c.t>=5)\n            tile_kernel<false,true>',
         b'        const bool ncu_range=s2_ncu_range_begin(0,false,n,nbatch,c.t);\n        if(c.warp_tail && c.t>=5)\n            tile_kernel<false,true>'),
        (b'        fuse_mark("fwd tile pass", ft0);',b'        s2_ncu_range_end(ncu_range);\n        fuse_mark("fwd tile pass", ft0);'),
        (b'        if(c.warp_tail && c.t>=5)\n            tile_kernel<true,true>',
         b'        const bool ncu_range=s2_ncu_range_begin(0,true,n,nbatch,c.t);\n        if(c.warp_tail && c.t>=5)\n            tile_kernel<true,true>'),
        (b'        fuse_mark("inv tile pass (+pw+scale)", ft0);',b'        s2_ncu_range_end(ncu_range);\n        fuse_mark("inv tile pass (+pw+scale)", ft0);')]
    for old,new in replacements:
        if raw.count(old) != 1:
            raise ValueError('ambiguous original tile launch site')
        raw = raw.replace(old,new)
    runtime.write_bytes(raw)
    outer = out/'src/cuda/stage2/ntt_coop_outer.cuh';raw=outer.read_bytes()
    old=b'    switch(m) {';new=b'    const bool ncu_range=s2_ncu_range_begin(1,INVERSE,n,nbatch,m);\n    switch(m) {'
    if raw.count(old) != 1:
        raise ValueError('ambiguous cooperative launcher')
    raw=raw.replace(old,new)
    old=b'    }\n}\n'
    if not raw.endswith(old):
        raise ValueError('unexpected cooperative launch ending')
    raw=raw[:-len(old)]+b'    }\n    s2_ncu_range_end(ncu_range);\n}\n';outer.write_bytes(raw)
    support = ['third_party/gmp-zen3/dist/include/gmp.h','third_party/gmp-zen3/dist/lib/gmp.lib',
               'third_party/gmp-zen3/dist/lib/gmp.dll.lib','third_party/gmp-zen3/dist/bin/gmp-10.dll']
    for name in support:
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    generated = {name:sha(out/name) for name in frozen['sources']}
    generated['src/cuda/stage2/ntt_profile_range.cuh']=sha(helper)
    changed = [name for name in frozen['sources'] if generated[name] != frozen['sources'][name]]
    if changed != ['src/cuda/stage2/ntt_coop_outer.cuh','src/cuda/stage2/ntt_runtime.cuh']:
        raise ValueError('unexpected instrumentation diff: ' + str(changed))
    # Prove exact inverse edits, independently of compilation or a later green check.
    restore=runtime.read_bytes()
    for old,new in reversed(replacements):restore=restore.replace(new,old)
    if restore != (exe.parent/'sources/src/cuda/stage2/ntt_runtime.cuh').read_bytes():
        raise ValueError('range edits changed original tile code')
    restore=outer.read_bytes().replace(b'    const bool ncu_range=s2_ncu_range_begin(1,INVERSE,n,nbatch,m);\n',b'').replace(b'    s2_ncu_range_end(ncu_range);\n',b'')
    if restore != (exe.parent/'sources/src/cuda/stage2/ntt_coop_outer.cuh').read_bytes():
        raise ValueError('range edits changed original cooperative code')
    receipt=dict(complete=True,base_exe=str(exe),base_binary_sha256=sha(exe),
        base_frozen_manifest_sha256=sha(exe.parent/'frozen_sources_manifest.json'),
        base_sources=frozen['sources'],generated_sources=generated,changed=changed,
        generator_sha256=sha(__file__),helper_sha256=sha(helper),support={n:sha(out/n) for n in support},
        scope='Host-only profiling hooks. Not a production release or GPU equivalence proof; SASS/resources and complete actual curve checks are still required.')
    (out/'instrumentation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(out/'tools/build/build_ecm_cuda_stage2.ps1')


if __name__ == '__main__':main()
