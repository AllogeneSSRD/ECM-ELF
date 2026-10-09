"""Verify exact NTT allocation/free order against extracted production code.

Native FuseCtx, fuse planning/allocation and arena code run on opaque CPU handles.
Only GPU launches/queries are stubbed. This proves normal successful owned NTT
payload events and cap-refusal prefixes, not NTT arithmetic or full admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from test_stage2_s4_memory import block

ROOT = Path(__file__).resolve().parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace_one(text, old, new):
    if text.count(old) != 1:
        raise ValueError('native annotation anchor is not unique: '+old)
    return text.replace(old, new)


def extract(text):
    begin = text.index('struct FuseCtx {')
    end = text.index('\n}', text.index('static void ntt_fuse_cache_tables('))+2
    fuse = text[begin:end]
    device = block(fuse, 'static bool fuse_shape_device_supported()')
    # Deterministic CPU descriptor selection for the measured-policy branch.
    # This does not query or claim to emulate driver free memory.
    fuse = replace_one(fuse, device, 'static bool fuse_shape_device_supported(){return true;}')
    fuse, removed = re.subn(r'\bbuild_\w+_kernel<<<[^;]+;', '(void)0;', fuse)
    if removed != 6:
        raise ValueError('native fuse launch count changed')
    for field, site in [('tblF', 'NttBaseTileF'), ('tblI', 'NttBaseTileI'),
                        ('scr', 'NttBaseScratch'), ('scr2', 'NttBaseRadix')]:
        anchor='fuse_base_allocate(&c.'+field+','
        fuse=replace_one(fuse, anchor, 'native_site='+site+';native_index=0;'+anchor)
    for field, site in [('passF', 'NttTablePassF'), ('radF', 'NttTableRadF'),
                        ('passI', 'NttTablePassI'), ('radI', 'NttTableRadI')]:
        anchor='CK(cudaMalloc(&c.'+field+'[p],'
        fuse=replace_one(fuse, anchor, 'native_site='+site+';native_index=p;'+anchor)
    begin=text.index('struct NttWorkspacePolicy {')
    end=text.index('/* device arithmetic selftest kernel:', begin)
    arena=text[begin:end]
    arena=replace_one(arena, 'auto allocate=[&](unsigned long long **p, int index) {',
        'auto allocate=[&](unsigned long long **p, int index) { native_site=NttMemorySite(NttWorkspaceA+index-1);native_index=0;')
    arena=replace_one(arena, 'auto allocate_legacy=[&](unsigned long long **p) {',
        'auto allocate_legacy=[&](unsigned long long **p) {native_site=p==&e.dA?NttKeyedA:p==&e.dB?NttKeyedB:NttKeyedQ;native_index=0;')
    arena=replace_one(arena, 'cudaMalloc((void**)&carry_scratch,need)',
        '(native_site=NttCarry,native_index=0,cudaMalloc((void**)&carry_scratch,need))')
    for field, site in [('small->dOut','NttDigitsOutput'), ('e.dOut','NttDigitsOutput'), ('e.dRes','NttDigitsResult')]:
        anchor='NttArena::try_malloc(&'+field+', '+('2 * nbatch' if field=='e.dRes' else 'out_slots * nbatch')+' * sizeof(unsigned long long))'
        arena=replace_one(arena, anchor, '(native_site='+site+',native_index=0,'+anchor+')')
    return fuse, arena, removed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--vcvars', type=Path, default=Path(r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    args=ap.parse_args();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    runtime=ROOT/'src/cuda/stage2/ntt_runtime.cuh';header=ROOT/'src/core/ecm_stage2_ntt_memory.h'
    template=Path(__file__).with_name('stage2_ntt_events_fixture.cpp')
    files=[runtime, header, template, Path(__file__).resolve(), ROOT/'tools/test/test_stage2_s4_memory.py',
           ROOT/'src/core/ecm_stage2_requests.h',ROOT/'src/core/ecm_stage2_geometry.h']
    identities={str(p.relative_to(ROOT)):sha(p) for p in files}
    result=dict(complete=False, sources=identities, gpu_calls=0)
    try:
        start=time.monotonic();fuse,arena,removed=extract(runtime.read_text(encoding='utf-8'))
        text=template.read_text(encoding='utf-8').replace('// @NATIVE_FUSE@',fuse).replace('// @NATIVE_ARENA@',arena)
        cpp=out/'fixture.cpp'
        include='"../../src/core/ecm_stage2_ntt_memory.h"'
        cpp.write_text(text.replace(include,'"'+header.as_posix()+'"'),encoding='utf-8')
        mutated=out/'wrong_release_order.h'
        bad=header.read_text(encoding='utf-8').replace('"ecm_stage2_requests.h"','"'+(ROOT/'src/core/ecm_stage2_requests.h').as_posix()+'"')
        bad=replace_one(bad,'std::make_pair(a.index,a.site)<std::make_pair(b.index,b.site)',
                            'std::make_pair(a.index,a.site)>std::make_pair(b.index,b.site)')
        mutated.write_text(bad,encoding='utf-8')
        bad_cpp=out/'wrong_fixture.cpp';bad_cpp.write_text(text.replace(include,'"'+mutated.as_posix()+'"'),encoding='utf-8')
        command=out/'compile.cmd'
        lines=['@echo off','call "'+str(args.vcvars)+'" >nul 2>&1','if errorlevel 1 exit /b 1']
        for src,exe in [(cpp,out/'fixture.exe'),(bad_cpp,out/'wrong_fixture.exe')]:
            lines+=['cl /nologo /std:c++17 /EHsc /O2 /utf-8 "'+str(src)+'" /Fe:"'+str(exe)+'" /Fo:"'+str(src.with_suffix('.obj'))+'"','if errorlevel 1 exit /b 1']
        command.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
        compiled=subprocess.run(['cmd.exe','/d','/c',str(command)],capture_output=True,timeout=180,env=env)
        (out/'compile.log').write_bytes(compiled.stdout+compiled.stderr)
        if compiled.returncode:raise ValueError('CPU compilation failed; see compile.log')
        run=subprocess.run([str(out/'fixture.exe')],capture_output=True,timeout=120,env=env)
        (out/'fixture.log').write_bytes(run.stdout+run.stderr)
        if run.returncode:raise ValueError('CPU event proof failed; see fixture.log')
        result['fixture']=json.loads(run.stdout)
        if result['fixture']['bad'] or result['fixture']['gpu_calls']:raise ValueError('invalid CPU-only result')
        wrong=subprocess.run([str(out/'wrong_fixture.exe')],capture_output=True,timeout=120,env=env)
        (out/'wrong_fixture.log').write_bytes(wrong.stdout+wrong.stderr)
        if wrong.returncode!=1 or b'event_order_or_payload' not in wrong.stderr:raise ValueError('wrong free order was not rejected')
        if identities!={str(p.relative_to(ROOT)):sha(p) for p in files}:raise ValueError('source changed during proof')
        result.update(complete=True,elapsed_seconds=time.monotonic()-start,removed_gpu_launches=removed,
                      binary_sha256=sha(out/'fixture.exe'),generated_sha256=sha(cpp),
                      compile_log_sha256=sha(out/'compile.log'),fixture_log_sha256=sha(out/'fixture.log'),
                      mutated_header_sha256=sha(mutated),mutated_binary_sha256=sha(out/'wrong_fixture.exe'),
                      mutation_returncode=wrong.returncode,mutation_log_sha256=sha(out/'wrong_fixture.log'),
                      extraction_sha256=hashlib.sha256((fuse+arena).encode()).hexdigest())
    except Exception as exc:
        result['error']=str(exc);raise
    finally:
        (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['fixture']))


if __name__=='__main__':main()
