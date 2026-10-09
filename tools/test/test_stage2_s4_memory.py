"""CPU-only S4 allocation-event proof against extracted production statements.

Opaque CUDA allocation handles record events only. No CUDA runtime, kernels,
GPU queries or arithmetic are executed; this is not complete Stage2 admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block(text, start):
    """Extract a uniquely anchored native allocation/lookup block."""
    if text.count(start) != 1:
        raise ValueError('native anchor is not unique: '+start)
    begin = text.index(start)
    opened = text.index('{', begin)
    depth = 0
    for i in range(opened, len(text)):
        depth += (text[i] == '{')-(text[i] == '}')
        if depth == 0:
            return text[begin:i+1]
    raise ValueError('unclosed native block')


def statement(text, anchor):
    if text.count(anchor) != 1:
        raise ValueError('native statement is not unique: '+anchor)
    begin = text.index(anchor)
    return text[begin:text.index(';', begin)+1]


def extract(text):
    begin = text.index('struct S4Ctx {')
    ctx = text[begin:text.index('\n};', begin)+3]
    start = text.index('static int s4_reduce_selftest(')
    selftest = text[start:text.index('\n}', start)+2]
    cases = re.search(r'const unsigned long long cases = (\d+);', selftest)
    if not cases or int(cases[1]) != 96:
        raise ValueError('mandatory selftest case count changed')
    # Native shape and reducer destruction order are explicitly checked: this
    # fixture isolates allocations, so host/GMP member destruction is omitted.
    shape_destructor = block(text, '~Shape()')
    reducer_destructor = block(text, '~S4Reduce()')
    if 'if (dy) cudaFree(dy);' not in shape_destructor:
        raise ValueError('shape destructor changed')
    if not (reducer_destructor.index('delete s') < reducer_destructor.index('cudaFree(dn)') <
            reducer_destructor.index('cudaFree(dbad)')):
        raise ValueError('reducer destruction order changed')
    if text.index('S4Ctx s4;') >= text.index('S4Reduce red;'):
        raise ValueError('context declaration order changed')
    return dict(
        NATIVE_CTX=ctx,
        NATIVE_OUTPUT=block(text, 'if(allocation_need>C.d_out_cap)'),
        NATIVE_PACK=block(text, 'if (!g_s4_pack_direct && pack_words > C.d_pack_cap)'),
        NATIVE_FIND=block(text, 'Shape *find(unsigned long long slot_bits, unsigned long long slot_words, int bpw) const'),
        NATIVE_MODULUS=statement(text, 'CK(cudaMalloc(&R.dn,'),
        NATIVE_CONSTANT=statement(text, 'CK(cudaMalloc(&S->dy,'),
        NATIVE_SELFTEST_ALLOC='\n'.join(statement(selftest, s) for s in
            ('CK(cudaMalloc(&dd, dig.size()', 'CK(cudaMalloc(&dout, (size_t)(cases * R.w)')),
        NATIVE_SELFTEST_FREE='\n'.join(statement(selftest, s) for s in ('cudaFree(dd);', 'cudaFree(dout);')),
        NATIVE_COUNTER=block(text, 'if (!R.dbad)'),
        NATIVE_METADATA_CTX=block(text, 'struct Metadata {')+' meta;',
        NATIVE_METADATA_ALLOC=statement(text, 'meta.capacity=3*(pad/2);')+'\n'+
            statement(text, 'CK(cudaMalloc(&meta.device,'),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--vcvars', type=Path, default=Path(r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT/'src/cuda/ecm_cuda_stage2.cu'
    template = Path(__file__).with_name('stage2_s4_memory_fixture.cpp')
    headers = ('ecm_stage2_s4_memory.h', 'ecm_stage2_requests.h', 'ecm_stage2_geometry.h')
    files = [source, template, Path(__file__).resolve(), *(ROOT/'src/core'/h for h in headers)]
    identities = {str(f.relative_to(ROOT)):sha(f) for f in files}
    result = dict(complete=False, sources=identities, gpu_calls=0, scope='s4_component_allocation_events')
    try:
        native = extract(source.read_text(encoding='utf-8'))
        cpp = template.read_text(encoding='utf-8')
        for h in headers:
            cpp = cpp.replace('"../../src/core/'+h+'"', '"'+(ROOT/'src/core'/h).as_posix()+'"')
        for name, text in native.items():
            marker = '// @'+name+'@'
            if cpp.count(marker) != 1:
                raise ValueError('fixture marker changed: '+name)
            cpp = cpp.replace(marker, text)
        generated = out/'fixture.cpp'
        generated.write_text(cpp, encoding='utf-8')
        result['extracted_source_sha256'] = {k:hashlib.sha256(v.encode()).hexdigest() for k,v in native.items()}
        cmd = out/'compile.cmd'
        cmd.write_text('@echo off\ncall "'+str(a.vcvars)+'" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
            'cl /nologo /std:c++17 /EHsc /O2 /utf-8 "'+str(generated)+'" /Fe:"'+str(out/'fixture.exe')+
            '" /Fo:"'+str(out/'fixture.obj')+'"\n', encoding='utf-8')
        env = {k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
        begin = time.monotonic()
        build = subprocess.run(['cmd.exe','/d','/c',str(cmd)], capture_output=True, timeout=180, env=env)
        (out/'compile.log').write_bytes(build.stdout+build.stderr)
        if build.returncode:
            raise ValueError('CPU fixture compilation failed; see compile.log')
        run = subprocess.run([str(out/'fixture.exe')], capture_output=True, timeout=120, env=env)
        (out/'fixture.log').write_bytes(run.stdout+run.stderr)
        if run.returncode:
            raise ValueError('CPU fixture failed; see fixture.log')
        result['fixture'] = json.loads(run.stdout)
        if result['fixture']['bad'] or result['fixture']['gpu_calls']:
            raise ValueError('invalid CPU-only result')
        # Sensitivity proof: an incorrect 128-window model must fail against
        # the extracted real 96-window allocator, even though both finish with
        # the same live bytes. Only an ignored copy is changed.
        model_header = ROOT/'src/core/ecm_stage2_s4_memory.h'
        model = model_header.read_text(encoding='utf-8')
        if model.count('96*8') != 2:
            raise ValueError('selftest model mutation anchors changed')
        mutant = out/'mutant.h'
        mutant.write_text(model.replace('96*8','128*8').replace('"ecm_stage2_geometry.h"',
            '"'+(ROOT/'src/core/ecm_stage2_geometry.h').as_posix()+'"'), encoding='utf-8')
        mutant_cpp = out/'mutant.cpp'
        mutant_cpp.write_text(cpp.replace('"'+model_header.as_posix()+'"',
            '"'+mutant.as_posix()+'"'), encoding='utf-8')
        mutant_cmd = out/'compile_mutant.cmd'
        mutant_cmd.write_text(cmd.read_text(encoding='utf-8').replace(str(generated),str(mutant_cpp))
            .replace('fixture.exe','mutant.exe').replace('fixture.obj','mutant.obj'), encoding='utf-8')
        bad_build = subprocess.run(['cmd.exe','/d','/c',str(mutant_cmd)], capture_output=True, timeout=180, env=env)
        (out/'mutant_compile.log').write_bytes(bad_build.stdout+bad_build.stderr)
        if bad_build.returncode:
            raise ValueError('mutation fixture failed to compile')
        bad_run = subprocess.run([str(out/'mutant.exe')], capture_output=True, timeout=120, env=env)
        (out/'mutant.log').write_bytes(bad_run.stdout+bad_run.stderr)
        if bad_run.returncode != 1 or not any(s in bad_run.stderr for s in
                (b'S4 peak bytes differ',b'S4 event differs')):
            raise ValueError('wrong selftest model was not rejected')
        result['mutation'] = dict(expected_failure=True, returncode=bad_run.returncode,
            reason=bad_run.stderr.decode('utf-8').strip(), header_sha256=sha(mutant),
            generated_sha256=sha(mutant_cpp), binary_sha256=sha(out/'mutant.exe'))
        if {str(f.relative_to(ROOT)):sha(f) for f in files} != identities:
            raise ValueError('source changed during check')
        result.update(complete=True, elapsed_seconds=time.monotonic()-begin,
                      binary_sha256=sha(out/'fixture.exe'), generated_sha256=sha(generated))
    except Exception as exc:
        result['error'] = str(exc)
        raise
    finally:
        (out/'checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['fixture']))


if __name__ == '__main__':
    main()
