"""Compile the actual NTT arena against a CPU allocation ledger and compare its
transitions to the predictive model. Does not initialize CUDA or execute GPU work.
Fuse descriptors are synthetic: this checks allocation policy, not NTT math or
the device-dependent sizes produced by fuse_describe in production.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vcvars', type=Path, default=Path(r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT/'src/cuda/stage2/ntt_runtime.cuh'
    template = Path(__file__).with_name('stage2_ntt_memory_fixture.cpp')
    files = [source, template, Path(__file__).resolve(), ROOT/'src/core/ecm_stage2_ntt_memory.h',
             ROOT/'src/core/ecm_stage2_requests.h', ROOT/'src/core/ecm_stage2_geometry.h',
             ROOT/'tools/test/stage2_request_program_fixture.cpp']
    identities = {str(p.relative_to(ROOT)): sha(p) for p in files}
    text = source.read_text(encoding='utf-8')
    begin = text.index('struct NttWorkspacePolicy {')
    end = text.index('/* device arithmetic selftest kernel:', begin)
    native = text[begin:end]
    cpp = out/'fixture.cpp'
    # The template normally lives in tools/test; generated fixture has an
    # absolute include so ignored output directories may be anywhere in the repo.
    fixture = template.read_text(encoding='utf-8').replace(
        '"../../src/core/ecm_stage2_ntt_memory.h"',
        '"'+(ROOT/'src/core/ecm_stage2_ntt_memory.h').as_posix()+'"')
    cpp.write_text(fixture.replace('// @NATIVE_ARENA@', native), encoding='utf-8')
    command = out/'compile.cmd'
    command.write_text('@echo off\ncall "'+str(args.vcvars)+'" >nul 2>&1\n'
                       'if errorlevel 1 exit /b 1\n'
                       'cl /nologo /std:c++17 /EHsc /O2 /utf-8 "'+str(cpp)+'" '
                       '/Fe:"'+str(out/'fixture.exe')+'" /Fo:"'+str(out/'fixture.obj')+'"\n'
                       'if errorlevel 1 exit /b 1\n'
                       'cl /nologo /std:c++17 /EHsc /O2 /utf-8 "'+str(ROOT/'tools/test/stage2_request_program_fixture.cpp')+'" '
                       '/Fe:"'+str(out/'request_fixture.exe')+'" /Fo:"'+str(out/'request_fixture.obj')+'"\n', encoding='utf-8')
    result = dict(complete=False, sources=identities, extracted_source_sha256=hashlib.sha256(native.encode()).hexdigest(), gpu_calls=0)
    environment = {key: value for key, value in os.environ.items() if not key.startswith('NTT_')}
    try:
        begin_time = time.monotonic()
        compile_result = subprocess.run(['cmd.exe', '/d', '/c', str(command)], capture_output=True, timeout=180, env=environment)
        (out/'compile.log').write_bytes(compile_result.stdout+compile_result.stderr)
        if compile_result.returncode:
            raise ValueError('CPU fixture compilation failed; see compile.log')
        run = subprocess.run([str(out/'fixture.exe')], capture_output=True, timeout=120, env=environment)
        (out/'fixture.log').write_bytes(run.stdout+run.stderr)
        if run.returncode:
            raise ValueError('CPU fixture failed; see fixture.log')
        result['fixture'] = json.loads(run.stdout)
        if result['fixture']['bad'] or result['fixture']['gpu_calls']:
            raise ValueError('invalid CPU-only result')
        regression = subprocess.run([str(out/'request_fixture.exe')], capture_output=True, timeout=120, env=environment)
        (out/'request_fixture.log').write_bytes(regression.stdout+regression.stderr)
        if regression.returncode:
            raise ValueError('request topology regression failed; see request_fixture.log')
        result['request_regression'] = json.loads(regression.stdout)
        if result['request_regression']['bad']:
            raise ValueError('invalid request topology regression result')
        if {str(p.relative_to(ROOT)): sha(p) for p in files} != identities:
            raise ValueError('source changed during check')
        result.update(complete=True, elapsed_seconds=time.monotonic()-begin_time,
                      binary_sha256=sha(out/'fixture.exe'), generated_sha256=sha(cpp))
    except Exception as exc:
        result['error'] = str(exc)
        raise
    finally:
        (out/'checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['fixture']))


if __name__ == '__main__':
    main()
