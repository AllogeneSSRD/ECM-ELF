"""CPU-only check of giant component lifetimes against production S3 allocation
code and optional completed GPU ledgers. Never loads CUDA or runs a GPU curve.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/bench'))
from stage2_memory_ledger import parse
from bench_stage2_production import sha


def workspace_sites(source):
    """Use the ledger's frozen compiled source, never current line numbers."""
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('struct S3Workspace'))
    end = next(i for i in range(start, len(lines)) if lines[i] == '};')
    return {f'ecm_cuda_stage2.cu:{i+1}' for i in range(start, end)
            if 'cudaMalloc(' in lines[i]}


def transient_sites(source):
    lines = source.splitlines()
    starts = [next(i for i,line in enumerate(lines) if line.startswith(prefix))
              for prefix in ('struct ResidentGiant {','static void giant_chunk_chain(')]
    ends = [next(i for i in range(starts[0],len(lines)) if lines[i]=='};'),
            next(i for i in range(starts[1],len(lines)) if lines[i].startswith('static void dev_block_products('))]
    return ({f'ecm_cuda_stage2.cu:{i+1}' for start,end in zip(starts,ends)
             for i in range(start,end) if 'cudaMalloc(' in lines[i]} |
            {f'ecm_cuda_stage2.cu:{i+1}' for i,line in enumerate(lines)
             if 'cudaMalloc(&resident.segments,' in line})


def check_ledger(matrix_path, fixture):
    matrix = json.loads(matrix_path.read_text(encoding='utf-8'))
    if not matrix['complete'] or not matrix['memory_ledger']:
        raise ValueError('completed resident owned ledger is required')
    rows = []
    for run in matrix['runs']:
        if int(run['device_leaf']['fallback_chunks']) or not int(run['device_leaf']['chunks']) or not int(run['device_leaf']['device_trees']):
            raise ValueError('this ledger comparison requires the resident chunk path')
        exe = Path(run['command'][0])
        source = exe.parent/'sources/src/cuda/ecm_cuda_stage2.cu'
        if sha(exe) != matrix['identity']['binary_sha256'] or sha(source) != matrix['identity']['sources']['src/cuda/ecm_cuda_stage2.cu']:
            raise ValueError('old runtime binary/source identity differs')
        debug = Path(run['debug_log'])
        if sha(debug) != run['debug_sha256']:
            raise ValueError('old runtime log changed')
        ledger = parse(debug.read_text(encoding='utf-8'))
        compiled = source.read_text(encoding='utf-8')
        sites = workspace_sites(compiled)
        def live(snapshot):
            return sum(int(s['bytes']) for s in ledger['sites']
                       if s['scope']=='live' and s['snapshot']==snapshot and s['site'] in sites)
        w = (int(run['shape']['S_bits'])+63)//64
        # Before inverse only the five constants and the small-prime point lease
        # exist. Recover its capacity from actual sites, not a guessed B1 rule.
        numerator = live('before_inverse')-40*w
        if numerator < 0 or numerator % (8+16*w):
            raise ValueError('unexpected initial S3 allocation contract')
        initial = numerator//(8+16*w)
        p = int(run['shape']['P'].split('=')[-1])
        n = int(run['shape']['giant_points'])
        q = int(run['point_plan']['points'])
        if int(run['device_leaf']['chunks'])!=(n+q-1)//q or int(run['device_leaf']['device_trees'])!=(n+p-1)//p:
            raise ValueError('resident chunk/tree coverage differs from predicted geometry')
        # Historical matrices predate compact products. Their caller policy is
        # part of the frozen evidence, not the default of the current binary.
        proc = subprocess.run([str(fixture),str(p),str(n),str(w),str(q),str(initial),'0'], capture_output=True, timeout=10)
        if proc.returncode:
            raise ValueError(proc.stderr.decode())
        prediction = json.loads(proc.stdout)
        checks = dict(initial=live('before_inverse'),after_giant=live('after_giant_loop'))
        for key, actual in checks.items():
            if prediction[key] != actual:
                raise ValueError(f'{run["name"]}: {key} prediction differs from owned ledger')
        # Resident scratch must have been released by these phase boundaries;
        # S3 retains its seed/segment/base capacities into descent.
        for snapshot in ('after_frontier_admission','after_descent'):
            if live(snapshot) != prediction['after_giant']:
                raise ValueError('S3 capacity unexpectedly changed across descent')
        component_sites = sites | transient_sites(compiled)
        interval_component = sum(int(s['bytes']) for s in ledger['sites']
            if s['scope']=='interval_peak' and s['snapshot']=='after_giant_loop' and s['site'] in component_sites)
        if interval_component != prediction['giant_peak']:
            raise ValueError(f'{run["name"]}: giant simultaneous sites {interval_component} differ from component peak {prediction["giant_peak"]}')
        rows.append(dict(matrix=str(matrix_path.resolve()),matrix_sha256=sha(matrix_path),
                         name=run['name'],binary_sha256=sha(exe),source_sha256=sha(source),
                         debug_sha256=sha(debug),initial_points=initial,prediction=prediction,
                         verified_checkpoints=4,verified_giant_interval_peak=interval_component))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ledger-matrix', type=Path, action='append', default=[])
    parser.add_argument('--regression-source',type=Path,help='Frozen pre-fix CUDA source; require its S3 bytes counter to fail')
    parser.add_argument('--vcvars', type=Path, default=Path(r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT/'src/cuda/ecm_cuda_stage2.cu'
    template = Path(__file__).with_name('stage2_giant_memory_fixture.cpp')
    files = [source,template,Path(__file__).resolve(),ROOT/'src/core/ecm_stage2_giant_memory.h',
             ROOT/'src/core/ecm_stage2_geometry.h', ROOT/'tools/bench/stage2_memory_ledger.py']
    identities = {str(p.relative_to(ROOT)):sha(p) for p in files}
    text = source.read_text(encoding='utf-8')
    start = text.index('struct S3Workspace {')
    end = text.index('\n};',start)+3
    native = text[start:end]
    cpp = out/'fixture.cpp'
    fixture = template.read_text(encoding='utf-8').replace('"../../src/core/ecm_stage2_giant_memory.h"',
        '"'+(ROOT/'src/core/ecm_stage2_giant_memory.h').as_posix()+'"')
    cpp.write_text(fixture.replace('// @NATIVE_S3@',native),encoding='utf-8')
    command = out/'compile.cmd'
    command.write_text('@echo off\ncall "'+str(args.vcvars)+'" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
        'cl /nologo /std:c++17 /EHsc /O2 /utf-8 "'+str(cpp)+'" /Fe:"'+str(out/'fixture.exe')+
        '" /Fo:"'+str(out/'fixture.obj')+'"\n',encoding='utf-8')
    result = dict(complete=False,sources=identities,gpu_calls=0,
                  extracted_source_sha256=hashlib.sha256(native.encode()).hexdigest())
    environment = {k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    try:
        begin = time.monotonic()
        compile_result = subprocess.run(['cmd.exe','/d','/c',str(command)],capture_output=True,timeout=180,env=environment)
        (out/'compile.log').write_bytes(compile_result.stdout+compile_result.stderr)
        if compile_result.returncode:
            raise ValueError('CPU fixture compilation failed; see compile.log')
        run = subprocess.run([str(out/'fixture.exe')],capture_output=True,timeout=120,env=environment)
        (out/'fixture.log').write_bytes(run.stdout+run.stderr)
        if run.returncode:
            raise ValueError('CPU fixture failed; see fixture.log')
        result['fixture'] = json.loads(run.stdout)
        if result['fixture']['bad'] or result['fixture']['gpu_calls']:
            raise ValueError('invalid CPU-only result')
        result['ledgers'] = [row for matrix in args.ledger_matrix for row in check_ledger(matrix,out/'fixture.exe')]
        if args.regression_source:
            old = args.regression_source.resolve()
            old_sha = sha(old)
            old_text = old.read_text(encoding='utf-8')
            old_start = old_text.index('struct S3Workspace {')
            old_native = old_text[old_start:old_text.index('\n};',old_start)+3]
            old_cpp = out/'regression.cpp'
            old_cpp.write_text('#define S3_TEST_LEGACY_SOURCE 1\n'+fixture.replace('// @NATIVE_S3@',old_native),encoding='utf-8')
            old_command = out/'compile_regression.cmd'
            old_command.write_text(command.read_text(encoding='utf-8').replace(str(cpp),str(old_cpp))
                .replace('fixture.exe','regression.exe').replace('fixture.obj','regression.obj'),encoding='utf-8')
            old_build = subprocess.run(['cmd.exe','/d','/c',str(old_command)],capture_output=True,timeout=180,env=environment)
            (out/'regression_compile.log').write_bytes(old_build.stdout+old_build.stderr)
            if old_build.returncode:
                raise ValueError('pre-fix regression compilation failed')
            old_run = subprocess.run([str(out/'regression.exe')],capture_output=True,timeout=120,env=environment)
            (out/'regression.log').write_bytes(old_run.stdout+old_run.stderr)
            if not old_run.returncode or b'S3 bytes counter includes released capacities' not in old_run.stderr or sha(old)!=old_sha:
                raise ValueError('pre-fix source did not reproduce the S3 accounting error')
            result['regression'] = dict(source=str(old),source_sha256=old_sha,
                binary_sha256=sha(out/'regression.exe'),expected_failure=True,returncode=old_run.returncode)
        if {str(p.relative_to(ROOT)):sha(p) for p in files} != identities:
            raise ValueError('source changed during check')
        result.update(complete=True,elapsed_seconds=time.monotonic()-begin,
                      binary_sha256=sha(out/'fixture.exe'),generated_sha256=sha(cpp))
    except Exception as exc:
        result['error'] = str(exc)
        raise
    finally:
        (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(fixture=result['fixture'],ledger_runs=len(result['ledgers']))))


if __name__ == '__main__':
    main()
