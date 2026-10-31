"""Bounded fixed-D point-route experiments from an audited default matrix.

One GPU, one saved point, one variable at a time. Never consumes a live queue
or writes a tune file. Diagnostic leaf hashes are separate from timing runs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from bench_stage2_version import load, write, stamp, shape_fields, check_gpu_processes
from bench_stage2_n_scaling import fields, number
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_clock(clock, leaf_checks=False):
    values=[float(clock[k]) for k in ('total','init','main')]
    # Additional GMP leaf fixtures deliberately mark clean=0. Their receipts
    # can validate arithmetic but must never enter the timing population.
    expected='0' if leaf_checks else '1'
    if clock['clean']!=expected or any(not math.isfinite(v) or v<0 for v in values) or abs(values[0]-values[1]-values[2])>.003:
        raise ValueError('Engine clock/check boundary differs')


def overrides(mode, p, words):
    if mode == 'baseline':
        return {}
    if mode == 'split6':
        # Exact six-P coordinate capacity, not the unrelated NTT batch budget.
        return {'NTT_GIANT_POINT_BUDGET_KB': str((6*p*16*words)//1024)}
    if mode in ('chain32', 'chain128'):
        return {'NTT_GIANT_CHAIN_BLOCK': mode[5:]}
    if mode in ('balanced', 'rounded', 'rounded_reference'):
        return {}
    raise ValueError('Unknown route experiment: ' + mode)


def rounded_bound(b2,d,p,capacity,minimum):
    if not d or not p or capacity%p or capacity<minimum:
        raise ValueError('No chain-capable rounded geometry')
    points=((b2//d+2+p-1)//p)*p
    tail=points%capacity
    if tail and tail<minimum:points+=((minimum-tail+p-1)//p)*p
    return max(b2,(points-2)*d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True, help='Completed bounded layout study')
    parser.add_argument('--case', required=True)
    parser.add_argument('--sequence', default='baseline,split6,split6,baseline')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exe', type=Path, help='Optional isolated point-route probe, with its build manifest')
    parser.add_argument('--trace-checks', action='store_true', help='Diagnostic only: compare normalized leaf hashes')
    parser.add_argument('--nsys', type=Path, help='Diagnostic only: Nsight Systems executable; capture CUDA/NVTX')
    a = parser.parse_args()
    study = a.study.resolve(); plan = load(study/'plan.json')
    baseline = load(Path(plan['defaults'])/'plan.json')
    if not plan['complete'] or not baseline['complete']:
        raise ValueError('Need completed audited source studies')
    source = next(r for r in baseline['runs'] if r['case'] == a.case and r['engine'] == 'gpu-carrier')
    case = next(c for c in baseline['cases'] if c['id'] == a.case)
    modes = a.sequence.split(',')
    if not 2 <= len(modes) <= 12:
        raise ValueError('Use 2..12 bounded calls, including a baseline')
    if 'baseline' not in modes:
        raise ValueError('Need a same-session baseline')
    words = (int(source['shape']['S_bits'])+63)//64
    source_text=Path(source['prefix']+'.engine.log').read_text(encoding='utf-8')+'\n'+Path(source['prefix']+'.debug.log').read_text(encoding='utf-8')
    capacity=int(fields(source_text,'giant_chunk_plan')['points'])
    minimum=int(fields(source_text,'giant_chain_policy')['effective_min'])
    rounded=rounded_bound(source['B2'],source['D'],source['P'],capacity,minimum)
    variants = {m: overrides(m, source['P'], words) for m in modes}
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'bin').mkdir(); (out/'bin/.gitignore').write_text('*\n', encoding='ascii')
    binary_dir=a.exe.resolve().parent if a.exe else study/'bin'
    if a.exe:
        build=load(binary_dir/'manifest.json')
        overlay='reused' in build and 'rebuilt' in build
        if (not overlay and build.get('kind')!='isolated_point_route_probe') or not build['complete'] or build['exe_sha256'].lower()!=sha(a.exe):
            raise ValueError('Invalid point-route probe build')
        if not overlay and build['base_exe_sha256'].lower()!=plan['exe_sha256']:
            raise ValueError('Probe does not reuse this study executable')
        if overlay:
            for unit in build['reused'].values():
                if sha(unit['object'])!=unit['sha256'].lower():raise ValueError('Overlay device object changed')
        shutil.copy2(binary_dir/'manifest.json',out/'build_manifest.json')
    for name in ('ecm_cuda_stage2.exe', 'gmp-10.dll'):
        shutil.copy2(binary_dir/name, out/'bin'/name)
    exe = out/'bin/ecm_cuda_stage2.exe'
    if (not a.exe and sha(exe) != plan['exe_sha256']) or sha(out/'bin/gmp-10.dll') != plan['dll_sha256']:
        raise ValueError('Source executable/dependency changed')
    save = out/'point.save'; shutil.copy2(case['save'], save)
    config = out/'ecm.toml'; shutil.copy2(source['command'][source['command'].index('--config')+1], config)
    shutil.copy2(__file__, out/'collector_snapshot.py')
    report = dict(complete=False, started_utc=stamp(), source_study=str(study), case=case,
        source_run=source['id'], exe_sha256=sha(exe), dll_sha256=sha(out/'bin/gmp-10.dll'),
        save_sha256=sha(save), sequence=modes, variants=variants,
        diagnostic=bool(a.trace_checks or a.nsys), leaf_checks=a.trace_checks, D=source['D'], P=source['P'], B2=source['B2'],
        allowed_processes=plan['allowed_processes'], runs=[])
    write(out/'summary.json', report)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('NTT_', 'ECM_')) and
           k not in ('CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING')}
    env['NTT_NO_PROGRESS'] = '1'
    if a.nsys:
        env['NTT_HOST_TRACE'] = '1'
    if a.trace_checks:
        env['NTT_S4_CARRY_TRACE'] = '1'
        env['NTT_DEVICE_GLEAF_CHECK'] = '1'
        env['NTT_TARGET_LEAF_HASH'] = '1'
    references = {}
    for index, mode in enumerate(modes, 1):
        folder = out/f'{index:02}_{mode}'; folder.mkdir()
        prefix = folder/'run'
        command = [str(exe)] + [x.replace(source['prefix'], str(prefix)) for x in source['command'][1:]]
        for flag, value in (('--save', save), ('--config', config), ('--d', source['D'])):
            command[command.index(flag)+1] = str(value)
        command += ['--tune-profile', str(out/'absent_tune.toml')]
        if a.exe and overlay:
            if mode=='chain128':raise ValueError('Production supports only chain32/64')
            command += ['--giant-tail-balance',str(int(mode=='balanced')), '--giant-chain','32' if mode=='chain32' else '64',
                        '--b2-round-chain',str(int(mode=='rounded'))]
            if mode=='rounded_reference':
                # The paired control covers exactly the rounded interval, but
                # gets there through an explicit B2 rather than rounding.
                command[command.index('--b2')+1]=str(rounded)
        write(folder/'command.json', dict(argv=command, environment_overrides=
            {k:v for k,v in (env | variants[mode]).items() if k.startswith('NTT_')}))
        started = stamp(); monitor = TelemetryRecorder(1, folder/'telemetry.jsonl')
        start = time.perf_counter()
        invocation=command
        if a.nsys:
            invocation=[str(a.nsys.resolve()), 'profile', '--trace=cuda,nvtx', '--sample=none',
                '--cpuctxsw=none', '--cuda-memory-usage=true', '-o', str(folder/'capture')]+command
            write(folder/'profile.command.json', invocation)
        try:
            check_gpu_processes(monitor.monitor, out, folder/'before.json', allowed=plan['allowed_processes'])
            with (folder/'console.log').open('wb') as log:
                process = subprocess.run(invocation, cwd=folder, env=env | variants[mode],
                    stdout=log, stderr=subprocess.STDOUT, timeout=600)
            check_gpu_processes(monitor.monitor, out, folder/'after.json', allowed=plan['allowed_processes'])
        finally:
            monitor.close()
        if process.returncode:
            raise ValueError('Worker failed; raw evidence retained')
        if a.nsys:
            with (folder/'export.log').open('wb') as log:
                subprocess.run([str(a.nsys.resolve()), 'export', '--type=sqlite', '--output',
                    str(folder/'capture.sqlite'), str(folder/'capture.nsys-rep')],
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        raw = load(prefix.with_suffix('.jsonl')); receipt = normalize_result(raw)
        requested=int(command[command.index('--b2')+1])
        actual=rounded if mode=='rounded' else requested
        identity = dict(N_hex=case['N_hex'], sigma=case['sigma'], B1=case['b1'],
                        B2=actual, carrier_exponent=source['carrier_exponent'], curve='ed')
        if any(receipt[k] != v for k,v in identity.items()) or raw['source-build'] != report['exe_sha256']:
            raise ValueError('Input/build identity differs')
        if raw['status'] != 'NF' or receipt['factors'] or receipt['bad_factors']:
            raise ValueError('Expected complete NF timing input')
        text = prefix.with_suffix('.engine.log').read_text(encoding='utf-8')+'\n'+prefix.with_suffix('.debug.log').read_text(encoding='utf-8')
        clock = fields(text, 'stage2_full_wall'); check = fields(text, 's4_multiply_stats')
        shape = shape_fields(text)
        if (number(shape['D']), number(shape['P'])) != (source['D'], source['P']):
            raise ValueError('Fixed geometry changed')
        validate_clock(clock,a.trace_checks)
        if min(int(check[k]) for k in ('gmp_checked', 'gmp_selftest_cases')) <= 0 or any(int(check[k]) for k in ('gmp_check_bad', 'gmp_selftest_bad')):
            raise ValueError('Mandatory arithmetic checks failed/absent')
        stats = {k:fields(text, k, False) for k in ('real_baby', 'real_batched_split',
            'real_batched_wall', 'device_gleaf', 'giant_chunk_plan', 'real_giant_seed',
            'real_giant_seed_pair', 'real_giant_chain', 'groot_layout', 'target_descent_values', 'descent_values')}
        if a.trace_checks:
            leaf_checks=stats['device_gleaf']
            if int(leaf_checks['checked_groups'])!=int(leaf_checks['groups']) or int(leaf_checks['checked_groups'])<=0 or \
               int(leaf_checks['checked_leaf_words'])!=int(leaf_checks['device_leaf_words']) or int(leaf_checks['checked_leaf_words'])<=0:
                raise ValueError('Diagnostic GMP leaf coverage incomplete')
            leaf = stats['target_descent_values']
            if not leaf or int(leaf['words']) <= 0:
                raise ValueError('Normalized leaf hash missing')
            signature = {k:leaf[k] for k in ('words', 'hash')}
            if actual in references and signature != references[actual]:
                raise ValueError('Normalized leaf digest changed')
            references[actual] = signature
        row = dict(index=index, mode=mode, prefix=str(prefix), started_utc=started, finished_utc=stamp(),
                   engine_seconds=float(clock['total']), process_seconds=time.perf_counter()-start,
                   clock=clock, s4=check, shape=shape, stats=stats, requested_B2=requested, actual_B2=actual,
                   giant_policies=[fields(line,'giant_chain_policy') for line in text.splitlines() if line.startswith('giant_chain_policy:')])
        report['runs'].append(row); write(out/'summary.json', report)
        print(index, mode, row['engine_seconds'], stats['real_batched_split'], flush=True)
    report['means'] = {m:statistics.mean(r['engine_seconds'] for r in report['runs'] if r['mode']==m) for m in variants}
    report['complete'] = True; report['finished_utc'] = stamp()
    write(out/'summary.json', report)
    print(report['means'], flush=True)


if __name__ == '__main__':
    main()
