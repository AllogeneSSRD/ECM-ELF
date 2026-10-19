"""Repeat the complete exact-N Edwards reference matrix with a numbered release.

GPU1 only. Reuse verified Stage1 saves and historical Prime95 timings, with
ordinary/carrier runs kept separate. Identity uses version numbers and frozen
files; result checksums and mandatory arithmetic verification remain enabled.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from bench_edwards_gpu_stage1 import validate
from bench_stage2_n_scaling import Monitor, fields, number

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/ecm_report'))
from result_format import normalize_result


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    temp = path.with_suffix(path.suffix+'.new')
    temp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def without_digests(value):
    if isinstance(value, dict):
        return {k:without_digests(v) for k,v in value.items() if 'sha256' not in k.lower()}
    if isinstance(value, list):
        return [without_digests(v) for v in value]
    return value


def prepare(args, out):
    old = load(args.reference/'plan.json')
    if not old['prepared'] or len(old['cases']) != 13:
        raise ValueError('Need the complete verified 13-input reference')
    out.mkdir(parents=True, exist_ok=False)
    binary = out/'bin'
    binary.mkdir()
    for path in [args.exe.resolve(), args.exe.resolve().parent/'gmp-10.dll']:
        shutil.copy2(path, binary/path.name)
    shutil.copy2(__file__, out/'collector_snapshot.py')
    inputs = out/'inputs'
    inputs.mkdir()
    cases = []
    for original in old['cases']:
        case = without_digests(original)
        target = inputs/Path(case['save']).name
        shutil.copy2(case['save'], target)
        case['save'] = str(target)
        n = int(case['N_hex'], 16)
        rows = validate(target, n, case['b1'], 1, case['sigma'], 'ed')
        if rows[0][1] != 'POINT':
            raise ValueError('Reference is not a complete Stage1 point')
        modulus = (1 << case['exponent'])-1
        for factor in case['known']:
            factor = int(factor)
            if factor <= 1 or modulus % factor:
                raise ValueError('Known factor does not divide M')
            modulus //= factor
        if modulus != n or n.bit_length() != case['bits']:
            raise ValueError('Exact cofactor identity differs')
        cases.append(case)
    report = dict(schema=1, program_version=args.version, started_utc=stamp(),
        reference_version=None, reference_note='Historical full matrix did not record a numeric version',
        reference=str(args.reference.resolve()), exe=str(binary/args.exe.name),
        device=1, repeats=args.repeats, arena_mb=6300, fold_mb=640, batch_mb=256,
        cases=cases, prime95_rows=without_digests(old['prime95_rows']),
        runs=[], failures=[], complete=False,
        protocol=dict(warmups=0, first_invocation='retained',
            order='repeat -> reference cell -> ordinary/carrier; reverse engines on alternate repeats',
            primary='stage2_full_wall.total=init+main; includes mandatory checks and GCD',
            process='launch through observed exit; polling interval 0.25 seconds',
            excluded='Stage1, process startup, outer selection and preflight point checks',
            settings='GPU1 only, no power/clock/priority changes; no concurrent task-owned compilation'))
    write(out/'plan.json', report)
    return report


def run(args):
    out = args.output.resolve()
    report = load(out/'plan.json') if args.resume else prepare(args, out)
    if report['program_version'] != args.version or report['complete']:
        raise ValueError('Version differs or study is already complete')
    if (out/'collector_snapshot.py').read_bytes() != Path(__file__).read_bytes():
        raise ValueError('Collector changed; preserve this study and start a new one')
    if report['failures']:
        raise ValueError('Recorded failures require review')
    ini = out/'bench.ini'
    ini.write_text('user=\ncomputer=\nstage2_tune_runtime=0\nstage2_auto_carrier=0\nstage2_ntt_counts=0\n', encoding='utf-8')
    runs = out/'runs'
    runs.mkdir(exist_ok=True)
    cells = []
    for ref in report['prime95_rows']:
        if ref['curve_type'] != 'Edwards' or not ref['benchmark_eligible']:
            continue
        case = next(c for c in report['cases'] if c['N_hex']==ref['N_hex']
                    and c['sigma']==int(ref['s']) and c['b1']==ref['b1'])
        cells.append((case, ref))
    if len(cells) != 26:
        raise ValueError('Reference matrix does not contain 26 conditions')
    env = {k:v for k,v in os.environ.items() if not k.startswith(('NTT_', 'ECM_'))
           and k not in ('CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING')}
    env['NTT_NO_PROGRESS'] = '1'
    monitor = Monitor(1)
    report['gpu'] = {k:v for k,v in monitor.identity.items() if k != 'uuid'}
    accepted = 0
    try:
        for repeat in range(1, report['repeats']+1):
            for case, ref in cells:
                modes = ['gpu', 'gpu-carrier'] if repeat%2 else ['gpu-carrier', 'gpu']
                for mode in modes:
                    tag = f'{case["id"]}_b{ref["b2"]}_{mode}_r{repeat}'
                    if any(r['id']==tag for r in report['runs']):
                        continue
                    prefix = runs/tag
                    if prefix.with_suffix('.driver.log').exists():
                        raise ValueError('Unrecorded invocation exists; preserve evidence: '+tag)
                    carrier = case['exponent'] if mode=='gpu-carrier' else 0
                    command = [report['exe'], '--ini', str(ini), '--save', case['save'],
                        '--b2', str(ref['b2']), '--curves', '1', '--exponent', str(case['exponent']),
                        '--results', str(prefix.with_suffix('.jsonl')), '--submission', str(prefix.with_suffix('.json.txt')),
                        '--device', '1', '--d', '0', '--short-calibration', '0', '--tune-runtime', '0',
                        '--arena-mb', '6300', '--owner-budget-mb', '640', '--batch-mb', '256',
                        '--factor-only', '--relocation', '0', '--log-level', 'phases',
                        '--log', str(prefix.with_suffix('.engine.log')),
                        '--debug-log-file', str(prefix.with_suffix('.debug.log')),
                        '--carrier-exponent', str(carrier)]
                    for factor in case['known']:
                        command += ['--known-factor', str(factor)]
                    print('RUN', len(report['runs'])+1, '/', 26*2*report['repeats'], tag, flush=True)
                    started = stamp()
                    start = time.perf_counter()
                    observations = []
                    with prefix.with_suffix('.driver.log').open('wb') as log:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=out, env=env)
                        while process.poll() is None:
                            observations.append(monitor.sample())
                            if time.perf_counter()-start > args.timeout:
                                process.kill();process.wait();raise TimeoutError(tag)
                            time.sleep(.25)
                    wall = time.perf_counter()-start
                    write(prefix.with_suffix('.telemetry.json'), observations)
                    if process.returncode:
                        raise ValueError('Worker failed: '+tag)
                    result = normalize_result(load(prefix.with_suffix('.jsonl')))
                    if result.get('program', {}).get('version') != args.version:
                        raise ValueError('Worker does not report the requested numeric version')
                    identity = dict(N_hex=case['N_hex'], B1=case['b1'], B2=ref['b2'],
                                    sigma=case['sigma'], curve='ed', carrier_exponent=carrier)
                    if any(result[k]!=v for k,v in identity.items()) or result['bad_factors']:
                        raise ValueError('Result identity or arithmetic differs')
                    n = int(case['N_hex'], 16)
                    factors = list(map(int, result.get('factors', [])))
                    if any(not 1<f<n or n%f for f in factors):
                        raise ValueError('Invalid factor')
                    text = prefix.with_suffix('.engine.log').read_text(encoding='utf-8')
                    if prefix.with_suffix('.debug.log').exists():
                        text += '\n'+prefix.with_suffix('.debug.log').read_text(encoding='utf-8')
                    phase = fields(text, 'stage2_full_wall')
                    shape = fields(text, 'real_shape')
                    s4 = fields(text, 's4_multiply_stats')
                    if phase['clean']!='1' or abs(float(phase['total'])-float(phase['init'])-float(phase['main']))>.003:
                        raise ValueError('Unclosed timing ledger')
                    if not int(s4['gmp_selftest_cases']) or not int(s4['gmp_checked']) or int(s4['gmp_selftest_bad']) or int(s4['gmp_check_bad']):
                        raise ValueError('Missing or failed arithmetic verification')
                    labels = ['real_baby','real_batched_split','real_batched_prepare','real_batched_wall',
                              'real_giant_base','real_giant_chain','scaled_frontier_device','scaled_root_device',
                              'ntt_workspace_stats','real_init','real_f','real_finv','stage2_phase_wall']
                    row = dict(id=tag, case=case['id'], engine=mode, repeat=repeat,
                        version=args.version, B2=ref['b2'], nominal_B2=ref['b2_requested'],
                        engine_seconds=float(phase['total']), process_seconds=wall,
                        phase_seconds=phase, shape=shape, s4=s4, factors=factors, full_sample=not factors,
                        D=number(shape['D']), P=number(shape['P']), carrier_exponent=carrier,
                        stats={k:fields(text,k,False) for k in labels},
                        command=command, prefix=str(prefix), started_utc=started, finished_utc=stamp())
                    report['runs'].append(row)
                    write(out/'plan.json', report)
                    accepted += 1
                    print('DONE',len(report['runs']),round(row['engine_seconds'],6),'seconds',flush=True)
                    if args.max_runs and accepted>=args.max_runs:
                        return
        report.update(complete=True, finished_utc=stamp())
        write(out/'plan.json', report)
    except Exception as error:
        report['failures'].append(dict(utc=stamp(), error=str(error)))
        write(out/'plan.json', report)
        raise
    finally:
        monitor.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--version',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-runs',type=int,default=0)
    parser.add_argument('--timeout',type=float,default=7200)
    args=parser.parse_args()
    if not args.resume and not (args.exe and args.reference):
        parser.error('New study needs --exe and --reference')
    run(args)
