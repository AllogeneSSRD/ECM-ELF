"""Measure D-selection preparation with matched verified worktodo queues on GPU1.

All arms use the same queue/checkpoint entry and replayed NF point. This measures
timing observations, not new independent ECM coverage. Never changes GPU controls.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics as st
import subprocess
import sys
import time

from bench_stage2_version import check_gpu_processes, load, stamp, write
from bench_stage2_n_scaling import Monitor, fields, number
from bench_edwards_gpu_stage1 import validate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/ecm_report'))
from result_format import normalize_result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log_rows(text, label):
    return [fields(line, label) for line in text.splitlines() if line.startswith(label+':')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('exe', 'profile', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--exponent', type=int, default=2003)
    p.add_argument('--b2', type=int, default=27015978990)
    p.add_argument('--curves', type=int, default=12)
    p.add_argument('--order', default='carrier,on,off,off,on,carrier')
    p.add_argument('--prepare-cache', type=int, choices=(0, 1), default=1)
    p.add_argument('--allow-learning', action='store_true', help='Retain changing profiles/trials as learning data, not a frozen timing control')
    p.add_argument('--timeout', type=float, default=1800)
    a = p.parse_args(); modes = a.order.split(',')
    if a.curves < 4 or not modes or any(m not in ('carrier', 'normal', 'on', 'off', 'uncached') for m in modes):
        p.error('Need >=4 curves and carrier/normal/on/off/uncached arms')
    original = load(a.reference/'plan.json')
    case = next(c for c in original['cases'] if c['exponent'] == a.exponent)
    n = int(case['N_hex'], 16); whole = (1 << case['exponent'])-1
    if n <= 1 or whole % n or n == whole or a.b2 <= case['b1']:
        raise ValueError('Need a proper verified cofactor and B2 > B1')
    validate(Path(case['save']), n, case['b1'], 1, case['sigma'], 'ed')
    source_ids = {str(x.resolve()): sha(x) for x in (a.exe, a.profile, Path(case['save']))}
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    binary = out/'bin'; binary.mkdir()
    for name in (a.exe.name, 'gmp-10.dll', 'build_manifest.json'):
        shutil.copy2(a.exe.resolve().parent/name, binary/name)
    manifest = load(binary/'build_manifest.json')
    for name, digest in manifest['source_hashes'].items():
        src = ROOT/name
        if sha(src) != digest.lower():
            raise ValueError('Compiled source changed: '+name)
        dst = out/'sources'/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    shutil.copy2(__file__, out/'collector_snapshot.py')
    point = next(s for s in Path(case['save']).read_text(encoding='utf-8-sig').splitlines() if s.startswith('METHOD='))
    save = out/'replay.save'; save.write_text((point+'\n')*a.curves, encoding='utf-8')
    env = {k:v for k,v in os.environ.items() if not k.startswith(('NTT_', 'ECM_'))
           and k not in ('CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING')}
    env['NTT_NO_PROGRESS'] = '1'; monitor = Monitor(1)
    report = dict(complete=False, frozen_control=not a.allow_learning, started_utc=stamp(), input={k:case[k] for k in ('id','bits','exponent','b1','sigma')},
        B2=a.b2, curves_per_queue=a.curves, order=modes, source_ids=source_ids, gpu=monitor.identity,
        protocol='Matched ECMSTAGE2 queues; no warmup; parent curve_done includes durable queue progress. '
                 'on/off share the same frozen measured profile; only runtime exploration differs. '
                 'All curves retained, including the last three with the short-queue search cutoff.', runs=[], failures=[])
    write(out/'report.json', report)
    try:
        for index, mode in enumerate(modes, 1):
            folder = out/f'{index:02}_{mode}'; folder.mkdir()
            ini = folder/'ecm.toml'; ini.write_text('user=\ncomputer=\nstage2_finished=finished.txt\n', encoding='utf-8')
            queue = folder/'worktodo.txt'
            line = f'ECMSTAGE2=1,2,{case["exponent"]},-1,"{save}",{a.b2},0,{a.curves},"{whole//n}"\n'
            queue.write_text(line, encoding='utf-8'); (folder/'worktodo.initial.txt').write_text(line, encoding='utf-8')
            profile = folder/'stage2_ecm_tune.toml'
            measured = mode in ('on', 'off', 'uncached')
            if measured:
                shutil.copy2(a.profile, profile)
            before = sha(profile) if measured else None
            cmd = [str(binary/a.exe.name), '--ini', str(ini), '--worktodo', str(queue), '--once', '--device', '1',
                '--d', '0', '--auto-carrier', '1', '--tune-runtime', '1' if mode in ('on', 'uncached') else '0',
                '--runtime-prepare-cache', '0' if mode == 'uncached' else str(a.prepare_cache),
                '--tune-runtime-samples', '10', '--tune-profile', str(profile), '--short-calibration', '0',
                '--arena-mb', '6300', '--owner-budget-mb', '640', '--batch-mb', '256', '--relocation', '0',
                '--ntt-counts', '0', '--factor-only', '--log-level', 'phases', '--log', str(folder/'engine.log'),
                '--results', str(folder/'result.jsonl'), '--submission', str(folder/'submission.txt')]
            if not measured:
                cmd += ['--carrier-exponent', str(case['exponent'] if mode == 'carrier' else 0)]
            write(folder/'command.json', cmd)
            check_gpu_processes(monitor, out, folder/'before.json')
            print('RUN', index, '/', len(modes), mode, flush=True)
            started = time.monotonic(); telemetry = []
            with (folder/'driver.log').open('wb') as log:
                process = subprocess.Popen(cmd, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    telemetry.append(monitor.sample())
                    if time.monotonic()-started > a.timeout:
                        process.kill();process.wait();raise TimeoutError(str(folder))
                    time.sleep(.25)
            wall = time.monotonic()-started; write(folder/'telemetry.json', telemetry)
            check_gpu_processes(monitor, out, folder/'after.json')
            if process.returncode:
                raise RuntimeError('Queue failed: '+str(folder))
            raw = [json.loads(s) for s in (folder/'result.jsonl').read_text(encoding='utf-8').splitlines()]
            results = [normalize_result(r) for r in raw]
            log = (folder/'engine.log').read_text(encoding='utf-8')
            phases = log_rows(log, 'stage2_full_wall'); checks = log_rows(log, 's4_multiply_stats')
            done = re.findall(r'curve_done:.*?\bwall=([\d.]+)', (folder/'driver.log').read_text(encoding='utf-8'))
            if not len(raw) == len(phases) == len(checks) == len(done) == a.curves:
                raise ValueError('Missing complete result, timing or arithmetic check')
            if queue.read_text(encoding='utf-8-sig').strip():
                raise ValueError('Queue did not complete')
            unchanged = sha(profile) == before if measured else not profile.exists()
            if not unchanged and (not measured or not a.allow_learning):
                raise ValueError('Measured profile changed; this is not a frozen preparation control')
            curves = []
            for i, (rr, r, phase, check, parent_wall) in enumerate(zip(raw, results, phases, checks, done), 1):
                if rr['status'] != 'NF' or r['bad_factors'] or r['factors'] or r['N_hex'] != case['N_hex'] or r['B2'] != a.b2:
                    raise ValueError('Input identity or NF status differs')
                if phase['clean'] != '1' or abs(float(phase['total'])-float(phase['init'])-float(phase['main'])) > .003:
                    raise ValueError('Engine timing did not close')
                if min(int(check['gmp_checked']), int(check['gmp_selftest_cases'])) <= 0 or int(check['gmp_check_bad']) or int(check['gmp_selftest_bad']):
                    raise ValueError('Mandatory arithmetic verification failed')
                choice = r['tune_plan']
                if measured and not a.allow_learning and (choice.get('runtime_probe') or choice.get('runtime_samples', 0) < 10):
                    raise ValueError('Selected shape is still learning')
                curves.append(dict(index=i, remaining=a.curves-i+1, engine_seconds=float(phase['total']),
                    parent_seconds=float(parent_wall), outer_seconds=r['seconds'], D=r['d'], carrier=r['carrier_exponent'],
                    selection=choice, observation_added=bool(rr.get('tune_observation')),
                    gmp_checked=int(check['gmp_checked']), selftest_cases=int(check['gmp_selftest_cases'])))
            row = dict(mode=mode, folder=str(folder), process_seconds=wall,
                curve_wall_sum=sum(c['parent_seconds'] for c in curves), engine_sum=sum(c['engine_seconds'] for c in curves),
                selected_shapes=sorted({(c['D'], c['carrier']) for c in curves}), profile_unchanged=unchanged, curves=curves)
            report['runs'].append(row); write(out/'report.json', report)
            print('DONE', mode, 'parent', round(row['curve_wall_sum'], 4), 'engine', round(row['engine_sum'], 4), flush=True)
        if any(sha(path) != digest for path, digest in source_ids.items()):
            raise ValueError('Original inputs changed')
        if not a.allow_learning:
            shapes = {tuple(shape) for r in report['runs'] if r['mode'] in ('on', 'off', 'uncached') for shape in r['selected_shapes']}
            if len(shapes) > 1:
                raise ValueError('Frozen-profile arms selected different geometries')
        report['summary'] = {}
        for mode in dict.fromkeys(modes):
            runs = [r for r in report['runs'] if r['mode'] == mode]
            report['summary'][mode] = dict(queues=len(runs), curves=sum(len(r['curves']) for r in runs),
                median_curve_wall_sum=st.median(r['curve_wall_sum'] for r in runs),
                median_engine_sum=st.median(r['engine_sum'] for r in runs))
        report.update(complete=True, finished_utc=stamp()); write(out/'report.json', report)
        print(json.dumps(report['summary'], indent=2), flush=True)
    except Exception as e:
        report['failures'].append(dict(utc=stamp(), error=str(e)));write(out/'report.json', report);raise


if __name__ == '__main__':
    main()
