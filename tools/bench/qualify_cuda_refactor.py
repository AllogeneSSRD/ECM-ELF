"""Paired old/new CUDA performance gate with immutable inputs and raw samples.

The JSON plan contains explicit subprocess argument arrays (no shell), input,
environment, and Stage1 save / Stage2 result contracts. $RUN and $CONFIG expand
to collector-owned fresh paths. Timings include all samples, including slow
ones. --resume --pairs N adds pairs without replacing the original evidence.
"""
import argparse
import ctypes
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/test'))
from test_cuda_p3_runtime import stable
from bench_edwards_gpu_stage1 import validate, reference
from bench_stage2_n_scaling import Monitor, fields
from result_format import normalize_result


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def stamp(): return datetime.now(timezone.utc).isoformat()


def ratio_interval(ratios):
    rng = random.Random(20261020)
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    return dict(median=statistics.median(ratios), low95=boot[249], high95=boot[9749],
                pairs=len(ratios), bootstrap_resamples=10000, seed=20261020)


def passes_metrics(metrics):
    assert set(metrics) == {'wall_seconds', 'internal_seconds'}, 'both timing boundaries are required'
    return all(v['median'] <= 1.05 and v['high95'] <= 1.05 for v in metrics.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=21)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    assert a.pairs >= 21, 'qualification requires at least 21 pairs per case'
    plan = load(a.plan)
    assert plan.get('acceptance_ready'), 'input preparation plan is not bound to the final acceptance gates'
    assert plan['threshold'] == 1.05 and plan['device'] == 1
    prerequisites = {str(Path(path).resolve()):sha(path) for path in plan['prerequisites']}
    assert prerequisites and all(load(path)['complete'] for path in prerequisites), 'correctness/build gate incomplete'
    helper_paths = [ROOT/path for path in ('tools/bench/bench_edwards_gpu_stage1.py',
        'tools/bench/bench_stage2_n_scaling.py','tools/test/test_cuda_p3_runtime.py',
        'tools/test/test_edwards_gpu_pipeline.py','tools/stat/suyama_mont_ref.py','tools/ecm_report/result_format.py')]
    helpers = {str(path):sha(path) for path in helper_paths}
    out = a.output.resolve()
    if a.resume:
        result = load(out/'result.json')
        assert result['collector_sha256'] == sha(__file__)
        assert result['plan_sha256'] == sha(a.plan)
        assert result['prerequisites'] == prerequisites
        assert result['helpers'] == helpers
        assert not result.get('invalid_samples'), 'invalid samples require investigation in a new study'
        assert a.pairs > result['requested_pairs']
    else:
        out.mkdir(parents=True, exist_ok=False)
        shutil.copy2(__file__, out/'collector.py')
        shutil.copy2(a.plan, out/'plan.json')
        result = dict(complete=False, accepted=False, started_utc=stamp(),
            collector_sha256=sha(__file__), plan_sha256=sha(a.plan), samples=[],
            prerequisites=prerequisites,
            helpers=helpers,
            invalid_samples=[], summaries=[], binaries={}, device=plan['device'],
            protocol=dict(warmups_per_side=2, order='AB/BA alternating by pair',
                threshold=1.05, gate='paired time ratio median and two-sided 95% percentile bootstrap upper <= 1.05',
                wall='perf_counter immediately around subprocess.run through observed process exit',
                internal='Stage1 Montgomery CUDA event gputime; Edwards arithmetic API wall; Stage2 full_wall.total',
                nesting='independent boundaries; never summed',
                telemetry='whole selected GPU NVML samples every 250 ms; no clock/power changes',
                oracle='save/point and result checks outside all measured intervals'))
        for case in plan['cases']:
            for side in ('old', 'new'):
                src = Path(case[side+'_exe']).resolve()
                assert sha(src) == case[side+'_sha256']
                dst = out/'bin'/side/src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists(): assert sha(dst) == sha(src)
                else: shutil.copy2(src, dst)
                dll = src.parent/'gmp-10.dll'
                target = dst.parent/dll.name
                if target.exists(): assert sha(target) == sha(dll)
                else: shutil.copy2(dll, target)
                result['binaries'][str(dst)] = sha(dst)
                result['binaries'][str(target)] = sha(target)
        for path in helper_paths:
            dst = out/'dependencies'/path.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dst)
        assert len({v for k,v in result['binaries'].items() if k.endswith('.dll')}) == 1, 'GMP must match'
    result['complete'] = result['accepted'] = False
    result['requested_pairs'] = a.pairs
    cpu = platform.processor()
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
            r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    host = dict(cpu=cpu, logical_cpus=os.cpu_count(), os=platform.platform(), python=sys.version)
    if a.resume: assert result['host'] == host, 'host conditions changed; cannot pool samples'
    result['host'] = host
    def persist():
        tmp = out/'result.json.new'
        tmp.write_text(json.dumps(result, indent=2)+'\n')
        tmp.replace(out/'result.json')
    config = out/'fixture.toml'
    config.write_text('[stage1]\nauto_stage2 = 0\nedwards_stage2_transfer = false\n'
                      '[stage2]\nstage2_short_calibration=false\nstage2_tune_runtime=false\n')
    env = {k:v for k,v in os.environ.items() if not k.startswith(('NTT_', 'ECM_', 'CUDA_FORCE_', 'CUDA_DISABLE_'))}
    clean_env = dict(env)
    monitor = Monitor(plan['device'])
    previous_gpu = result.get('gpu')
    if a.resume:
        assert all(previous_gpu[k] == monitor.identity[k] for k in ('index','name','uuid'))
    result['gpu'] = monitor.identity
    def process_snapshot(case, phase):
        text = subprocess.check_output(['nvidia-smi','-i',str(plan['device']),'--query-compute-apps=gpu_uuid,pid,process_name',
                                       '--format=csv']).decode('utf-8','replace')
        rows = list(csv.DictReader(io.StringIO(text), skipinitialspace=True))
        selected = [r for r in rows if r.get('gpu_uuid','').strip() == monitor.identity['uuid']]
        prefix = os.path.normcase(str(out/'bin'))+os.sep
        foreign = [r for r in selected if not os.path.normcase(r['process_name'].strip()).startswith(prefix)]
        folder = out/'processes'; folder.mkdir(exist_ok=True)
        (folder/f'{case}_{a.pairs}_{phase}.json').write_text(json.dumps(dict(utc=stamp(),
            rows=rows, selected=selected, foreign=foreign,
            scope='before/after case snapshots; not continuous process attribution'),indent=2)+'\n')
        assert not foreign, ('selected GPU contains another compute context',foreign)
    info = subprocess.check_output(['nvidia-smi', '-i', str(plan['device']),
        '--query-gpu=index,name,uuid,driver_version,compute_cap,power.limit', '--format=csv'])
    if a.resume: assert (out/'gpu.csv').read_bytes() == info, 'GPU/driver conditions changed'
    else: (out/'gpu.csv').write_bytes(info)
    limit = ctypes.c_uint()
    status = monitor.lib.nvmlDeviceGetPowerManagementLimit(monitor.handle, ctypes.byref(limit))
    result['gpu']['power_limit_mw'] = limit.value if status == 0 else None
    result['gpu']['power_limit_status'] = status
    if a.resume: assert result['gpu'] == previous_gpu, 'power policy changed; cannot pool samples'
    stop = threading.Event()
    telemetry_errors = []
    def telemetry():
        try:
            with (out/'telemetry.jsonl').open('a') as log:
                while not stop.is_set():
                    log.write(json.dumps(monitor.sample())+'\n'); log.flush()
                    stop.wait(.25)
        except BaseException as error: telemetry_errors.append(repr(error))
    thread = threading.Thread(target=telemetry, daemon=True)
    thread.start()
    signatures = {}
    def run(case, side, pair, warmup=False):
        name = case['name']+'_'+side+('_warm' if warmup else '_r')+str(pair)
        folder = out/'runs'/name
        folder.mkdir(parents=True, exist_ok=False)
        exe = out/'bin'/side/Path(case[side+'_exe']).name
        args = [str(v).replace('$RUN', str(folder)).replace('$CONFIG', str(config)) for v in case['args']]
        command = [str(exe), *args]
        selected_env = clean_env | case.get('environment', {})
        (folder/'invocation.json').write_text(json.dumps(dict(command=command,
            environment=case.get('environment', {}), input=case.get('input'), started_utc=stamp()), indent=2)+'\n')
        started = time.perf_counter()
        proc = subprocess.run(command, cwd=folder, env=selected_env,
            input=case.get('input', '').encode(), capture_output=True, timeout=3600)
        wall = time.perf_counter()-started
        raw = proc.stdout+proc.stderr
        (folder/'stdout.log').write_bytes(raw)
        text = raw.decode('utf-8', 'replace')
        for name in ('debug.log', 'engine.log'):
            path = folder/name
            if path.exists(): text += '\n'+path.read_text(encoding='utf-8', errors='replace')
        row = dict(case=case['name'], side=side, pair=pair, warmup=warmup,
                   wall_seconds=wall, exit_code=proc.returncode, folder=str(folder))
        # Persist the observation before validation; a rejected run stays visible.
        with (out/'observations.jsonl').open('a') as log: log.write(json.dumps(row)+'\n')
        assert proc.returncode == 0, (case['name'], side, proc.returncode, text[-1500:])
        if case['kind'] == 'stage1':
            n = int(case['N_hex'], 16)
            sig = validate(folder/'point.save', n, case['b1'], case['curves'], case['sigma'], case['model'])
            assert len(sig) == case['curves']
            if case['model'].startswith('ed'):
                values = re.findall(r'GPU Edwards arithmetic: algorithm=(\w+) api_seconds=([\d.]+)', text)
                binary = case['model'] == 'ed-binary'
                assert len(values) == 1 and values[0][0] == ('binary' if binary else 'naf')
                selected = re.findall(r'GPU Edwards Stage1: device=(\d+).*window=(\d+) doubling=(\d+)', text)
                assert len(selected) == 1 and int(selected[0][0]) == plan['device']
                assert int(selected[0][1]) == (0 if binary else 6)
                assert int(selected[0][2]) == (5 if case['model']=='ed-5' else 4)
                internal = float(values[0][1])
            else:
                assert f'GPU: will use device {plan["device"]}:' in text
                values = re.findall(r'GPU stage1 returned: \d+ gputime=([\d.eE+-]+) ms', text)
                assert len(values) == 1
                internal = float(values[0])/1000
            if case['name'] not in signatures:
                for index in sorted({0, len(sig)-1}):
                    expected = (tuple(case['oracle'][str(index)]) if 'oracle' in case else
                                reference(n, sig[index][0], case['b1'], case['model']))
                    assert sig[index][1:] == expected, (case['name'], index, 'independent point oracle')
        else:
            sig = [stable(normalize_result(json.loads(line))) for line in (folder/'result.jsonl').read_text().splitlines()]
            assert len(sig) == 1 and not sig[0]['bad_factors']
            n = int(sig[0]['N_hex'], 16)
            assert n == int(case['N_hex'], 16)
            assert all(1<int(f)<n and n%int(f)==0 for f in sig[0]['factors'])
            assert not re.search(r'(?:mismatches|gmp_check_bad|gmp_selftest_bad)=[1-9]', text)
            for token in ('mont_selftest: cases=2048 mismatches=0', 'gmp_check_bad=0', 'pending=0'):
                assert token in text, (case['name'], token)
            internal = float(fields(text, 'stage2_full_wall')['total'])
            counts = fields(text, 's4_multiply_stats')
            sig = dict(result=sig, counts={k:counts[k] for k in
                ('launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks')})
        assert internal > 0
        if case['name'] in signatures: assert signatures[case['name']] == sig, (case['name'], 'output/workload changed')
        else: signatures[case['name']] = sig
        row.update(internal_seconds=internal, validated=True,
                   signature_sha256=hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest())
        assert all(r['signature_sha256'] == row['signature_sha256'] for r in result['samples']
                   if r['case'] == case['name']), (case['name'], 'retained output contract changed')
        result['samples'].append(row)
        persist()
        return row
    try:
        persist()
        for case in plan['cases']:
            process_snapshot(case['name'], 'before')
            for side in ('old', 'new'):
                src = Path(case[side+'_exe'])
                assert sha(src) == case[side+'_sha256']
            if 'save' in case: assert sha(case['save']) == case['save_sha256']
            if 'boundary_input' in case:
                assert sha(case['boundary_input']['path']) == case['boundary_input']['sha256']
            existing = [r for r in result['samples'] if r['case'] == case['name']]
            if a.resume:
                # Reconstruct the output contract from a retained valid pair.
                old_case = out/'runs'/(case['name']+'_old_r0')
                if case['kind'] == 'stage1':
                    signatures[case['name']] = validate(old_case/'point.save', int(case['N_hex'],16),
                        case['b1'], case['curves'], case['sigma'], case['model'])
                # Stage2 resume obtains the contract from its new warmup below.
            for warm in range(2):
                for side in ('old', 'new'):
                    run(case, side, f'{a.pairs}_{warm}' if a.resume else warm, True)
            done = {r['pair'] for r in existing if not r['warmup']}
            assert all(sum(r['pair']==i and not r['warmup'] for r in existing) == 2 for i in done)
            for pair in range(a.pairs):
                if pair in done: continue
                for side in (('old','new') if pair % 2 == 0 else ('new','old')): run(case, side, pair)
                if pair % 5 == 0: print('PAIR', case['name'], pair+1, '/', a.pairs, flush=True)
            rows = [r for r in result['samples'] if r['case']==case['name'] and not r['warmup']]
            summary = dict(case=case['name'], kind=case['kind'], metrics={})
            for metric in ('wall_seconds','internal_seconds'):
                ratios = [next(r[metric] for r in rows if r['pair']==i and r['side']=='new') /
                          next(r[metric] for r in rows if r['pair']==i and r['side']=='old') for i in range(a.pairs)]
                summary['metrics'][metric] = ratio_interval(ratios)
            summary['accepted'] = passes_metrics(summary['metrics'])
            result['summaries'] = [s for s in result['summaries'] if s['case'] != case['name']]+[summary]
            process_snapshot(case['name'], 'after')
            persist()
            print('GATE', json.dumps(summary), flush=True)
        assert all(sha(path) == value for path,value in result['binaries'].items()), 'binary changed'
        assert helpers == {str(path):sha(path) for path in helper_paths}, 'validation helper changed'
        stop.set(); thread.join(timeout=5)
        assert not telemetry_errors, telemetry_errors
        result['telemetry_sha256'] = sha(out/'telemetry.jsonl')
        result['accepted'] = all(s['accepted'] for s in result['summaries'])
        result['complete'] = True
        result['finished_utc'] = stamp()
        persist()
    except BaseException as error:
        result['invalid_samples'].append(dict(error=repr(error), utc=stamp()))
        persist()
        raise
    finally:
        stop.set(); thread.join(timeout=5); monitor.close()
    if not result['accepted']: raise SystemExit(2)


if __name__ == '__main__': main()
