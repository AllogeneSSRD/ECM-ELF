"""Fixed-D, direct-worker event-wait A/B; measures the owned process lifetime.

Input is an existing full-matrix plan.json (verified text Stage1 saves). Nothing
in production or on the GPU is reconfigured. Raw logs are never overwritten.
"""
import argparse
import ctypes
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecm_report.result_format import normalize_result

# Saves are bounded to 16384 bits; their decimal result identity can exceed
# Python's default 4300-digit conversion limit.
sys.set_int_max_str_digits(0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fields(text, tag):
    rows = re.findall(r'^' + re.escape(tag) + r': (.*)$', text, re.M)
    if not rows:
        raise ValueError('Missing ' + tag)
    return dict(re.findall(r'(\w+)=([^\s|]+)', rows[-1]))


def process_times(child):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
    kernel.GetProcessTimes.restype = w.BOOL
    times = [w.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(w.HANDLE(int(child._handle)), *map(ctypes.byref, times)):
        raise ctypes.WinError(ctypes.get_last_error())
    ticks = [(x.dwHighDateTime << 32) | x.dwLowDateTime for x in times]
    if not ticks[1]:
        raise ValueError('Process must have exited')
    wall = (ticks[1] - ticks[0]) / 1e7
    cpu = (ticks[2] + ticks[3]) / 1e7
    return dict(wall_seconds=wall, user_seconds=ticks[3] / 1e7,
                kernel_seconds=ticks[2] / 1e7, cpu_seconds=cpu,
                logical_cores=cpu / wall)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exponents', type=int, nargs='+', default=[5003])
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--b2', type=int, default=26000000000)
    p.add_argument('--d', type=int, default=120120)
    p.add_argument('--arena-mb', type=int, default=4096)
    p.add_argument('--fold-mb', type=int, default=640)
    p.add_argument('--carrier', action='store_true')
    p.add_argument('--graph', action='store_true', help='Exercise existing short-NTT Graph submission')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--variants', nargs='+', default=['legacy', 'sleep', 'fifo', 'both'],
                   choices=['baseline', 'legacy', 'sleep', 'fifo', 'both', 'readback', 'blocking', 'hybrid'])
    a = p.parse_args()
    if a.repeats < 1:
        p.error('repeats must be positive')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, w.BOOL, w.LPCWSTR]
    kernel.CreateMutexW.restype = w.HANDLE
    # Hold for this process lifetime. Prevent overlapping invocations of this
    # benchmark; this does not reserve the GPU against external applications.
    mutex = kernel.CreateMutexW(None, True, f'Local\\ECM-ELF-Stage2-Benchmark-GPU{a.device}')
    if not mutex or ctypes.get_last_error() == 183:
        raise RuntimeError('Another Stage2 benchmark owns this device lock')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use a fresh output directory')
    exe = a.exe.resolve()
    plan = json.loads(a.plan.read_text(encoding='utf-8-sig'))
    cases = [c for c in plan['cases'] if c['exponent'] in a.exponents]
    if sorted(c['exponent'] for c in cases) != sorted(a.exponents):
        raise ValueError('Missing or ambiguous cases')
    binary_hash = sha(exe)
    identity = dict(binary_sha256=binary_hash, plan_sha256=sha(a.plan),
                    script_sha256=sha(__file__), saves={c['id']: sha(c['save']) for c in cases})
    (out/'benchmark.py').write_bytes(Path(__file__).read_bytes())
    report = dict(complete=False, identity=identity, controls=vars(a) | {'exe': str(exe)}, runs=[])
    config = out/'ecm.toml'
    config.write_text('[stage2]\nstage2_tune_runtime=false\nstage2_short_calibration=false\n', encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env.update(CUDA_LAUNCH_BLOCKING='0', NTT_NO_PROGRESS='1', NTT_CUDA_WAIT_MODE='0')
    env['NTT_FOLD_DEVICE_MAX_MB'] = str(a.fold_mb)
    if a.arena_mb:
        env['NTT_ARENA_CAP_KB'] = str(a.arena_mb*1024)
    env['NTT_S4_BATCH_MB'] = '64'
    env['NTT_SHORT_GRAPH'] = str(int(a.graph))
    def persist():
        (out/'measurements.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    persist()
    references = {}
    policies = {'baseline': {}, 'legacy': {'NTT_HOST_EVENT_WAIT': '0', 'NTT_S4_TIMING_FIFO': '0'},
                'sleep': {'NTT_HOST_EVENT_WAIT': '1', 'NTT_S4_TIMING_FIFO': '0'},
                'fifo': {'NTT_HOST_EVENT_WAIT': '0', 'NTT_S4_TIMING_FIFO': '1'},
                'both': {'NTT_HOST_EVENT_WAIT': '1', 'NTT_S4_TIMING_FIFO': '1'}}
    for values in policies.values():
        values['NTT_HOST_READBACK_WAIT'] = '0'
    policies['readback'] = {'NTT_HOST_EVENT_WAIT': '1', 'NTT_S4_TIMING_FIFO': '1', 'NTT_HOST_READBACK_WAIT': '1'}
    policies['blocking'] = {'NTT_HOST_EVENT_WAIT': '0', 'NTT_S4_TIMING_FIFO': '1',
                            'NTT_HOST_READBACK_WAIT': '0', 'NTT_CUDA_WAIT_MODE': '4'}
    policies['hybrid'] = policies['blocking'] | {'NTT_HOST_EVENT_WAIT':'1'}
    for case in cases:
        saved = Path(case['save'])
        line = saved.read_bytes().partition(b'\n')[0]
        if not line.startswith((b'METHOD=ECM;', b'METHOD=ECM-EDWARDS-AM;')):
            raise ValueError('Expected the verified first text record')
        fingerprint = 14695981039346656037
        for byte in line:
            fingerprint = ((fingerprint ^ byte) * 1099511628211) & ((1 << 64)-1)
        for repeat in range(a.repeats):
            # Reverse alternate rounds to reduce monotonic thermal/order bias.
            for variant in a.variants[::1 if repeat % 2 == 0 else -1]:
                name = f'{len(report["runs"]):03d}_{case["id"]}_{variant}'
                log, debug, result = [out/(name+s) for s in ('.log', '.debug.log', '.jsonl')]
                if sha(exe) != binary_hash or sha(saved) != identity['saves'][case['id']]:
                    raise ValueError('Frozen binary/input changed')
                cmd = [str(exe), '--config', str(config), '--save', str(saved), '--device', str(a.device),
                       '--b2', str(a.b2), '--d', str(a.d), '--arena-mb', str(a.arena_mb),
                       '--owner-budget-mb', str(a.fold_mb), '--factor-only', '--results', str(result),
                       '--debug-log-file', str(debug), '--curve-worker', '--record-offset', '0',
                       '--record-index', '1', '--record-hash', str(fingerprint)]
                if a.carrier:
                    cmd += ['--carrier-exponent', str(case['exponent'])]
                telemetry = subprocess.check_output(['nvidia-smi', f'--id={a.device}',
                    '--query-gpu=uuid,name,clocks.sm,utilization.gpu,memory.free', '--format=csv,noheader'], text=True).strip()
                start = time.perf_counter()
                gpu_log = out/(name+'.gpu.csv')
                with log.open('wb') as stream, gpu_log.open('wb') as gpu_stream:
                    monitor = subprocess.Popen(['nvidia-smi', f'--id={a.device}',
                        '--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits', '--loop-ms=500'], stdout=gpu_stream, stderr=subprocess.STDOUT)
                    child = subprocess.Popen(cmd, cwd=out, env=env | policies[variant], stdout=stream, stderr=subprocess.STDOUT)
                    try:
                        child.wait(timeout=600)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                        raise
                    finally:
                        monitor.terminate()
                        monitor.wait(timeout=10)
                cpu = process_times(child)
                cpu['launcher_wall_seconds'] = time.perf_counter() - start
                if child.returncode:
                    raise RuntimeError(f'{name} failed ({child.returncode}); see {log}')
                text = log.read_text(encoding='utf-8', errors='replace') + '\n' + debug.read_text(encoding='utf-8', errors='replace')
                r = normalize_result(json.loads(result.read_text(encoding='utf-8')))
                oracle = fields(text, 's4_oracle_stats')
                multiply = fields(text, 's4_multiply_stats')
                if (r['B1'] != case['b1'] or r['B2'] != a.b2 or r['sigma'] != case['sigma']
                    or int(r['N_hex'], 16) != int(case['N_hex'], 16) or r['bad_factors']):
                    raise ValueError('Actual target/parameters changed')
                if any(t not in text for t in ('gmp_selftest_bad=0', 'gmp_check_bad=0', 'pending=0', 'clean=1')):
                    raise ValueError('Missing arithmetic verification')
                if oracle['selected'] != oracle['compared']:
                    raise ValueError('Uncompared snapshots')
                signature = (fields(text, 'descent_values')['hash'], r['factors'],
                             oracle['signature'], oracle['samples'], oracle['selected'])
                if case['id'] in references and references[case['id']] != signature:
                    raise ValueError('Leaf/factor/oracle coverage mismatch')
                references[case['id']] = signature
                wait_stats = fields(text, 'stage2_host_wait_stats') if 'stage2_host_wait_stats:' in text else None
                full = fields(text, 'stage2_full_wall')
                row = dict(name=name, variant=variant, case=case, repeat=repeat, command=cmd,
                           telemetry_before=telemetry, cpu=cpu, full=full, oracle=oracle, multiply=multiply,
                           wait_stats=wait_stats, signature=signature, log=str(log), log_sha256=sha(log),
                           debug=str(debug), debug_sha256=sha(debug), gpu_log=str(gpu_log), gpu_sha256=sha(gpu_log), result=r)
                report['runs'].append(row)
                persist()
                print(name, f'full={full["total"]}s cpu={cpu["logical_cores"]:.3f} cores', flush=True)
    report['complete'] = True
    report['summary'] = [dict(case=c['id'], variant=v,
        full_seconds=statistics.median(float(r['full']['total']) for r in report['runs'] if r['case']['id']==c['id'] and r['variant']==v),
        logical_cores=statistics.median(r['cpu']['logical_cores'] for r in report['runs'] if r['case']['id']==c['id'] and r['variant']==v))
        for c in cases for v in a.variants]
    persist()
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
