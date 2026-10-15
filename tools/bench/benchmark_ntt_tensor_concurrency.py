"""Same-card CUDA/Tensor NTT concurrency, with all outputs checked and frozen identity."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics as st
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fields(text, label):
    return [dict(re.findall(r'(\w+)=([^\s]+)', line)) for line in text.splitlines() if line.startswith(label + ':')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--repeats', type=int, default=9)
    p.add_argument('--execution', choices=['native', 'graph'], default='graph')
    p.add_argument('--skip-gate', action='store_true')
    a = p.parse_args()
    if a.device < 0 or not 1 <= a.repeats <= 101:
        p.error('invalid device/repeats')
    out, exe = a.output.resolve(), a.exe.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((exe.parent / 'manifest.json').read_text(encoding='utf-8-sig'))
    def verify():
        if sha(exe) != manifest['sha256'].lower() or sha(exe.parent / 'gmp-10.dll') != manifest['gmp_dll_sha256'].lower():
            raise ValueError('Binary/DLL changed')
        for name, digest in manifest['sources'].items():
            if sha(ROOT / name) != digest.lower():
                raise ValueError('Compiled source changed: ' + name)
    verify()
    frozen = out / 'build'
    frozen.mkdir()
    for path in (exe, exe.parent / 'gmp-10.dll', exe.parent / 'manifest.json', Path(__file__), a.cases):
        shutil.copy2(path, frozen / path.name)
    for name in manifest['sources']:
        target = frozen / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    cases = json.loads(a.cases.read_text(encoding='utf-8-sig'))
    names = set()
    if not isinstance(cases, list) or not cases:
        raise ValueError('Nonempty cases array required')
    for case in cases:
        if not re.fullmatch(r'[A-Za-z0-9_]+', case.get('name', '')) or case['name'] in names:
            raise ValueError('Unique alphanumeric/underscore case names required')
        names.add(case['name'])
        limits = {'k': (6, 26), 'batch': (1, 4096), 'iterations': (1, 256)}
        for key, (low, high) in limits.items():
            if type(case.get(key)) is not int or not low <= case[key] <= high:
                raise ValueError('Invalid case field: ' + key)
        if case.get('threads', 256) not in (128, 256, 512) or (1 << case['k'])*case['batch'] > 1 << 25:
            raise ValueError('Invalid CTA or working set')
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env['CUDA_LAUNCH_BLOCKING'] = '0'
    query = ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,memory.total,memory.used,power.limit,utilization.gpu', '--format=csv']
    (out / 'devices_before.csv').write_bytes(subprocess.check_output(query))
    identity = dict(exe=str(exe), sha256=sha(exe), manifest=manifest, cases=cases, execution=a.execution,
                    repeats=a.repeats, date=datetime.now(timezone.utc).isoformat(), device=a.device,
                    timing='GPU makespan for all chained jobs including fork/join; fills, reference, validation, '
                           'allocations and capture excluded. No discarded warmups; CUDA reference executes first.')
    (out / 'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    summaries, measurements = [], []
    with (out / 'telemetry.csv').open('wb') as telemetry_log:
        monitor = subprocess.Popen(['nvidia-smi', '-i', str(a.device), '--query-gpu=timestamp,uuid,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used', '--format=csv', '-lms', '500'], stdout=telemetry_log, stderr=subprocess.STDOUT)
        try:
            if not a.skip_gate:
                for flag, want in [('--check', 0), ('--fault', 3)]:
                    cmd = [str(exe), str(a.device), flag]
                    run = subprocess.run(cmd, env=env, capture_output=True, timeout=300)
                    text = (run.stdout + run.stderr).decode('utf-8', errors='replace')
                    (out / (flag[2:] + '.log')).write_text(text, encoding='utf-8')
                    gate = fields(text, 'tc_length_gate')
                    if run.returncode != want or len(gate) != 1 or int(gate[0]['bad']) != (1 if want else 0):
                        raise ValueError('Independent GMP/fault gate failed')
                    if not want and (len(fields(text, 'tc_concurrency_gate')) != 1 or fields(text, 'tc_concurrency_gate')[0]['bad'] != '0'):
                        raise ValueError('Concurrent GMP gate failed')
                    if not re.search(r'tc_concurrency_memory: .*live_bytes=0 fuse_live_bytes=0', text):
                        raise ValueError('Gate allocation leak')
                    print(flag, gate[0], flush=True)
            for case in cases:
                verify()
                cmd = [str(exe), str(a.device), '--bench', str(case['k']), str(case['batch']), '12',
                       str(case.get('threads', 256)), str(a.repeats), str(case['iterations']), str(int(a.execution == 'graph'))]
                run = subprocess.run(cmd, env=env, capture_output=True, timeout=600)
                text = (run.stdout + run.stderr).decode('utf-8', errors='replace')
                (out / (case['name'] + '.log')).write_text(text, encoding='utf-8')
                (out / (case['name'] + '.command.json')).write_text(json.dumps(dict(command=cmd, environment={'CUDA_LAUNCH_BLOCKING': '0'}, returncode=run.returncode), indent=2))
                samples, plans = fields(text, 'tc_concurrency_bench'), fields(text, 'tc_concurrency_plan')
                if run.returncode or len(samples) != 13*a.repeats or len(plans) != 1:
                    raise ValueError('Invalid sample count/exit; evidence retained: ' + case['name'])
                for sample in samples:
                    expected = 2 * (1 << case['k']) * case['batch'] * int(sample['jobs'])
                    if int(sample['bad']) or int(sample['checked_words']) != expected:
                        raise ValueError('Arithmetic check failed: ' + case['name'])
                if not re.search(r'tc_concurrency_memory: .*live_bytes=0 fuse_live_bytes=0', text):
                    raise ValueError('Benchmark allocation leak')
                rows = {}
                for scene in range(13):
                    selected = [r for r in samples if int(r['scene']) == scene]
                    if len(selected) != a.repeats:
                        raise ValueError('Unequal scene counts')
                    times = [float(r['seconds']) for r in selected]
                    median = st.median(times)
                    row = dict(case=case['name'], k=case['k'], batch=case['batch'], threads=case.get('threads', 256),
                               iterations=case['iterations'], execution=a.execution, scene=scene, name=selected[0]['name'],
                               jobs=int(selected[0]['jobs']), median_seconds=median, mad_seconds=st.median(abs(v-median) for v in times),
                               host_median_seconds=st.median(float(r['host_seconds']) for r in selected), repeats=len(selected),
                               convolutions_per_s=int(selected[0]['jobs'])*case['batch']*case['iterations']/median,
                               checked_words=sum(int(r['checked_words']) for r in selected), bad=0)
                    rows[row['name']] = row
                    summaries.append(row)
                def sec(name):
                    return rows[name]['median_seconds']
                best_c = min(sec('CC_serial'), sec('CC_parallel'))
                comparison = dict(case=case['name'], narrow_gain_vs_t40=100*(1-sec('single_N40')/sec('single_T40')),
                    narrow_gain_vs_cuda=100*(1-sec('single_N40')/sec('single_C')),
                    ct40_parallel_speedup=sec('CT40_serial')/sec('CT40_parallel'),
                    cn40_parallel_speedup=sec('CN40_serial')/sec('CN40_parallel'),
                    ct40_gain_vs_best_cuda=100*(1-sec('CT40_parallel')/best_c),
                    cn40_gain_vs_best_cuda=100*(1-sec('CN40_parallel')/best_c))
                measurements.append(dict(case=case, plan=plans[0], raw=samples, comparison=comparison,
                                         resources=fields(text, 'tc_gold_tile_resources')))
                (out / 'measurements.json').write_text(json.dumps(dict(complete=False, identity=identity, runs=measurements), indent=2))
                with (out / 'summary.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
                    writer.writeheader();writer.writerows(summaries)
                print(json.dumps(comparison), flush=True)
        finally:
            monitor.terminate();monitor.wait(timeout=10)
    verify()
    (out / 'measurements.json').write_text(json.dumps(dict(complete=True, identity=identity, runs=measurements), indent=2))


if __name__ == '__main__':
    main()
