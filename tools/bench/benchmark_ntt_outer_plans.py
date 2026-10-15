"""Gate and interleave production-runtime outer plans; retain all raw evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fields(text, prefix):
    return [dict(re.findall(r'(\w+)=(\S+)', line)) for line in
            re.findall(r'^' + re.escape(prefix) + r': (.*)$', text, re.M)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--lengths', type=int, nargs='+', default=[20, 21, 22, 23, 24])
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--operations', type=int, nargs='+', default=[2])
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=8)
    args = parser.parse_args()
    if not all(16 <= k <= 25 for k in args.lengths) or not all(1 <= b <= 8 for b in args.batches):
        parser.error('lengths 16..25 and batches 1..8 required')
    if not all(op in (0, 1, 2) for op in args.operations) or not 1 <= args.repeats <= 101 or not 1 <= args.iterations <= 128:
        parser.error('invalid operation/repeat/iteration')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    exe = args.exe.resolve()
    manifest = json.loads((exe.parent / 'manifest.json').read_text(encoding='utf-8-sig'))
    if sha(exe) != manifest['sha256'].lower():
        raise ValueError('binary does not match build manifest')
    for name, want in manifest['sources'].items():
        if sha(ROOT / name) != want.lower():
            raise ValueError('build source mismatch: ' + name)
        destination = out / 'sources' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env['CUDA_LAUNCH_BLOCKING'] = '0'
    report = dict(complete=False, binary_sha256=sha(exe), build=manifest, tool_sha256=sha(__file__),
                  device=args.device, timing='device events; setup, fill and check excluded; no discarded warmup',
                  runs=[], statistics=[], checks={})

    def save():
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    def run(name, arguments, expected=0):
        if sha(exe) != report['binary_sha256']:
            raise ValueError('binary changed')
        command = [str(exe), str(args.device), *map(str, arguments)]
        with (out / (name + '.log')).open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=600, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        report['runs'].append(dict(name=name, command=command, returncode=result.returncode,
                                   log_sha256=sha(out / (name + '.log'))))
        save()
        if result.returncode != expected:
            raise RuntimeError(f'{name}: exit {result.returncode}, expected {expected}')
        return (out / (name + '.log')).read_text(encoding='utf-8')

    telemetry_file = (out / 'gpu.csv').open('w', encoding='utf-8')
    telemetry = subprocess.Popen(['nvidia-smi', '-i', str(args.device),
        '--query-gpu=timestamp,index,uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu',
        '--format=csv', '-lms', '500'], stdout=telemetry_file, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        gate = run('gate', ['--check'])
        if 'ntt_fuse_coop_check: cases=96 words=27131904 bad=0' not in gate or 'ntt_fuse_coop_switch_check: calls=4 words=3145728 bad=0' not in gate:
            raise ValueError('independent GMP/lifecycle coverage missing')
        fault = run('fault', ['--fault'], 3)
        if 'ntt_fuse_coop_check: cases=96 words=27131904 bad=1 ' not in fault:
            raise ValueError('fault not detected')
        report['checks'].update(gmp=True, fault=True)
        for k in args.lengths:
            for batch in args.batches:
                for op in args.operations:
                    name = f'k{k}_b{batch}_op{op}'
                    text = run(name, ['--bench', k, batch, args.repeats, args.iterations, op])
                    rows = fields(text, 'outer_plan_sample')
                    plans = fields(text, 'outer_plan')[1:]  # first is the dense reference
                    if len(rows) != 5 * args.repeats or len(plans) != 5 or 'outer_plan_done: live_bytes=0' not in text:
                        raise ValueError('incomplete scan or live plan leak')
                    for row in rows:
                        if int(row['bad']) or int(row['checked_words']) != 2 * (1 << k) * batch:
                            raise ValueError('full-output check failed')
                    item = dict(k=k, batch=batch, operation=op, plans=plans, samples=rows, modes={})
                    for mode in range(5):
                        selected = [r for r in rows if int(r['mode']) == mode]
                        if len(selected) != args.repeats:
                            raise ValueError('unbalanced mode count')
                        times = [float(r['seconds']) for r in selected]
                        median = statistics.median(times)
                        item['modes'][str(mode)] = dict(median_seconds=median,
                            mad_seconds=statistics.median(abs(t-median) for t in times),
                            min_seconds=min(times), max_seconds=max(times))
                    baseline = item['modes']['0']['median_seconds']
                    for data in item['modes'].values():
                        data['gain_percent'] = 100 * (1-data['median_seconds']/baseline)
                    report['statistics'].append(item)
                    save()
                    print(name + ': ' + ' '.join(f"M{mode+4}={item['modes'][str(mode)]['gain_percent']:+.2f}%"
                          for mode in range(1, 5)), flush=True)
        report['complete'] = True
        save()
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=10)
        telemetry_file.close()


if __name__ == '__main__':
    main()
