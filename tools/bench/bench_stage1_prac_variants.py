#!/usr/bin/env python3
"""Pair PRAC register/DBL policies on the ordinary production sampling path.

The second repeat reverses configuration order. All policies use the same
binary, curve count, container and TPI. Rates are partial-run projections.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

from bench_cuda_prac import sample


CONFIGS = {
    'resident': ('resident', 'baseline', 0),
    'natural': ('prac', 'baseline', 255),
    'cap168': ('prac', 'baseline', 168),
    'compact': ('prac', 'compact', 255),
    'compact168': ('prac', 'compact', 168),
    'outline': ('prac', 'outline-add', 255),
    'outline168': ('prac', 'outline-add', 168),
    'single': ('prac', 'single-add', 255),
    'single168': ('prac', 'single-add', 168),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, default=Path('build_cuda_cmake/prac/ecm_cuda.exe'))
    p.add_argument('--b1', nargs='+', type=int, default=[10_000_000, 260_000_000])
    p.add_argument('--curves', nargs='+', type=int, default=[1536])
    p.add_argument('--configs', nargs='+', choices=list(CONFIGS), default=['resident', 'natural', 'cap168', 'compact', 'compact168'])
    p.add_argument('--target-ms', nargs='+', type=float, default=[100])
    p.add_argument('--seconds', type=float, default=15)
    p.add_argument('--warmup', type=float, default=5)
    p.add_argument('--startup-timeout', type=float, default=600)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--exp-cache', type=Path)
    p.add_argument('--exponent', choices=['lcm', 'choose12'], default='lcm')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 0 <= a.warmup < a.seconds or min(a.curves) < 1 or min(a.b1) < 2 or a.repeats < 1:
        p.error('Positive curves/repeats, B1 >= 2 and 0 <= warmup < seconds required')
    if any(not 10 <= t <= 500 for t in a.target_ms):p.error('Targets must be in 10..500 ms')
    exe = a.exe.resolve(strict=True)
    a.exp_cache = (a.exp_cache or exe.parent).resolve()
    root = a.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, measurement='partial-progress-projection-not-completed-wall-time',
                  exe=str(exe), binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  bits=4423, requested_tpi=16, pairing='same curves and submitted grid', results=[])
    for repeat in range(a.repeats):
        for b1 in a.b1:
            for curves in a.curves:
                configs=[(name,t) for name in a.configs for t in (a.target_ms if CONFIGS[name][0]=='prac' else [100])]
                for name,target_ms in (configs if repeat % 2 == 0 else list(reversed(configs))):
                    algo, variant, registers = CONFIGS[name]
                    options = SimpleNamespace(**vars(a))
                    options.curves = curves; options.tpi = 16
                    options.prac_variant = variant; options.prac_registers = registers
                    options.prac_target_ms = target_ms
                    row = sample(exe, root / f'{name}_b{b1}_c{curves}_target{target_ms}_r{repeat}', 4423, b1, algo, options)
                    geometry = row.get('geometry')
                    if not geometry or geometry['tpi'] != 16 or geometry['container_bits'] != 4608:
                        raise RuntimeError('Expected 4608/TPI16 geometry')
                    row['config'] = name; row['repeat'] = repeat
                    report['results'].append(row)
                    grouped = {}
                    for x in report['results']:
                        grouped.setdefault((x['B1'], x['curves'], x['config'],x['prac_target_ms']), []).append(x['projected_s_per_curve'])
                    report['aggregates'] = [dict(B1=k[0], curves=k[1], config=k[2],target_ms=k[3],repeats=len(v),
                        projected_s_per_curve=statistics.median(v), curves_per_second=1/statistics.median(v),
                        min=min(v), max=max(v)) for k, v in grouped.items()]
                    (root / 'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(root / 'summary.json', flush=True)


if __name__ == '__main__': main()
