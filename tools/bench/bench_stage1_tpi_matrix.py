#!/usr/bin/env python3
"""Compare equal submitted block grids: TPI32 uses half of TPI16's curves."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from bench_cuda_prac import sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, default=Path('build_cuda_cmake/prac/ecm_cuda.exe'))
    p.add_argument('--bits', nargs='+', type=int, default=[2203, 4423])
    p.add_argument('--b1', nargs='+', type=int, default=[10000000])
    p.add_argument('--curves16', nargs='+', type=int, default=[384, 768, 1536])
    p.add_argument('--tpis', nargs='+', type=int, choices=[16, 32], default=[16, 32])
    p.add_argument('--algorithms', nargs='+', choices=['resident', 'prac'], default=['resident', 'prac'])
    p.add_argument('--prac-registers', type=int, choices=[0, 168, 255], default=255)
    p.add_argument('--prac-variant', choices=['baseline', 'compact'], default='baseline')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--seconds', type=float, default=12)
    p.add_argument('--warmup', type=float, default=4)
    p.add_argument('--startup-timeout', type=float, default=600)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--exp-cache', type=Path)
    p.add_argument('--exponent', choices=['lcm', 'choose12'], default='lcm')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.prac_variant == 'compact' and (a.bits != [4423] or a.tpis != [16] or a.prac_registers == 0):
        p.error('compact requires --bits 4423 --tpis 16 and registers 168/255')
    if any(c < 2 or c % 2 for c in a.curves16) or not 0 <= a.warmup < a.seconds or a.repeats < 1:
        p.error('Positive even curve counts, repetitions, and 0 <= warmup < seconds required')
    exe = a.exe.resolve(strict=True); a.exp_cache = (a.exp_cache or exe.parent).resolve()
    root = a.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, measurement='partial-progress-projection-not-completed-wall-time',
                  exe=str(exe), binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  pairing='same submitted grid blocks; resource-limited residency may differ', results=[])
    for r in range(a.repeats):
        for b1 in a.b1:
            for n in a.bits:
                for base in a.curves16:
                    # Reverse order on the second repeat to reduce warmup/order bias.
                    for tpi in (a.tpis if r % 2 == 0 else list(reversed(a.tpis))):
                        a.tpi = tpi; a.curves = base if tpi == 16 else base // 2
                        for algo in (a.algorithms if r % 2 == 0 else list(reversed(a.algorithms))):
                            folder = root / f'{algo}_n{n}_b{b1}_t{tpi}_c{a.curves}_r{r}'
                            row = sample(exe, folder, n, b1, algo, a)
                            row['curves16_pair'] = base; report['results'].append(row)
                            grouped = {}
                            for x in report['results']:
                                key = (x['bits'], x['B1'], x['algorithm'], x['requested_tpi'], x['curves'])
                                grouped.setdefault(key, []).append(x['projected_s_per_curve'])
                            report['aggregates'] = [dict(bits=k[0], B1=k[1], algorithm=k[2], tpi=k[3], curves=k[4], prac_variant=a.prac_variant,
                                repeats=len(v), projected_s_per_curve=statistics.median(v),
                                curves_per_second=1/statistics.median(v), min=min(v), max=max(v)) for k,v in grouped.items()]
                            (root / 'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(root / 'summary.json', flush=True)


if __name__ == '__main__': main()
