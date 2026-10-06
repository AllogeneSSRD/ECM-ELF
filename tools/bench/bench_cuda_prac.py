#!/usr/bin/env python3
"""Sample production-B1 Stage1 s/curve without completing whole batches.

Each run has an independent working directory. A killed ladder sample and a
checkpoint-only resident/PRAC sample are never treated as completed Stage1.
Reports are projections from a partial run, not final wall seconds per curve.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import hashlib
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

PROGRESS = re.compile(r'GPU: \[[^\]]+\].*?([\d.]+)%.*?~([\d.]+) s/curve.*?elapsed ([\d.]+)s')
PRECISE = re.compile(r'GPU: (PRAC|resident) slice next=(\d+)/(\d+).*?projected=([\d.]+) s/curve')
GEOMETRY = re.compile(r'CGBN<(\d+),(\d+)>, curves=(\d+), blocks=(\d+), blocks/SM=(\d+)')
SLICE_COST = re.compile(r'length=(\d+), slice-ms=([\d.]+)')


def sample(exe: Path, folder: Path, bits: int, b1: int, algo: str, args) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'n.txt').write_text(f'(2^{bits}-1)\n', encoding='ascii')
    env = dict(os.environ, ECM_GPU_STAGE1_ALGO=algo)
    env['ECM_PRAC_REG_TARGET'] = str(args.prac_registers)
    env['ECM_STAGE1_TPI'] = str(args.tpi)
    variant = getattr(args, 'prac_variant', 'baseline') if algo == 'prac' else 'baseline'
    env['ECM_PRAC_VARIANT'] = variant
    env['ECM_PRAC_PLAN_CACHE'] = str(args.exp_cache)
    target_ms = getattr(args, 'prac_target_ms', 100)
    env['ECM_PRAC_TARGET_MS'] = str(target_ms)
    for key in tuple(env):
        if key.startswith('ECM_PRAC_WINDOW'): env.pop(key)
    env.pop('ECM_GPU_DUMP', None)
    env['ECM_GPU_STAGE1_SAMPLE_SECONDS'] = str(args.seconds) if algo != 'ladder' else '0'
    command = [str(exe), '-gpu', '-d', str(args.device), '--gpu-param', '0',
               '-sigma', '0:26', '-gpucurves', str(args.curves), '--ckpt', '0', '-v',
               '--exponent', args.exponent, '--exp-cache', str(args.exp_cache),
               '-savea', 'completed.save', str(b1), '0']
    log = folder / 'run.log'
    launched = time.monotonic()
    samples = []; precise = []; slice_costs = []; offset = 0; pending = ''; first = None; killed = False
    print(f'START algo={algo} variant={variant} reg={args.prac_registers} n={bits} B1={b1} C={args.curves} TPI={args.tpi} device={args.device}', flush=True)
    with (folder / 'n.txt').open('rb') as source, log.open('wb') as out:
        process = subprocess.Popen(command, cwd=folder, env=env, stdin=source,
                                   stdout=out, stderr=subprocess.STDOUT)
        try:
            while True:
                time.sleep(0.2)
                with log.open('rb') as incoming:
                    incoming.seek(offset); chunk = incoming.read(); offset += len(chunk)
                pending += chunk.decode('utf-8', errors='replace').replace('\r', '\n')
                lines = pending.split('\n'); pending = lines.pop()
                for line in lines:
                    match = PROGRESS.search(line)
                    if match:
                        if first is None: first = time.monotonic()
                        pct, rate, elapsed = map(float, match.groups())
                        if elapsed >= args.warmup: samples.append(dict(elapsed=elapsed, pct=pct, s_per_curve=rate))
                    match = PRECISE.search(line)
                    if match and samples:
                        precise.append(dict(next=int(match[2]), total=int(match[3]),
                                            s_per_curve=float(match[4]), elapsed=samples[-1]['elapsed']))
                    match = SLICE_COST.search(line)
                    if match and samples:
                        slice_costs.append(dict(length=int(match[1]), milliseconds=float(match[2]), elapsed=samples[-1]['elapsed']))
                finished = process.poll() is not None
                if finished: break
                now = time.monotonic()
                if first is None and now - launched > args.startup_timeout:
                    raise TimeoutError('No GPU progress before startup timeout')
                if first is not None and now - first > args.seconds + 2:
                    killed = True; process.terminate(); process.wait(timeout=15)
                    break
        finally:
            if process.poll() is None: process.kill(); process.wait()
    chosen = precise if precise else samples
    if not chosen: raise RuntimeError(f'No post-warmup rate samples: {log}')
    # Keep the last five seconds to exclude adaptive initial slice sizes.
    tail = [r for r in chosen if r['elapsed'] >= chosen[-1]['elapsed'] - 5]
    median = statistics.median(r['s_per_curve'] for r in tail)
    completed_save = folder / 'completed.save'
    completed_records = 0
    if completed_save.exists():
        completed_records = sum('SIGMA=' in line and 'X=' in line
                                for line in completed_save.read_text(encoding='utf-8').splitlines())
    result = dict(algorithm=algo, bits=bits, B1=b1, curves=args.curves, device=args.device,
                  requested_tpi=args.tpi,
                  exponent=args.exponent, prac_registers=args.prac_registers, prac_variant=variant,
                  prac_target_ms=target_ms if algo=='prac' else None,
                  projected_s_per_curve=median,
                  rate_min=min(r['s_per_curve'] for r in tail), rate_max=max(r['s_per_curve'] for r in tail),
                  partial_run=True, sample_count=len(tail), elapsed=chosen[-1]['elapsed'],
                  progress_percent=samples[-1]['pct'] if samples else None,
                  resolution=0.000001 if precise else 0.01, samples=chosen,
                  log=str(log), command=command, exit_code=process.returncode, terminated=killed,
                  final_save_present=completed_records > 0, final_save_records=completed_records)
    text = log.read_text(encoding='utf-8', errors='replace')
    if algo == 'prac' and f'PRAC slice target={target_ms:.3f} ms' not in text:
        raise RuntimeError(f'Binary did not select requested PRAC slice target: {log}')
    if algo == 'prac' and f'PRAC variant={variant}' not in text:
        raise RuntimeError(f'Binary did not select requested PRAC variant: {log}')
    result['slice_costs'] = slice_costs
    geometry = GEOMETRY.search(text)
    if geometry:
        result['geometry'] = dict(zip(('tpi', 'container_bits', 'curves', 'grid_blocks', 'resident_blocks_per_sm'),
                                      map(int, geometry.groups())))
        if args.tpi and int(geometry[1]) != args.tpi: raise RuntimeError('Requested TPI was not selected')
    result['exponent_cache_hit'] = 'stage1 exponent built: cache hit' in text
    result['prac_cache_hit'] = 'PRAC plan cache hit' in text
    result['curves_per_second'] = 1 / median
    if completed_records >= args.curves:
        result['partial_run'] = False  # Tiny runs can finish before the sample limit.
    print(f'RESULT algo={algo} n={bits} B1={b1} projected={median:.6f} s/curve', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=Path('build_cuda_cmake/prac/ecm_cuda.exe'))
    parser.add_argument('--bits', nargs='+', type=int, default=[2203, 4423, 8191])
    parser.add_argument('--b1', nargs='+', type=int, default=[10_000_000])
    parser.add_argument('--algorithms', nargs='+', choices=['ladder', 'resident', 'prac'], default=['ladder', 'resident', 'prac'])
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--warmup', type=float, default=5)
    parser.add_argument('--startup-timeout', type=float, default=600)
    parser.add_argument('--curves', type=int, default=1536)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--prac-registers', type=int, choices=[0, 168, 255], default=0)
    parser.add_argument('--prac-variant', choices=['baseline', 'compact', 'outline-add'], default='baseline')
    parser.add_argument('--prac-target-ms', type=float, default=100)
    parser.add_argument('--tpi', type=int, choices=[0, 16, 32], default=0)
    parser.add_argument('--exp-cache', type=Path, help='Shared validated B1/PRAC cache; defaults to exe directory')
    parser.add_argument('--exponent', choices=['lcm', 'choose12'], default='lcm')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.seconds or args.curves < 1 or args.repeats < 1:
        parser.error('Expected 0 <= warmup < seconds and positive curves/repeats')
    if not 10 <= args.prac_target_ms <= 500:parser.error('PRAC target must be in 10..500 ms')
    if args.tpi and 'ladder' in args.algorithms:
        parser.error('TPI overrides require --algorithms resident prac (the original ladder is unchanged)')
    if args.prac_variant != 'baseline' and (args.bits != [4423] or args.tpi == 32 or args.prac_registers == 0):
        parser.error('Experimental point variants require --bits 4423, TPI16/default and registers 168/255')
    exe = args.exe.resolve(strict=True)
    args.exp_cache = (args.exp_cache or exe.parent).resolve()
    root = (args.output or Path('docs/data') / ('prac_cuda_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = dict(schema=1, measurement='partial-progress-projection-not-completed-wall-time',
                  exe=str(exe), binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), results=[])
    for repeat in range(args.repeats):
        for b1 in args.b1:
            for bits in args.bits:
                for algo in args.algorithms:
                    folder = root / f'{algo}_n{bits}_b{b1}_r{repeat}'
                    report['results'].append(sample(exe, folder, bits, b1, algo, args))
                    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(root / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
