#!/usr/bin/env python3
"""Measure PRAC prime windows from reset points, never complete Stage1 saves."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from bench_cuda_prac import GEOMETRY

INFO = re.compile(r'PRAC_WINDOW position=(\w+) first=(\d+) count=(\d+) p_first=(\d+) p_last=(\d+) work=(\d+) full_work=(\d+) warmup=(\d+)')
DONE = re.compile(r'PRAC_WINDOW_DONE rounds=(\d+) measured=(\d+) kernel_ms=([\d.]+) work=(\d+) projected=([\d.]+) s/curve wall=([\d.]+) seed_bytes=(\d+) restore_bytes=(\d+)')
SLICING = re.compile(r'PRAC_WINDOW_SLICING chunk=(\d+) launches_per_round=(\d+)')
COST = re.compile(r'PRAC_WINDOW_COST measured_wall_ms=([\d.]+) measured_launches=(\d+) boundary_logical_bytes_per_round=(\d+)')


def sample(exe, folder, bits, b1, curves, device, tpi, registers, variant,
           position, count, seconds, warmup, cache, dump=False, sigma=26, exponent='lcm', chunk=0):
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith('ECM_PRAC_WINDOW'): env.pop(key)
    env.update(ECM_GPU_STAGE1_ALGO='prac', ECM_PRAC_REG_TARGET=str(registers),
        ECM_PRAC_VARIANT=variant, ECM_STAGE1_TPI=str(tpi),
        ECM_PRAC_WINDOW=position, ECM_PRAC_WINDOW_COUNT=str(count),
        ECM_PRAC_WINDOW_CHUNK=str(chunk),
        ECM_PRAC_WINDOW_WARMUP=str(warmup), ECM_PRAC_WINDOW_DUMP='1' if dump else '0',
        ECM_GPU_STAGE1_SAMPLE_SECONDS=str(seconds), ECM_PRAC_PLAN_CACHE=str(cache))
    env.pop('ECM_GPU_DUMP', None)
    env.pop('ECM_PRAC_TARGET_MS', None)  # Fixed window chunks are independent of production feedback.
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('.ecm_ckpt_*')}
    command = [str(exe), '-gpu', '-d', str(device), '--gpu-param', '0', '-sigma', f'0:{sigma}',
        '-gpucurves', str(curves), '--ckpt', '0', '--exponent', exponent,
        '--exp-cache', str(cache), '-v', '-savea', 'completed.save', str(b1), '0']
    print(f'START window={position} chunk={chunk or count} n={bits} B1={b1} C={curves} reg={registers} variant={variant}', flush=True)
    start = time.monotonic()
    proc = subprocess.run(command, input=f'(2^{bits}-1)\n'.encode('ascii'), env=env, cwd=folder,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=seconds + 180)
    text = proc.stdout.decode('utf-8', errors='replace')
    (folder / 'run.log').write_text(text, encoding='utf-8')
    info, done, geometry = INFO.search(text), DONE.search(text), GEOMETRY.search(text)
    if not info or not done or not geometry or proc.returncode != 1:
        raise RuntimeError(f'Incomplete window measurement: {folder}\n{text[-2000:]}')
    if f'PRAC variant={variant}' not in text:
        raise RuntimeError('Requested window variant was not selected')
    if 'checkpoint resumed' in text or 'Stage1 PRAC completed' in text:
        raise RuntimeError('Window entered a production checkpoint/completion path')
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('.ecm_ckpt_*')}
    if before != after: raise RuntimeError('Window modified production checkpoints')
    if any('SIGMA=' in p.read_text(encoding='utf-8') for p in folder.glob('*.save')):
        raise RuntimeError('Window published a final Stage1 save')
    rounds, measured, ms, work, projected, wall, seed, restore = done.groups()
    slicing = SLICING.search(text)
    if chunk and not slicing: raise RuntimeError('Binary does not support window slicing')
    actual_chunk = int(slicing[1]) if slicing else int(info[3])
    launches_per_round = int(slicing[2]) if slicing else 1
    if actual_chunk != (chunk or int(info[3])) or launches_per_round != (int(info[3])+actual_chunk-1)//actual_chunk:
        raise RuntimeError('Wrong window slicing geometry')
    cost = COST.search(text)
    if chunk and not cost: raise RuntimeError('Missing sliced window wall-time accounting')
    if cost and int(cost[2]) != launches_per_round*int(measured):raise RuntimeError('Wrong measured launch count')
    if int(work) != int(info[6]) or int(measured) < 3:
        raise RuntimeError('Window work/round count mismatch')
    selected = dict(zip(('tpi', 'container_bits', 'curves', 'grid_blocks', 'resident_blocks_per_sm'),
                        map(int, geometry.groups())))
    if (tpi and selected['tpi'] != tpi) or selected['curves'] != curves:
        raise RuntimeError('Wrong window geometry')
    result = dict(bits=bits, B1=b1, curves=curves, device=device, requested_tpi=tpi,
        registers=registers, variant=variant, position=position, requested_count=count,
        first=int(info[2]), count=int(info[3]), p_first=int(info[4]), p_last=int(info[5]),
        requested_chunk=chunk, chunk=actual_chunk, launches_per_round=launches_per_round,
        measured_launches=launches_per_round*int(measured), total_launches=launches_per_round*int(rounds),
        work=int(work), full_work=int(info[7]), warmup=int(info[8]), rounds=int(rounds), measured=int(measured),
        kernel_ms=float(ms), window_wall_seconds=float(wall), process_wall_seconds=time.monotonic()-start,
        projected_s_per_curve=float(projected), projected_curves_per_second=1/float(projected),
        m_equiv_per_curve_per_second=int(work)*int(measured)*1000/float(ms),
        m_equiv_aggregate_per_second=int(work)*int(measured)*curves*1000/float(ms),
        seed_bytes=int(seed), restore_bytes=int(restore), geometry=selected,
        exponent=exponent, sigma=sigma, exponent_cache_hit='stage1 exponent built: cache hit' in text,
        prac_cache_hit='PRAC plan cache hit' in text, exit_code=proc.returncode,
        checkpoints_unchanged=True, final_save_records=0, diagnostic_dump=dump,
        measurement='reset-point subproduct windows; projections, not complete curves',
        command=command, log=str(folder/'run.log'))
    if cost:
        result.update(measured_wall_ms=float(cost[1]), boundary_logical_bytes_per_round=int(cost[3]),
            wall_projected_s_per_curve=int(info[7])*float(cost[1])/(int(work)*int(measured))/1000/curves)
    print(f"RESULT projected={float(projected):.6f} s/curve p={info[4]}..{info[5]}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, default=Path('build_cuda_cmake/prac/ecm_cuda.exe'))
    p.add_argument('--bits', type=int, nargs='+', default=[4423])
    p.add_argument('--b1', type=int, nargs='+', default=[10000000, 260000000])
    p.add_argument('--curves', type=int, default=1536)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--tpi', type=int, choices=[0, 16, 32], default=0)
    p.add_argument('--registers', type=int, nargs='+', choices=[0, 168, 255], default=[255, 168])
    p.add_argument('--variants', nargs='+', choices=['baseline', 'compact', 'outline-add', 'single-add', 'single-compact'], default=['baseline', 'compact'])
    p.add_argument('--windows', nargs='+', choices=['prefix', 'middle', 'tail'], default=['prefix', 'middle', 'tail'])
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--chunks', type=int, nargs='+', default=[0], help='Per-launch counts; 0 means one launch per window')
    p.add_argument('--seconds', type=float, default=6)
    p.add_argument('--warmup', type=int, default=2, help='Excluded reset rounds, not seconds')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--exp-cache', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); exe = a.exe.resolve(strict=True)
    if not 1 <= a.count <= 32 or not 0 <= a.warmup <= 32 or not 0 < a.seconds <= 600 or a.curves < 1 or a.repeats < 1:
        p.error('Invalid count, warmup, seconds, curves or repeats')
    if any(not 0 <= c <= a.count for c in a.chunks):p.error('Chunks must be in 0..count')
    if (any(v != 'baseline' for v in a.variants) or 168 in a.registers) and a.bits != [4423]:
        p.error('Experimental point variants/168 require --bits 4423')
    if a.tpi == 32 and (any(v not in ('baseline','single-compact') for v in a.variants) or
                        (168 in a.registers and 'baseline' in a.variants)):
        p.error('TPI32 point candidates/168 support only single-compact; baseline requires registers 0/255')
    if any(v != 'baseline' for v in a.variants) and 0 in a.registers:
        p.error('Experimental point variants require registers 168/255')
    cache = (a.exp_cache or exe.parent).resolve(); root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, exe=str(exe), binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  measurement='reset-point subproduct windows; projections, not complete curves', results=[])
    configs = [(v,r,c) for v in a.variants for r in a.registers for c in a.chunks]
    for repeat in range(a.repeats):
        for n in a.bits:
            for b in a.b1:
                for position in (a.windows if repeat % 2 == 0 else list(reversed(a.windows))):
                    for variant, registers, chunk in (configs if repeat % 2 == 0 else list(reversed(configs))):
                        folder = root/f'n{n}_b{b}_{position}_{variant}_reg{registers}_chunk{chunk}_r{repeat}'
                        row = sample(exe, folder, n, b, a.curves, a.device, a.tpi, registers,
                            variant, position, a.count, a.seconds, a.warmup, cache,chunk=chunk)
                        report['results'].append(row)
                        grouped = {}
                        for x in report['results']:
                            key = (x['bits'],x['B1'],x['position'],x['variant'],x['registers'],x['chunk'])
                            grouped.setdefault(key,[]).append(x['projected_s_per_curve'])
                        report['aggregates'] = [dict(bits=k[0],B1=k[1],position=k[2],variant=k[3],registers=k[4],chunk=k[5],
                            repeats=len(v),projected_s_per_curve=statistics.median(v),min=min(v),max=max(v))
                            for k,v in grouped.items()]
                        (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(root/'summary.json',flush=True)


if __name__ == '__main__': main()
