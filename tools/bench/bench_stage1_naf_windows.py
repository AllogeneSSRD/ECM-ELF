"""Measure cold-process Edwards Stage1 phases across NAF windows on Windows."""

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

import psutil

from bench_edwards_gpu_stage1 import reference, validate


SIGMA = 20260922
CASES = {1279: (12000, 8), 4253: (6000, 16), 8191: (2000, 16)}
BACKENDS = {
    'cpu-ifma': (None, 'simd', 8),
    'cpu-scalar': (None, 'scalar', 1),
    'rtx4060': (1, None, 24),
    'rtx4070ti': (0, None, 60),
}
GPU_FIELDS = 'timestamp,index,name,driver_version,clocks.sm,power.draw,power.limit,utilization.gpu'


def gpu_info(device=None):
    selector = ['-i', str(device)] if device is not None else []
    return subprocess.check_output(['nvidia-smi', *selector, f'--query-gpu={GPU_FIELDS}',
                                    '--format=csv,noheader,nounits'], text=True, timeout=5)


def monitor_gpu(path, device, stop):
    with path.open('w') as log:
        log.write(GPU_FIELDS + '\n')
        while not stop.is_set():
            try:
                log.write(gpu_info(device))
            except (OSError, subprocess.SubprocessError) as error:
                log.write(f'# {error}\n')
            log.flush()
            stop.wait(0.25)


def marker(lines, pattern):
    matches = [(seconds, match) for seconds, line in lines if (match := re.search(pattern, line))]
    if len(matches) != 1:
        raise ValueError(f'expected one marker for {pattern!r}, found {len(matches)}')
    return matches[0]


def write_report(path, report):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(report, indent=2) + '\n')
    pending.replace(path)


def run(exe, folder, backend, bits, w, repeat, logical_cpu):
    device, impl, batch_or_sms = BACKENDS[backend]
    b1, tpi = CASES[bits]
    curves = batch_or_sms if impl else 2 * batch_or_sms * (128 // tpi)
    folder.mkdir()
    config = folder / 'ecm.toml'
    config.write_text('config_version = 3\n')
    command = [str(exe), '--config', str(config), '--param', 'e', '-sigma', str(SIGMA),
               '-gpucurves', str(curves), '--naf-w', str(w), '--ckpt', '0', '--exp-cache', 'off',
               '--tmp-dir', str(folder), '-save', 'completed.save', '-v', str(b1), '0']
    if impl:
        command += ['--target', 'cpu', '--impl', impl, '--stage1-threads', '1',
                    '--affinity', str(logical_cpu)]
    else:
        command += ['--target', 'gpu', '--gpu-backend', 'cuda', '--gpu-algorithm', 'cgbn',
                    '-d', str(device)]
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ECM_', 'NTT_', 'ED_SOA_'))}
    env['ECM_EDWARDS_NAF_TIMING'] = '1'
    if device is not None:
        env['ECM_EDWARDS_DBL_ALGO'] = '4'
    lines = []
    with (folder / 'run.log').open('wb') as raw, (folder / 'line_times.tsv').open('w') as timings:
        t0 = time.perf_counter()
        with subprocess.Popen(command, cwd=folder, env=env, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as child:
            child.stdin.write(f'(2^{bits}-1)\n'.encode())
            child.stdin.close()
            for line in child.stdout:
                seconds = time.perf_counter() - t0
                raw.write(line)
                decoded = line.decode(errors='replace').rstrip('\r\n')
                timings.write(f'{seconds:.9f}\t{decoded}\n')
                lines.append((seconds, decoded))
            returncode = child.wait()
            wall = time.perf_counter() - t0
    if returncode:
        raise RuntimeError(f'{folder}: exit {returncode}')
    startup, banner = marker(lines, r'ECM-ELF (\d+\.\d+\.\d+) \(Stage1\)')
    row = dict(backend=backend, bits=bits, b1=b1, curves=curves, w=w, repeat=repeat,
               version=banner[1], startup_s=startup, wall_s=wall, log=folder.name)
    if impl:
        end, measure = marker(lines, r'\[naf-window\] cpu_loop_seconds=([\d.]+)')
        loop = float(measure[1])
        start = end - loop
        if impl == 'simd':
            marker(lines, rf'CPU implementation : simd .*dict w={w}\)')
    else:
        marker(lines, rf'GPU Edwards Stage1:.*window={w} doubling=4')
        start, setup = marker(lines, r'GPU Edwards dictionary: ([\d.]+) s')
        end, _ = marker(lines, r'GPU Edwards arithmetic: algorithm=naf')
        loop = end - start
        _, metrics = marker(lines, r'GPU Edwards NAF: columns=(\d+) additions=(\d+) '
                            r'container_bits=(\d+) tpi=(\d+) dictionary_bytes=(\d+) '
                            r'dictionary_gpu_ms=([\d.]+) loop_gpu_ms=([\d.]+)')
        columns, adds, container_bits, actual_tpi, dictionary_bytes = map(int, metrics.groups()[:5])
        if actual_tpi != tpi or dictionary_bytes != curves * (container_bits // 8) * 4 * 2**(w - 2):
            raise ValueError(f'{folder}: unexpected CGBN geometry or dictionary allocation')
        row.update(columns=columns, additions=adds, container_bits=container_bits, tpi=actual_tpi,
                   dictionary_bytes=dictionary_bytes, dictionary_wall_s=float(setup[1]),
                   dictionary_gpu_s=float(metrics[6]) / 1000, loop_gpu_s=float(metrics[7]) / 1000)
    if not startup < start < end < wall:
        raise ValueError(f'{folder}: inconsistent phase boundaries {(startup, start, end, wall)}')
    row.update(init_s=start-startup, loop_s=loop, tail_s=wall-end)
    records = [json.loads(line) for line in (folder / 'stage1_results.jsonl').read_text().splitlines()]
    batch_seconds = {record['stage1-engine']['batch-seconds'] for record in records}
    builds = {record['source-build'] for record in records}
    if len(records) != curves or len(batch_seconds) != 1 or len(builds) != 1:
        raise ValueError(f'{folder}: invalid internal result count or batch duration')
    row['batch_s'] = batch_seconds.pop()
    row['source_build'] = builds.pop()
    modulus = (1 << bits) - 1
    points = validate(folder / 'completed.save', modulus, b1, curves, SIGMA, 'ed')
    return row, points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backends', choices=BACKENDS, nargs='+', default=list(BACKENDS))
    parser.add_argument('--bits', type=int, choices=CASES, nargs='+', default=list(CASES))
    parser.add_argument('--windows', type=int, choices=range(3, 13), nargs='+', default=[4, 6, 8, 10, 12])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--logical-cpu', type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or any(len(set(values)) != len(values)
                               for values in (args.backends, args.bits, args.windows)):
        parser.error('use positive repeats and unique backends, widths and windows')
    exe = args.exe.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True)
    if any(BACKENDS[name][0] is not None for name in args.backends):
        (out / 'gpu_inventory.csv').write_text(GPU_FIELDS + '\n' + gpu_info())
    report = dict(complete=False, logical_cpu=args.logical_cpu, sigma=SIGMA, samples=[])
    report_file = out / 'report.json'
    endpoints = {}
    checked = set()
    process = psutil.Process()
    original_affinity = process.cpu_affinity()
    try:
        for backend in args.backends:
            device = BACKENDS[backend][0]
            process.cpu_affinity([args.logical_cpu] if device is None else original_affinity)
            stop = threading.Event()
            monitor = None
            if device is not None:
                monitor = threading.Thread(target=monitor_gpu, args=(out / f'{backend}.csv', device, stop))
                monitor.start()
            try:
                for bits in args.bits:
                    baseline = 12 if device is None else 6
                    for repeat in range(-1, args.repeats):
                        offset = repeat % len(args.windows)
                        windows = args.windows[offset:] + args.windows[:offset]
                        if repeat == -1:
                            windows = [baseline]
                        elif repeat % 2:
                            windows.reverse()
                        for w in windows:
                            name = f'{backend}_m{bits}_w{w}_r{repeat}'
                            print('START', name, flush=True)
                            row, points = run(exe, out / name, backend, bits, w, repeat, args.logical_cpu)
                            for identity in ('version', 'source_build'):
                                if report.setdefault(identity, row[identity]) != row[identity]:
                                    raise ValueError(f'{name}: executable identity changed during collection')
                            for sigma, status, x in points:
                                key = (bits, sigma)
                                if endpoints.setdefault(key, (status, x)) != (status, x):
                                    raise ValueError(f'{name}: saved point mismatch at {sigma}')
                            for sigma, status, x in (points[0], points[-1]):
                                key = (bits, sigma)
                                if key not in checked:
                                    expected = reference((1 << bits) - 1, sigma, row['b1'], 'ed')
                                    if expected != (status, x):
                                        raise ValueError(f'{name}: reference point mismatch at {sigma}')
                                    checked.add(key)
                            report['samples'].append(row)
                            print(f'  {row["startup_s"]:.4f} + {row["init_s"]:.4f} + '
                                  f'{row["loop_s"]:.4f} + {row["tail_s"]:.4f} s', flush=True)
                            write_report(report_file, report)
            finally:
                stop.set()
                if monitor:
                    monitor.join()
    finally:
        process.cpu_affinity(original_affinity)
    report.update(complete=True, distinct_endpoints=len(endpoints), reference_points=len(checked))
    write_report(report_file, report)


if __name__ == '__main__':
    main()
