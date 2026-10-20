"""Uninstrumented Edwards Stage1 process comparison on GPU1.

Wall/CPU cover process creation through exit, including setup, callbacks, saves
and results. All save records are checked and must match across both binaries.
Independent arithmetic checks belong to the separate CPU/GPU correctness gate.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import threading
import time

from bench_edwards_gpu_stage1 import sha, validate
from diagnose_stage1_host_wait import gpu_sample


def process_times(process):
    api = ctypes.WinDLL('kernel32', use_last_error=True).GetProcessTimes
    api.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    api.restype = wintypes.BOOL
    values = [wintypes.FILETIME() for _ in range(4)]
    if not api(int(process._handle), *(ctypes.byref(x) for x in values)):
        raise ctypes.WinError(ctypes.get_last_error())
    created, exited, kernel, user = [(x.dwHighDateTime << 32) | x.dwLowDateTime for x in values]
    return (exited-created)*1e-7, (kernel+user)*1e-7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--exponent', type=int, required=True)
    parser.add_argument('--b1', type=int, required=True)
    parser.add_argument('--curves', type=int, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 3 or args.curves < 1:
        raise ValueError('Need positive curves and at least three repeats')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshots = out/'tools'
    snapshots.mkdir()
    script = Path(__file__).resolve()
    for path in [script, script.with_name('bench_edwards_gpu_stage1.py'),
                 script.with_name('diagnose_stage1_host_wait.py'),
                 script.parents[1]/'test/test_edwards_gpu_pipeline.py',
                 script.parents[1]/'stat/suyama_mont_ref.py']:
        shutil.copy2(path, snapshots/path.name)
    report = dict(complete=False, device=1, exponent=args.exponent, b1=args.b1,
                  curves=args.curves, sigma_start=17, binaries={}, runs=[],
                  collector_sha256=sha(__file__),
                  timing_boundary='GetProcessTimes process creation to exit, all setup and publication included')
    binaries = {}
    for mode, source in [('baseline', args.baseline), ('candidate', args.candidate)]:
        source = source.resolve()
        target = out/'bin'/mode
        target.mkdir(parents=True)
        for path in [source, source.parent/'gmp-10.dll']:
            shutil.copy2(path, target/path.name)
        binaries[mode] = target/source.name
        report['binaries'][mode] = dict(exe_sha256=sha(source), gmp_sha256=sha(target/'gmp-10.dll'))
    env = {k:v for k,v in os.environ.items() if not k.startswith(('ECM_', 'NTT_'))
           and k not in ('CUDA_LAUNCH_BLOCKING', 'CUDA_VISIBLE_DEVICES')}
    env['ECM_EDWARDS_STAGE1_ALGO'] = 'naf'
    reference = None
    for repeat in range(args.repeats):
        for mode in (['baseline', 'candidate'] if repeat % 2 == 0 else ['candidate', 'baseline']):
            folder = out/f'{repeat}_{mode}'
            folder.mkdir()
            ini = folder/'ecm.toml'
            ini.write_text(f'user=\ncomputer=\nckpt_seconds=0\nlog_file=\nstage1_results_file={folder/"internal.jsonl"}\n', encoding='utf-8')
            save = folder/'points.save'
            command = list(map(str, [binaries[mode], '--curve', 'ed', '--target', 'gpu', '-d', 1,
                '-sigma', 17, '-gpucurves', args.curves, '--naf-w', 6, '-ini', ini,
                '--ckpt', 0, '--exp-cache', folder/'cache', '--tmp-dir', folder/'scratch',
                '-save', save, args.b1, args.b1]))
            stop = threading.Event()
            samples = []
            def telemetry():
                while not stop.is_set():
                    try:samples.append(dict(monotonic=time.monotonic(), gpu=gpu_sample()))
                    except subprocess.TimeoutExpired:samples.append(dict(error='telemetry timeout'))
                    stop.wait(.3)
            thread = threading.Thread(target=telemetry)
            thread.start()
            try:
                with (folder/'run.log').open('wb') as log:
                    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                          cwd=folder, env=env) as process:
                        try:process.communicate(f'2^{args.exponent}-1\n'.encode(), timeout=900)
                        except subprocess.TimeoutExpired:
                            process.kill();process.wait();raise
                        wall, cpu = process_times(process)
                        if process.returncode:
                            raise ValueError('Worker failed; inspect preserved log')
            finally:
                stop.set()
                thread.join(timeout=15)
            rows = validate(save, (1 << args.exponent)-1, args.b1, args.curves, 17, 'ed')
            if reference is None:reference = rows
            if reference != rows:raise ValueError('Complete save records differ')
            text = (folder/'run.log').read_text(encoding='utf-8', errors='replace')
            match = re.search(r'GPU Edwards arithmetic: algorithm=naf api_seconds=([\d.]+)', text)
            launches = re.search(r'loop_gpu_ms=[\d.]+ launches=(\d+)', text)
            if not match or not launches:raise ValueError('Missing production NAF metrics')
            row = dict(mode=mode, repeat=repeat, wall_seconds=wall, cpu_seconds=cpu, cpu_cores=cpu/wall,
                       curves_per_hour=args.curves*3600/wall, arithmetic_seconds=float(match[1]),
                       launches=int(launches[1]), save_sha256=sha(save), telemetry=samples, command=command)
            report['runs'].append(row)
            (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
            print(json.dumps({k:row[k] for k in ('mode', 'wall_seconds', 'cpu_cores', 'launches')}), flush=True)
    report['summaries'] = {}
    for mode in binaries:
        rows = [r for r in report['runs'] if r['mode']==mode]
        report['summaries'][mode] = dict(samples=len(rows),
            median_wall_seconds=statistics.median(r['wall_seconds'] for r in rows),
            median_cpu_cores=statistics.median(r['cpu_cores'] for r in rows),
            median_curves_per_hour=statistics.median(r['curves_per_hour'] for r in rows),
            wall_range=[min(r['wall_seconds'] for r in rows), max(r['wall_seconds'] for r in rows)])
    report.update(complete=True, identical_save_records=True)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report['summaries']), flush=True)


if __name__ == '__main__':
    main()
