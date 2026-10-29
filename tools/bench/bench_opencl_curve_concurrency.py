"""Independent valid curves through production CLI: fixed work, varied workers.

Each task has its own save, configuration, log and result folder. This tool does
not concurrently consume a production worktodo and does not change GPU settings.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

from bench_opencl_stage2_variants import fields
from stage2_windows_job import WindowsJob
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'test'))
from check_opencl_stage2_worker import save_line

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opencl-build', type=Path, required=True)
    p.add_argument('--cuda-build', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--inputs', type=Path, default=Path('data/experiments/opencl_production_20261029/ab01/inputs'))
    p.add_argument('--exponents', default='5003,8011')
    p.add_argument('--backends', default='opencl_device,opencl_cpu,cuda')
    p.add_argument('--parallel', default='1,2,4')
    p.add_argument('--curves', type=int, default=4)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--b1', type=int, default=20)
    p.add_argument('--b2', type=int, default=2600000)
    p.add_argument('--d', type=int, default=2310)
    p.add_argument('--arena-mb', type=int, default=512)
    p.add_argument('--opencl-device', type=int, default=1)
    p.add_argument('--cuda-device', type=int, default=1)
    a = p.parse_args()
    parallel = list(map(int, a.parallel.split(',')))
    modes = a.backends.split(',')
    if a.out.exists() or not 1 <= a.repeats <= 10 or not 1 <= a.curves <= 32 or min(parallel) < 1 or max(parallel) > a.curves:
        p.error('Use a new directory, repeats=1..10, curves=1..32, parallel=1..curves')
    if any(m not in ('opencl_device', 'opencl_cpu', 'cuda') for m in modes) or ('cuda' in modes and not a.cuda_build):
        p.error('Select known backends and supply CUDA build if requested')
    if not 2 <= a.b1 <= 1000 or not a.b2 > a.b1 or not 6 <= a.d <= 200000000 or a.d % 2:
        p.error('Fast saves require B1=2..1000, B2>B1 and even D=6..200000000')
    a.out.mkdir(parents=True)
    binaries = {}
    for backend, build, name in [('opencl', a.opencl_build, 'ecm_opencl_stage2.exe'), ('cuda', a.cuda_build, 'ecm_cuda_stage2.exe')]:
        if not any(m.startswith(backend) for m in modes):
            continue
        dest = a.out / 'bin' / backend
        dest.mkdir(parents=True)
        for filename in (name, 'gmp-10.dll', 'build_manifest.json'):
            shutil.copy2(build / filename, dest / filename)
        if backend == 'opencl':
            shutil.copytree(build / 'opencl_resources', dest / 'opencl_resources')
        binaries[backend] = {'exe': str((dest / name).resolve()), 'sha256': sha(dest / name),
                             'files': {str(f.relative_to(dest)): sha(f) for f in dest.rglob('*') if f.is_file()}}
        binaries[backend]['version'] = subprocess.check_output([binaries[backend]['exe'], '--version'], text=True).strip()
    sources = [Path(__file__), Path(__file__).with_name('stage2_windows_job.py'), Path(__file__).with_name('bench_opencl_stage2_variants.py'),
               Path(__file__).resolve().parents[1] / 'test/check_opencl_stage2_worker.py']
    for src in sources:
        dest = a.out / 'tools' / src.name; dest.parent.mkdir(exist_ok=True); shutil.copy2(src, dest)
    manifest = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'kind': 'independent_curves_multi_process',
                'binaries': binaries, 'B1': a.b1, 'B2': a.b2, 'D': a.d, 'curves': a.curves, 'repeats': a.repeats,
                'parallel': parallel, 'modes': modes, 'warmup': 0, 'gpu_settings_changed': False,
                'inherited_environment': {k: v for k, v in os.environ.items() if k.startswith(('NTT_', 'ECM_OPENCL_'))},
                'accounting': 'Windows job includes parent drivers and descendants; attach immediately after Popen; tiny pre-attach CPU is not counted',
                'tools': {src.name: sha(src) for src in sources}, 'inputs': []}
    # Generate distinct, valid PARAM0 curves from the same target cofactor N.
    # The source Edwards save contributes N only; these are fresh small-B1 saves.
    for exponent in map(int, a.exponents.split(',')):
        original = list(a.inputs.glob(f'm{exponent}_n*_s*.save'))
        if len(original) != 1:
            raise RuntimeError('Need exactly one source modulus save')
        text = original[0].read_text(encoding='utf-8-sig')
        n = int(re.search(r'\bN=0x([0-9a-fA-F]+)', text)[1], 16)
        inputs = []
        sigma = 26
        while len(inputs) < a.curves:
            try:
                line = save_line(n, sigma, a.b1)
            except ValueError:
                sigma += 1
                continue
            file = a.out / 'inputs' / f'm{exponent}_n{n.bit_length()}_s{sigma}.save'
            file.parent.mkdir(exist_ok=True); file.write_text(line, encoding='utf-8')
            inputs.append({'path': str(file.resolve()), 'sha256': sha(file), 'sigma': sigma})
            sigma += 1
        manifest['inputs'].append({'exponent': exponent, 'bits': n.bit_length(), 'source_N_save_sha256': sha(original[0]), 'saves': inputs})
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    poll = subprocess.Popen(['nvidia-smi', f'--id={a.cuda_device}', '--query-gpu=timestamp,clocks.current.graphics,utilization.gpu,memory.used,memory.total',
                             '--format=csv,noheader', '-lms', '250'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1, creationflags=flags)
    tag = ['outside']
    telemetry = (a.out / 'gpu.csv').open('w', newline='', encoding='utf-8')
    writer = csv.writer(telemetry); writer.writerow(['cohort', 'received_monotonic', 'timestamp', 'clock', 'utilization', 'memory_used', 'memory_total'])
    def drain():
        for line in poll.stdout:
            values = next(csv.reader([line], skipinitialspace=True))
            if len(values) == 5:
                writer.writerow([tag[0], time.perf_counter(), *values]); telemetry.flush()
    reader = threading.Thread(target=drain, daemon=True); reader.start()
    cohorts = []
    try:
        for inp in manifest['inputs']:
            for repeat in range(a.repeats):
                choices = [(mode, c) for mode in modes for c in parallel]
                if repeat % 2:
                    choices.reverse()
                for mode, count in choices:
                    name = f'm{inp["exponent"]}_r{repeat}_{mode}_c{count}'
                    folder = a.out / name; folder.mkdir()
                    backend = 'cuda' if mode == 'cuda' else 'opencl'
                    exe = binaries[backend]['exe']
                    device = a.cuda_device if backend == 'cuda' else a.opencl_device
                    env = os.environ.copy()
                    if backend == 'opencl':
                        env['ECM_OPENCL_POINTS'] = mode.removeprefix('opencl_')
                    job = WindowsJob(); active = []; done = []; next_task = 0
                    tag[0] = name; start = time.perf_counter()
                    try:
                        while active or next_task < len(inp['saves']):
                            while len(active) < count and next_task < len(inp['saves']):
                                save = inp['saves'][next_task]; work = folder / f'curve{next_task}'; work.mkdir()
                                config = work / 'ecm.toml'
                                config.write_text(f'''[stage2]
stage2_log_level="phases"
stage2_arena_mb={a.arena_mb}
stage2_batch_mb=64
stage2_fold_mb=640
stage2_auto_carrier=false
stage2_tune_runtime=false
stage2_short_calibration=false
stage2_results_file="results.jsonl"
stage2_log_file="worker.log"
''', encoding='utf-8')
                                cmd = [exe, '--config', str(config.resolve()), '--save', save['path'], '--device', str(device), '--b2', str(a.b2),
                                       '--d', str(a.d), '--curves', '1', '--factor-only', '--carrier-exponent', '0', '--auto-carrier', '0',
                                       '--tune-runtime', '0', '--short-calibration', '0', '--ntt-counts', '1']
                                (work / 'command.json').write_text(json.dumps({'cmd': cmd, 'points': env.get('ECM_OPENCL_POINTS')}, indent=2), encoding='utf-8')
                                log = (work / 'stdout.log').open('w', encoding='utf-8'); begin = time.perf_counter()
                                process = subprocess.Popen(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
                                try:
                                    job.assign(process)
                                except BaseException:
                                    process.terminate(); process.wait(); log.close(); raise
                                active.append((process, log, begin, work, next_task)); next_task += 1
                            finished = []
                            for item in active:
                                process, log, begin, work, number = item
                                code = process.poll()
                                if code is None:
                                    if time.perf_counter() - begin > 300:
                                        raise TimeoutError(str(work))
                                    continue
                                log.close(); seconds = time.perf_counter() - begin
                                if code:
                                    raise RuntimeError(f'{work}: exit {code}')
                                result = [json.loads(s) for s in (work / 'results.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
                                text = (work / 'worker.log').read_text(encoding='utf-8-sig')
                                full = fields(text, 'stage2_full_wall')
                                shape = re.search(r'real_shape: D=(\d+) P=phi\(D\)/2=(\d+)', text)
                                if len(result) != 1 or result[0].get('factors') or result[0]['stage2-engine']['hits'] or result[0]['stage2-engine']['bad_factors'] or not full:
                                    raise RuntimeError('Need a complete checked NF curve: ' + str(work))
                                if not shape or int(shape[1]) != a.d or result[0]['stage2-engine']['requested_B2'] != a.b2:
                                    raise RuntimeError('Executed D/B2 differs from the requested fixed shape')
                                if result[0]['source-build'].lower() != binaries[backend]['sha256']:
                                    raise RuntimeError('Executed binary identity mismatch')
                                done.append({'task': number, 'sigma': inp['saves'][number]['sigma'], 'process_seconds': seconds,
                                             'D': a.d, 'P': int(shape[2]), 'B2': a.b2,
                                             'full': full, 'points': fields(text, 'opencl_points'), 'split': fields(text, 'opencl_split'),
                                             'payload': fields(text, 'opencl_payload'), 'log': str(work / 'worker.log'),
                                             'log_sha256': sha(work / 'worker.log'), 'result_sha256': sha(work / 'results.jsonl')})
                                finished.append(item)
                            for item in finished:
                                active.remove(item)
                            if active:
                                time.sleep(.01)
                        parent_wall = time.perf_counter() - start
                        account = job.drain()
                        wall = time.perf_counter() - start
                    finally:
                        if active or job.accounting()['active_processes']:
                            job.terminate()
                            for process, log, *_ in active:
                                process.wait(); log.close()
                        job.close(); tag[0] = 'outside'
                    cohort = {'name': name, 'exponent': inp['exponent'], 'bits': inp['bits'], 'mode': mode, 'parallel': count, 'repeat': repeat,
                              'curves': len(done), 'wall_seconds': wall, 'curves_per_second': len(done) / wall,
                              'parent_wall_seconds': parent_wall, 'descendant_drain_seconds': wall - parent_wall,
                              'cpu_core_equivalents': account['cpu_seconds'] / wall, 'accounting': account, 'tasks': done}
                    cohorts.append(cohort)
                    with (a.out / 'measurements.jsonl').open('a', encoding='utf-8') as file:
                        file.write(json.dumps(cohort) + '\n')
                    print(f'{name}: {wall:.3f}s, {len(done)/wall:.4f} curves/s, CPU={cohort["cpu_core_equivalents"]:.2f} cores, commit={account["peak_job_commit_bytes"]/2**20:.1f} MiB', flush=True)
    finally:
        poll.terminate(); poll.wait(); reader.join(timeout=5); telemetry.close()
    manifest['complete'] = True; manifest['cohorts'] = len(cohorts)
    (a.out / 'completed.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
