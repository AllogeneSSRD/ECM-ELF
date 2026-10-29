"""Replay real saves with named OpenCL builds/environments; preserve all samples.

--variants is a JSON array of {name, build, environment: {KEY: VALUE}}.
Every executable/resource is frozen before use. Environment overrides are local
to each child; this tool never changes GPU settings or production queues.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(text, prefix):
    line = next((s for s in text.splitlines() if s.startswith(prefix + ':')), '')
    return {k: float(v) for k, v in re.findall(r'(\w+)=([0-9]+(?:\.[0-9]+)?)', line)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--inputs', type=Path, default=Path('data/experiments/opencl_production_20261029/ab01/inputs'))
    p.add_argument('--exponents', default='503,2003,5003,8011')
    p.add_argument('--b2', default='260000,2600000')
    p.add_argument('--d', type=int, default=2310)
    p.add_argument('--arena-mb', type=int, default=512)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--telemetry-device', type=int, default=1)
    a = p.parse_args()
    if a.out.exists() or not 1 <= a.repeats <= 20:
        p.error('Choose a new output directory and repeats=1..20')
    specs = json.loads(a.variants.read_text(encoding='utf-8-sig'))
    if not specs or len({v['name'] for v in specs}) != len(specs):
        p.error('Need unique named variants')
    a.out.mkdir(parents=True)
    manifest = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'variants': [],
                'D': a.d, 'arena_mb': a.arena_mb, 'repeats': a.repeats, 'warmup': 0, 'device': a.device,
                'gpu_settings_changed': False, 'inherited_ntt_environment': {
                    k: v for k, v in os.environ.items() if k.startswith(('ECM_OPENCL_', 'NTT_'))}}
    for spec in specs:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', spec['name']):
            p.error('Unsafe variant name')
        build = Path(spec['build'])
        dest = a.out / 'bin' / spec['name']
        dest.mkdir(parents=True)
        for name in ('ecm_opencl_stage2.exe', 'gmp-10.dll', 'build_manifest.json'):
            shutil.copy2(build / name, dest / name)
        shutil.copytree(build / 'opencl_resources', dest / 'opencl_resources')
        spec['exe'] = str((dest / 'ecm_opencl_stage2.exe').resolve())
        spec['sha256'] = sha(Path(spec['exe']))
        spec['version'] = subprocess.check_output([spec['exe'], '--version'], text=True).strip()
        spec['files'] = {str(f.relative_to(dest)): sha(f) for f in dest.rglob('*') if f.is_file()}
        manifest['variants'].append(spec)
    manifest['opencl_devices'] = subprocess.check_output([specs[0]['exe'], '--list-devices'], text=True)
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    with (a.out / 'gpu.csv').open('w', encoding='utf-8') as telemetry:
        poll = subprocess.Popen(['nvidia-smi', f'--id={a.telemetry_device}',
                                 '--query-gpu=timestamp,clocks.current.graphics,power.draw,utilization.gpu',
                                 '--format=csv', '-lms', '500'], stdout=telemetry,
                                stderr=subprocess.DEVNULL, creationflags=flags)
        try:
            for exponent in map(int, a.exponents.split(',')):
                saves = list(a.inputs.glob(f'm{exponent}_n*_s*.save'))
                if len(saves) != 1:
                    raise RuntimeError(f'Need exactly one save for M{exponent}')
                frozen = a.out / 'inputs' / saves[0].name
                frozen.parent.mkdir(exist_ok=True)
                shutil.copy2(saves[0], frozen)
                bits = int(re.search(r'_n(\d+)_', frozen.name)[1])
                for b2 in map(int, a.b2.split(',')):
                    for repeat in range(a.repeats):
                        order = specs if repeat % 2 == 0 else list(reversed(specs))
                        for spec in order:
                            folder = a.out / f'm{exponent}_b{b2}_r{repeat}_{spec["name"]}'
                            folder.mkdir()
                            config = folder / 'ecm.toml'
                            config.write_text(f'''[stage2]
stage2_log_level="phases"
stage2_auto_carrier=false
stage2_tune_runtime=false
stage2_short_calibration=false
stage2_arena_mb={a.arena_mb}
stage2_batch_mb=64
stage2_fold_mb=640
stage2_results_file="results.jsonl"
stage2_log_file="worker.log"
''', encoding='utf-8')
                            cmd = [spec['exe'], '--config', str(config.resolve()), '--save', str(frozen.resolve()),
                                   '--device', str(a.device), '--b2', str(b2), '--d', str(a.d), '--curves', '1',
                                   '--factor-only', '--carrier-exponent', '0', '--auto-carrier', '0',
                                   '--tune-runtime', '0', '--short-calibration', '0', '--ntt-counts', '1']
                            env = os.environ.copy()
                            env.update({k: str(v) for k, v in spec.get('environment', {}).items()})
                            (folder / 'command.json').write_text(json.dumps({'cmd': cmd, 'environment': spec.get('environment', {})}, indent=2), encoding='utf-8')
                            start = time.perf_counter()
                            with (folder / 'stdout.log').open('w', encoding='utf-8') as log:
                                result = subprocess.run(cmd, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                        timeout=300, creationflags=flags)
                            seconds = time.perf_counter() - start
                            if result.returncode:
                                raise RuntimeError(f'{folder}: exit {result.returncode}')
                            text = (folder / 'worker.log').read_text(encoding='utf-8-sig')
                            results = [json.loads(s) for s in (folder / 'results.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
                            full = fields(text, 'stage2_full_wall')
                            shape = re.search(r'real_shape: D=(\d+) P=phi\(D\)/2=(\d+)', text)
                            if not full or not shape or int(shape[1]) != a.d or len(results) != 1 or results[0].get('factors', []):
                                raise RuntimeError(f'Not a matching complete NF curve: {folder}')
                            if results[0]['source-build'].lower() != spec['sha256']:
                                raise RuntimeError('Result binary identity mismatch')
                            row = {'variant': spec['name'], 'exponent': exponent, 'bits': bits, 'B2': b2,
                                   'D': a.d, 'P': int(shape[2]), 'repeat': repeat, 'full': full,
                                   'points': {**fields(text, 'opencl_points'), **fields(text, 'opencl_point_io')},
                                   'point_command': fields(text, 'opencl_point_command'),
                                   'point_window': fields(text, 'opencl_point_window'),
                                   'checks': fields(text, 'opencl_checks'), 'payload': fields(text, 'opencl_payload'),
                                   'ntt_lengths': next((s for s in text.splitlines() if s.startswith('ntt_lengths:')), ''),
                                   'split': {**fields(text, 'opencl_split'), **fields(text, 'opencl_tail')},
                                   'programs': fields(text, 'opencl_programs'),
                                   'process_seconds': seconds, 'save_sha256': sha(frozen), 'binary_sha256': spec['sha256'],
                                   'log': str(folder / 'worker.log'), 'log_sha256': sha(folder / 'worker.log'),
                                   'result_sha256': sha(folder / 'results.jsonl')}
                            with (a.out / 'measurements.jsonl').open('a', encoding='utf-8') as f:
                                f.write(json.dumps(row) + '\n')
                            print(f'{spec["name"]} N={bits} B2={b2} r={repeat}: engine={full["total"]:.6f}s process={seconds:.3f}s points={row["points"]}', flush=True)
        finally:
            poll.terminate()
            poll.wait()
    (a.out / 'completed.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
