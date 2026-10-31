"""Isolated same-input preparation A/B; optionally execute complete saved curves.

Planning replays are not GPU throughput measurements. Original save, profile,
executable and configuration are hashed before/after; the source queue is unused.
"""
import argparse
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
    sys.set_int_max_str_digits(20000)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('exe', 'profile', 'save', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--baseline-exe', type=Path)
    p.add_argument('--record', type=int, default=364)
    p.add_argument('--exponent', type=int, default=5987)
    p.add_argument('--b2', type=int, default=1300000000000)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--curves', type=int, default=6)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--cache', choices=('0', '1', 'both'), default='both')
    a = p.parse_args()
    if a.record < 1 or a.curves < 1 or a.device < 0 or not 2 <= a.exponent <= 16384:
        p.error('record/curves >= 1; device >= 0; exponent in 2..16384')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    inputs = [a.exe.resolve(), a.profile.resolve(), a.save.resolve()]
    if a.baseline_exe:
        inputs.append(a.baseline_exe.resolve())
    identities = {str(f): sha(f) for f in inputs}
    points = [s for s in a.save.read_text(encoding='utf-8-sig').splitlines() if s.startswith('METHOD=')]
    if a.record > len(points):
        p.error(f'record {a.record} exceeds {len(points)} save points')
    point = points[a.record - 1] + '\n'
    n = int(re.search(r'\bN=0x([\da-fA-F]+);', point)[1], 16)
    assert ((1 << a.exponent) - 1) % n == 0
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_') and k != 'CUDA_VISIBLE_DEVICES'}
    modes = [('cache_' + m, a.exe, m) for m in (('0', '1') if a.cache == 'both' else (a.cache,))]
    if a.baseline_exe:
        modes.insert(0, ('baseline', a.baseline_exe, '1'))
    report = dict(identities=identities, execute=a.execute, record=a.record,
                  independent_curves=False, runs=[])
    for label, executable, mode in modes:
        run_dir = out / label; run_dir.mkdir()
        (run_dir / 'points.save').write_text(point * a.curves, encoding='utf-8')
        profile = run_dir / 'tune.toml'; profile.write_bytes(a.profile.read_bytes())
        (run_dir / 'worktodo.txt').write_text(
            f'ECMSTAGE2=1,2,{a.exponent},-1,"points.save",{a.b2},0,{a.curves},"{((1 << a.exponent)-1)//n}"\n',
            encoding='utf-8')
        (run_dir / 'ecm.toml').write_text('[stage2]\n'
            'stage2_auto_carrier = true\nstage2_auto_b2 = false\n'
            'stage2_tune_runtime = true\nstage2_tune_runtime_samples = 10\n'
            'stage2_prediction_policy = "model"\nstage2_short_calibration = false\n', encoding='utf-8')
        command = [str(executable.resolve()), '--config', str(run_dir / 'ecm.toml'),
            '--device', str(a.device), '--worktodo', str(run_dir / 'worktodo.txt'), '--once',
            '--batch-mb', '64', '--arena-mb', '0', '--owner-budget-mb', '960',
            '--short-calibration', '0', '--tune-profile', str(profile), '--tune-runtime', '1',
            '--runtime-prepare-cache', mode, '--results', str(run_dir / 'internal.jsonl'),
            '--submission', str(run_dir / 'results.txt'), '--log', str(run_dir / 'engine.log')]
        if not a.execute:
            command.append('--plan-only')
        (run_dir / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        start = time.perf_counter()
        monitor = None
        if a.execute and shutil.which('nvidia-smi'):
            monitor = subprocess.Popen(['nvidia-smi', '-i', str(a.device),
                '--query-gpu=timestamp,index,driver_version,clocks.current.sm,clocks.current.memory,pstate,power.draw,temperature.gpu,utilization.gpu',
                '--format=csv', '-lms', '500', '-f', str(run_dir / 'telemetry.csv')],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            with (run_dir / 'console.log').open('wb') as log:
                result = subprocess.run(command, cwd=run_dir, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=1800 if a.execute else 300)
        finally:
            if monitor is not None:
                monitor.terminate(); monitor.wait(timeout=10)
        elapsed = time.perf_counter() - start
        text = (run_dir / 'console.log').read_text(encoding='utf-8', errors='replace')
        if result.returncode:
            raise RuntimeError(f'cache={mode}, exit={result.returncode}: {text[-3000:]}')
        rows = []
        # Queue logs contain forwarded child JSON even when the console hides it.
        log_text = text + ('\n' + (run_dir / 'engine.log').read_text(encoding='utf-8', errors='replace')
                           if (run_dir / 'engine.log').exists() else '')
        seen = set()
        for line in log_text.splitlines():
            if not line.startswith('{'):
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('type') != 'tune_selection' or line in seen:
                continue
            seen.add(line); rows.append(row)
        assert len(rows) == a.curves, f'expected {a.curves} production selections, got {len(rows)}'
        for row in rows:
            if not row.get('runtime_probe'):
                measured = [c['mean_seconds'] for c in row['runtime_candidates'] if c['samples'] >= 2]
                assert row['mean_seconds'] == min(measured), 'fastest measured route displaced'
        assert sha(profile) == identities[str(a.profile.resolve())], 'frozen saturated profile changed'
        report['runs'].append(dict(label=label, binary_sha256=sha(executable), cache=int(mode), wall_seconds=elapsed, selections=rows))
        (out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    assert all(sha(Path(f)) == digest for f, digest in identities.items()), 'source input changed'
    report['complete'] = True
    (out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(complete=True, output=str(out), runs=[dict(cache=r['cache'], wall_seconds=r['wall_seconds'],
        preparation=[s.get('runtime_prepare', {}).get('total_seconds') for s in r['selections']]) for r in report['runs']])))


if __name__ == '__main__':
    main()
