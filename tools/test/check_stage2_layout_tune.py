"""Qualify layout-aware formal/runtime tuning on copied real Stage1 points.

Repeated points are timing fixtures, not independent ECM coverage. Production
profiles, queues and saves stay unchanged; each run requires a fresh output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from gpu_telemetry import TelemetryRecorder
from stage2_tune_tables import summaries


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('exe', 'save', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--profile', type=Path)
    p.add_argument('--mode', choices=('formal', 'runtime'), required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--exponent', type=int, default=5987)
    p.add_argument('--d', type=int, default=1141140)
    p.add_argument('--b2', type=int, default=1300000000000)
    a = p.parse_args()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    sources = [a.exe.resolve(), a.save.resolve()] + ([a.profile.resolve()] if a.profile else [])
    identities = {str(path): sha(path) for path in sources}
    point = next(s for s in a.save.read_text(encoding='utf-8-sig').splitlines() if s.startswith('METHOD=')) + '\n'
    n = int(re.search(r'\bN=0x([\da-fA-F]+);', point)[1], 16)
    whole = (1 << a.exponent) - 1
    assert whole % n == 0
    (out / 'points.save').write_text(point * 2, encoding='utf-8')
    config = out / 'ecm.toml'
    config.write_text('[stage2]\nstage2_auto_carrier=true\nstage2_auto_b2=false\n'
                      'stage2_tune_runtime=true\nstage2_short_calibration=false\n', encoding='utf-8')
    profile = out / 'tune.toml'
    if a.profile:
        profile.write_bytes(a.profile.read_bytes())
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_') and k != 'CUDA_VISIBLE_DEVICES'}
    common = [str(a.exe.resolve()), '--config', str(config), '--device', str(a.device),
              '--batch-mb', '64', '--arena-mb', '0', '--owner-budget-mb', '960',
              '--short-calibration', '0', '--short-ntt-batch-tune', '0']
    report = dict(identities=identities, complete=False, independent_curves=False, mode=a.mode,
                  D=a.d, B2=a.b2, actual_bits=n.bit_length(), carrier=a.exponent, runs=[])

    def run(label, extra, timeout):
        command = common + extra
        (out / (label + '.command.json')).write_text(json.dumps(command, indent=2), encoding='utf-8')
        start = time.perf_counter()
        telemetry = TelemetryRecorder(a.device, out / (label + '.telemetry.jsonl'))
        try:
            with (out / (label + '.console.log')).open('wb') as log:
                result = subprocess.run(command, cwd=out, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=timeout)
        finally:
            telemetry.close()
        text = (out / (label + '.console.log')).read_text(encoding='utf-8', errors='replace')
        assert result.returncode == 0, text[-4000:]
        selections = []
        for line in text.splitlines():
            if line.startswith('{'):
                row = json.loads(line)
                if row.get('type') == 'tune_selection':
                    selections.append(row)
        report['runs'].append(dict(label=label, seconds=time.perf_counter()-start,
                                   selections=selections, telemetry_samples=telemetry.samples))
        return selections

    if a.mode == 'formal':
        run('formal', ['--tune', 'ecm', '--tune-save', str(out / 'points.save'),
            '--tune-carrier-exponent', str(a.exponent), '--tune-d', str(a.d),
            '--tune-b2', str(a.b2), '--tune-repeats', '2', '--tune-tail-samples', '0',
            '--tune-budget-seconds', '600', '--tune-file', str(profile)], 720)
        rows = [r for r in summaries(tomllib.loads(profile.read_text(encoding='utf-8')))
                if r['source'] == 'measured' and r['d'] == a.d and r['b2'] == a.b2]
        carrier = {r['groot_balanced']: r for r in rows if r['carrier_exponent'] == a.exponent}
        assert set(carrier) == {0, 1} and all(r['repeats'] >= 2 for r in carrier.values()), rows
        assert all(r.get('groot_balanced') in (0, 1) for r in rows)
        report['measurements'] = rows
    else:
        if not a.profile:
            p.error('runtime qualification requires an existing real profile')
        queue = out / 'worktodo.txt'
        queue.write_text(f'ECMSTAGE2=1,2,{a.exponent},-1,"points.save",{a.b2},0,2,"{whole // n}"\n', encoding='utf-8')
        preflight = run('preflight', ['--worktodo', str(queue), '--once', '--plan-only',
            '--b2', str(a.b2), '--d', str(a.d), '--auto-carrier', '1', '--tune-runtime', '1',
            '--tune-profile', str(profile)], 90)[0]
        assert preflight['groot_balanced'] == 1 and preflight['runtime_allocation']['ordinary_reduced'], preflight
        selections = run('learn', ['--worktodo', str(queue), '--once', '--b2', str(a.b2),
            '--d', str(a.d), '--auto-carrier', '1', '--tune-runtime', '1',
            '--tune-profile', str(profile), '--results', str(out / 'results.jsonl'),
            '--submission', str(out / 'results.txt'), '--debug-log-file', str(out / 'debug.log'),
            '--log', str(out / 'engine.log'), '--log-level', 'batches'], 420)
        assert len(selections) == 2 and all(s['groot_balanced'] == 1 and
            s['carrier_exponent'] == a.exponent and s['runtime_probe'] for s in selections), selections
        assert all(s['runtime_allocation']['ordinary_reduced'] for s in selections)
        receipts = [json.loads(s) for s in (out / 'results.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        assert len(receipts) == 2 and all(r['status'] == 'NF' for r in receipts)
        rows = summaries(tomllib.loads(profile.read_text(encoding='utf-8')))
        learned = [r for r in rows if r.get('groot_balanced') == 1 and
                   r['carrier_exponent'] == a.exponent and r['d'] == a.d and r['b2'] == a.b2]
        assert len(learned) == 1 and learned[0]['repeats'] == 2, learned
        # Read-only planning uses the newly published row without executing a
        # third duplicate curve or changing the finite queue's progress.
        adopt_queue = out / 'adopt.worktodo.txt'
        adopt_queue.write_text(f'ECMSTAGE2=1,2,{a.exponent},-1,"points.save",{a.b2},0,1,"{whole // n}"\n', encoding='utf-8')
        stable = run('adopt', ['--worktodo', str(adopt_queue), '--once', '--plan-only',
            '--b2', str(a.b2), '--d', str(a.d),
            '--auto-carrier', '1', '--tune-runtime', '0', '--tune-profile', str(profile)], 90)[0]
        candidates = [r for r in stable['runtime_candidates'] if r['samples'] >= 2]
        assert not stable['runtime_probe'] and stable['mean_seconds'] == min(r['mean_seconds'] for r in candidates)
        report['learned'] = learned[0]
        report['adopted'] = stable
    assert all(sha(Path(path)) == identity for path, identity in identities.items())
    report['complete'] = True
    (out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(complete=True, mode=a.mode, runs=len(report['runs']))))


if __name__ == '__main__':
    main()
