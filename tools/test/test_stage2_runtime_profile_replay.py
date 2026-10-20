"""Replay a real runtime tune profile through production planning, without ECM work.

All inputs are frozen under a fresh ignored output directory. The source queue,
profile, INI and save remain untouched; repeated points are planning fixtures,
not independent curves or a performance measurement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('exe', 'profile', 'save', 'ini', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--baseline-exe', type=Path)
    p.add_argument('--exponent', type=int, required=True)
    p.add_argument('--b2', type=int, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--arena-mb', type=int, default=6400)
    p.add_argument('--fold-mb', type=int, default=960)
    p.add_argument('--batch-mb', type=int, default=64)
    p.add_argument('--expected-d', type=int)
    p.add_argument('--expect-forward-d', action='store_true')
    p.add_argument('--expect-forward-rejection', action='store_true',
                   help='Require a larger D to reach native planning and be rejected by the free-memory bound')
    a = p.parse_args()
    if not 2 <= a.exponent <= 16384:
        p.error('exponent must be in 2..16384')
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(20000)
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    inputs = [a.exe.resolve(), a.profile.resolve(), a.save.resolve(), a.ini.resolve()]
    if a.baseline_exe:
        inputs.append(a.baseline_exe.resolve())
    identities = {str(path): sha(path) for path in inputs}
    profile = out / 'profile.toml'; profile.write_bytes(a.profile.read_bytes())
    ini = out / 'ecm.toml'; ini.write_bytes(a.ini.read_bytes())
    with a.save.open(encoding='utf-8-sig') as f:
        point = next(line for line in f if re.search(r'\bN=0x[\da-fA-F]+;', line))
    n = int(re.search(r'\bN=0x([\da-fA-F]+);', point)[1], 16)
    whole = (1 << a.exponent) - 1
    assert n > 1 and whole % n == 0, 'save does not divide the requested Mersenne number'
    known = str(whole // n)
    (out / 'curve.save').write_text(point, encoding='utf-8')
    (out / 'four.save').write_text(point * 4, encoding='utf-8')
    for name, saved, count in [('one.txt', 'curve.save', 1), ('four.txt', 'four.save', 4)]:
        (out / name).write_text(
            f'ECMSTAGE2=1,2,{a.exponent},-1,"{saved}",{a.b2},0,{count},"{known}"\n', encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_') and k != 'CUDA_VISIBLE_DEVICES'}
    report = dict(identities=identities, timing_scope='planning only; no Stage2 kernels or timing claim')

    def call(label, exe, explore):
        command = [str(exe.resolve()), '--ini', str(ini), '--device', str(a.device),
                   '--worktodo', str(out / ('four.txt' if explore else 'one.txt')), '--once', '--plan-only',
                   '--b2', str(a.b2), '--batch-mb', str(a.batch_mb), '--arena-mb', str(a.arena_mb),
                   '--owner-budget-mb', str(a.fold_mb), '--short-calibration', '0',
                   '--tune-profile', str(profile), '--tune-runtime', str(int(explore)),
                   '--results', str(out / (label + '.jsonl')), '--log', str(out / (label + '.engine.log'))]
        (out / (label + '.command.json')).write_text(json.dumps(command, indent=2), encoding='utf-8')
        run = subprocess.run(command, cwd=out, env=env, capture_output=True, text=True,
                             errors='replace', timeout=120)
        (out / (label + '.log')).write_text(run.stdout + run.stderr, encoding='utf-8')
        assert run.returncode == 0, (label, run.stderr, run.stdout[-2000:])
        rows = []
        for line in run.stdout.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('type') == 'tune_selection':
                rows.append(row)
        assert rows and all(r.get('selected') for r in rows), (label, 'missing selection')
        (out / (label + '.selection.json')).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        report[label] = rows[0]
        return rows[0]

    if a.baseline_exe:
        call('baseline', a.baseline_exe, False)
    stable = call('readonly', a.exe, False)
    candidates = [c for c in stable['runtime_candidates'] if c['samples'] >= 2]
    assert not stable['runtime_probe'] and stable['reason'] == 'runtime_measured_fastest'
    assert stable['mean_seconds'] == min(c['mean_seconds'] for c in candidates)
    if a.expected_d:
        assert stable['D'] == a.expected_d, stable['D']
    search = call('search', a.exe, True)
    if a.expect_forward_d:
        assert search['runtime_probe'] and search['D'] > stable['D'], search
        assert any(s['reason'] == 'improving_edge' and s['proposed_D'] == search['D']
                   for s in search['runtime_search']), search
    if a.expect_forward_rejection:
        assert not search['runtime_probe'] and search['D'] == stable['D'], search
        rejected = [s.get('last_memory_rejection') for s in search['runtime_search']
                    if s.get('memory_rejected', 0)]
        assert any(r and r['D'] > stable['D'] and r['initial_free_snapshot_fits'] is False and
                   r['required_free_bytes'] > r['free_bytes'] for r in rejected), search
    assert sha(profile) == identities[str(a.profile.resolve())], 'planning changed the frozen profile'
    assert all(sha(Path(path)) == digest for path, digest in identities.items()), 'source input changed'
    report['complete'] = True
    (out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(complete=True, stable_D=stable['D'], proposed_D=search['D'],
                         baseline_D=report.get('baseline', {}).get('D'), source_inputs_unchanged=True)))


if __name__ == '__main__':
    main()
