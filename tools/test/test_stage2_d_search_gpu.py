"""Exercise automatic D exploration on a replayed, verified NF save on GPU1.

No independent ECM coverage or performance claim. All queue/profile/result files
are private copies. Does not change GPU controls or production state.
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
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/ecm_report'))
sys.path.insert(0, str(ROOT / 'tools/test'))
from result_format import normalize_result
from stage2_tune_tables import summaries


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('exe', 'save', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--exponent', type=int, default=503)
    parser.add_argument('--b2', type=int, default=27015978990)
    parser.add_argument('--curves', type=int, default=56)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    point = next(s for s in args.save.read_text(encoding='utf-8-sig').splitlines() if s.startswith('METHOD='))
    n = int(re.search(r'\bN=0x([\da-fA-F]+);', point)[1], 16)
    whole = (1 << args.exponent) - 1
    assert whole % n == 0 and whole != n and args.curves >= 32
    identities = {str(p.resolve()): sha(p) for p in (args.exe, args.save)}
    binary = out / 'bin'; binary.mkdir()
    for name in (args.exe.name, 'gmp-10.dll', 'build_manifest.json'):
        shutil.copy2(args.exe.resolve().parent / name, binary / name)
    manifest = json.loads((binary / 'build_manifest.json').read_text(encoding='utf-8-sig'))
    for name, digest in manifest['source_hashes'].items():
        src = ROOT / name
        assert sha(src).lower() == digest.lower(), 'Compiled source changed: ' + name
        dst = out / 'sources' / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    shutil.copy2(__file__, out / 'collector_snapshot.py')
    saved = out / 'replay.save'; saved.write_text((point + '\n') * args.curves, encoding='utf-8')
    queue = out / 'worktodo.txt'
    queue.write_text(f'ECMSTAGE2=1,2,{args.exponent},-1,replay.save,{args.b2},0,{args.curves},"{whole//n}"\n', encoding='utf-8')
    ini = out / 'ecm.ini'; ini.write_text('user=\ncomputer=\nstage2_finished=finished.txt\n', encoding='utf-8')
    profile = out / 'stage2_ecm_tune.toml'
    common = [str(binary / args.exe.name), '--ini', str(ini), '--device', '1', '--batch-mb', '256',
              '--arena-mb', '6300', '--owner-budget-mb', '640', '--short-calibration', '0',
              '--factor-only', '--relocation', '0', '--ntt-counts', '0', '--tune-profile', str(profile),
              '--tune-runtime', '1', '--tune-runtime-samples', '10', '--auto-carrier', '1',
              '--submission', str(out / 'submission.txt')]
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NTT_', 'ECM_'))
           and k not in ('CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING')}
    env['NTT_NO_PROGRESS'] = '1'

    def call(name, options):
        command = common + list(map(str, options))
        (out / (name + '.command.json')).write_text(json.dumps(command, indent=2), encoding='utf-8')
        start = time.monotonic()
        result = subprocess.run(command, cwd=out, env=env, capture_output=True, text=True, errors='replace', timeout=1800)
        (out / (name + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(name + ' failed; inspect retained log')
        return time.monotonic() - start

    print('Running automatic D/carrier queue', flush=True)
    results = out / 'internal.jsonl'
    wall = call('production', ['--worktodo', queue, '--once', '--d', '0', '--results', results,
                               '--log', out / 'engine.log', '--log-level', 'phases'])
    raw = [json.loads(line) for line in results.read_text(encoding='utf-8').splitlines()]
    rows = [normalize_result(r) for r in raw]
    assert len(rows) == args.curves and not queue.read_text(encoding='utf-8-sig').strip()
    assert all(r['status'] == 'stage2_completed' and not r['factors'] and not r['bad_factors'] for r in rows)
    assert all(r['B2'] == args.b2 and int(r['N_hex'], 16) == n and r['requested_D'] == 0 for r in rows)
    assert [r['carrier_exponent'] for r in rows[:4]] == [args.exponent, args.exponent, 0, 0]
    measured = summaries(tomllib.loads(profile.read_text(encoding='utf-8')))
    assert all(s['repeats'] <= 10 and s['verified'] == 1 for s in measured)
    families = {carrier: sorted({r['d'] for r in rows if r['carrier_exponent'] == carrier}) for carrier in (0, args.exponent)}
    assert all(len(ds) >= 2 for ds in families.values()), families
    assert max(map(len, families.values())) > 2, 'Did not progress beyond one alternate D'
    reasons = set();extended = False;observations = 0
    for raw_row, row in zip(raw, rows):
        choice = row['tune_plan']
        assert choice['D'] == row['d'] and choice['carrier_exponent'] == row['carrier_exponent']
        if not choice['runtime_probe']:
            assert choice['runtime_samples'] >= 2
        for search in choice.get('runtime_search', []):
            assert search['explored'] <= 8 and search['candidate_limit'] in (4, 8)
            extended |= search['candidate_limit'] == 8
            if choice['runtime_probe'] and search['proposed_D'] == row['d'] and search['carrier_exponent'] == row['carrier_exponent']:
                reasons.add(search['reason'])
        observation = raw_row.get('tune_observation')
        if observation:
            observations += 1
            sample = observation['sample']
            assert sample['bad'] == sample['hits'] == 0 and sample['clean'] == 1
            assert min(sample['checked'], sample['selftest_cases']) > 0
    assert reasons & {'scale', 'refine', 'region_restart', 'distant_restart'}, reasons
    assert extended and observations > 4, 'Refinement round or checked production observations missing'
    before = sha(profile)
    for carrier in (0, args.exponent):
        target = out / f'locked_{carrier}.jsonl'
        call('locked_' + str(carrier), ['--save', saved, '--curves', '1', '--b2', args.b2,
            '--d', '30030', '--carrier-exponent', carrier, '--tune-runtime', '0', '--results', target,
            '--log', out / f'locked_{carrier}.engine.log'])
        row = normalize_result(json.loads(target.read_text(encoding='utf-8')))
        assert row['d'] == 30030 and row['carrier_exponent'] == carrier and not row['factors']
    assert sha(profile) == before and all(sha(path) == digest for path, digest in identities.items())
    report = dict(complete=True, functional_only=True, CUDA_device=1, input_identities=identities,
                  replayed_curves=args.curves, B2=args.b2, arithmetic_families=families,
                  search_reasons=sorted(reasons), explicit_locks=True, disabled_profile_unchanged=True,
                  refinement_round_unlocked=extended, checked_observations=observations,
                  production_wall_seconds=wall, version=rows[0]['program']['version'])
    (out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
