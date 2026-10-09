"""Exercise full Stage2 tune publication and measured D/carrier selection on a GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True, help='Previously verified M503 cofactor, B1=20, sigma=26')
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    exe = a.exe.resolve()
    ini = out/'bench.ini'
    ini.write_text('verbose=false\nstage2_debug_log=false\n', encoding='utf-8')
    base = [str(exe),'--ini',str(ini),'--device',str(a.device),
            '--batch-mb','256','--arena-mb','6300','--owner-budget-mb','640']

    def run(name, args, ok=True):
        proc = subprocess.run(base+args, cwd=ROOT, capture_output=True, text=True, errors='replace', timeout=600)
        (out/(name+'.log')).write_text(proc.stdout+proc.stderr, encoding='utf-8')
        if ok and proc.returncode:
            raise RuntimeError(proc.stdout+proc.stderr)
        if not ok and not proc.returncode:
            raise ValueError('Invalid tune invocation accepted: '+name)
        return proc.stdout

    profile = out/'prime.toml'
    text = run('prime_tune', ['--tune','ecm','--tune-exponents','521','--tune-d','30030,60060',
               '--tune-b2','2600000000','--tune-repeats','2','--tune-file',str(profile)])
    data = tomllib.loads(profile.read_text(encoding='utf-8'))
    assert data['summary'] == dict(complete=1, measured=2, skipped=0, failed=0)
    evidence = Path(text.split(' evidence=')[-1].strip())
    save = evidence/'m521.save'
    samples = list(data['ecm'].values())
    expected = min(samples, key=lambda s:s['median_seconds']+2*s['mad_seconds'])
    assert expected['d'] == 60060  # This fixture must demonstrate a real larger-D win.
    for source in ['binary','manifest','sha256','path']:
        assert source not in profile.read_text(encoding='utf-8').lower()
    for result in sorted(evidence.glob('case_*_*.jsonl')):
        r = json.loads(result.read_text(encoding='utf-8'))
        log = result.with_suffix('.log').read_text(encoding='utf-8')
        wall = next(x for x in log.splitlines() if x.startswith('stage2_full_wall:'))
        measured = float(wall.split(' total=')[1].split()[0])
        assert abs(r['total_seconds']-measured) < 0.000001
        assert r['hits'] == r['bad'] == 0 and r['clean'] == r['fold_resident'] == r['frontier_resident'] == 1
        assert r['selftest_cases'] and r['checked']

    def select(name, save, profile, b2, extras=()):
        text = run(name, ['--save',str(save),'--b2',str(b2),'--plan-only','--tune-profile',str(profile),*extras])
        rows = [json.loads(x) for x in text.splitlines() if x.startswith('{')]
        choice = next(x for x in rows if x.get('type') == 'tune_selection')
        plan = next(x for x in rows if x.get('type') == 'stage2_plan')
        return choice,plan

    choice, plan = select('auto_larger_d', save, profile, 2600000000)
    assert choice['selected'] and choice['D'] == plan['D'] == 60060
    assert plan['curve_workspace_memory']['initial_free_snapshot_fits']
    fixed, plan = select('fixed_d', save, profile, 2600000000, ['--d','30030'])
    assert fixed['selected'] and fixed['D'] == plan['D'] == 30030
    fallback, _ = select('unmeasured_b2', save, profile, 26000000000)
    assert not fallback['selected'] and fallback['reason'] == 'no_matching_measured_scope'
    mismatch, _ = select('policy_mismatch', save, profile, 2600000000, ['--batch-mb','64'])
    assert not mismatch['selected'] and mismatch['reason'] == 'device_or_policy_mismatch'
    debug, _ = select('debug_log_mismatch', save, profile, 2600000000, ['--debug-log-file',str(out/'debug.log')])
    assert not debug['selected'] and debug['reason'] == 'debug_log_not_calibrated'
    paired = out/'paired.toml'
    pair_text = run('paired_tune', ['--tune','ecm','--tune-save',str(a.save.resolve()),'--tune-carrier-exponent','503',
                    '--tune-d','180180','--tune-b2','26000000000','--tune-repeats','2','--tune-file',str(paired)])
    pair = tomllib.loads(paired.read_text(encoding='utf-8'))
    scopes = list(pair['ecm'].values())
    assert {s['carrier_exponent'] for s in scopes} == {0,503}
    best = min(scopes, key=lambda s:s['median_seconds']+2*s['mad_seconds'])
    chosen, _ = select('auto_arithmetic', a.save.resolve(), paired, 26000000000)
    assert chosen['selected'] and chosen['carrier_exponent'] == best['carrier_exponent']
    for exponent in [0,503]:
        fixed, plan = select('fixed_carrier_'+str(exponent), a.save.resolve(), paired, 26000000000,
                              ['--carrier-exponent',str(exponent)])
        assert fixed['selected'] and fixed['carrier_exponent'] == plan['carrier_exponent'] == exponent
    # INI config follows the same reader; explicit CLI profile has precedence.
    ini.write_text('verbose=false\nstage2_debug_log=false\nstage2_tune_profile='+str(profile)+'\n', encoding='utf-8')
    text = run('ini_profile', ['--save',str(save),'--b2','2600000000','--plan-only'])
    assert '"selected":true' in text and '"D":60060' in text
    result_file = out/'production.jsonl'
    run('production_curve', ['--save',str(save),'--b2','2600000000','--curves','1',
        '--results',str(result_file),'--log',str(out/'production_engine.log')])
    result = json.loads(result_file.read_text(encoding='utf-8'))
    assert result['tune_plan']['selected'] and result['tune_plan']['D'] == 60060
    assert result['requested_D'] == 0 and result['hits'] == result['bad_factors'] == 0
    ini.write_text('verbose=false\nstage2_debug_log=false\n', encoding='utf-8')
    retained = out/'retained.toml'
    retained.write_text('existing profile\n', encoding='utf-8')
    run('unsupported_prime', ['--tune','ecm','--tune-exponents','503','--tune-file',str(retained)], ok=False)
    assert retained.read_text(encoding='utf-8') == 'existing profile\n'
    run('ignored_ntt_budget', ['--tune','ecm','--tune-memory-mb','512','--tune-file',str(retained)], ok=False)
    run('invalid_pair', ['--tune','ecm','--tune-save',str(a.save.resolve()),'--tune-carrier-exponent','521',
                         '--tune-file',str(retained)], ok=False)
    assert retained.read_text(encoding='utf-8') == 'existing profile\n'
    run('all_skipped', ['--tune','ecm','--tune-exponents','521','--tune-d','30030',
                        '--tune-b2','100','--tune-repeats','1','--tune-file',str(retained)], ok=False)
    assert retained.read_text(encoding='utf-8') == 'existing profile\n'
    assert list(out.glob('retained.toml.partial.*'))
    run('effort_limit', ['--tune','ecm','--tune-exponents','521','--tune-d','30030',
                         '--tune-b2','2600000000','--tune-max-batches','1','--tune-file',str(retained)], ok=False)
    assert retained.read_text(encoding='utf-8') == 'existing profile\n'
    wide_prime = out/'wide_prime.toml'
    catalogue = [107,127,521,607,1279,2203,2281,3217,4253,4423,9689,9941,11213]
    run('wide_prime', ['--tune','ecm','--tune-exponents',','.join(map(str,catalogue)),'--tune-d','60060',
                       '--tune-b2','2600000000','--tune-repeats','1','--tune-file',str(wide_prime)])
    high = tomllib.loads(wide_prime.read_text(encoding='utf-8'))
    high_sample = next(s for s in high['ecm'].values() if s['target_bits'] == 11213)
    assert high['summary']['measured'] == 13 and high['summary']['skipped'] == 0
    assert {s['target_bits'] for s in high['ecm'].values()} == set(catalogue)
    assert high['profile']['format'] == 3 and 'xadd6' in high['policy']['environment']
    assert high_sample['target_bits'] == 11213 and high_sample['arithmetic_bits'] == 11213
    assert high_sample['hits'] == high_sample['bad'] == 0 and high_sample['checked'] > 0
    result = dict(prime_tune_shapes=2, paired_tune_shapes=2, curves_per_shape=3,
                  measured_larger_d=60060, measured_carrier_choice=best['carrier_exponent'],
                  production_selection_verified=True, failed_publication_retains_existing=True,
                  high_width_prime_bits=11213, prime_catalogue_curves=26, effort_budget_rejection=True,
                  prime_evidence=str(evidence), paired_evidence=pair_text.split(' evidence=')[-1].strip())
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
