"""Verify B2 prediction and independent full-curve holdouts in production."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--rejected-profile', type=Path, required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [a.exe.resolve(), a.profile.resolve(), a.save.resolve(), a.rejected_profile.resolve()]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    identities = {str(path):sha(path) for path in paths}
    profile = tomllib.loads(a.profile.read_text(encoding='utf-8'))
    assert profile['profile']['prediction_model']=='giant_route_cost_v2'
    sources = [s for s in profile['ecm'].values() if s['d']==60060]
    assert len(sources) == 5
    assert {s['target_bits'] for s in sources} == {521}
    assert {s['b1'] for s in sources} == {20}
    assert all(s['giant_ladder_steps']==0 for s in sources), 'this fixture expects chain-only calibration'
    ini = out/'bench.ini'
    ini.write_text('verbose=false\nstage2_debug_log=false\n', encoding='utf-8')
    common = [str(a.exe.resolve()), '--ini', str(ini), '--save', str(a.save.resolve()),
              '--device', str(a.device), '--batch-mb', str(profile['policy']['batch_mb']),
              '--arena-mb', str(profile['policy']['arena_mb']), '--owner-budget-mb', str(profile['policy']['fold_mb']),
              '--tune-profile', str(a.profile.resolve()), '--log-level', 'quiet']

    def run(name, options):
        proc = subprocess.run(common+options, cwd=ROOT, capture_output=True, text=True,
                              errors='replace', timeout=300)
        (out/(name+'.log')).write_text(proc.stdout+proc.stderr, encoding='utf-8')
        assert proc.returncode == 0, (name,proc.stdout,proc.stderr)
        return [json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{')]

    def select(name,b2,extras=()):
        rows = run(name,['--b2',str(b2),'--plan-only',*extras])
        return (next(x for x in rows if x.get('type')=='tune_selection'),
                next(x for x in rows if x.get('type')=='stage2_plan'))

    exact,_ = select('exact',2600000000)
    assert exact['selected'] and exact['model']=='measured_exact_scope_v1'
    assert exact['D']==60060 and 'estimated_seconds' not in exact
    missing = out/'no_prediction.toml'
    missing.write_text(a.profile.read_text(encoding='utf-8').replace('prediction_model = "giant_route_cost_v2"\n',''),encoding='utf-8')
    legacy,_ = select('missing_optin',5200000000,['--tune-profile',str(missing)])
    assert not legacy['selected']
    rejected,_ = select('rejected_three_anchor_fit',5200000000,['--tune-profile',str(a.rejected_profile.resolve())])
    assert not rejected['selected']
    for name,bound in [('above',52000000000),('below',1000000000)]:
        choice,_ = select(name,bound)
        assert not choice['selected']
    lock,_ = select('fixed_insufficient_d',5200000000,['--d','30030'])
    assert not lock['selected']
    ordinary,_ = select('fixed_ordinary',5200000000,['--carrier-exponent','0'])
    assert ordinary['selected'] and ordinary['carrier_exponent']==0
    mx = sum(s['giant_points'] for s in sources)/len(sources)
    my = sum(s['median_seconds'] for s in sources)/len(sources)
    rate = sum((s['giant_points']-mx)*(s['median_seconds']-my) for s in sources)/sum((s['giant_points']-mx)**2 for s in sources)
    fixed = my-rate*mx
    holdouts = []
    for index,b2 in enumerate([5200000000,13000000000,20000000000]):
        assert b2 not in {s['b2'] for s in sources}
        choice,plan = select('holdout_plan_'+str(index),b2)
        assert choice['selected'] and choice['model']=='giant_route_cost_v2'
        assert choice['D']==plan['D']==60060 and plan['B2']==b2 and plan['I']==b2//60060+2
        assert choice['fit_samples']==5 and choice['fit_max_relative_error']<=.08
        assert choice['fit_b2_min']<b2<choice['fit_b2_max']
        assert plan['curve_workspace_memory']['initial_free_snapshot_fits']
        assert abs(choice['estimated_seconds']-(fixed+rate*plan['I'])) < 1e-10
        assert choice['rank_seconds']>=choice['estimated_seconds'] and 'median_seconds' not in choice
        result_path,log = out/('curve_'+str(index)+'.jsonl'),out/('curve_'+str(index)+'.log')
        run('holdout_driver_'+str(index),['--b2',str(b2),'--curves','1','--results',str(result_path),'--log',str(log)])
        result = json.loads(result_path.read_text(encoding='utf-8'))
        assert result['status']=='stage2_completed' and result['requested_D']==0
        assert result['B2']==b2 and result['tune_plan']['model']=='giant_route_cost_v2'
        assert result['hits']==result['bad_factors']==0
        text = log.read_text(encoding='utf-8')
        assert 'gmp_selftest_bad=0' in text and 'gmp_check_bad=0' in text
        assert re.search(r'real_batched_folddevice: requested=1 enabled=1 fallback=none\b',text)
        assert re.search(r'scaled_frontier_device: requested=1 enabled=1\b',text)
        wall = next(x for x in text.splitlines() if x.startswith('stage2_full_wall:'))
        assert 'clean=1' in wall
        actual = float(re.search(r'\btotal=([0-9.]+)',wall).group(1))
        error = abs(actual-choice['estimated_seconds'])/actual
        assert error<=.08, (b2,error,actual,choice)
        holdouts.append(dict(b2=b2,predicted=choice['estimated_seconds'],actual=actual,relative_error=error,selection=choice))
    assert identities == {str(path):sha(path) for path in paths}
    report = dict(binary_sha256=sha(a.exe.resolve()),profile_sha256=sha(a.profile.resolve()),device=a.device,
                  target_bits=521,b1=20,d=60060,holdouts=holdouts,arithmetic_bad=0,exact_priority=True,
                  unsafe_fit_refused=True,no_extrapolation=True,missing_optin_retains_legacy=True,
                  total_scope='stage2_full_wall.total; Stage1/process/planning/publication excluded')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
