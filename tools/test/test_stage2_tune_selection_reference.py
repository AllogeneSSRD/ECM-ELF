"""Exercise the independent cost reference used by full-curve rank validation."""
import argparse
import importlib.util
import json
import hashlib
import math
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if bool(args.profile) != bool(args.evidence):
        parser.error('provide profile and completed evidence together')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('reference', ROOT/'tools/bench/validate_stage2_tune_selection.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    b2s = [2600000000, 5200000000, 10400000000]
    samples = [dict(b2=b2, d=60060, giant_points=b2//60060+2,
                    median_seconds=.5+(b2//60060+2)*.00001,
                    mad_seconds=.001, fold_resident=1, frontier_resident=1) for b2 in b2s]
    expected = .5+(7800000000//60060+2)*.00001
    actual = module.predict(samples, 7800000000, True)
    assert abs(actual['seconds']-expected) < 1e-12
    assert abs(actual['rank']-(expected+.002)) < 1e-12
    assert actual['fit_relative_error'] < 1e-12
    exact = module.predict(samples, b2s[1], False)
    assert exact['seconds'] == samples[1]['median_seconds']
    assert exact['model'] == 'measured_exact_scope_v1'
    assert module.predict(samples, 7800000000, False) is None
    assert module.predict(samples[:2], 7800000000, True) is None
    assert module.predict([], 7800000000, True) is None
    assert module.predict(samples, b2s[0]-1, True) is None
    assert module.predict(samples, b2s[-1]+1, True) is None
    assert module.predict(samples[::-1], 7800000000, True) == actual
    bad = [dict(s) for s in samples]
    bad[1]['fold_resident'] = 0
    assert module.predict(bad, 7800000000, True) is None
    assert module.predict(bad, b2s[1], True) is None
    bad = [dict(s) for s in samples]
    bad[1]['giant_points'] = bad[0]['giant_points']
    assert module.predict(bad, 7800000000, True) is None
    bad = [dict(s) for s in samples]
    bad[1]['median_seconds'] *= 1.2
    assert module.predict(bad, 7800000000, True) is None
    for intercept, slope in [(-.5, .0001), (10, -.00001)]:
        bad = [dict(s, median_seconds=intercept+slope*s['giant_points']) for s in samples]
        assert module.predict(bad, 7800000000, True) is None
    report = dict(linear_reference=True, exact_priority=True, model_optin_required=True,
                  no_extrapolation=True, fit_rejections=True, fixed_time_error_limit=.08,
                  fixed_rank_loss_limit=.05)
    if args.evidence:
        evidence = json.loads(args.evidence.read_text(encoding='utf-8'))
        assert evidence['complete'] and evidence['repeats'] >= 2
        assert hashlib.sha256(args.profile.read_bytes()).hexdigest() == evidence['profile_sha256']
        profile = tomllib.loads(args.profile.read_text(encoding='utf-8'))
        count = 0
        for holdout in evidence['holdouts']:
            assert holdout['complete'] and holdout['rank_loss'] <= module.RANK_LOSS_LIMIT
            assert holdout['automatic_relative_error'] <= module.TIME_ERROR_LIMIT
            for candidate in holdout['candidates']:
                group = [s for s in profile['ecm'].values()
                         if s['target_bits']==evidence['target_bits'] and s['b1']==evidence['b1']
                         and s['d']==candidate['d'] and s['carrier_exponent']==candidate['carrier']
                         and s['fold_resident'] and s['frontier_resident']]
                prediction = module.predict(group,holdout['b2'],True)
                assert prediction and prediction['model']==candidate['selection']['model']
                assert math.isclose(prediction['seconds'],candidate['selection']['estimated_seconds'],rel_tol=1e-10)
                assert math.isclose(prediction['rank'],candidate['selection']['rank_seconds'],rel_tol=1e-10)
                assert candidate['relative_error'] <= module.TIME_ERROR_LIMIT
                count += 1
        report.update(recorded_native_candidates=count,current_reference_matches=True,
                      evidence_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest())
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
