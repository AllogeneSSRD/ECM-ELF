"""Check Auto benefit harness references without running CUDA or compilation."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
import validate_stage2_tune_selection as fixed
from analyze_stage2_tune_workload import one_json, workload
from validate_stage2_tune_auto_profit import benefit, profit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('profile', 'training-plans', 'holdout-dir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    identities = {str(path.resolve()): sha(path) for path in (a.profile, Path(__file__))}
    for module in (fixed, fixed.components, sys.modules['analyze_stage2_tune_workload'],
                   sys.modules['stage2_tune_route_cost'], sys.modules['validate_stage2_tune_auto_profit']):
        path = Path(module.__file__).resolve()
        identities[str(path)] = sha(path)
    profile = tomllib.loads(a.profile.read_text(encoding='utf-8-sig'))
    groups = {}
    for sample in profile['ecm'].values():
        groups.setdefault(tuple(sample[k] for k in fixed.SCOPE), []).append(sample)
    measurements, models = fixed.component_reference(profile, groups, a.training_plans, identities)
    assert len(models) == len(groups) == 4
    report_path = a.holdout_dir / 'result.json'
    identities[str(report_path.resolve())] = sha(report_path)
    recorded = json.loads(report_path.read_text())
    assert recorded['complete'] and recorded['profile_sha256'] == sha(a.profile)
    queries = 0
    for i, holdout in enumerate(recorded['holdouts']):
        for candidate in holdout['candidates']:
            path = a.holdout_dir / f"holdout_{i}_d{candidate['d']}_c{candidate['carrier']}_plan.console.log"
            identities[str(path.resolve())] = sha(path)
            plan = next(json.loads(line) for line in path.read_text().splitlines()
                        if line.startswith('{') and json.loads(line).get('type') == 'stage2_plan')
            key, group = next((k, g) for k, g in groups.items()
                              if k[-1] == candidate['d'] and k[2] == candidate['carrier'])
            result = fixed.predict(group, holdout['b2'], profile['profile'].get('prediction_model'),
                                   models[key], plan, measurements)
            assert result['model'] == 'phase_ntt_loop_v1'
            for field in ('seconds', 'rank'):
                assert math.isclose(result[field], candidate['prediction'][field], rel_tol=1e-10)
            assert profit(plan['B1'], plan['B2'], .03204918749997887, result['rank'], 1) > 0
            queries += 1
    assert queries == 8
    exact = 0
    for key, group in groups.items():
        for sample in group:
            result = fixed.predict(group, sample['b2'], profile['profile'].get('prediction_model'), models[key])
            assert result['model'] == 'measured_exact_scope_v1' and result['seconds'] == sample['median_seconds']
            exact += 1
    legacy = copy.deepcopy(profile)
    legacy['profile']['format'] = 3
    legacy['profile'].pop('component_model')
    legacy['summary'].pop('ntt_measured')
    legacy.pop('ntt')
    assert fixed.component_reference(legacy, groups, None, {}) == ({}, {})
    refusals = 0

    def reject(plans):
        nonlocal refusals
        try:
            fixed.component_reference(profile, groups, plans, {})
        except AssertionError:
            refusals += 1
        else:
            raise AssertionError('invalid training plans accepted')

    empty = out / 'empty'
    empty.mkdir()
    reject(empty)
    duplicate = out / 'duplicate'
    duplicate.mkdir()
    first = next(a.training_plans.glob('case_*.plan.jsonl'))
    shutil.copyfile(first, duplicate / 'case_1.plan.jsonl')
    shutil.copyfile(first, duplicate / 'case_duplicate.plan.jsonl')
    reject(duplicate)
    mismatch = out / 'mismatch'
    mismatch.mkdir()
    bad = one_json(first)
    bad['tree_workspace']['batch_bytes'] += 1
    (mismatch / first.name).write_text(json.dumps(bad) + '\n')
    reject(mismatch)
    missing = copy.deepcopy(profile)
    row = workload(one_json(first))[0]
    del missing['ntt']['length_' + str(row['length'])]['slices_' + str(row['slices'])]
    missing['summary']['ntt_measured'] -= 1
    _, partial = fixed.component_reference(missing, groups, a.training_plans, {})
    assert len(partial) < len(models)
    assert math.isclose(benefit(20, 2600000000),
                        .11343 + .88657 * (math.log10(2600000000 / 20) / 2) **
                        (1.96617 - .06781 * math.log10(20)), rel_tol=1e-14)
    assert all(sha(Path(path)) == digest for path, digest in identities.items())
    report = dict(complete=True, groups=len(models), queries=queries, exact_anchors=exact,
                  bad_plan_refusals=refusals, legacy_fallback=True, missing_shape_ineligible=True,
                  identities=identities)
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'identities'}))


if __name__ == '__main__':
    main()
