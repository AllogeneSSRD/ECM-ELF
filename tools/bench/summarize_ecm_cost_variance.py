"""Describe repeated Auto B2 measurements without discarding runs or refitting.

Median errors here are diagnostics, never replacements for the per-run release
gate. NVML/host state covers the whole process, including cold initialization;
it cannot establish the cause or the phase of a scheduling delay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from calibrate_stage2_d import parse
from ecm_cost_coverage import check_study_coverage
from ecm_cost_model import admits, observed, predict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = list(values)
    if not values:
        return None
    return dict(count=len(values), min=min(values), median=statistics.median(values),
                max=max(values), mean=statistics.mean(values))


def state_summary(row):
    state = row.get('state', {})
    samples = state.get('samples', [])
    result = {key: distribution(s[key] for s in samples if s.get(key) is not None)
              for key in ('gpu', 'sm_mhz', 'mem_mhz', 'temperature_c',
                          'power_mw', 'system_cpu_percent', 'used')}
    result.update(samples=len(samples), monitor_errors=state.get('errors', []),
                  window='whole_process_including_startup')
    return result


def summarize(study, model):
    if not study.get('complete'):
        raise ValueError('Calibration is incomplete')
    coverage = check_study_coverage(study)
    if study['identity'] != model['identity']:
        raise ValueError('Model/calibration identity differs')
    groups = {}
    for row in study['stage2']:
        if sha(row['log']) != row['log_sha256']:
            raise ValueError('Raw log changed: ' + row['name'])
        if parse(Path(row['log']).read_text(encoding='utf-8')) != row['phases']:
            raise ValueError('Stored phases differ: ' + row['name'])
        scope = [s for s in model['stage2'] if s['owner_mb'] == row['owner_mb']
                 and s['regime'] == row['regime'] and admits(s, row['features'])]
        if len(scope) != 1:
            raise ValueError('Need exactly one matching scope: ' + row['name'])
        key = (row['bits'], row['D'], row['B2'], row['owner_mb'], row['kind'], row['regime'])
        groups.setdefault(key, (scope[0], []))[1].append(row)
    repeats = []
    per_scope = {}
    for key, (scope, rows) in sorted(groups.items()):
        bits, d, b2, owner, kind, regime = key
        prediction, _ = predict(bits, d, b2, scope['rates'], model['chain_min'])
        fast, slow = min(rows, key=lambda r: r['phases']['full']), max(rows, key=lambda r: r['phases']['full'])
        fast_phases, slow_phases = observed(fast), observed(slow)
        delta = slow['phases']['full'] - fast['phases']['full']
        phase_deltas = {k: slow_phases[k] - fast_phases[k] for k in fast_phases}
        # Full/init/main are independently printed to six decimal places.
        roundoff = delta - sum(phase_deltas.values())
        if abs(roundoff) > 2.1e-6:
            raise ValueError('Exclusive phase deltas do not sum to full time: ' + fast['name'])
        actual = statistics.median(r['phases']['full'] for r in rows)
        errors = [100 * (prediction['full'] / r['phases']['full'] - 1) for r in rows]
        # For one deterministic prediction x and two times a <= b, the best
        # possible worst relative error is (b-a)/(b+a), at x=2ab/(a+b).
        # More repetitions cannot make two already incompatible bands overlap.
        fastest, slowest = fast['phases']['full'], slow['phases']['full']
        error_floor = 100 * (slowest-fastest) / (slowest+fastest)
        report = dict(bits=bits, D=d, B2=b2, owner_mb=owner, kind=kind, regime=regime,
                      scope_id=scope['id'], repeats=len(rows),
                      full_seconds=distribution(r['phases']['full'] for r in rows),
                      slow_over_fast=slow['phases']['full'] / fast['phases']['full'],
                      delta_seconds=delta, exclusive_phase_deltas_seconds=phase_deltas,
                      timing_roundoff_seconds=roundoff,
                      predicted_seconds=prediction['full'], raw_error_percent=errors,
                      best_possible_raw_max_abs_percent=error_floor,
                      single_prediction_10_percent_possible=error_floor <= 10,
                      diagnostic_median_error_percent=100 * (prediction['full'] / actual - 1),
                      fast_name=fast['name'], slow_name=slow['name'],
                      samples=[dict(name=r['name'], process_seconds=r['process_seconds'],
                                    phases=observed(r), oracle=r.get('oracle'),
                                    state=state_summary(r), log=r['log'], log_sha256=r['log_sha256'])
                               for r in rows])
        repeats.append(report)
        per_scope.setdefault(scope['id'], []).append(report)
    scopes = []
    for scope in model['stage2']:
        groups_here = per_scope[scope['id']]
        hold = [r for r in groups_here if r['kind'] == 'holdout']
        raw = [abs(e) for r in hold for e in r['raw_error_percent']]
        scopes.append(dict(id=scope['id'], bits=scope['bits'], D=scope['d_values'],
                           owner_mb=scope['owner_mb'], regime=scope['regime'],
                           release_holdout_passed=scope['usable'],
                           holdout_raw_max_abs_percent=max(raw),
                           diagnostic_holdout_median_max_abs_percent=max(abs(r['diagnostic_median_error_percent']) for r in hold),
                           worst_repeat_ratio=max(r['slow_over_fast'] for r in groups_here)))
    return dict(schema=1, kind='repeat_variance_diagnostic', complete=True,
                discarded_runs=0, refitted=False, changes_release_gate=False,
                state_is_causal_evidence=False, schedule_coverage=coverage,
                total_curves=len(study['stage2']), exact_repeated_points=len(repeats),
                scope_count=len(scopes), holdout_passed=sum(s['release_holdout_passed'] for s in scopes),
                holdout_failed=[s for s in scopes if not s['release_holdout_passed']],
                repeated_point_spread=distribution(r['slow_over_fast'] for r in repeats),
                points_over_10_percent_repeat_spread=sum(r['slow_over_fast'] > 1.1 for r in repeats),
                points_incompatible_with_single_prediction_10_percent=sum(not r['single_prediction_10_percent_possible'] for r in repeats),
                holdout_points_incompatible_with_single_prediction_10_percent=sum(r['kind'] == 'holdout' and not r['single_prediction_10_percent_possible'] for r in repeats),
                scopes=scopes, repeated_points=repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.study.resolve(), args.profile.resolve()}:
        raise ValueError('Output must not overwrite an input')
    study = json.loads(args.study.read_text(encoding='utf-8'))
    model = json.loads(args.profile.read_text(encoding='utf-8'))
    if model['source_sha256'] != sha(args.study):
        raise ValueError('Model source hash differs')
    if model['model_code_sha256'] != sha(Path(__file__).with_name('ecm_cost_model.py')):
        raise ValueError('Cost implementation changed')
    result = summarize(study, model)
    result['identity'] = dict(study=str(args.study.resolve()), study_sha256=sha(args.study),
                              profile=str(args.profile.resolve()), profile_sha256=sha(args.profile),
                              tool_sha256=sha(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('repeated_points', 'scopes', 'identity')}, indent=2))


if __name__ == '__main__':
    main()
