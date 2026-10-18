"""Import completed, frozen full-curve benchmarks as final tune measurements.

No GPU work. Accepts the N-scaling collector's verified no-factor formal runs;
warmups, factor-producing controls and incomplete studies are not timing anchors.
Also accepts new production results containing checked tune_observation receipts.
The output is an intermediate format-8 profile; the native --tune-merge command
validates, compacts and merges repeat-weighted mean/variance when combining this evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"ecm_report"))
from result_format import normalize_result
from stage2_tune_matrix import aggregate, write_profile
import re
import statistics
import tomllib


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def literal(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    if not math.isfinite(value):
        raise ValueError('Non-finite performance value')
    return repr(value)


def import_study(path, condition, audit):
    study = json.loads(path.read_text(encoding='utf-8'))
    if not study['complete'] or study['failures']:
        raise ValueError('Need a completed study without failed runs')
    if sha(study['executable']) != study['executable_sha256']:
        raise ValueError('Frozen executable changed')
    if sha(Path(study['executable']).parent/'gmp-10.dll') != study['dll_sha256']:
        raise ValueError('Frozen arithmetic dependency changed')
    for key in ('gl_fixed_mode','outer_unroll_u','add_sub_mask'):
        if study['build'][key] != condition[key]:
            raise ValueError('Condition arithmetic differs from the frozen build')
    memory = study['memory']
    if any(condition[key] != memory[name] for key, name in
           [('batch_mb', 'batch_mb'), ('arena_mb', 'arena_mb'), ('fold_mb', 'fold_mb')]):
        raise ValueError('Condition budgets differ from the frozen benchmark')
    if condition['uuid_hex'] != study['device']['uuid'].removeprefix('GPU-').replace('-', ''):
        raise ValueError('Condition GPU differs from the benchmark')
    groups = {}
    cases = {case['id']: case for case in study['cases']}
    for row in study['runs']:
        if row['category'] not in ('timing', 'control'):
            continue
        case = cases[row['case']]
        if sha(case['save']) != case['save_sha256'] or sha(row['log']) != row['log_sha256']:
            raise ValueError('Frozen save/log changed')
        results = Path(row['command'][row['command'].index('--results')+1])
        if sha(results) != row['result_sha256']:
            raise ValueError('Frozen result changed')
        result, checks = normalize_result(row['result']), row['s4']
        if (result['bad_factors'] or int(row['wall']['clean']) != 1 or
                any(int(checks[k]) for k in ('gmp_selftest_bad', 'gmp_check_bad')) or
                not int(checks['gmp_selftest_cases']) or not int(checks['gmp_checked'])):
            raise ValueError('Incomplete arithmetic evidence')
        n = int(case['N_hex'], 16)
        if result['N_hex'] != case['N_hex'] or result['B1'] != case['B1'] or result['B2'] != row['B2']:
            raise ValueError('Result/input identity mismatch')
        if result['hits'] or result['factors']:
            if not result['factors'] or any(not 1 < int(f) < n or n % int(f) for f in result['factors']):
                raise ValueError('Invalid factor-producing control')
            continue
        text = Path(row['debug_log']).read_text(encoding='utf-8', errors='replace')
        record=audit.get(str(Path(row['log'])))
        if record is None or sha(row['debug_log'])!=record['debug_sha256']:
            raise ValueError('Debug route evidence differs from the frozen analysis audit')
        chunk = re.search(r'^giant_chunk_plan:.*?\bpoints=(\d+)', text, re.M)
        if not chunk:
            raise ValueError('Missing actual giant chunk plan')
        kind = 'mersenne' if ((n+1) & n) == 0 or result['carrier_exponent'] else 'generic'
        fold = int(row['stats']['real_batched_folddevice']['enabled'])
        frontier = int(row['stats']['scaled_frontier_device']['enabled'])
        key = (row['bits'], row['B1'], row['B2'], row['D'], result['carrier_exponent'], kind, fold, frontier)
        groups.setdefault(key, []).append((row, int(chunk[1])))
    samples = []
    for key, runs in groups.items():
        bits, b1, b2, d, carrier, kind, fold, frontier = key
        chunks = {chunk for _, chunk in runs}
        if len(chunks) != 1:
            raise ValueError('Mixed giant routes in one candidate')
        times = [float(row['wall']['total']) for row, _ in runs]
        mean = statistics.fmean(times)
        samples.append(dict(condition=0, target_bits=bits, arithmetic_bits=carrier or bits,
            carrier_exponent=carrier, modulus_kind=kind, b1=b1, b2=b2, d=d, p=runs[0][0]['P'],
            fold_resident=fold, frontier_resident=frontier,
            execution_path='resident' if fold and frontier else 'host_frontier' if fold else 'host_fold_frontier',
            program_version=runs[-1][0].get('result',{}).get('program',{}).get('version',condition.get('program_version','')),
            source='measured', benchmark_kind='frozen_full_curve_benchmark', clean=1, bad=0, hits=0,
            selftest_cases=min(int(row['s4']['gmp_selftest_cases']) for row, _ in runs),
            checked=min(int(row['s4']['gmp_checked']) for row, _ in runs), repeats=len(runs),
            mean_seconds=mean, stddev_seconds=statistics.stdev(times) if len(times)>1 else 0,
            giant_chunk_points=chunks.pop(), giant_chain_min=32768, giant_force_ladder=0))
    return samples


def import_runtime(paths, sample_limit=10):
    """Recover optional cache writes from durable new production receipts."""
    conditions, best, inputs, counts = [], {}, [], {}
    for path in paths:
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        # Match the queue receipt boundary: a torn final line is not committed.
        torn = bool(lines and not lines[-1].endswith(b'\n'))
        if torn:
            lines.pop()
        admitted = 0
        for line in lines:
            row = normalize_result(json.loads(line))
            observation = row.get('tune_observation')
            if not observation:
                continue  # Older results have no complete structured evidence.
            condition, sample = observation['condition'], dict(observation['sample'])
            if (row['status'] != 'stage2_completed' or row['hits'] or row['bad_factors'] or row['factors'] or
                    sample['source'] != 'measured' or sample['benchmark_kind'] != 'runtime_save' or
                    sample['clean'] != 1 or sample['bad'] or sample['hits'] or
                    not sample['selftest_cases'] or not sample['checked'] or sample['repeats'] != 1):
                raise ValueError('Unchecked runtime observation')
            legacy_version=condition.get('program_version')
            sample.setdefault('program_version',legacy_version or row.get('program',{}).get('version',''))
            condition={k:v for k,v in condition.items() if k not in ('program_version','updated_at')}
            n, carrier = int(row['N_hex'], 16), row['carrier_exponent']
            if (sample['target_bits'] != n.bit_length() or sample['arithmetic_bits'] != (carrier or n.bit_length()) or
                    sample['carrier_exponent'] != carrier or not 0 <= carrier <= 16384 or
                    (carrier and ((1 << carrier) - 1) % n) or
                    sample['b1'] != row['B1'] or sample['b2'] != row['B2']):
                raise ValueError('Runtime target/bounds identity mismatch')
            seconds, predicted = sample['mean_seconds'], sample['predicted_seconds']
            phases = ('shape', 'setup', 'baby', 'ftree', 'main_setup', 'inverse_setup',
                      'giant_loop', 'descent', 'accum', 'finalize')
            values = [sample[f'phase_{key}_seconds'] for key in phases]
            if (sample['phase_accounting'] != 'exclusive_engine_v1' or seconds <= 0 or predicted <= 0 or
                    any(not math.isfinite(x) or x < 0 for x in values) or
                    not math.isclose(sum(values), seconds, rel_tol=1e-9, abs_tol=1e-8) or
                    not math.isclose(sample['init_seconds'] + sample['main_seconds'], seconds, rel_tol=1e-9) or
                    not math.isclose(sample['measured_model_ratio'], seconds / predicted, rel_tol=1e-9) or
                    sample['prediction_basis'] != 'before_measurement'):
                raise ValueError('Inconsistent runtime prediction/timing boundary')
            if condition not in conditions:
                conditions.append(condition)
            sample['condition'] = conditions.index(condition)
            group = tuple(sample.get(k) for k in ('condition', 'target_bits', 'arithmetic_bits', 'carrier_exponent',
                                           'modulus_kind', 'b1', 'b2', 'd', 'fold_resident',
                                           'frontier_resident', 'execution_path', 'giant_chunk_points',
                                           'giant_chain_min', 'giant_force_ladder'))
            sample.setdefault('runtime_sample_id', row.get('queue_receipt') or hashlib.sha256(line).hexdigest())
            sample.setdefault('runtime_probe', 1)
            key = group + (sample['runtime_sample_id'],)
            if key not in best and counts.get(group, 0) < sample_limit:
                best[key] = sample
                counts[group] = counts.get(group, 0) + 1
            admitted += 1
        inputs.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                           observations=admitted, ignored_torn_line=torn))
    if not best:
        raise ValueError('No new complete runtime tune observations; older wall times alone cannot be imported')
    return conditions, aggregate(list(best.values())), inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--study', type=Path)
    mode.add_argument('--runtime-results', type=Path, action='append', help='New production JSONL; repeatable')
    parser.add_argument('--runtime-samples', type=int, default=10, help='Independent runtime observations per matching scope (2..1000)')
    parser.add_argument('--condition-profile', type=Path)
    parser.add_argument('--audit-analysis', type=Path,
                        help='Completed analysis containing the frozen raw_artifact_hashes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.runtime_samples <= 1000:
        parser.error('runtime-samples must be 2..1000')
    if args.runtime_results:
        if args.condition_profile or args.audit_analysis:
            parser.error('runtime-results carries its own condition; omit condition-profile/audit-analysis')
        conditions, samples, inputs = import_runtime(args.runtime_results, args.runtime_samples)
        receipt = dict(runtime_inputs=inputs)
    else:
        if not args.condition_profile or not args.audit_analysis:
            parser.error('study requires condition-profile and audit-analysis')
        study = args.study.resolve() / 'measurements.json'
        doc = tomllib.loads(args.condition_profile.read_text(encoding='utf-8-sig'))
        data = json.loads(study.read_text(encoding='utf-8'))
        condition = next(c for c in doc['condition'].values() if
                         c.get('batch_mb') == data['memory']['batch_mb'] and
                         c.get('arena_mb') == data['memory']['arena_mb'] and c.get('fold_mb') == data['memory']['fold_mb'])
        analysis=json.loads(args.audit_analysis.read_text(encoding='utf-8'))
        if not analysis['complete'] or Path(analysis['study']).resolve()!=study.parent:
            raise ValueError('Analysis is not the matching completed study')
        audit={str(Path(r['log'])):r for r in analysis['raw_artifact_hashes']}
        samples = import_study(study, condition, audit)

        conditions = [condition]
        receipt = dict(study=str(study.parent), manifest_sha256=sha(study), binary_sha256=data['executable_sha256'],
                       audit_analysis=str(args.audit_analysis.resolve()), audit_sha256=sha(args.audit_analysis),
                       condition_profile_sha256=sha(args.condition_profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence=args.output.with_suffix('.evidence.json')
    if args.output.exists() or evidence.exists():
        raise FileExistsError('Import requires new summary/evidence output paths')
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(write_profile(conditions,samples))
    receipt.update(summary_sha256=sha(args.output), importer_sha256=sha(__file__),
                 candidates=[{key:s[key] for key in ('target_bits','arithmetic_bits','carrier_exponent',
                     'b1','b2','d','p','execution_path','mean_seconds','stddev_seconds','repeats')} for s in samples])
    with evidence.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(f'Imported {len(samples)} final candidates: {args.output}')


if __name__ == '__main__':
    main()
