"""Import completed, frozen full-curve benchmarks as final tune measurements.

No GPU work. Accepts the N-scaling collector's verified no-factor formal runs;
warmups, factor-producing controls and incomplete studies are not timing anchors.
The output is an intermediate format-7 profile; the native --tune-merge command
validates, compacts and keeps the fastest candidate when merging this evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
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
        result, checks = row['result'], row['s4']
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
        median = statistics.median(times)
        samples.append(dict(condition=0, target_bits=bits, arithmetic_bits=carrier or bits,
            carrier_exponent=carrier, modulus_kind=kind, b1=b1, b2=b2, d=d, p=runs[0][0]['P'],
            fold_resident=fold, frontier_resident=frontier,
            execution_path='resident' if fold and frontier else 'host_frontier' if fold else 'host_fold_frontier',
            source='measured', benchmark_kind='frozen_full_curve_benchmark', clean=1, bad=0, hits=0,
            selftest_cases=min(int(row['s4']['gmp_selftest_cases']) for row, _ in runs),
            checked=min(int(row['s4']['gmp_checked']) for row, _ in runs), repeats=len(runs),
            median_seconds=median, mad_seconds=statistics.median(abs(t-median) for t in times),
            giant_chunk_points=chunks.pop(), giant_chain_min=32768, giant_force_ladder=0))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--condition-profile', type=Path, required=True)
    parser.add_argument('--audit-analysis', type=Path, required=True,
                        help='Completed analysis containing the frozen raw_artifact_hashes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
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
    lines = ['# Final medians imported from verified frozen full-curve evidence.',
             '[profile]', 'format = 7', 'algorithm_revision = 1', 'unit = "full_stage2"',
             'revision = 0', '[condition.0]']
    lines += [f'{key} = {literal(value)}' for key, value in sorted(condition.items())]
    for i, sample in enumerate(samples):
        lines += [f'\n[width.{sample["target_bits"]}.table.{i}]']
        lines += [f'{key} = {literal(value)}' for key, value in sorted(sample.items())
                  if key not in ('d','p','b2','median_seconds','mad_seconds')]
        keys = ['d','p','b2','median_seconds','mad_seconds']
        lines += ['matrix = [', '    ['+', '.join(map(literal, keys))+'],',
                  '    ['+', '.join(literal(sample[k]) for k in keys)+'],', ']']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence=args.output.with_suffix('.evidence.json')
    if args.output.exists() or evidence.exists():
        raise FileExistsError('Import requires new summary/evidence output paths')
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    receipt=dict(study=str(study.parent), manifest_sha256=sha(study),
                 binary_sha256=data['executable_sha256'], summary_sha256=sha(args.output),
                 audit_analysis=str(args.audit_analysis.resolve()), audit_sha256=sha(args.audit_analysis),
                 condition_profile_sha256=sha(args.condition_profile), importer_sha256=sha(__file__),
                 candidates=[{key:s[key] for key in ('target_bits','arithmetic_bits','carrier_exponent',
                     'b1','b2','d','p','execution_path','median_seconds','mad_seconds','repeats')} for s in samples])
    with evidence.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(f'Imported {len(samples)} final candidates: {args.output}')


if __name__ == '__main__':
    main()
