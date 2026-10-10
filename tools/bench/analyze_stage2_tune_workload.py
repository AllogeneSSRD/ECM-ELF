"""Extract exact NTT workload features from frozen Stage2 tune plans (CPU only).

Logical polynomial pairs, physical chunks and their lengths are separate units.
An optional single-slice NTT profile supplies comparison features, never an ECM
time prediction. Raw identities go in evidence.json, not performance TOML.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tomllib

PHASES = ('ftree', 'gtrees', 'fold', 'descent', 'inverse')
ENGINE_PHASES = ('shape', 'setup', 'baby', 'ftree', 'main_setup', 'inverse_setup',
                 'giant_loop', 'descent', 'accum', 'finalize')
TIMERS = ('init_seconds', 'main_seconds', 'giant_seconds', 'gtrees_seconds',
          'fold_seconds', 'descent_seconds', 'inverse_seconds', 'accum_seconds')
IDENTITY = ('uuid_hex', 'sm_major', 'sm_minor', 'cuda_runtime', 'cuda_driver',
            'gl_fixed_mode', 'outer_unroll_u')
MAX_BYTES = 64 * 1048576


def integer(x, minimum=0):
    if type(x) is not int or not minimum <= x <= (1 << 63)-1:
        raise ValueError('invalid bounded integer')
    return x


def positive(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        raise ValueError('invalid positive finite time')
    return x


def read(path):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('input exceeds 64 MiB')
    return path.read_bytes()


def one_json(path):
    rows = read(path).decode('utf-8-sig').splitlines()
    if len(rows) != 1:
        raise ValueError('one structured record per file required')
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON field')
            result[key] = value
        return result
    return json.loads(rows[0], object_pairs_hook=unique)


def workload(plan):
    """O(compressed requests), including repeated middle G batches without expansion."""
    program, s4, tree = plan['request_program'], plan['s4_memory'], plan['tree_workspace']
    if (program['version'] != 2 or not program['valid'] or not program['supported'] or
            not program['residency_required'] or not program['full_fold_degree_required'] or
            not program['no_eviction_model'] or program['process_peak_complete'] or
            program['admission_model'] or not s4['valid'] or not tree['supported']):
        raise ValueError('unsupported conditional request workload')
    shapes = {}
    for shape in s4['shapes']:
        operand, n, slots = (integer(shape[k], 1) for k in ('operand', 'N', 'slots'))
        if operand in shapes or n & (n-1) or n > 1 << 27 or slots != 2*operand-1:
            raise ValueError('invalid/duplicate packing shape')
        shapes[operand] = (n, slots)
    batch = integer(tree['batch_bytes'], 1)
    limit = integer(tree['chunk_max'])
    physical = tree['physical_chunks']
    if type(physical) is not bool:
        raise ValueError('invalid physical chunk flag')
    buffers = integer(plan['workspace_buffers'], 1) if physical else 3
    if buffers not in (2, 3):
        raise ValueError('unsupported workspace buffer count')
    hist = defaultdict(lambda: dict(request_occurrences=0, calls=0, pairs=0, output_coefficients=0))
    totals = [dict(groups=0, pairs=0, chunks=0) for _ in PHASES]
    for block in program['blocks']:
        repeat = integer(block['repeat'], 1)
        for req in block['requests']:
            phase = integer(req['phase'])
            ma, mb, pairs, first, count = (integer(req[k], int(k != 'first'))
                                         for k in ('ma', 'mb', 'pairs', 'first', 'count'))
            if phase >= len(PHASES) or first+count > ma+mb-1:
                raise ValueError('invalid phase/output interval')
            if req['input'] not in ('host', 'tree_raw', 'fold_owner', 'frontier_owner'):
                raise ValueError('unsupported request input')
            n, slots = shapes[max(ma, mb)]
            slices = pairs
            bytes_per_slice = 8*(buffers*n+slots+(2 if physical else 0))
            while slices and slices*bytes_per_slice > batch:
                slices //= 2
            slices = max(1, slices)
            if limit:
                slices = min(slices, limit)
            full, tail = divmod(pairs, slices)
            totals[phase]['groups'] += repeat
            totals[phase]['pairs'] += pairs*repeat
            totals[phase]['chunks'] += (full+bool(tail))*repeat
            for c, calls in ((slices, full), (tail, int(bool(tail)))):
                if not calls:
                    continue
                entry = hist[(phase, n, c)]
                entry['request_occurrences'] += repeat
                entry['calls'] += calls*repeat
                entry['pairs'] += calls*c*repeat
                entry['output_coefficients'] += calls*c*count*repeat
    for i, observed in enumerate(program['phases']):
        if i >= len(PHASES) or observed['phase'] != PHASES[i]:
            raise ValueError('invalid phase table order')
        if any(totals[i][k] != integer(observed[k]) for k in totals[i]):
            raise ValueError('independent workload differs from native phase totals')
    if len(program['phases']) != len(PHASES):
        raise ValueError('missing native phase totals')
    rows = []
    for (phase, n, slices), counts in sorted(hist.items()):
        row = dict(phase=PHASES[phase], length=n, slices=slices, **counts)
        row['field_words'] = n*counts['pairs']
        row['field_nlog2n'] = n*(n.bit_length()-1)*counts['pairs']
        for x in row.values():
            if type(x) is int:
                integer(x)
        rows.append(row)
    return rows


def ntt_samples(profile):
    if (profile['profile']['format'] != 1 or profile['profile']['unit'] != 'field_convolution' or
            profile['summary']['failed'] or not profile['summary']['usable']):
        raise ValueError('incomplete/unsupported NTT profile')
    measured = {}
    for key, s in profile['ntt'].items():
        n = integer(s['length'], 1)
        if n & (n-1) or key != f'length_{n}':
            raise ValueError('invalid NTT length table')
        if s['status'] == 'skipped_memory':
            continue
        if (s['status'] != 'measured' or s['unit'] != 'field_convolution' or s['batch'] != 1 or
                s['bad'] or s['verified_words_per_sample'] != n or
                len(s['seconds']) != profile['profile']['repeats']):
            raise ValueError('unchecked NTT benchmark')
        seconds = [positive(x) for x in s['seconds']]
        median = statistics.median(seconds)
        if not math.isclose(positive(s['median_seconds']), median, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError('inconsistent NTT median')
        if not math.isclose(positive(s['conv_iter_per_s']), 1/median, rel_tol=1e-9):
            raise ValueError('inconsistent NTT throughput')
        measured[n] = median
    if len(measured) != integer(profile['summary']['measured'], 1):
        raise ValueError('inconsistent NTT summary')
    return measured


def paired_trials(plan, trials):
    if not trials:
        raise ValueError('formal receipts required')
    contracts = {t.get('phase_accounting') for t in trials}
    if len(contracts) != 1 or contracts - {None, 'exclusive_engine_v1'}:
        raise ValueError('mixed/unsupported engine phase contract')
    exclusive = 'exclusive_engine_v1' in contracts
    for t in trials:
        if (t['d'] != plan['D'] or t['p'] != plan['P'] or t['giant_points'] != plan['I'] or
                t['hits'] or t['bad'] or t['clean'] != 1 or t['fold_resident'] != 1 or
                t['frontier_resident'] != 1 or not t['selftest_cases'] or not t['checked']):
            raise ValueError('different/unchecked/nonresident executed curve')
        total, init, main = (positive(t[k]) for k in ('total_seconds', 'init_seconds', 'main_seconds'))
        if not math.isclose(total, init+main, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('broken engine timing boundary')
        for key in TIMERS:
            if not math.isfinite(t[key]) or t[key] < 0:
                raise ValueError('invalid legacy phase timer')
        if exclusive:
            values = [t['phase_'+name+'_seconds'] for name in ENGINE_PHASES]
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                   not math.isfinite(x) or x < 0 for x in values):
                raise ValueError('invalid exclusive phase timer')
            if not (math.isclose(sum(values[:4]), init, rel_tol=1e-9, abs_tol=1e-9) and
                    math.isclose(sum(values[4:]), main, rel_tol=1e-9, abs_tol=1e-9)):
                raise ValueError('exclusive phases do not conserve engine time')
        elif any(k.startswith('phase_') for k in t):
            raise ValueError('exclusive phases missing contract')
    # Paired observations, not sums of independently computed phase medians.
    result = {key: [t[key] for t in trials] for key in ('total_seconds',)+TIMERS}
    if exclusive:
        result['phase_accounting'] = 'exclusive_engine_v1'
        result.update({'phase_'+name+'_seconds': [t['phase_'+name+'_seconds'] for t in trials]
                       for name in ENGINE_PHASES})
    return result


def published_costs(sample, timers):
    """Cross-check raw phase receipts; worker intervals exist in the parent profile."""
    result = {}
    if sample.get('phase_accounting') != timers.get('phase_accounting'):
        raise ValueError('published/raw phase accounting differs')
    if 'phase_accounting' in timers:
        for timer, samples in [('init_seconds', 'init_samples'), ('main_seconds', 'main_samples'),
                               *[('phase_'+name+'_seconds', 'phase_'+name+'_samples')
                                 for name in ENGINE_PHASES]]:
            if sample[samples] != timers[timer] or not math.isclose(
                    sample[timer], statistics.median(timers[timer]), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('published/raw paired phases differ')
    elif any(k.startswith('phase_') or k in ('init_samples', 'main_samples') for k in sample):
        raise ValueError('published phases missing contract')
    if 'worker_accounting' in sample:
        if sample['worker_accounting'] != 'spawn_wait_exit_v1':
            raise ValueError('unsupported worker accounting')
        for key in ('worker', 'worker_overhead'):
            values = sample[key+'_samples']
            if len(values) != len(timers['total_seconds']) or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or
                    not math.isfinite(x) or x < 0 for x in values):
                raise ValueError('invalid paired worker interval')
            if not math.isclose(sample[key+'_seconds'], statistics.median(values),
                                rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('inconsistent paired worker median')
            result[key+'_samples'] = values
            result[key+'_seconds'] = sample[key+'_seconds']
        for wall, engine, residual in zip(sample['worker_samples'], timers['total_seconds'],
                                         sample['worker_overhead_samples']):
            if not math.isclose(wall, engine+residual, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('worker interval does not conserve engine plus residual')
        spread = statistics.median(abs(x-sample['worker_seconds']) for x in sample['worker_samples'])
        if not math.isclose(sample['worker_mad_seconds'], spread, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('inconsistent worker MAD')
        result.update(worker_accounting=sample['worker_accounting'], worker_mad_seconds=spread)
    elif any(k.startswith('worker_') for k in sample):
        raise ValueError('worker costs missing contract')
    return result


def toml_value(value):
    if type(value) is bool:
        return str(value).lower()
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return '['+', '.join(toml_value(x) for x in value)+']'
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError('nonfinite output')
        return repr(value)
    raise ValueError('unsupported performance field')


def table(name, values):
    return '\n['+name+']\n'+''.join(k+' = '+toml_value(v)+'\n' for k, v in values.items())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True, help='native full ECM tune evidence directory')
    p.add_argument('--profile', type=Path, required=True, help='matching full ECM performance TOML')
    p.add_argument('--ntt-profile', type=Path, help='optional single-slice NTT performance TOML')
    p.add_argument('--output', type=Path, required=True, help='fresh ignored output directory')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    evidence = dict(complete=False, sources=[], cases=[])
    sources = {}

    def source(path):
        data = read(path)
        digest = hashlib.sha256(data).hexdigest()
        sources[path.resolve()] = digest
        return data

    try:
        source(Path(__file__))
        profile = tomllib.loads(source(a.profile).decode('utf-8-sig'))
        if (profile['profile']['format'] not in (2, 3) or profile['summary']['failed'] or
                not profile['summary']['complete'] or profile['profile']['unit'] != 'full_stage2'):
            raise ValueError('complete full ECM profile required')
        repeats = integer(profile['profile']['repeats'], 1)
        measured_ntt = {}
        identity_match = False
        if a.ntt_profile:
            ntt = tomllib.loads(source(a.ntt_profile).decode('utf-8-sig'))
            measured_ntt = ntt_samples(ntt)
            identity_match = all(profile['device'][k] == ntt['device'][k] for k in IDENTITY)
            if not identity_match:
                raise ValueError('NTT/full ECM device or arithmetic identity mismatch')
        text = '# Stage2 workload features; not a production cost profile.\n'
        text += table('profile', dict(format=1, unit='logical_field_convolutions',
            ranking_qualified=False, legacy_timer_contract='legacy_overlapping',
            phase_timer_contract='per_sample_optional',
            timing_boundary='stage2_engine_init_plus_main',
            ntt_single_slice_identity_match=identity_match,
            ntt_policy_qualified=False, ntt_feature_is_time_prediction=False))
        text += table('device', profile['device'])
        text += table('policy', {k: v for k, v in profile['policy'].items() if not isinstance(v, dict)})
        env = profile['policy'].get('environment', {})
        if isinstance(env, dict):
            text += table('policy.environment', env)
        consumed = set()
        for path in sorted(a.evidence.glob('case_*.plan.jsonl'), key=lambda x: int(x.name.split('_')[1].split('.')[0])):
            source(path)
            plan = one_json(path)
            if not plan['curve_workspace_memory']['valid'] or not plan['curve_workspace_memory']['finished']:
                continue
            case = int(path.name.split('_')[1].split('.')[0])
            receipt_paths = [a.evidence/f'case_{case}_{i}.jsonl' for i in range(1, repeats+1)]
            if not all(x.is_file() for x in receipt_paths):
                continue  # Memory-admitted but no measured resident route may be a valid tune skip.
            trials = []
            for receipt_path in receipt_paths:
                source(receipt_path)
                trials.append(one_json(receipt_path))
            timers = paired_trials(plan, trials)
            samples = [(key, s) for key, s in profile['ecm'].items() if
                s['target_bits'] == plan['target_bits'] and s['arithmetic_bits'] == plan['bits'] and
                s['carrier_exponent'] == plan['carrier_exponent'] and s['b1'] == plan['B1'] and
                s['b2'] == plan['B2'] and s['d'] == plan['D']]
            if len(samples) != 1 or samples[0][0] in consumed:
                raise ValueError('missing/duplicate measured plan scope')
            key, sample = samples[0]
            if sample['seconds'] != timers['total_seconds']:
                raise ValueError('paired receipt times differ from published sample')
            if not math.isclose(sample['median_seconds'], statistics.median(sample['seconds']), rel_tol=1e-9):
                raise ValueError('inconsistent full ECM median')
            costs = published_costs(sample, timers)
            consumed.add(key)
            rows = workload(plan)
            scope = {k: sample[k] for k in ('target_bits', 'arithmetic_bits', 'carrier_exponent',
                'modulus_kind', 'b1', 'b2', 'd', 'p', 'giant_points', 'repeats')}
            scope['total_pairs'] = sum(r['pairs'] for r in rows)
            scope['total_physical_calls'] = sum(r['calls'] for r in rows)
            scope['ntt_length_covered_pairs'] = sum(r['pairs'] for r in rows if r['length'] in measured_ntt)
            scope['ntt_missing_lengths'] = sorted({r['length'] for r in rows if r['length'] not in measured_ntt})
            # Exact paired arrays retain correlation. Other timers are explicitly overlapping.
            text += table(f'ecm.{key}', dict(**scope, **timers, **costs))
            for i, row in enumerate(rows):
                row = dict(row)
                if row['length'] in measured_ntt:
                    row['ntt_single_slice_seconds'] = measured_ntt[row['length']]
                    row['serial_single_slice_reference_seconds'] = row['pairs']*measured_ntt[row['length']]
                text += table(f'workload.{key}.bin_{i}', row)
            evidence['cases'].append(dict(case=case, sample=key, bins=len(rows),
                pairs=scope['total_pairs'], physical_calls=scope['total_physical_calls']))
        if len(consumed) != len(profile['ecm']) or len(consumed) != profile['summary']['measured']:
            raise ValueError('must cover every published sample; cannot silently drop failed/missing scope')
        text += table('summary', dict(complete=True, failed=0, measured=len(consumed),
            workload_bins=sum(x['bins'] for x in evidence['cases'])))
        for path, digest in sources.items():
            if hashlib.sha256(read(path)).hexdigest() != digest:
                raise ValueError('input changed during analysis')
        result = tomllib.loads(text)
        if len(result['ecm']) != len(consumed):
            raise ValueError('generated TOML roundtrip failed')
        (a.output/'workload.toml').write_text(text, encoding='utf-8')
        evidence['complete'] = True
        print(f'workload: measured={len(consumed)} bins={sum(x["bins"] for x in evidence["cases"])} ranking_qualified=0')
    except Exception as exc:
        evidence['error'] = str(exc)
        raise
    finally:
        evidence['sources'] = [dict(path=str(path), sha256=digest) for path, digest in sources.items()]
        (a.output/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
