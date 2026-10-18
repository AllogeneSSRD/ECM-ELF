"""Format-8 scalar matrices. Full receipts remain in the original JSONL."""
import json
import math

SHAPE = ('condition', 'target_bits', 'arithmetic_bits', 'carrier_exponent', 'modulus_kind',
         'b1', 'b2', 'd', 'fold_resident', 'frontier_resident', 'execution_path',
         'giant_chunk_points', 'giant_chain_min', 'giant_force_ladder', 'program_version')
PHASES = ('shape', 'setup', 'baby', 'ftree', 'main_setup', 'inverse_setup',
          'giant_loop', 'descent', 'accum', 'finalize')
KEEP = set(SHAPE) | {'p', 'source', 'benchmark_kind', 'verified', 'repeats', 'mean_seconds',
    'stddev_seconds', 'measurement_ids', 'preferred_order', 'validation_count',
    'validation_max_relative_error', 'giant_seconds', 'gtrees_seconds', 'fold_seconds',
    'model_relative_error_allowance'} | {f'phase_{p}_seconds' for p in PHASES}


def literal(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    if not math.isfinite(value):
        raise ValueError('Non-finite tune value')
    return repr(value)


def encode_ids(ids):
    return ''.join(f'{len(s.encode("utf-8"))}:{s}' for s in sorted(ids))


def decode_ids(value):
    result = set()
    value = value.encode('utf-8')
    while value:
        length, _, tail = value.partition(b':')
        n = int(length)
        if not n or n > len(tail):
            raise ValueError('Invalid measurement identity')
        token = tail[:n].decode('utf-8')
        if token in result:
            raise ValueError('Duplicate measurement identity')
        result.add(token)
        value = tail[n:]
    return result


def aggregate(samples):
    groups = {}
    for original in samples:
        row = {k:v for k,v in original.items() if k in KEEP}
        if row['source'] == 'measured':
            row['verified'] = 1
            row.setdefault('measurement_ids', encode_ids({original.get('runtime_sample_id') or
                f'summary:{row["mean_seconds"]}:{row["stddev_seconds"]}:{row["repeats"]}'}))
        if not original.get('runtime_probe', 1):
            row['preferred_order'] = original.get('runtime_completed_ns', 0)
        key = tuple(row.get(k) for k in SHAPE) + (row.get('benchmark_kind') == 'runtime_save',)
        if key not in groups:
            groups[key] = row
            continue
        old = groups[key]
        if old['source']!='measured':
            groups[key]=row
            continue
        if row['source']!='measured':
            continue
        a, b = decode_ids(old['measurement_ids']), decode_ids(row['measurement_ids'])
        if b <= a:
            continue
        if a <= b:
            groups[key] = row
            continue
        if a & b:
            raise ValueError('Partially overlapping snapshots: recover original JSONL')
        n, m = old['repeats'], row['repeats']
        delta = row['mean_seconds']-old['mean_seconds']
        m2 = (n-1)*old['stddev_seconds']**2+(m-1)*row['stddev_seconds']**2+delta**2*n*m/(n+m)
        for k in old.keys() & row.keys():
            if k.endswith('_seconds') and k != 'stddev_seconds':
                old[k] += (row[k]-old[k])*m/(n+m)
        old.update(repeats=n+m, stddev_seconds=math.sqrt(max(0, m2/(n+m-1))),
                   measurement_ids=encode_ids(a | b))
        order = max(old.get('preferred_order',0),row.get('preferred_order',0))
        if order:
            old['preferred_order'] = order
        checks = old.get('validation_count',0)+row.get('validation_count',0)
        if checks:
            old['validation_count'] = checks
            old['validation_max_relative_error'] = max(old.get('validation_max_relative_error',0),
                                                       row.get('validation_max_relative_error',0))
    return list(groups.values())


def write_profile(conditions, samples):
    lines = ['# Stage2 mean timings; identical parameters share one repeated measurement row.',
             '[profile]', 'format = 8', 'unit = "full_stage2"']
    for i, condition in enumerate(conditions):
        if not condition.get('program_version'):
            raise ValueError('Missing program version')
        lines += [f'\n[condition.{i}]']
        lines += [f'{k} = {literal(v)}' for k,v in sorted(condition.items())]
    widths = {}
    for s in aggregate(samples):
        widths.setdefault(s['target_bits'], []).append(s)
    omit = {'target_bits','arithmetic_bits','p','execution_path','program_version'}
    first = ('condition','carrier_exponent','modulus_kind','b1','b2','d','fold_resident',
             'frontier_resident','source','benchmark_kind','repeats','mean_seconds','stddev_seconds')
    for width, rows in sorted(widths.items()):
        names = set().union(*(r.keys() for r in rows))-omit
        columns = [k for k in first if k in names]+sorted(names-set(first))
        rows.sort(key=lambda r:(r['condition'],r['carrier_exponent'],r['b2'],r['d'],r['b1'],r['source']))
        matrix = [[literal(c) for c in columns]]+[[literal(r.get(c,'')) for c in columns] for r in rows]
        col_widths = [max(len(r[i]) for r in matrix) for i in range(len(columns))]
        lines += [f'\n[width.{width}]', 'matrix = [']
        lines += ['    ['+', '.join(v.rjust(w) for v,w in zip(row,col_widths))+'],' for row in matrix]
        lines += [']']
    return '\n'.join(lines)+'\n'
