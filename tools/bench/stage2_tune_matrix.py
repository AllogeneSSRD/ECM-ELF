"""Format-8 scalar matrices. Full receipts remain in the original JSONL."""
import json
import math
import re
from pathlib import Path
from datetime import datetime, timezone

SHAPE = ('condition', 'target_bits', 'arithmetic_bits', 'carrier_exponent', 'modulus_kind',
         'b1', 'b2', 'd', 'fold_resident', 'frontier_resident', 'execution_path',
         'giant_chunk_points', 'giant_chain_min', 'giant_force_ladder')
PHASES = ('shape', 'setup', 'baby', 'ftree', 'main_setup', 'inverse_setup',
          'giant_loop', 'descent', 'accum', 'finalize')
KEEP = set(SHAPE) | {'program_version', 'updated_at', 'p', 'source', 'benchmark_kind', 'verified', 'repeats', 'mean_seconds',
    'stddev_seconds', 'measurement_ids', 'preferred_order', 'validation_count',
    'validation_max_relative_error', 'giant_seconds', 'gtrees_seconds', 'fold_seconds',
    'model_relative_error_allowance'} | {f'phase_{p}_seconds' for p in PHASES}


def matrix_column_order():
    path=Path(__file__).resolve().parents[2]/'src/core/ecm_stage2_tune_columns.def'
    result=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line or line.startswith('//'):continue
        match=re.fullmatch(r'ECM_STAGE2_TUNE_COLUMN\(([a-z_][a-z_0-9]*)\)',line)
        if not match or match[1] in result:raise ValueError('Invalid shared tune column definition')
        result.append(match[1])
    return tuple(result)

COLUMN_ORDER=matrix_column_order()

def utc_updated_at(nanoseconds):
    seconds,fraction=divmod(int(nanoseconds),1_000_000_000)
    return datetime.fromtimestamp(seconds,timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')+f'.{fraction:09d}Z'

def merge_provenance(target, source):
    a,b=target.get('updated_at',''),source.get('updated_at','')
    if b>a or (b==a and not target.get('program_version')):
        for k in ('program_version','updated_at'):
            if k in source:target[k]=source[k]
            else:target.pop(k,None)

def validate_provenance(row):
    version=row.get('program_version','')
    if not isinstance(version,str) or len(version)>256:raise ValueError('Invalid source version')
    time=row.get('updated_at','')
    if time:
        if not isinstance(time,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z',time):
            raise ValueError('updated_at requires UTC nanosecond format')
        datetime.strptime(time[:19],'%Y-%m-%dT%H:%M:%S')


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
        if not row.get('updated_at') and original.get('runtime_completed_ns'):
            row['updated_at']=utc_updated_at(original['runtime_completed_ns'])
        validate_provenance(row)
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
            merge_provenance(row,old)
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
        merge_provenance(old,row)
    return list(groups.values())


def write_profile(conditions, samples):
    lines = ['# Stage2 mean timings; identical parameters share one repeated measurement row.',
             '[profile]', 'format = 8', 'unit = "full_stage2"']
    canonical=[];remap={};prepared=[]
    for i,c in enumerate(conditions):
        condition={k:v for k,v in c.items() if k not in ('program_version','updated_at')}
        if condition not in canonical:canonical.append(condition)
        remap[i]=canonical.index(condition)
    for s in samples:
        row=dict(s);old=int(row['condition'])
        if not row.get('program_version') and conditions[old].get('program_version'):
            row['program_version']=conditions[old]['program_version']
        row['condition']=remap[old];prepared.append(row)
    for i,condition in enumerate(canonical):
        lines += [f'\n[condition.{i}]']
        lines += [f'{k} = {literal(v)}' for k,v in sorted(condition.items())]
    widths = {}
    for s in aggregate(prepared):
        widths.setdefault(s['target_bits'], []).append(s)
    omit = {'target_bits','arithmetic_bits','p','execution_path'}
    for width, rows in sorted(widths.items()):
        names = (set().union(*(r.keys() for r in rows)) | {'program_version','updated_at'})-omit
        columns = [k for k in COLUMN_ORDER if k in names]+sorted(names-set(COLUMN_ORDER))
        rows.sort(key=lambda r:(r['condition'],r['carrier_exponent'],r['b2'],r['d'],r['b1'],r['source']))
        matrix = [[literal(c) for c in columns]]+[[literal(r.get(c,'')) for c in columns] for r in rows]
        col_widths = [max(len(r[i]) for r in matrix) for i in range(len(columns))]
        lines += [f'\n[width.{width}]', 'matrix = [']
        lines += ['    ['+', '.join(v.rjust(w) for v,w in zip(row,col_widths))+'],' for row in matrix]
        lines += [']']
    return '\n'.join(lines)+'\n'
