"""Audit and summarize bench_stage2_n_scaling.py records; fit empirical laws.

Writes portable per-run CSV, grouped CSV, and JSON. Warmups and failed attempts
never enter means. One-shot intact-Mersenne controls never get a sample stdev.
"""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from calibrate_stage2_d import shape as ntt_shape


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fields(text, label):
    rows = re.findall(r'^' + re.escape(label) + r': (.*)$', text, re.M)
    return dict(re.findall(r'(\w+)=([^\s]+)', rows[-1])) if rows else {}


def dump_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fit_power(rows, min_bits, joint=False):
    import numpy as np
    selected = [r for r in rows if r['bits'] >= min_bits]
    x = np.array([[1, math.log(r['bits'] / 1000)] +
                  ([math.log(r['B2'] / 26_000_000_000)] if joint else []) for r in selected])
    y = np.log([r['mean_seconds'] for r in selected])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    predicted = np.exp(x @ coefficients)
    actual = np.exp(y)
    errors = (predicted - actual) / actual
    withheld = np.empty(len(selected))
    for exponent in {r['exponent'] for r in selected}:
        hold = np.array([r['exponent'] == exponent for r in selected])
        trained = np.linalg.lstsq(x[~hold], y[~hold], rcond=None)[0]
        withheld[hold] = np.exp(x[hold] @ trained)
    hold_errors = (withheld - actual) / actual
    return dict(min_bits=min_bits, points=len(selected), C_seconds=float(np.exp(coefficients[0])),
                alpha=float(coefficients[1]), **({'beta': float(coefficients[2])} if joint else {}),
                log_r2=float(1 - np.sum((np.log(predicted) - y) ** 2) / np.sum((y - np.mean(y)) ** 2)),
                mape_percent=float(100 * np.mean(np.abs(errors))),
                max_abs_error_percent=float(100 * np.max(np.abs(errors))),
                leave_one_exponent_out_mape_percent=float(100 * np.mean(np.abs(hold_errors))),
                leave_one_exponent_out_max_error_percent=float(100 * np.max(np.abs(hold_errors))),
                predictions=[dict(exponent=r['exponent'], bits=r['bits'], B2=r['B2'],
                                  actual_seconds=r['mean_seconds'], predicted_seconds=float(p),
                                  error_percent=float(100 * e)) for r, p, e in zip(selected, predicted, errors)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='Filename prefix, e.g. data/benchmarks/stage2_n_20261008')
    p.add_argument('--allow-partial', action='store_true')
    a = p.parse_args()
    study = a.study.resolve()
    data = json.loads((study / 'measurements.json').read_text(encoding='utf-8'))
    if not data['complete'] and not a.allow_partial:
        p.error('Study not complete; use --allow-partial only for an interim view')
    if sha(data['executable']) != data['executable_sha256']:
        raise ValueError('Binary identity changed')
    if sha(study / 'database_rows.json') != data['database_rows_sha256']:
        raise ValueError('Database snapshot changed')
    for c in data['cases']:
        if sha(c['save']) != c['save_sha256']:
            raise ValueError('Stage1 save changed')
    rows, groups, audit = [], defaultdict(list), []
    for run in data['runs']:
        log = Path(run['log'])
        if sha(log) != run['log_sha256']:
            raise ValueError('Raw log changed: ' + str(log))
        prefix = log.with_suffix('')
        result = prefix.with_suffix('.result.jsonl')
        if sha(result) != run['result_sha256']:
            raise ValueError('Result changed')
        debug = Path(run['debug_log'])
        text = log.read_text(encoding='utf-8') + '\n' + debug.read_text(encoding='utf-8')
        audit.append(dict(log=str(log), log_sha256=sha(log), debug_sha256=sha(debug),
                          result_sha256=sha(result), telemetry_sha256=sha(run['telemetry'])))
        module = run['stats']['ntt_workspace_stats']
        owner = run['stats']['real_batched_folddevice']
        split = run['stats']['real_batched_split']
        reduction = fields(text, 's4_reduce_mode')
        point = fields(text, 'point_mersenne_mode')
        gleaf = fields(text, 'device_gleaf')
        model = fields(text, 'd_model_scope')
        breakdown = fields(text, 'real_batched_breakdown')
        ledger = fields(text, 'real_batched_wall')
        inversions = fields(text, 'real_batched_gfinv')
        paired = fields(text, 'real_giant_seed_pair')
        scan = fields(text, 'd_scan_wall')
        psize, bits = run['P'], run['bits']
        fold_length, bpw, digits = ntt_shape(psize + 1, bits)
        top = 1 << ((psize - 1).bit_length() - 1)
        tree_length = ntt_shape(top + 1, bits)[0]
        # P/2 planning bound and the actual padded tree top can differ.
        row = dict(sequence_index=run['sequence_index'], category=run['category'],
                   started_utc=run['started_utc'], finished_utc=run['finished_utc'],
                   variant=run['variant'], exponent=run['exponent'], bits=bits, words=run['words'],
                   B1=run['B1'], sigma=run['sigma'], B2=run['B2'], repeat=run['repeat'],
                   total_seconds=float(run['wall']['total']), init_seconds=float(run['wall']['init']),
                   main_seconds=float(run['wall']['main']), shape_seconds=float(run['wall']['shape']),
                   d_scan_seconds=float(scan.get('seconds', 0)),
                   process_seconds=run['process_seconds'], D=run['D'], P=psize,
                   giant_points=int(run['shape']['giant_points']),
                   G=int(run['shape']['num_poly_g']), fold_ntt_length=fold_length,
                   tree_top_ntt_length=tree_length, packing_bpw=bpw, packing_digits=digits,
                   ntt_full_peak_mib=int(module['full_peak_bytes']) / 2**20,
                   ntt_big_peak_mib=int(module['big_peak_bytes']) / 2**20,
                   fold_peak_mib=int(owner['peak_bytes']) / 2**20,
                   fold_enabled=int(owner['enabled']), fold_fallback=owner['fallback'],
                   reduction=reduction.get('algorithm'), point_mersenne=int(point.get('enabled', 0)),
                   d_model_calibrated=model.get('calibrated'),
                   s4_reduce_seconds=float(run['s4']['t_reduce']),
                   s4_coefficients=int(run['s4']['coeffs_reduced']),
                   poly_muls=int(run['s4']['poly_muls']), s4_launches=int(run['s4']['launches']),
                   gmp_checked=int(run['s4']['gmp_checked']),
                   leaf_hash=run['stats']['descent_values'].get('hash'),
                   factor_count=len(run['result']['factors']), factors=';'.join(run['result']['factors']),
                   device_gleaf_coord_mib=int(gleaf.get('coord_peak_bytes', 0)) / 2**20,
                   device_gleaf_bad_groups=int(gleaf.get('bad_groups', 0)),
                   device_gleaf_good_segments=int(gleaf.get('good_segments', 0)),
                   device_gleaf_patch_words=int(gleaf.get('patch_words', 0)),
                   device_gleaf_bad_point_d2h_bytes=int(gleaf.get('bad_point_d2h_bytes', 0)),
                   gfinv_nonunits=int(inversions.get('nonunits', 0)),
                   giant_base_nonunits=int(paired.get('base_nonunits', 0)),
                   reported_ntt_seconds=float(breakdown.get('ntt_seconds', 0)),
                   arena_overflow=int(breakdown.get('overflow', 0)))
        for key in ('giant', 'gtrees', 'fold', 'descent', 'inv', 'accum', 'name', 'f_tree_incl'):
            row[key + '_seconds'] = float(split[key])
        for key in ('pre', 'loop_wall', 'post', 'gleaves', 'loop_host'):
            row[key + '_seconds'] = float(ledger.get(key, 0))
        baby = run['stats']['real_baby']
        row['baby_seconds'] = float(baby['ladder']) + float(baby['affine'])
        row['f_tree_init_seconds'] = row['init_seconds'] - row['baby_seconds']
        row['outside_full_wall_seconds'] = row['process_seconds'] - row['total_seconds']
        row['other_seconds'] = row['total_seconds'] - sum(
            row[key + '_seconds'] for key in
            ('giant', 'gtrees', 'fold', 'descent', 'inv', 'accum', 'name', 'init', 'gleaves'))
        for key, alias in [('used_bytes', 'observed_gpu_used'), ('gpu_percent', 'gpu_util'),
                           ('sm_clock_mhz', 'sm_clock'), ('temperature_c', 'temperature'), ('power_mw', 'power')]:
            value = run['gpu_observations'][key]
            row[alias + '_mean'] = value['mean'] if value else None
            row[alias + '_max'] = value['max'] if value else None
        rows.append(row)
        if run['category'] != 'warmup':
            groups[(run['variant'], run['exponent'], run['B2'])].append(row)
    summary, warnings = [], []
    for (variant, exponent, b2), samples in sorted(groups.items()):
        times = [r['total_seconds'] for r in samples]
        record = {k: samples[0][k] for k in ('variant', 'exponent', 'bits', 'words', 'B1', 'sigma', 'B2')}
        record.update(samples=len(samples), mean_seconds=statistics.mean(times),
                      median_seconds=statistics.median(times), min_seconds=min(times), max_seconds=max(times),
                      std_seconds=statistics.stdev(times) if len(times) > 1 else None)
        record['cv_percent'] = 100 * record['std_seconds'] / record['mean_seconds'] if record['std_seconds'] is not None else None
        record['range_percent'] = 100 * (max(times) - min(times)) / record['mean_seconds']
        for key in ('D', 'P', 'giant_points', 'G', 'fold_ntt_length', 'tree_top_ntt_length', 'packing_bpw',
                    'reduction', 'point_mersenne', 'fold_enabled', 'fold_fallback', 'leaf_hash',
                    's4_coefficients', 'poly_muls', 's4_launches', 'gmp_checked', 'factor_count', 'factors',
                    'device_gleaf_bad_groups', 'device_gleaf_good_segments', 'device_gleaf_patch_words',
                    'device_gleaf_bad_point_d2h_bytes', 'gfinv_nonunits', 'giant_base_nonunits', 'arena_overflow'):
            values = {r[key] for r in samples}
            record[key] = samples[0][key] if len(values) == 1 else ';'.join(str(v) for v in sorted(values))
            if len(values) != 1:
                warnings.append(f'{variant} M{exponent} B2={b2}: {key} varied across samples')
        keys = [k for k in samples[0] if k.endswith('_seconds') and k != 'total_seconds']
        keys += ['ntt_full_peak_mib', 'ntt_big_peak_mib', 'fold_peak_mib', 'device_gleaf_coord_mib',
                 'gpu_util_mean', 'sm_clock_mean', 'temperature_max', 'power_mean', 'observed_gpu_used_max']
        peak_keys = {'ntt_full_peak_mib', 'ntt_big_peak_mib', 'fold_peak_mib', 'device_gleaf_coord_mib',
                     'temperature_max', 'observed_gpu_used_max'}
        for key in keys:
            values = [r[key] for r in samples if r[key] is not None]
            record[key] = (max(values) if key in peak_keys else statistics.mean(values)) if values else None
        record['s4_reduce_percent'] = 100 * record['s4_reduce_seconds'] / record['mean_seconds']
        for phase in ('giant', 'gtrees', 'fold', 'descent', 'inv', 'accum', 'f_tree_incl',
                      'init', 'baby', 'f_tree_init', 'gleaves', 'other'):
            record[phase + '_percent'] = 100 * record[phase + '_seconds'] / record['mean_seconds']
        summary.append(record)
    if data['complete']:
        if len(data['runs']) != len(data['sequence']):
            raise ValueError('Completed study missing invocations')
        for c in data['cases']:
            for b2 in data['bounds']:
                expected = data['repetitions'] if c['variant'] == 'cofactor' else 1
                if len(groups[(c['variant'], c['exponent'], b2)]) != expected:
                    raise ValueError('Incomplete cell')
    primary = [r for r in summary if r['variant'] == 'cofactor']
    fits = {}
    if len(primary) >= 9:
        fits['joint_all_bits'] = fit_power(primary, 1, True)
        fits['joint_at_least_1000_bits'] = fit_power(primary, 1000, True)
        fits['per_B2_at_least_1000_bits'] = {
            str(b2): fit_power([r for r in primary if r['B2'] == b2], 1000)
            for b2 in data['bounds'] if sum(r['B2'] == b2 and r['bits'] >= 1000 for r in primary) >= 3}
    beta, alpha = [], []
    for exponent in data['exponents']:
        group = sorted([r for r in primary if r['exponent'] == exponent], key=lambda r: r['B2'])
        for lo, hi in zip(group, group[1:]):
            beta.append(dict(exponent=exponent, bits=lo['bits'], lower_B2=lo['B2'], upper_B2=hi['B2'],
                             beta=math.log(hi['mean_seconds'] / lo['mean_seconds']) / math.log(hi['B2'] / lo['B2'])))
    for b2 in data['bounds']:
        group = sorted([r for r in primary if r['B2'] == b2], key=lambda r: r['bits'])
        for lo, hi in zip(group, group[1:]):
            alpha.append(dict(B2=b2, lower_bits=lo['bits'], upper_bits=hi['bits'],
                              alpha=math.log(hi['mean_seconds'] / lo['mean_seconds']) / math.log(hi['bits'] / lo['bits']),
                              ntt_length_ratio=hi['fold_ntt_length'] / lo['fold_ntt_length']))
    output = a.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_csv(str(output) + '_runs.csv', rows)
    dump_csv(str(output) + '_summary.csv', summary)
    report = dict(schema=1, complete=data['complete'], study=str(study),
                  created_utc=data['created_utc'], completed_utc=data.get('completed_utc'),
                  device=data['device'], executable_sha256=data['executable_sha256'],
                  dll_sha256=data['dll_sha256'], ini_sha256=data['ini_sha256'],
                  database_rows_sha256=data['database_rows_sha256'],
                  collector_sha256=sorted({r['collector_sha256'] for r in data['runs']}),
                  B1=data['B1'], memory=data['memory'], bounds=data['bounds'],
                  counts=dict(warmup=sum(r['category'] == 'warmup' for r in rows),
                              timing=sum(r['category'] == 'timing' for r in rows),
                              control=sum(r['category'] == 'control' for r in rows),
                              failures=len(data['failures'])),
                  inputs=data['cases'], summary=summary, fits=fits, local_B2_exponents=beta, adjacent_N_exponents=alpha,
                  warnings=warnings, raw_artifact_hashes=audit,
                  input_manifest_sha256=sha(study / 'measurements.json'),
                  analysis_sha256=sha(__file__))
    Path(str(output) + '_analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(counts=report['counts'], warnings=warnings, fits={k: {a: b for a, b in v.items() if a != 'predictions'}
                        for k, v in fits.items() if k != 'per_B2_at_least_1000_bits'}), indent=2))


if __name__ == '__main__':
    main()
