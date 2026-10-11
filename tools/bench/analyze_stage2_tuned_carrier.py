"""Audit tuned carrier A/B evidence and produce paired CSV/JSON and PNG/SVG."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics

from bench_stage2_n_scaling import collect, fields, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='Benchmark summary prefix')
    p.add_argument('--figures', type=Path, required=True, help='Figure prefix')
    a = p.parse_args()
    data = read(a.study / 'measurements.json')
    if not data['complete'] or data['failures']:
        raise ValueError('require complete successful study; raw failures must be investigated')
    for path, expected in [(Path(data['executable']), data['executable_sha256']),
                           (Path(data['executable']).parent/'gmp-10.dll', data['dll_sha256']),
                           (a.study/'measured.toml', data['profile_sha256']),
                           (a.study/'bench.ini', data['ini_sha256']),
                           (a.study/'seed.toml', data['seed_sha256'])]:
        if sha(path) != expected:
            raise ValueError('frozen input changed: ' + str(path))
    cases = {c['id']: c for c in data['cases']}
    for c in cases.values():
        if sha(c['save']) != c['save_sha256'] or ((1 << c['exponent'])-1) % int(c['N_hex'], 16):
            raise ValueError('invalid carrier input')
    audit = []
    groups = defaultdict(list)
    for row in data['runs']:
        log, debug = Path(row['log']), Path(row['debug_log'])
        prefix = log.with_suffix('')
        for path, expected in [(log, row['log_sha256']), (debug, row['debug_sha256']),
                               (prefix.with_suffix('.result.jsonl'), row['result_sha256']),
                               (Path(row['telemetry']), row['telemetry_sha256'])]:
            if sha(path) != expected:
                raise ValueError('raw evidence changed: ' + str(path))
        item = {k: row[k] for k in ('case', 'B2', 'category', 'repeat', 'arm')}
        checked = collect(data, cases[row['case']], item, prefix, row['process_seconds'], read(row['telemetry']))
        if checked['wall'] != row['wall'] or checked['result'] != row['result']:
            raise ValueError('collector result mismatch')
        if any(int(row['s4'].get(k, 0)) <= 0 for k in ('gmp_selftest_cases', 'gmp_checked', 'full_checks')):
            raise ValueError('mandatory arithmetic coverage missing')
        carrier = row['exponent'] if row['arm'] == 'carrier' else 0
        if row['result']['carrier_exponent'] != carrier:
            raise ValueError('arithmetic lock mismatch')
        text = log.read_text(encoding='utf-8')+'\n'+debug.read_text(encoding='utf-8')
        reduction, point = fields(text, 's4_reduce_mode'), fields(text, 'point_mersenne_mode')
        if carrier and (reduction.get('algorithm') != 'mersenne' or point.get('enabled') != '1'):
            raise ValueError('carrier did not enable the expected fast arithmetic')
        arithmetic_bits = int(row['shape']['S_bits'])
        if arithmetic_bits != (carrier or row['bits']):
            raise ValueError('unexpected arithmetic width')
        derived = dict(row, arithmetic_bits=arithmetic_bits, reduction=reduction.get('algorithm'),
                       point_mersenne=point.get('enabled'))
        counts = re.findall(r'^ntt_length_counts: log2_length:num_of_ntt (.+)$', text, re.M)
        histogram = {int(k): int(n) for k, n in re.findall(r'(\d+):(\d+)', counts[-1])} if counts else {}
        derived['max_ntt_log2'] = max(histogram) if histogram else None
        derived['ntt_length_counts'] = histogram
        derived['active_clocks'] = [s['sm_clock_mhz'] for s in read(row['telemetry'])
                                    if s['gpu_percent'] is not None and s['gpu_percent'] >= 50
                                    and s['sm_clock_mhz'] is not None]
        audit.append(dict(log=str(log), log_sha256=sha(log), debug_sha256=sha(debug),
                          result_sha256=row['result_sha256'], telemetry_sha256=row['telemetry_sha256']))
        if row['category'] == 'timing':
            groups[row['case'], row['B2'], row['arm']].append(derived)
    pairs = []
    for c in data['cases']:
        for b in data['bounds']:
            pair = dict(exponent=c['exponent'], bits=c['bits'], B1=c['B1'], sigma=c['sigma'], B2=b)
            for arm in ('generic', 'carrier'):
                runs = groups[c['id'], b, arm]
                if len(runs) != data['repetitions']:
                    raise ValueError('incomplete timing cell')
                if len({r['D'] for r in runs}) != 1:
                    raise ValueError('D changed within repetitions')
                times = [float(r['wall']['total']) for r in runs]
                active_clocks = [v for r in runs for v in r['active_clocks']]
                split = runs[0]['stats']['real_batched_split']
                ntt = runs[0]['stats']['ntt_workspace_stats']
                owner = runs[0]['stats']['real_batched_folddevice']
                pair[arm] = dict(samples=len(times), mean_seconds=statistics.mean(times),
                                 median_seconds=statistics.median(times),
                                 std_seconds=statistics.stdev(times) if len(times)>1 else None,
                                 process_seconds=statistics.mean(r['process_seconds'] for r in runs),
                                 D=runs[0]['D'], P=runs[0]['P'], G=int(runs[0]['shape']['num_poly_g']),
                                 arithmetic_bits=runs[0]['arithmetic_bits'], reduction=runs[0]['reduction'],
                                 point_mersenne=runs[0]['point_mersenne'],
                                 max_ntt_log2=runs[0]['max_ntt_log2'],
                                 ntt_length_counts=runs[0]['ntt_length_counts'],
                                 tune_plan=runs[0]['result'].get('tune_plan'),
                                 ntt_full_peak_mib=max(int(r['stats']['ntt_workspace_stats']['full_peak_bytes'])/2**20 for r in runs),
                                 fold_peak_mib=max(int(r['stats']['real_batched_folddevice']['peak_bytes'])/2**20 for r in runs),
                                 fold_enabled=int(owner['enabled']), fold_fallback=owner['fallback'],
                                 observed_gpu_used_max_mib=max(r['gpu_observations']['used_bytes']['max']/2**20 for r in runs),
                                 sm_clock_mean=statistics.mean(r['gpu_observations']['sm_clock_mhz']['mean'] for r in runs),
                                 active_clock_samples=len(active_clocks),
                                 active_clock_median=statistics.median(active_clocks) if active_clocks else None,
                                 active_clock_min=min(active_clocks) if active_clocks else None,
                                 active_clock_max=max(active_clocks) if active_clocks else None,
                                 temperature_max=max(r['gpu_observations']['temperature_c']['max'] for r in runs),
                                 s4_reduce_seconds=statistics.mean(float(r['s4']['t_reduce']) for r in runs),
                                 **{key+'_seconds': statistics.mean(float(r['stats']['real_batched_split'][key]) for r in runs)
                                    for key in ('giant','gtrees','fold','descent')},
                                 factor_coverage=math.lcm(*(int(f) for f in runs[0]['result']['factors'])))
            pair.update(speedup=pair['generic']['mean_seconds']/pair['carrier']['mean_seconds'],
                        change_percent=100*(pair['carrier']['mean_seconds']/pair['generic']['mean_seconds']-1),
                        factor_coverage_equal=pair['generic']['factor_coverage']==pair['carrier']['factor_coverage'])
            pairs.append(pair)
    training = []
    for t in data['training']:
        log = a.study/'training'/(t['key']+'.log')
        if sha(log) != t['log_sha256'] or t['returncode'] != 0:
            raise ValueError('training evidence changed or failed')
        text = log.read_text(encoding='utf-8')
        windows = [dict(re.findall(r'(\w+)=(\S+)', line)) for line in text.splitlines()
                   if line.startswith('ecm_tune_window_done:')]
        training.append(dict(case=t['case'], B2=t['B2'], seconds=t['seconds'], windows=windows,
                             log=str(log), log_sha256=sha(log)))
    totals = {arm: sum(r[arm]['mean_seconds'] for r in pairs) for arm in ('generic', 'carrier')}
    output = a.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(complete=True, study=str(a.study.resolve()), binary_sha256=data['executable_sha256'],
                  profile_sha256=data['profile_sha256'], device=data['device'], memory=data['memory'],
                  profile_source=data.get('profile_source'), seed_sha256=data['seed_sha256'],
                  counts=dict(training=len(training), warmup=sum(r['category']=='warmup' for r in data['runs']),
                              timing=sum(r['category']=='timing' for r in data['runs'])),
                  totals=totals, speedup=totals['generic']/totals['carrier'],
                  analysis_sha256=sha(__file__), input_manifest_sha256=sha(a.study/'measurements.json'),
                  semantics='same target N/save/B1/sigma/B2 and binary; arithmetic width and tuned D may differ; '
                            'full=init+main, excludes Stage1/planning/startup/tune; component peaks are not additive; '
                            'n=1 has no repeatability estimate', pairs=pairs, training=training, raw_hashes=audit)
    if data.get('skip_training') and data.get('profile_source'):
        receipt = Path(data['profile_source']).parent/'report.json'
        if receipt.exists():
            trained = read(receipt)
            if (not trained['complete'] or trained['output_profile_sha256'] != data['seed_sha256']
                    or trained['binary_sha256'] != data['executable_sha256']):
                raise ValueError('external training receipt does not match holdout')
            for t in trained['receipts']:
                if sha(t['log']) != t['log_sha256'] or t['returncode']:
                    raise ValueError('external training log changed/failed')
            report['external_training'] = dict(receipt=str(receipt), receipt_sha256=sha(receipt),
                                               jobs=trained['jobs'], receipts=trained['receipts'])
    write(str(output)+'_analysis.json', report)
    flat = []
    for row in pairs:
        r = {k: v for k, v in row.items() if k not in ('generic', 'carrier')}
        for arm in ('generic', 'carrier'):
            r.update({arm+'_'+k: v for k, v in row[arm].items() if k != 'tune_plan'})
        flat.append(r)
    with Path(str(output)+'_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    figure = a.figures.resolve()
    figure.parent.mkdir(parents=True, exist_ok=True)
    for kind in ('time', 'speedup', 'memory'):
        fig, axes = plt.subplots(1, len(data['bounds']), figsize=(15, 5), squeeze=False)
        for ax, b in zip(axes[0], data['bounds']):
            rows = sorted((r for r in pairs if r['B2']==b), key=lambda r:r['bits'])
            if kind == 'speedup':
                ax.plot([r['bits'] for r in rows], [r['speedup'] for r in rows], 'o-', color='#2468a2')
                ax.axhline(1, linestyle='--', color='#888888')
            else:
                for arm, label, color, marker in [('generic','Ordinary target N','#858585','o'),
                                                  ('carrier','Mersenne carrier + tuned D','#2468a2','s')]:
                    key = 'mean_seconds' if kind=='time' else 'ntt_full_peak_mib'
                    ax.plot([r['bits'] for r in rows], [r[arm][key] for r in rows],
                            marker=marker, label=label, color=color)
            ax.set_title(f'B2={b:.1e}')
            ax.set_xlabel('Actual target N (bits)')
            ax.set_ylabel({'time':'Complete Stage2 (s)', 'speedup':'Ordinary / carrier (>1: carrier faster)',
                           'memory':'NTT module full_peak (MiB)'}[kind])
            ax.xaxis.set_major_locator(MultipleLocator(1000))
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
        if kind != 'speedup':
            axes[0,0].legend(fontsize=8)
        fig.suptitle('Same cofactor saves: current native tune, ordinary versus Mersenne carrier')
        fig.text(.01, -.03, 'RTX 4060 Laptop, user setting 2400 MHz; same binary, frozen tune profile and memory budgets. '
                 'Automatic D differs by route; Stage1/tune/startup excluded. n='+str(data['repetitions'])+
                 '; component peaks are not process peaks. Source: '+output.name+'_analysis.json', fontsize=8)
        fig.tight_layout()
        for extension in ('png','svg'):
            fig.savefig(str(figure)+'_'+kind+'.'+extension, dpi=180, bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(dict(counts=report['counts'], totals=totals, speedup=report['speedup']), indent=2))


if __name__ == '__main__':
    main()
