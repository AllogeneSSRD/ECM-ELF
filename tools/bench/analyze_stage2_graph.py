"""Summarize complete fixed-D Graph A/B, own-event traces, and GPU telemetry."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics as st


def telemetry(path):
    values = []
    with path.open(encoding='utf-8-sig') as f:
        for raw in csv.DictReader(f):
            raw = {k.strip(): v.strip() for k, v in raw.items()}
            if not raw or not raw.get('timestamp'):
                continue
            row = {'unix': datetime.strptime(raw['timestamp'], '%Y/%m/%d %H:%M:%S.%f').timestamp()}
            for key, label in [('clocks.current.sm [MHz]', 'mhz'), ('power.draw [W]', 'watts'),
                               ('temperature.gpu', 'celsius'), ('utilization.gpu [%]', 'util'), ('memory.used [MiB]', 'memory_mib')]:
                try:
                    row[label] = float(raw[key].split()[0])
                except (ValueError, KeyError):
                    row[label] = None
            values.append(row)
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, action='append', required=True)
    p.add_argument('--profile', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figure', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    summaries, runs, gpu = [], [], []
    for path in a.experiment:
        data = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
        if not data['complete'] or data['identity']['diagnostic']:
            raise ValueError('Only complete, unprofiled performance matrices accepted')
        samples = [json.loads(s) for s in (path / 'runs.jsonl').read_text().splitlines()]
        device = telemetry(path / 'gpu.csv')
        gpu += device
        runs += samples
        for item in data['summaries']:
            row = dict(experiment=path.name, case=item['case'], gain_percent=item['gain_percent'],
                       b2=item['input']['b2'], d=item['input']['d'], batch_mb=item['input']['batch_mb'],
                       carrier=item['input'].get('carrier', 0), fold_mb=item['input'].get('fold_mb', 640))
            for side, flag in [('off', 0), ('on', 1)]:
                values = item[side]
                selected = [v for v in samples if v['case'] == item['case'] and v['enabled'] == flag]
                active = [t for v in selected for t in device if v['start_unix'] <= t['unix'] <= v['end_unix'] and (t['util'] or 0) >= 50]
                for key in ('engine_seconds', 'engine_seconds_mad', 'init_seconds', 'main_seconds', 'reduce_seconds', 'full_peak_bytes'):
                    row[side + '_' + key] = values[key]
                for key in ('mhz', 'watts', 'celsius', 'memory_mib'):
                    measured = [v[key] for v in active if v[key] is not None and (key != 'watts' or v[key] <= 150)]
                    row[side + '_' + key] = st.median(measured) if measured else None
                row[side + '_phases'] = values['phases']
                row[side + '_graph'] = values['graph']
                row['actual_bits'] = int(selected[0]['shape']['N_bits'])
            hist = {int(k): v for k, v in item['off']['histogram'].items()}
            row['transforms'] = sum(hist.values())
            row['short_transform_fraction'] = sum(v for k, v in hist.items() if k <= 19) / row['transforms']
            row['short_element_fraction'] = sum(v*(1 << k) for k, v in hist.items() if k <= 19) / sum(v*(1 << k) for k, v in hist.items())
            row['short_butterfly_fraction'] = sum(v*(1 << k)*k for k, v in hist.items() if k <= 19) / sum(v*(1 << k)*k for k, v in hist.items())
            summaries.append(row)
    profiles = {str(path): json.loads((path / 'summary.json').read_text(encoding='utf-8')) for path in a.profile}
    active = [r for r in gpu if (r['util'] or 0) >= 50]
    def spread(key, accept=lambda x: True):
        data = [r[key] for r in active if r[key] is not None and accept(r[key])]
        return dict(min=min(data), median=st.median(data), max=max(data), samples=len(data)) if data else None
    metadata = dict(runs=len(runs), selftests=sum(v['selftests'] for v in runs), gmp_windows=sum(v['checked'] for v in runs),
                    active_mhz=spread('mhz'), active_power_w=spread('watts', lambda x: x <= 150),
                    active_temperature=spread('celsius'), excluded_power_outliers=sum((r['watts'] or 0) > 150 for r in gpu))
    totals = {}
    for path in a.experiment:
        group = [s for s in summaries if s['experiment'] == path.name]
        off = sum(s['off_engine_seconds'] for s in group)
        on = sum(s['on_engine_seconds'] for s in group)
        totals[path.name] = dict(off_seconds=off, on_seconds=on, gain_percent=100*(1-on/off), cases=len(group))
    result = dict(summaries=summaries, telemetry=metadata, totals=totals, profiles=profiles)
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    flat = [{k: v for k, v in row.items() if not isinstance(v, dict)} for row in summaries]
    with (a.output / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [s['case'].replace('m8011_cofactor', 'N7995').replace('_', ' ') + '\n' + s['experiment'] for s in summaries]
    y = np.arange(len(summaries))
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, len(y)*.58)))
    for side, delta, color in [('off', -.18, '#888888'), ('on', .18, '#287aaf')]:
        values = [s[side+'_engine_seconds'] for s in summaries]
        axes[0].barh(y+delta, values, height=.34, xerr=[s[side+'_engine_seconds_mad'] for s in summaries],
                     label=side, color=color, capsize=2)
    axes[0].set_xscale('log')
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend()
    axes[0].set_xlabel('Complete Stage2 init+main (s, log scale); median ± MAD')
    gains = [s['gain_percent'] for s in summaries]
    axes[1].barh(y, gains, color=['#999999' if int(s['on_graph']['launches']) == 0 else '#287aaf' if g > 0 else '#bc6556' for s, g in zip(summaries, gains)])
    axes[1].set_yticks(y, ['' for _ in y])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color='#555555', lw=.7)
    for pos, s in enumerate(summaries):
        gain = s['gain_percent']
        axes[1].text(gain+(.05 if gain >= 0 else -.05), pos,
            f"{gain:+.2f}% | graph launches={s['on_graph']['launches']}", va='center',
            ha='left' if gain >= 0 else 'right', fontsize=8)
    axes[1].set_xlim(min(gains)-4, max(gains)+6)
    axes[1].set_xlabel('Time saved (%); positive is faster. Grey = Graph inactive')
    fig.suptitle('Fixed-D complete Stage2 Graph A/B | RTX 4060 Laptop | default 55 W / 1800 MHz cap')
    fig.tight_layout(rect=[0, .02, 1, .96])
    a.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figure, dpi=160)
    plt.close(fig)
    # Non-overlapping pipeline components are displayed within each side only.
    # These host measurements need not sum exactly to engine total.
    fig, ax = plt.subplots(figsize=(15, max(6, len(y)*.55)))
    phases = ['f_tree_incl', 'inv', 'giant', 'gtrees', 'fold', 'descent', 'accum']
    colors = ['#a1bcd4', '#b9d8ae', '#dbaa5d', '#3d83af', '#997ab6', '#65b2a3', '#b87575']
    for side, delta in [('off', -.18), ('on', .18)]:
        left = np.zeros(len(y))
        for phase, color in zip(phases, colors):
            values = np.array([s[side+'_phases'][phase] / s[side+'_engine_seconds'] * 100 for s in summaries])
            ax.barh(y+delta, values, left=left, height=.34, color=color, label=phase if side == 'off' else None)
            left += values
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Reported component / complete engine time (%); upper=off, lower=on')
    ax.legend(ncol=4)
    ax.set_title('Stage2 components | independent medians; rounding/residual overhead may leave a gap')
    fig.tight_layout()
    fig.savefig(a.figure.with_name(a.figure.stem + '_phases.png'), dpi=160)
    plt.close(fig)
    systems = [(Path(name).name, value) for name, value in profiles.items() if 'kernels' in value]
    if systems:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for name, value in systems:
            # One forward tile only; no launch gaps or complete convolution.
            groups = {}
            for kernel in value['kernels']:
                if kernel['length'] == 2048 and '(bool)0' in kernel['name'] and kernel['batch'] in (720, 1440, 2880):
                    bucket = groups.setdefault(kernel['batch'], [0, 0])
                    bucket[0] += kernel['count']
                    bucket[1] += kernel['seconds']
            batches = sorted(groups)
            axes[0].plot(batches, [b*groups[b][0]/groups[b][1]/1e6 for b in batches], marker='o', label=name)
        axes[0].set_ylim(0, 6)
        axes[0].set_xticks([720, 1440, 2880])
        axes[0].set_xlabel('Actual transforms in one batch')
        axes[0].set_ylabel('Forward tile transforms/s (millions); L=2048')
        axes[0].set_title('Already beyond the small-batch throughput ramp')
        axes[0].legend(fontsize=8)
        y = np.arange(len(systems))
        active = [100*v['window']['event_union']/v['window']['span'] for _, v in systems]
        axes[1].barh(y, active, label='Own GPU event union', color='#287aaf')
        axes[1].barh(y, [100-v for v in active], left=active, label='No own GPU event', color='#cccccc')
        axes[1].set_yticks(y, [name for name, _ in systems])
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Own first-to-last GPU event window (%)')
        axes[1].set_title('No-own-event interval is not whole-card idle')
        axes[1].legend(fontsize=8)
        fig.suptitle('Nsight Systems diagnostics: profiler overhead excluded from A/B')
        fig.tight_layout()
        fig.savefig(a.figure.with_name(a.figure.stem + '_systems.png'), dpi=160)
        plt.close(fig)
    print(json.dumps(dict(totals=totals, telemetry=metadata), indent=2))


if __name__ == '__main__':
    main()
