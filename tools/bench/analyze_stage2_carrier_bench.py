"""Summarize a complete same-binary carrier/workspace matrix and draw its phases.

Only formal timing rows contribute to means. CUDA reduction events are nested
diagnostics, never added to parent phase totals or allocation peaks.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics

from bench_stage2_production import fields, sha


def summarize(rows, section, field):
    v = [float(r[section][field]) for r in rows]
    return dict(mean=statistics.mean(v), min=min(v), max=max(v),
                stdev=statistics.stdev(v) if len(v) > 1 else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figure-prefix', type=Path)
    a = p.parse_args()
    data = json.loads(a.input.read_text(encoding='utf-8'))
    if not data['complete'] or data['mode'] != 'timing':
        raise ValueError('a complete formal timing matrix is required')
    if data.get('memory_ledger'):
        raise ValueError('allocation-ledger runs are diagnostics, not formal timing evidence')
    timed = [r for r in data['runs'] if r['category'] == 'timing']
    comparison = data.get('comparison', 'carrier')
    keys = {'carrier': ('generic', 'carrier'), 'workspace-bq': ('three_buffer', 'two_buffer'),
            'plan': ('baseline_d', 'candidate_d'),
            'chunk': ('legacy_chunk','workspace_chunk'),
            'phase-output': ('retained_output','trimmed_output'),
            'owner-cache': ('kept_cache','trimmed_cache'),
            'giant-chunk': ('legacy_points','bounded_points')}[comparison]
    arm_labels = {'carrier': ['Generic target N', 'Mersenne carrier M'],
                  'workspace-bq': ['Three buffers A/B/Q', 'Two buffers A/(B=Q)'],
                  'plan': [f'D={data["input"]["D"]}', f'D={data["input"].get("candidate_D", 0)}'],
                  'chunk': ['Legacy three-buffer estimate','Physical workspace estimate'],
                  'phase-output': ['Retained S4 output','Reclaimed S4 output'],
                  'owner-cache': ['Keep cached shapes','Reclaim cold shapes for owner'],
                  'giant-chunk': ['Round points up to batches','Round points down to budget']}[comparison]
    sequence = [keys[i] for i in (0, 1, 1, 0, 1, 0, 0, 1)]
    if [r['key'] for r in timed] != sequence:
        raise ValueError('formal ABBA+BAAB sequence changed')
    for r in data['runs']:
        if sha(r['log']) != r['log_sha256'] or sha(r['debug_log']) != r['debug_sha256']:
            raise ValueError('raw log changed: ' + r['name'])
    result = dict(input_sha256=sha(a.input), source_sha256=sha(__file__), input=data['input'],comparison=comparison,
                  identity=data['identity'], summary=data['summary'], arms={})
    for key in keys:
        rows = [r for r in timed if r['key'] == key]
        arm = dict(n=len(rows), wall={k: summarize(rows, 'wall', k) for k in ('shape', 'init', 'main', 'total')},
                   phases={k: summarize(rows, 'phases', k) for k in ('giant', 'gtrees', 'fold', 'descent', 'inv', 'accum', 'name')},
                   reduction=summarize(rows, 'coverage', 't_reduce'),
                   workspace_big_peak_mib=summarize(rows, 'workspace', 'big_peak_bytes'),
                   workspace_full_peak_mib=summarize(rows, 'workspace', 'full_peak_bytes'),
                   owner_peak_mib=summarize(rows, 'fold', 'peak_bytes'),
                   modulus=rows[0]['modulus'], samples=[float(r['wall']['total']) for r in rows])
        for metric in ('workspace_big_peak_mib', 'workspace_full_peak_mib', 'owner_peak_mib'):
            arm[metric] = {k: v/(1 << 20) for k, v in arm[metric].items()}
        arm['misc_seconds'] = arm['wall']['total']['mean']-arm['wall']['init']['mean']-sum(v['mean'] for v in arm['phases'].values())
        if comparison in ('workspace-bq', 'plan','chunk','phase-output','owner-cache','giant-chunk'):
            arm['layout'] = rows[0]['layout']
        if comparison in ('owner-cache','giant-chunk'):
            arm['cache_trim'] = [r['cache_trim'] for r in rows]
            arm['cache_stats'] = [r['cache_stats'] for r in rows]
            arm['phase_memory'] = [r['phase_memory'] for r in rows]
            arm['coverage'] = rows[0]['coverage']
        if comparison=='giant-chunk':
            arm['point_plan']=rows[0]['point_plan'];arm['point_done']=rows[0]['point_done']
            arm['device_leaf']=rows[0]['device_leaf'];arm['giant_seed']=rows[0]['giant_seed']
            arm['coordinate_peak_mib']={k:v/(1<<20) for k,v in summarize(rows,'device_leaf','coord_peak_bytes').items()}
        if comparison == 'phase-output':
            arm['phase_trim'] = rows[0]['phase_trim']
            arm['coverage'] = rows[0]['coverage']
        if comparison == 'chunk':
            arm['chunk_plan'] = rows[0]['chunk_plan']
            arm['coverage'] = rows[0]['coverage']
            arm['reduce_hook_calls'] = rows[0]['reduce_hook_calls']
            arm['owned_subset_observed_peak_mib'] = {
                k: v/(1 << 20) for k,v in summarize(rows,'chunk_plan','owned_subset_observed_peak_bytes').items()}
            shapes = {}
            for row in rows:
                text = Path(row['debug_log']).read_text(encoding='utf-8')
                for line in text.splitlines():
                    if not line.startswith('s4_reduce_stats:'): continue
                    shape = fields(line,'s4_reduce_stats')
                    shapes.setdefault(shape['P'],[]).append(shape)
            arm['reduction_shapes'] = {}
            for degree, records in sorted(shapes.items(),key=lambda v:int(v[0])):
                if len(records) != len(rows): raise ValueError('shape missing from an arm: '+degree)
                for field in ('launches','coeffs','gmp_checked'):
                    if len({s[field] for s in records}) != 1:
                        raise ValueError('shape work changed within an arm: '+degree+'/'+field)
                arm['reduction_shapes'][degree] = dict(
                    calls=int(records[0]['launches']), coefficients=int(records[0]['coeffs']),
                    checked=int(records[0]['gmp_checked']),
                    mean_kernel_seconds=statistics.mean(float(s['t_reduce']) for s in records))
        arm['shape'] = rows[0]['shape']
        arm['residency'] = {k: [r.get(k) for r in rows] for k in ('fold', 'root', 'frontier')}
        if arm['misc_seconds'] < -0.05:
            raise ValueError('parent phase budgets overlap unexpectedly')
        samples = []
        for row in rows:
            telemetry = a.input.parent/(row['name']+'_telemetry.json')
            if telemetry.exists():
                for sample in json.loads(telemetry.read_text(encoding='utf-8')):
                    cells = [v.strip() for v in sample.get('csv', '').split(',')]
                    if sample.get('returncode') != 0 or len(cells) != 7:
                        continue
                    try:
                        samples.append(dict(power_W=float(cells[2]), clock_MHz=float(cells[3]),
                                            temperature_C=float(cells[4]), utilization_percent=float(cells[5]),
                                            memory_used_mib=float(cells[6])))
                    except ValueError:
                        pass
        busy = [s for s in samples if s['utilization_percent'] >= 90]
        arm['telemetry'] = dict(samples=len(samples), busy_samples=len(busy),
            sampled_device_memory_peak_mib=max(s['memory_used_mib'] for s in samples) if samples else None,
            busy={k: dict(mean=statistics.mean(s[k] for s in busy), min=min(s[k] for s in busy),
                          max=max(s[k] for s in busy)) for k in ('power_W', 'clock_MHz', 'temperature_C')} if busy else {})
        result['arms'][key] = arm
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    with a.output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['arm', 'metric', 'mean_seconds', 'min_seconds', 'max_seconds', 'stdev_seconds'])
        for key, arm in result['arms'].items():
            for group in ('wall', 'phases'):
                for metric, values in arm[group].items():
                    writer.writerow([key, group+'.'+metric, *[values[k] for k in ('mean', 'min', 'max', 'stdev')]])
    if a.figure_prefix:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout='constrained')
        colors = ['#7d8e9c', '#b1bcc5', '#366c93', '#bb8250', '#7b9b8b', '#b8b09b', '#d7dce0']
        labels = [('init', 'Initialization'), ('giant', 'Giant points'), ('gtrees', 'G trees'),
                  ('fold', 'Fold'), ('descent', 'Descent'), ('inv', 'Polynomial inverse'), ('other', 'Other')]
        for index, (key, arm) in enumerate(result['arms'].items()):
            values = [arm['wall']['init']['mean']] + [arm['phases'][k]['mean'] for k in ('giant', 'gtrees', 'fold', 'descent', 'inv')]
            values.append(arm['wall']['total']['mean']-sum(values))
            seconds, percent = 0, 0
            total = arm['wall']['total']['mean']
            for value, (_, label), color in zip(values, labels, colors):
                axes[0].bar(index, value, bottom=seconds, color=color, label=label if index == 0 else None, width=.55)
                axes[1].bar(index, 100*value/total, bottom=percent, color=color, width=.55)
                if 100*value/total > 7:
                    axes[1].text(index, percent+50*value/total, f'{100*value/total:.1f}%', ha='center', va='center', fontsize=9)
                seconds += value
                percent += 100*value/total
            axes[0].errorbar(index, total, yerr=arm['wall']['total']['stdev'], color='#252525', capsize=4)
            axes[0].text(index, total+4, f'{total:.2f} s', ha='center')
        for ax in axes:
            ax.set_xticks([0, 1], arm_labels)
            ax.set_xlabel({'carrier': 'Arithmetic backend (same saved target N)',
                          'workspace-bq': 'Workspace layout (same arithmetic backend)',
                          'plan': 'Fixed D (same arithmetic, two-buffer pool and budgets)',
                          'chunk': 'Chunk policy (same D, two-buffer pool and budgets)',
                          'phase-output': 'Phase output lifetime (same D and budgets)',
                          'owner-cache': 'Cold cache admission (same D and budgets)',
                          'giant-chunk': 'Point chunk rounding (same D and budgets)'}[comparison])
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].set_ylabel('Mean complete Stage2 wall time (s)')
        axes[0].set_ylim(0, max(r['wall']['total']['mean'] for r in result['arms'].values())*1.17)
        axes[0].set_title('Complete wall time and parent phases', loc='left')
        axes[1].set_ylabel('Share of complete Stage2 wall time (%)')
        axes[1].set_title('Phase proportions', loc='left')
        axes[1].set_ylim(0, 100)
        fig.legend(loc='outside lower center', ncol=4, frameon=False)
        first = timed[0]
        nbits = int(first['result']['N_hex'], 16).bit_length()
        text = Path(first['log']).read_text(encoding='utf-8')
        gpu = re.search(r'stage2_real: device=\d+ \((.*?)\)', text)
        gpu_name = gpu[1] if gpu else 'device '+str(first['result']['device'])
        date = datetime.fromtimestamp(first['result']['timestamp_ms']/1000, timezone.utc).date().isoformat()
        carrier = data['input']['carrier_exponent']
        d_label = str(data['input']['D'])+(f' / {data["input"]["candidate_D"]}' if comparison == 'plan' else '')
        modulus_label = f'{nbits}-bit target'+(f' in M{carrier}' if carrier else ' (generic arithmetic)')
        fig.suptitle(f'{modulus_label} · {gpu_name}\n'
                     f'B2={data["input"]["B2"]:.1e}, D={d_label} · {date} UTC · ABBA+BAAB, n=4 per arm\n'
                     'Warmups excluded; error bars = sample SD', fontsize=11)
        a.figure_prefix.parent.mkdir(parents=True, exist_ok=True)
        for ext in ('png', 'svg'):
            fig.savefig(a.figure_prefix.with_suffix('.'+ext), dpi=170)
        plt.close(fig)
        if comparison in ('workspace-bq', 'plan','chunk','phase-output','owner-cache','giant-chunk'):
            fig, ax = plt.subplots(figsize=(8.8, 5.3), layout='constrained')
            metrics = [('workspace_big_peak_mib', 'Big buffers', '#366c93'),
                       ('workspace_full_peak_mib', 'Whole NTT workspace', '#7b9b8b'),
                       ('sampled_device_memory_peak_mib', 'Sampled GPU usage (2 s)', '#bb8250')]
            if comparison == 'chunk':
                metrics.append(('owned_subset_observed_peak_mib','Observed NTT + S4 subset','#8a789b'))
            if comparison in ('phase-output','owner-cache','giant-chunk'):
                metrics.append(('owner_peak_mib','Resident fold owner','#8a789b'))
            if comparison=='giant-chunk':
                metrics.append(('coordinate_peak_mib','Giant X/Z coordinates','#ad5c76'))
            for mi, (metric, label, color) in enumerate(metrics):
                for ai, (_, arm) in enumerate(result['arms'].items()):
                    value = arm['telemetry'][metric] if mi == 2 else arm[metric]['mean']
                    if value is None:
                        continue
                    xpos = ai+(mi-(len(metrics)-1)/2)*.19
                    ax.bar(xpos, value, width=.18, color=color, label=label if ai == 0 else None)
                    ax.text(xpos, value+45, f'{value:.0f}', ha='center', fontsize=9)
            ax.set_xticks([0, 1], arm_labels)
            ax.set_ylabel('MiB (overlapping series; never summed)')
            ax.set_title(f'{modulus_label} · B2={data["input"]["B2"]:.1e}, D={d_label}\n'
                         'Module capacity peaks and sampled device usage', loc='left')
            ax.margins(y=.2)
            ax.spines[['top', 'right']].set_visible(False)
            fig.legend(loc='outside lower center', ncol=2, frameon=False)
            for ext in ('png', 'svg'):
                fig.savefig(a.figure_prefix.parent/(a.figure_prefix.name+'_memory.'+ext), dpi=170)
            plt.close(fig)
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
