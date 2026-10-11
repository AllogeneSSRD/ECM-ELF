"""Summarize frozen A/B logs, GPU telemetry and short-NTT throughput; export figures."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics as stats
import sys
import tomllib


def telemetry(experiment):
    with (experiment / 'gpu.csv').open() as f:
        records = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    output = []
    for r in records:
        r['unix'] = datetime.strptime(r['timestamp'], '%Y/%m/%d %H:%M:%S.%f').timestamp()
        for key in ('utilization.gpu [%]','clocks.current.sm [MHz]','power.draw [W]','memory.used [MiB]','temperature.gpu'):
            value = r[key].split()[0]
            r[key] = float(value) if value != '[N/A]' else None
        output.append(r)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, action='append', required=True)
    p.add_argument('--ntt-profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figure', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    summaries, rows, devices = [], [], []
    for experiment in a.experiment:
        summaries += json.loads((experiment / 'summary.json').read_text())
        device = telemetry(experiment)
        devices += device
        for line in (experiment / 'runs.jsonl').read_text().splitlines():
            row = json.loads(line)
            samples = [r for r in device if row['start_unix'] <= r['unix'] <= row['end_unix'] and r['utilization.gpu [%]'] >= 50]
            row['telemetry_active_samples'] = len(samples)
            for key, name in [('clocks.current.sm [MHz]','sm_mhz'),('power.draw [W]','power_w'),('temperature.gpu','temperature_c')]:
                values = [r[key] for r in samples if r[key] is not None]
                row[name] = stats.median(values) if values else None
            rows.append(row)
    output = []
    for s in summaries:
        r = dict(case=s['case'], bits=s['off']['actual_bits'], engine_gain_percent=s['engine_gain_percent'],
            result_gain_percent=s['result_gain_percent'], process_gain_percent=s['process_gain_percent'])
        for side, flag in [('off',0),('on',1)]:
            for key in ('engine_seconds','engine_seconds_mad','result_seconds','process_seconds','chunks','changed','reduce_seconds','full_peak_bytes'):
                r[side+'_'+key] = s[side][key]
            active = [v for v in rows if v['case'] == s['case'] and v['enabled'] == flag and not v['warmup'] and v['sm_mhz'] is not None]
            r[side+'_sm_mhz'] = stats.median(v['sm_mhz'] for v in active) if active else None
            r[side+'_power_w'] = stats.median(v['power_w'] for v in active) if active else None
            r[side+'_checks'] = stats.median(v['checked'] for v in rows if v['case']==s['case'] and v['enabled']==flag and not v['warmup'])
        output.append(r)
    with (a.output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    active = [r for r in devices if r['utilization.gpu [%]'] >= 50]
    metadata = dict(active_samples=len(active), active_sm_mhz=dict(min=min(r['clocks.current.sm [MHz]'] for r in active),
        median=stats.median(r['clocks.current.sm [MHz]'] for r in active), max=max(r['clocks.current.sm [MHz]'] for r in active)),
        active_power_w=dict(min=min(r['power.draw [W]'] for r in active), median=stats.median(r['power.draw [W]'] for r in active), max=max(r['power.draw [W]'] for r in active)),
        arithmetic_runs=len(rows), arithmetic_checks=sum(r['checked'] for r in rows), selftest_cases=sum(r['selftests'] for r in rows))
    (a.output / 'summary.json').write_text(json.dumps(dict(summary=output, telemetry=metadata, runs=rows), indent=2))
    with (a.output / 'ntt_counts.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case','bits','log2_length','length','convolutions','transforms','elements','logical_radix2_butterflies'])
        writer.writeheader()
        for s in summaries:
            for k, count in s['off']['histogram'].items():
                k = int(k)
                writer.writerow(dict(case=s['case'], bits=s['off']['actual_bits'], log2_length=k,
                    length=1<<k, convolutions=count//3, transforms=count, elements=count*(1<<k),
                    logical_radix2_butterflies=count*(1<<k)*k//2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), gridspec_kw={'height_ratios':[1.25,1]})
    labels = [r['case'].replace('m8011_cofactor', 'N7995').replace('m', 'N', 1).replace('_', ' ') for r in output]
    y = np.arange(len(output))
    ax = axes[0,0]
    for delta, side, color in [(-.18,'off','#777777'),(.18,'on','#2479b3')]:
        values = [r[side+'_engine_seconds'] for r in output]
        ax.barh(y+delta, values, height=.35, color=color, label=side,
                xerr=[r[side+'_engine_seconds_mad'] for r in output], capsize=2)
        for pos, value in zip(y+delta, values):
            ax.text(value*1.02, pos, f'{value:.3f}', va='center', fontsize=8)
    ax.set_xscale('log');ax.set_xlim(.08, max(r['on_engine_seconds'] for r in output)*1.7)
    ax.set_yticks(y, labels, fontsize=8);ax.invert_yaxis();ax.legend()
    ax.set_xlabel('Engine total (s, log scale)');ax.set_title('Stage2 total: median of 3, one warmup excluded')
    ax = axes[0,1]
    gains = [r['engine_gain_percent'] for r in output]
    ax.barh(y, gains, color=['#999999' if not r['on_changed'] else '#2479b3' if v>0 else '#b75446' for v,r in zip(gains,output)])
    ax.axvline(0,color='#666666',lw=.8);ax.set_yticks(y,labels,fontsize=8);ax.invert_yaxis()
    for pos, value, r in zip(y, gains, output):
        ax.text(value + (.1 if value>=0 else -.1), pos, f'{value:+.2f}% (changed={int(r["on_changed"])})',
                va='center', ha='left' if value>=0 else 'right', fontsize=8)
    ax.set_xlim(min(gains)-4,max(gains)+4);ax.set_xlabel('Time saved (%); positive is faster')
    ax.set_title('No causal gain when changed=0; small deltas need more samples')
    ax = axes[1,0]
    profile=tomllib.loads(a.ntt_profile.read_text(encoding='utf-8-sig'))
    points=[dict(table,**dict(zip(table['columns'],row))) for node in profile['ntt'].values() for table in node['table'].values() for row in table['rows']]
    for k in sorted(set(r['log2_length'] for r in points)):
        group=sorted((r for r in points if r['log2_length']==k),key=lambda r:r['batch'])
        ax.plot([r['batch'] for r in group],[r['conv_iter_per_s'] for r in group],marker='o',label=f'2^{k}')
    ax.set_xscale('log',base=2);ax.set_xticks([1,4,8,16,32,64],[1,4,8,16,32,64]);ax.legend(title='NTT length')
    ax.set_xlabel('Convolutions per batch');ax.set_ylabel('Conv/s (2 forward + 1 inverse)')
    ax.set_title('Fresh NTT calibration: GPU events, median of 19')
    ax = axes[1,1]
    for side,color in [('off','#777777'),('on','#2479b3')]:
        delta=-.18 if side=='off' else .18
        ax.barh(y+delta,[r[side+'_chunks'] for r in output],height=.35,label=side,color=color)
    ax.set_yticks(y,labels,fontsize=8);ax.invert_yaxis();ax.legend()
    ax.set_xlabel('S4 chunks per full curve (count)');ax.set_title('Same mathematical NTT counts; different chunk boundaries')
    fig.suptitle('Short NTT batching A/B | RTX 4060 Laptop | sm89 / CUDA 13.3 | core usually 1800 MHz',fontsize=13)
    fig.text(.01,.01,'Source: frozen short_ntt_ab_20261010 experiments; Stage1 excluded; process and result times in CSV.\n'
        '256 MiB and fold=0 cases explicitly ignore profile memory metadata; current admission limits remain enforced.',fontsize=8)
    fig.tight_layout(rect=[0,.065,1,.96]);a.figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.figure,dpi=160);plt.close(fig)
    print(json.dumps(dict(summary=output,telemetry=metadata),indent=2))


if __name__ == '__main__':
    main()
