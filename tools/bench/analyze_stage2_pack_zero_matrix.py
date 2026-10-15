"""Aggregate complete pack/zero studies without pooling separate rechecks."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def telemetry(path):
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig')))
    active = []
    for original in rows:
        row = {k.strip(): v.strip() for k, v in original.items()}
        try:
            if float(row['utilization.gpu [%]'].split()[0]) >= 80:
                active.append((float(row['clocks.current.sm [MHz]'].split()[0]),
                               float(row['clocks.current.memory [MHz]'].split()[0]),
                               float(row['power.draw [W]'].split()[0])))
        except (ValueError, KeyError):
            continue
    if not active:
        return dict(high_load_samples=0)
    return dict(high_load_samples=len(active),
                sm_mhz=dict(min=min(x[0] for x in active), max=max(x[0] for x in active),
                            median=statistics.median(x[0] for x in active)),
                memory_mhz=dict(min=min(x[1] for x in active), max=max(x[1] for x in active),
                                median=statistics.median(x[1] for x in active)),
                power_w=dict(min=min(x[2] for x in active), max=max(x[2] for x in active),
                             median=statistics.median(x[2] for x in active)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', action='append', required=True, help='Unique label=complete study directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--prefix', default='stage2_pack_zero_expanded')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.prefix):
        p.error('simple figure prefix required')
    studies, rows = {}, []
    for spec in a.study:
        label, sep, source = spec.partition('=')
        if not sep or not label or label in studies:
            p.error('unique label=directory required')
        root = Path(source).resolve()
        full = json.loads((root/'summary.json').read_text(encoding='utf-8'))
        if not full['complete'] or full['identity']['diagnostic'] or full['identity']['variant'] != 'pack-zero':
            raise ValueError('complete, non-diagnostic pack-zero study required')
        runs = [json.loads(s) for s in (root/'runs.jsonl').read_text().splitlines()]
        if len(runs) != sum(s['off']['samples']+s['on']['samples'] for s in full['summaries']):
            raise ValueError('sample counts differ')
        selected = []
        for item in full['summaries']:
            off, on, case = item['off'], item['on'], item['input']
            if off['samples'] != on['samples']:
                raise ValueError('unbalanced samples')
            samples = [r for r in runs if r['case'] == item['case']]
            invariant = [(r['correctness'], r['histogram'], r['convolutions'], r['chunks'],
                          r['selftests'], r['checked'], r['launches']) for r in samples]
            if any(v != invariant[0] for v in invariant):
                raise ValueError('arithmetic/workload mismatch')
            # Module peak only, not a process-wide memory measurement.
            if off['full_peak_bytes'] != on['full_peak_bytes']:
                raise ValueError('NTT module peak changed')
            r = dict(study=label, case=item['case'], target_bits=int(samples[0]['shape']['N_bits']),
                     carrier=case.get('carrier', 0), b2=case['b2'], requested_d=case['d'],
                     batch_mb=case['batch_mb'], arena_mb=case.get('arena_mb', 6300), fold_mb=case.get('fold_mb', 640),
                     off_seconds=off['engine_seconds'], on_seconds=on['engine_seconds'],
                     off_process_seconds=off['process_seconds'], on_process_seconds=on['process_seconds'],
                     process_gain_percent=100*(1-on['process_seconds']/off['process_seconds']),
                     off_mad=off['engine_seconds_mad'], on_mad=on['engine_seconds_mad'],
                     gain_percent=item['gain_percent'], samples_per_side=off['samples'],
                     ntt_full_peak_bytes=on['full_peak_bytes'],
                     environment=json.dumps(case.get('environment', {}), sort_keys=True),
                     binary_sha256=full['identity']['identity']['binary_sha256'])
            selected.append(r)
            rows.append(r)
        off = sum(r['off_seconds'] for r in selected)
        on = sum(r['on_seconds'] for r in selected)
        studies[label] = dict(source=str(root), device=full['identity']['device'], cases=len(selected),
                              curves=len(runs), off_sum_case_medians=off, on_sum_case_medians=on,
                              gain_percent=100*(1-on/off), positive_cases=sum(r['gain_percent']>0 for r in selected),
                              min_gain_percent=min(r['gain_percent'] for r in selected),
                              max_gain_percent=max(r['gain_percent'] for r in selected),
                              telemetry=telemetry(root/'gpu.csv'),
                              binary_sha256=full['identity']['identity']['binary_sha256'])
        off_process=sum(r['off_process_seconds'] for r in selected)
        on_process=sum(r['on_process_seconds'] for r in selected)
        studies[label].update(off_sum_process_medians=off_process, on_sum_process_medians=on_process,
                              process_gain_percent=100*(1-on_process/off_process))
    a.output.mkdir(parents=True, exist_ok=False)
    a.figures.mkdir(parents=True, exist_ok=True)
    with (a.output/'matrix.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (a.output/'summary.json').write_text(json.dumps(dict(studies=studies, rows=rows), indent=2)+'\n')
    # Each panel is its own study; follow-up samples do not replace formal cells.
    fig, axes = plt.subplots(len(studies), 1, figsize=(14, max(4, 3.8*len(studies))), squeeze=False,
                             layout='constrained')
    for ax, (label, meta) in zip(axes[:, 0], studies.items()):
        group = [r for r in rows if r['study'] == label]
        errors = [100*r['on_seconds']/r['off_seconds']*np.hypot(
            r['on_mad']/r['on_seconds'], r['off_mad']/r['off_seconds']) for r in group]
        ax.bar(range(len(group)), [r['gain_percent'] for r in group], yerr=errors, capsize=2,
               color=['#237b69' if r['gain_percent']>=0 else '#b75b51' for r in group])
        ax.axhline(0, color='#555', lw=.8)
        labels = [f"{r['target_bits']}b"+(f"/M{r['carrier']}" if r['carrier'] else '')+f"\nB2={r['b2']:.1e}"
                  if re.fullmatch(r'm\d+_n\d+_b\d+(?:_carrier)?', r['case']) else r['case'].replace('_','\n')
                  for r in group]
        ax.set(xticks=range(len(group)), xticklabels=labels,
               ylabel='Engine time reduction (%)', title=f"{label} | {meta['cases']} cases, {meta['curves']} curves | "
               f"sum of medians {meta['off_sum_case_medians']:.3f} -> {meta['on_sum_case_medians']:.3f}s "
               f"({meta['gain_percent']:+.2f}%)")
        ax.tick_params(axis='x', labelsize=7, rotation=25)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Pack + zero fusion | same-binary interleaved A/B | no discarded warmup\n'
                 'Error bars: propagated MAD scale, not confidence intervals; clock details in summary.json')
    fig.savefig(a.figures/(a.prefix+'.png'), dpi=170); plt.close(fig)
    # Heatmaps retain exact N and arithmetic family, rather than merging carrier with generic.
    maps = []
    for label in studies:
        for carrier in (False, True):
            group = [r for r in rows if r['study']==label and bool(r['carrier'])==carrier
                     and re.fullmatch(r'm\d+_n\d+_b\d+(?:_carrier)?', r['case'])]
            if len(group)<4:
                continue
            maps.append((label+(' carrier' if carrier else ' generic'), group))
    if maps:
        ncols=min(2,len(maps)); nrows=(len(maps)+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.2*ncols, 4.7*nrows), squeeze=False, layout='constrained')
        used_axes=axes.ravel().tolist()[:len(maps)]
        for ax in axes.ravel().tolist()[len(maps):]:ax.set_axis_off()
        limit = max(4, max(abs(r['gain_percent']) for _, g in maps for r in g))
        for ax, (label, group) in zip(used_axes, maps):
            widths = sorted({r['target_bits'] for r in group}); bounds = sorted({r['b2'] for r in group})
            matrix = np.full((len(widths), len(bounds)), np.nan)
            for r in group:
                i,j = widths.index(r['target_bits']), bounds.index(r['b2'])
                if not np.isnan(matrix[i,j]): raise ValueError('duplicate heatmap cell')
                matrix[i,j] = r['gain_percent']
                ax.text(j,i,f"{r['gain_percent']:+.2f}%",ha='center',va='center',fontsize=9)
            im = ax.imshow(matrix, cmap='BrBG', vmin=-limit, vmax=limit, aspect='auto')
            ax.set(xticks=range(len(bounds)), xticklabels=[f'{x:.1e}' for x in bounds],
                   yticks=range(len(widths)), yticklabels=widths, xlabel='B2', ylabel='Actual target N bits', title=label)
            ax.tick_params(axis='x', rotation=30)
        fig.colorbar(im, ax=used_axes, label='Engine time reduction (%)', shrink=.7)
        fig.suptitle('Pack + zero fusion | exact width / B2 cells | positive = faster')
        fig.savefig(a.figures/(a.prefix+'_heatmap.png'), dpi=170); plt.close(fig)
    for label, meta in studies.items():
        print(label, json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()
