"""Read completed seed A/B evidence; export CSV, phase figures and a report fragment."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import statistics


def phases(row):
    s=row['stats'];split=s['real_batched_split'];wall=s['real_batched_wall']
    baby=sum(float(s['real_baby'][k]) for k in ('ladder','affine'))
    result={'Baby':baby,'F-tree/init':float(row['clock']['init'])-baby,
            'Inverse':float(split['inv']),'Giant/leaves':float(split['giant'])+float(wall['gleaves']),
            'G trees':float(split['gtrees']),'Fold':float(split['fold']),
            'Descent':float(split['descent']),'Acc/GCD':float(split['accum'])}
    result['Other']=row['seconds']-sum(result.values())
    if min(result.values())<-.02:raise ValueError('Phase timers overlap or have incompatible boundaries')
    return result


def curve_wall(row):
    text=(Path(row['folder'])/'console.log').read_text(encoding='utf-8')
    matches=re.findall(r'^curve_done:.*?wall=([\d.]+)',text,re.M)
    if len(matches)!=1:raise ValueError('Missing complete outer curve clock')
    return float(matches[0])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','output','figures'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--publish',type=Path)
    p.add_argument('--retest',type=Path,action='append',default=[])
    a=p.parse_args();d=json.loads((a.study/'summary.json').read_text())
    if not d['complete'] or d['diagnostic'] or d['failures']:raise ValueError('Need complete clean timing study')
    extras=[]
    for path in a.retest:
        other=json.loads((path/'summary.json').read_text())
        if not other['complete'] or other['diagnostic'] or other['failures'] or other.get('component','seed')!=d.get('component','seed') or other.get('reduce_kernel')!=d.get('reduce_kernel') or any(other[k]!=d[k] for k in
            ('exe_sha256','dll_sha256','source_matrix_sha256')):raise ValueError('Incompatible retest')
        if any(other['protocol'].get(k,1 if k=='device' else None)!=d['protocol'].get(k,1 if k=='device' else None)
               for k in ('device','batch_mb','arena_mb','fold_mb')):raise ValueError('Retest changed device or memory budgets')
        for cell in other['cells']:
            original=next((c for c in d['cells'] if c['id']==cell['id']),None)
            if original is None or any(cell[k]!=original[k] for k in ('D','P','B2','carrier','layout','chain')):
                raise ValueError('Retest changed geometry')
        extras.extend(other['runs'])
    statistic=statistics.median if extras else statistics.mean
    def central(v):
        # The central actual runs preserve additive phase accounting. Taking a
        # separate median of each phase would not sum to the median total.
        ordered=sorted(v,key=lambda r:r['seconds'])
        return [ordered[(len(v)-1)//2],ordered[len(v)//2]] if extras else v
    a.output.mkdir(parents=True,exist_ok=False);a.figures.mkdir(parents=True,exist_ok=False)
    component=d.get('component','seed');title='S4 reduction' if component=='reduce' else 'seed'
    prefix='stage2_s4_coop' if component=='reduce' else 'stage2_seed_coop'
    rows=[]
    for cell in d['cells']:
        case=next(c for c in d['cases'] if c['id']==cell['case'])
        group={m:[r for r in d['runs'] if r['cell']==cell['id'] and r['mode']==m] for m in ('scalar','coop32')}
        if any(len(v)!=d['protocol']['repeats'] for v in group.values()):raise ValueError('Incomplete A/B')
        for mode in group:group[mode].extend(r for r in extras if r['cell']==cell['id'] and r['mode']==mode)
        if len(group['scalar'])!=len(group['coop32']):raise ValueError('Unpaired retest')
        means={m:statistic(r['seconds'] for r in v) for m,v in group.items()}
        curve_means={m:statistic(curve_wall(r) for r in v) for m,v in group.items()}
        phase_means={m:{k:statistics.mean(phases(r)[k] for r in central(v)) for k in phases(v[0])} for m,v in group.items()}
        rows.append(dict(cell=cell['id'],target_bits=case['bits'],arithmetic_bits=cell['carrier'] or case['bits'],
            carrier=bool(cell['carrier']),D=cell['D'],P=cell['P'],B2=cell['B2'],
            scalar_seconds=means['scalar'],coop_seconds=means['coop32'],
            samples_per_backend=len(group['scalar']),s4_seconds={m:statistic(float(r['stats']['s4_multiply_stats']['t_reduce']) for r in v) for m,v in group.items()},
            scalar_curve_seconds=curve_means['scalar'],coop_curve_seconds=curve_means['coop32'],
            reduction_percent=100*(1-means['coop32']/means['scalar']),phases=phase_means))
    totals={m:sum(r[m+'_seconds'] for r in rows) for m in ('scalar','coop')}
    device=d['protocol'].get('device',1)
    first_log=(Path(d['runs'][0]['folder'])/'debug.log').read_text(encoding='utf-8')
    device_name=re.search(r'stage2_real: device=\d+ \(([^)]+)\)',first_log)
    device_label='GPU'+str(device)+(' / '+device_name[1] if device_name else '')
    if d.get('approved_background_snapshot'):device_label+=' / desktop background'
    result=dict(complete=True,source=str(a.study.resolve()),retests=[str(p.resolve()) for p in a.retest],statistic='median' if extras else 'mean',exe_sha256=d['exe_sha256'],totals=totals,
                device=device,device_label=device_label,reduce_kernel=d.get('reduce_kernel'),protocol=d['protocol'],
                reduction_percent=100*(1-totals['coop']/totals['scalar']),rows=rows)
    result['curve_totals']={m:sum(r[m+'_curve_seconds'] for r in rows) for m in ('scalar','coop')}
    result['curve_reduction_percent']=100*(1-result['curve_totals']['coop']/result['curve_totals']['scalar'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    columns=[k for k in rows[0] if k not in ('phases','s4_seconds')]
    with (a.output/'cells.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        writer.writerows({k:r[k] for k in columns} for r in rows)
    shutil.copy2(__file__,a.output/'analyzer_snapshot.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for carrier,ax in zip((False,True),axes):
        group=sorted((r for r in rows if r['carrier']==carrier),key=lambda r:r['target_bits'])
        x=[r['target_bits'] for r in group]
        for key,label,color in (('scalar_seconds','Scalar '+title,'#bc4b51'),('coop_seconds','Cooperative '+title,'#2a9d8f')):
            ax.plot(x,[r[key] for r in group],'o-',label=label,color=color)
        ax.set(title='Mersenne carrier' if carrier else 'Ordinary modulus',xlabel='Actual target bits',ylabel='Engine seconds')
        ax.xaxis.set_major_locator(MultipleLocator(1000));ax.grid(alpha=.2);ax.legend()
    sample_counts=sorted({r['samples_per_backend'] for r in rows})
    fig.suptitle('Fixed D/layout/chain; baby32/giant32'+('/seed32' if component=='reduce' else '')+
        '\n'+device_label+'; clocks in telemetry; n='+','.join(map(str,sample_counts))+('; medians with retests' if extras else ''))
    names=[]
    for suffix in ('png','svg'):
        name=prefix+'_times.'+suffix;fig.savefig(a.figures/name,dpi=180);names.append(name)
    plt.close(fig)
    top=max(r['target_bits'] for r in rows);groups=[r for r in rows if r['target_bits']==top]
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    labels=[];ys=[];left=[]
    for row in groups:
        for mode in ('scalar','coop32'):
            labels.append(('Carrier' if row['carrier'] else 'Ordinary')+' / '+mode)
            ys.append(row['phases'][mode]);left.append(0.)
    for i,key in enumerate(ys[0]):
        values=[r[key] for r in ys]
        ax.barh(range(len(ys)),values,left=left,label=key,color=plt.get_cmap('tab20')(i))
        left=[x+y for x,y in zip(left,values)]
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Engine seconds',
           title=f'{top}-bit target: mutually separated phase groups (nested timers excluded)')
    ax.invert_yaxis();ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.12));ax.grid(axis='x',alpha=.2)
    for suffix in ('png','svg'):
        name=prefix+'_phases.'+suffix;fig.savefig(a.figures/name,dpi=180);names.append(name)
    plt.close(fig)
    if a.publish:
        a.publish.mkdir(parents=True,exist_ok=True)
        for name in names:
            if name.endswith('.png'):shutil.copy2(a.figures/name,a.publish/name)
    lines=[f'| Target bits | Arithmetic | D | Scalar {title} s | Cooperative {title} s | Reduction |',
           '| ---: | --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['target_bits']} | {'carrier' if r['carrier'] else 'ordinary'} | {r['D']} | {r['scalar_seconds']:.3f} | {r['coop_seconds']:.3f} | {r['reduction_percent']:.2f}% |")
    (a.output/'report_fragment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
