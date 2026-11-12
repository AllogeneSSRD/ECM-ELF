"""Re-audit held-out point-backend curves and publish small-B2 comparisons."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import shutil
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

from bench_stage2_backend_matrix import capture,learned
from bench_stage2_serial_matrix import INPUT, shape_key
from bench_stage2_version import load,write
from bench_stage2_point_routes import sha
from bench_stage2_runtime_matrix import completed_records
from analyze_stage2_point_matrix import partition


def audit(root):
    plan=load(root/'plan.json')
    if not plan['complete'] or plan['failures'] or len(plan['cells'])!=26 or len(plan['training'])!=26 or len(plan['decisions'])!=26:
        raise ValueError('Need complete 13-width ordinary/carrier evidence')
    if sha(root/'bin/ecm_cuda_stage2.exe')!=plan['exe_sha256'] or sha(root/'bin/gmp-10.dll')!=plan['dll_sha256'] or sha(root/'source_plan.json')!=plan['source_sha256']:
        raise ValueError('Frozen source/binary mismatch')
    signatures={};clocks=Counter();checked=0;selftests=0;calls=0
    for entry in plan['training']+plan['evaluations']:
        cell=next(c for c in plan['cells'] if c['id']==entry['cell']);case=next(c for c in plan['cases'] if c['id']==cell['case'])
        folder=Path(entry['folder'])
        if sha(Path(case['save']))!=case['save_sha256']:raise ValueError('Save changed')
        for name,digest in entry['file_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('Evidence changed: '+str(folder/name))
        rows=capture(plan,root,folder,case,cell)
        if rows!=entry['rows']:raise ValueError('Capture differs from raw receipt/log')
        if entry['label']=='training' and not learned(completed_records(folder/'result.jsonl'),8192<(cell['carrier'] or case['bits'])<=16384):
            raise ValueError('Incomplete backend qualification')
        for row in rows:
            signature=(row['root'],row['ntt'],row['s4']['coeffs_reduced'],[(p['npts'],p['route'],p['block']) for p in row['policies']])
            if cell['id'] in signatures and signatures[cell['id']]!=signature:raise ValueError('Point backend changed mathematics/workload')
            signatures[cell['id']]=signature;checked+=int(row['s4']['gmp_checked'])
            selftests+=int(row['s4']['gmp_selftest_cases']);partition(row)
        calls+=len(rows)
        for sample in (folder/'telemetry.jsonl').read_text(encoding='utf-8').splitlines():
            sample=json.loads(sample)
            if sample.get('gpu_percent',0)>=50:clocks[sample['sm_clock_mhz']]+=1
    cells=[]
    for cell in plan['cells']:
        case=next(c for c in plan['cases'] if c['id']==cell['case']);decision=next(d for d in plan['decisions'] if d['cell']==cell['id'])
        frozen=root/'profiles'/(cell['id']+'.frozen.toml')
        if sha(frozen)!=decision['profile_sha256']:raise ValueError('Frozen tuning profile changed')
        values={}
        for mode in ('scalar','auto'):
            entries=[e for e in plan['evaluations'] if e['cell']==cell['id'] and (e['label']==mode or e['label'].startswith(mode+'_retest'))]
            if len(entries) not in (1,3):raise ValueError('Incomplete held-out treatment/retests')
            rows=[e['rows'][0] for e in entries]
            expected=(0,0) if mode=='scalar' else (decision['selection']['baby_coop'],decision['selection']['giant_coop'])
            if any((r['baby'],r['giant'])!=expected for r in rows):raise ValueError('Held-out execution differs from frozen decision')
            values[mode]=dict(n=len(rows),seconds=st.median(r['seconds'] for r in rows),
                minimum=min(r['seconds'] for r in rows),maximum=max(r['seconds'] for r in rows),
                curve_wall=st.median(r['curve_wall'] for r in rows),
                phases={key:st.mean(partition(r)[key] for r in rows) for key in partition(rows[0])})
        cells.append(dict(cell=cell['id'],case=case['id'],bits=case['bits'],exponent=case['exponent'],B1=case['b1'],B2=cell['B2'],
            family='carrier' if cell['carrier'] else 'ordinary',carrier=cell['carrier'],arithmetic_bits=cell['carrier'] or case['bits'],
            D=cell['D'],P=cell['P'],layout=cell['layout'],chain=cell['chain'],baby=decision['selection']['baby_coop'],
            giant=decision['selection']['giant_coop'],training_score=decision['selection']['comparison_seconds'],
            historical_seconds=cell['reference_seconds'],values=values,change_percent=100*(values['auto']['seconds']/values['scalar']['seconds']-1)))
    best=[];reference=load(root/'reference_layout.json') if plan['reference_layout_sha256'] else None
    if reference and sha(root/'reference_layout.json')!=plan['reference_layout_sha256']:raise ValueError('Historical reference changed')
    for case in plan['cases']:
        candidates=[r for r in cells if r['case']==case['id']]
        # The family is chosen from frozen training, never from held-out minima.
        winner=min(candidates,key=lambda r:r['training_score']);row=dict(winner)
        if reference:
            old=next(r for r in reference['rows'] if r['case']==case['id'])
            if (old['bits'],old['B1'],old['B2'])!=(row['bits'],row['B1'],row['B2']):raise ValueError('Historical input mismatch')
            row.update(layout_reference=old['tuned']['median_seconds'],prime95_seconds=old['prime95_seconds'])
        best.append(row)
    totals={mode:sum(r['values'][mode]['seconds'] for r in cells) for mode in ('scalar','auto')}
    training=[r for e in plan['training'] for r in e['rows']]
    evaluation=[r for e in plan['evaluations'] for r in e['rows']]
    return dict(complete=True,version=plan['program_version'],reference_version=plan['reference_version'],exe_sha256=plan['exe_sha256'],source=str(root),calls=calls,gmp_checked=checked,selftests=selftests,
        training_calls=len(training),training_engine_seconds=sum(r['seconds'] for r in training),evaluation_calls=len(evaluation),
        curve_wall_totals={mode:sum(r['values'][mode]['curve_wall'] for r in cells) for mode in ('scalar','auto')},
        busy_clocks=dict(clocks),protocol=plan['protocol'],cells=cells,best=best,totals=totals,
        historical_sum=sum(r['historical_seconds'] for r in cells),change_percent=100*(totals['auto']/totals['scalar']-1),
        regressions=[r['cell'] for r in cells if r['values']['auto']['seconds']>r['values']['scalar']['seconds']*1.05 and
            r['values']['auto']['seconds']-r['values']['scalar']['seconds']>.02])


def best_figure(result, logarithmic, paired=None, metric="wall_seconds_per_curve", serial=None):
    version=result["version"]
    rows=sorted(result["best"],key=lambda r:r["bits"])
    x=[r["bits"] for r in rows]
    fig,ax=plt.subplots(figsize=(12,5.2))
    ax.plot(x,[r["values"]["auto"]["seconds"] for r in rows],"o-",lw=2,color="#008879",
            label=version+" frozen best known parameters (held out)")
    ax.plot(x,[r["values"]["scalar"]["seconds"] for r in rows],".--",color="#c46a19",
            label=version+" scalar at the same family / geometry")
    for mode,color in [("scalar","#c46a19"),("auto","#008879")]:
        ax.fill_between(x,[r["values"][mode]["minimum"] for r in rows],
                        [r["values"][mode]["maximum"] for r in rows],color=color,alpha=.12)
    if "layout_reference" in rows[0]:
        ax.plot(x,[r["layout_reference"] for r in rows],".--",color="#5073a7",
                label="0.54.7 tuned reference (n=3)")
        ax.plot(x,[r["prime95_seconds"] for r in rows],"s:",color="#333333",ms=3,
                label="Prime95 4T exact-N historical reference (n=1)")
    ylabel="Stage2 engine init + main (s)"
    title="Small B2: 13 exact cofactor inputs; training and held-out samples separate"
    legend_font=None
    if paired:
        try:
            family=font_manager.findfont("Microsoft YaHei",fallback_to_default=False)
            serial_label, paired_label="单曲线优选", "2 并行"
            legend_font=font_manager.FontProperties(fname=family,size=8)
        except ValueError:
            serial_label, paired_label="best serial", "2 concurrent"
        scope="queue wall" if metric=="wall_seconds_per_curve" else "queue middle"
        for study,label,style,color in [(serial,serial_label,"^-","#bd3e59"),
                                       (paired,paired_label,"D-","#8e44ad")]:
            if study is None:
                continue
            measured={r["bits"]:r for r in study["rows"]}
            samples=[[sample[metric] for sample in measured[bit]["samples"]] for bit in x]
            ax.plot(x,[st.median(s) for s in samples],style,ms=4,lw=2,color=color,
                    label=f"{study['version']} {label} ({scope})")
            ax.fill_between(x,[min(s) for s in samples],[max(s) for s in samples],color=color,alpha=.16)
        ylabel="Seconds per curve"
        title="Edwards Stage2 | small B2, 13 exact cofactors"
    ax.set(xlabel="Target N bits",ylabel=ylabel,title=title)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.tick_params(axis="x",rotation=45)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8,prop=legend_font)
    if logarithmic:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    return fig


def verify_windows(result, study, metric, pair=False):
    measured={r["bits"]:r for r in study["rows"]}
    reference_bits={r["bits"] for r in result["best"]}
    if (study["failures"] or len(reference_bits)!=13 or len(result["best"])!=13 or
            measured.keys()!=reference_bits or len(study["rows"])!=13):
        raise ValueError("queue plot requires all 13 verified widths")
    for row in result["best"]:
        item=measured[row["bits"]]
        if any(row[k]!=item[k] for k in INPUT) or (pair and row["carrier"]!=item["carrier"]):
            raise ValueError("queue series uses a different input, bound or fixed arithmetic family")
        samples=item["samples"]
        if len(samples)!=study["repeats"] or any(
            s["curves"]!=study["curves_per_window"] or s[metric] is None or
            not math.isfinite(s[metric]) or s[metric]<=0 for s in samples
        ):
            raise ValueError("queue series uses inconsistent timing windows")
    devices={(s["device"]["name"],s["device"]["uuid"]) for r in study["rows"] for s in r["samples"]}
    if len(devices)!=1:
        raise ValueError("queue summaries use different GPUs")
    return next(iter(devices))


def queue_comparison(serial, paired, metric):
    pairs={r["bits"]:r for r in paired["rows"]}
    result=[]
    for single in sorted(serial["rows"],key=lambda r:r["bits"]):
        dual=pairs[single["bits"]]
        one=[sample[metric] for sample in single["samples"]]
        two=[sample[metric] for sample in dual["samples"]]
        serial_wall, paired_wall=st.median(one), st.median(two)
        result.append(dict(bits=single["bits"],serial_D=single["D"],serial_P=single["P"],
            paired_D=dual["D"],paired_P=dual["P"],same_parameters=shape_key(single)==shape_key(dual),
            serial_seconds=serial_wall,paired_seconds=paired_wall,
            change_percent=100*(paired_wall/serial_wall-1),
            minimum_percent=100*(min(two)/max(one)-1),maximum_percent=100*(max(two)/min(one)-1)))
    return result


def percent_figure(rows, version, gpu_name):
    fig,ax=plt.subplots(figsize=(12,5.2))
    x=np.arange(len(rows))
    y=[r["change_percent"] for r in rows]
    yerr=([v-r["minimum_percent"] for r,v in zip(rows,y)],
          [r["maximum_percent"]-v for r,v in zip(rows,y)])
    bars=ax.bar(x,y,color=["#007f73" if v<0 else "#be592d" for v in y],width=.68,
                yerr=yerr,error_kw=dict(ecolor="#42464b",linewidth=1,capsize=3))
    for bar,row in zip(bars,rows):
        if not row["same_parameters"]:
            bar.set(hatch="///",edgecolor="#333333",linewidth=.7)
    if any(not r["same_parameters"] for r in rows):
        ax.legend(handles=[Patch(facecolor="white",edgecolor="#333333",hatch="///",
                                 label="Different route or parameters")],loc="upper left",fontsize=8)
    ax.axhline(0,color="#444444",linewidth=.8)
    ax.bar_label(bars,labels=[f"{v:+.1f}%" for v in y],padding=4,fontsize=8)
    ax.set_xticks(x,[str(r["bits"]) for r in rows])
    ax.set(title=f"{version} Edwards Stage2 | two slots versus best tested serial",
           xlabel="Target N bits",ylabel="Change in seconds per curve")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100,decimals=0))
    ax.set_axisbelow(True)
    ax.grid(axis="y",alpha=.24)
    ax.margins(y=.17)
    fig.text(.5,.016,
        f"{gpu_name} | 100 × (two-slot time / serial time − 1) | negative values mean less time per curve\n"
        "Each mode: median of two 2-curve queue walls per curve. Whiskers show the range from the observed batches.",
        ha="center",fontsize=8,color="#555555")
    fig.tight_layout(rect=(0,.072,1,1))
    return fig


def parallel_figures(result, paired, metric, out, serial=None):
    device=verify_windows(result,paired,metric,pair=True)
    gpu_name=device[0].removeprefix("NVIDIA GeForce ")
    comparison=None
    if serial:
        keys=("version","curves_per_window","repeats")
        if any(serial[k]!=paired[k] for k in keys) or verify_windows(result,serial,metric)!=device:
            raise ValueError("serial and two-slot series use different versions, batches or GPUs")
        comparison=queue_comparison(serial,paired,metric)
    out.mkdir(parents=True,exist_ok=False)
    stem=f"stage2-small-b2-{paired['version']}-two-concurrent"
    scope="complete queue wall" if metric=="wall_seconds_per_curve" else "trimmed queue wall"
    measured={r["bits"]:r for r in paired["rows"]}
    changed=sum((r["D"],r["P"])!=(measured[r["bits"]]["D"],measured[r["bits"]]["P"]) for r in result["best"])
    modes=f"{paired['version']} serial and two slots" if serial else "two slots"
    geometry=f" Pair D/P changed at {changed} points to fit memory." if changed else ""
    if comparison:
        different=sum(not row["same_parameters"] for row in comparison)
        geometry=f" Paired and serial parameters differ at {different} points." if different else ""
    for logarithmic in (False,True):
        fig=best_figure(result,logarithmic,paired,metric,serial)
        fig.text(.5,.013,
            f"{gpu_name} | B1=50000 | {modes}: {scope} per curve, "
            f"{paired['curves_per_window']} curves per batch, {paired['repeats']} batch repeats\n"
            "Historical single-curve series retain their original timing boundaries."+geometry,
            ha="center",fontsize=8,color="#555555")
        fig.tight_layout(rect=(0,.07,1,1))
        suffix="log" if logarithmic else "linear"
        fig.savefig(out/f"{stem}-{suffix}.png",dpi=170)
        fig.savefig(out/f"{stem}-{suffix}.svg")
        plt.close(fig)
    if comparison:
        fig=percent_figure(comparison,paired["version"],gpu_name)
        stem=f"stage2-small-b2-{paired['version']}-parallel-time-change"
        fig.savefig(out/f"{stem}.png",dpi=170)
        fig.savefig(out/f"{stem}.svg")
        plt.close(fig)
    return comparison


def combine_parallel(paths):
    inputs=[load(path) for path in paths]
    keys=("version","curves_per_window","repeats")
    identity={key:inputs[0][key] for key in keys}
    if any(item[key]!=value for item in inputs for key,value in identity.items()):
        raise ValueError("two-slot summaries use different versions or sample counts")
    rows=[row for item in inputs for row in item["rows"]]
    qualified={row["bits"] for row in rows}
    failures=[error for item in inputs for error in item["failures"]]
    return dict(identity, rows=rows,
                failures=[error for error in failures if error["bits"] not in qualified],
                superseded_failures=[error for error in failures if error["bits"] in qualified])


def figures(result,out):
    out.mkdir(parents=True,exist_ok=False);version=result['version'];stem='stage2-backend-'+version+'-low'
    def save(fig,name,top=1):
        fig.text(.5,.008,'RTX 4060 Laptop | target 1500 MHz | B1=50000 | fixed D/layout/chain | held out n=1; flagged retests n=3 | no warmup',
                 ha='center',fontsize=8,color='#555555')
        fig.tight_layout(rect=(0,.045,1,top));fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.svg'));plt.close(fig)
    for logarithmic in (False,True):
        suffix='log' if logarithmic else 'linear'
        fig,axes=plt.subplots(1,2,figsize=(14,5.2))
        for ax,family in zip(axes,('ordinary','carrier')):
            rows=sorted((r for r in result['cells'] if r['family']==family),key=lambda r:r['bits']);x=[r['bits'] for r in rows]
            ax.plot(x,[r['historical_seconds'] for r in rows],'.--',color='#777777',label=result['reference_version']+' scalar reference (n=1)')
            for mode,color,label in [('scalar','#c46a19',version+' scalar/scalar'),('auto','#008879',version+' frozen automatic choice')]:
                ax.plot(x,[r['values'][mode]['seconds'] for r in rows],'o-',color=color,label=label,ms=3)
                ax.fill_between(x,[r['values'][mode]['minimum'] for r in rows],[r['values'][mode]['maximum'] for r in rows],alpha=.12,color=color)
            ax.set(title=family,xlabel='Target N bits',ylabel='Stage2 engine init + main (s)');ax.xaxis.set_major_locator(MultipleLocator(1000))
            if logarithmic:ax.set_yscale('log')
            else:ax.set_ylim(bottom=0)
            ax.grid(alpha=.2);ax.legend(fontsize=8);ax.tick_params(axis='x',rotation=45)
        save(fig,stem+'-families-'+suffix)
        save(best_figure(result,logarithmic),stem+'-best-times-'+suffix)
        fig,axes=plt.subplots(1,2,figsize=(14,5.2))
        for ax,family in zip(axes,('ordinary','carrier')):
            rows=sorted((r for r in result['cells'] if r['family']==family),key=lambda r:r['bits']);x=[r['bits'] for r in rows]
            for mode,color in [('scalar','#c46a19'),('auto','#008879')]:
                ax.plot(x,[r['values'][mode]['curve_wall'] for r in rows],'o-',ms=3,color=color,label=version+' '+mode)
            ax.set(title=family,xlabel='Target N bits',ylabel='curve_done.wall (s; includes startup checks / planning)')
            ax.xaxis.set_major_locator(MultipleLocator(1000));ax.tick_params(axis='x',rotation=45);ax.grid(alpha=.2);ax.legend(fontsize=8)
            if logarithmic:ax.set_yscale('log')
            else:ax.set_ylim(bottom=0)
        save(fig,stem+'-curve-wall-'+suffix)
    fig,axes=plt.subplots(2,2,figsize=(15,9));keys=['baby','ftree_init','inv','giant_combined','gtrees','fold','descent','accum','other'];colors=plt.get_cmap('tab10').colors
    labels=['Baby + normalization','F tree + init','Polynomial inverse','Giant + leaf preparation','G trees','Fold','Descent','Product + GCD','Other']
    if set(keys)!=set(result['cells'][0]['values']['auto']['phases']):raise ValueError('Unmapped phase')
    for column,family in enumerate(('ordinary','carrier')):
        rows=sorted((r for r in result['cells'] if r['family']==family),key=lambda r:r['bits']);x=np.arange(len(rows))
        for line in (0,1):
            ax=axes[line,column];base=np.zeros(len(rows))
            for i,key in enumerate(keys):
                y=np.array([r['values']['auto']['phases'][key] for r in rows]);total=np.array([sum(r['values']['auto']['phases'].values()) for r in rows])
                if line:y=100*y/total
                ax.bar(x,y,bottom=base,label=labels[i],color=colors[i%len(colors)],width=.8);base+=y
            ax.set_xticks(x,[str(r['bits']) for r in rows],rotation=60);ax.set(title=family,ylabel='Time share (%)' if line else 'Stage2 phase time (s)')
            if line:ax.set_ylim(0,100)
            ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.965),ncol=5,fontsize=8)
    fig.suptitle('Frozen automatic backend: mutually exclusive phases; equally spaced input cases',y=.995)
    save(fig,stem+'-phase-shares',.89)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--reference-summary",type=Path)
    p.add_argument("--parallel-summary",type=Path,action="append")
    p.add_argument("--serial-summary",type=Path,action="append")
    p.add_argument("--parallel-metric",choices=("wall_seconds_per_curve","middle_seconds_per_curve"),
                   default="wall_seconds_per_curve")
    p.add_argument("--figures",type=Path,required=True)
    p.add_argument("--publish",type=Path)
    a=p.parse_args()
    if a.parallel_summary or a.reference_summary or a.serial_summary:
        if not (a.parallel_summary and a.reference_summary) or a.study:
            p.error("use --reference-summary and --parallel-summary together for a queue plot")
        if a.serial_summary and a.parallel_metric!="wall_seconds_per_curve":
            p.error("serial and paired short batches use the whole queue wall")
        paired=combine_parallel(a.parallel_summary)
        serial=combine_parallel(a.serial_summary) if a.serial_summary else None
        comparison=parallel_figures(load(a.reference_summary),paired,a.parallel_metric,a.figures,serial)
        if a.output:
            a.output.mkdir(parents=True,exist_ok=False)
            write(a.output/"summary.json",paired)
            if comparison:
                write(a.output/"serial.json",serial)
                with (a.output/"serial_vs_parallel.csv").open("w",newline="",encoding="utf-8") as f:
                    writer=csv.DictWriter(f,fieldnames=comparison[0].keys())
                    writer.writeheader()
                    writer.writerows(comparison)
        if a.publish:
            a.publish.mkdir(parents=True,exist_ok=True)
            for f in a.figures.glob("*.png"):
                shutil.copy2(f,a.publish/f.name)
        return
    if not a.study or not a.output:
        p.error("the original audit requires --study and --output")
    result=audit(a.study.resolve())
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'summary.json',result);shutil.copy2(__file__,a.output/'analyzer_snapshot.py')
    with (a.output/'cells.csv').open('w',newline='',encoding='utf-8') as f:
        columns=['bits','family','B2','D','P','layout','chain','baby','giant','scalar_seconds','auto_seconds','change_percent','n']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in result['cells']:
            row={k:r[k] for k in columns if k in r};row.update(scalar_seconds=r['values']['scalar']['seconds'],auto_seconds=r['values']['auto']['seconds'],n=r['values']['auto']['n']);writer.writerow(row)
    lines=['| N bits | 实际B2 | 路径 | D / P | chain | baby / giant | 标量s | auto s | n |',
        '|---:|---:|---|---|---:|---|---:|---:|---:|']
    for r in sorted(result['best'],key=lambda r:r['bits']):
        lines.append(f'| {r["bits"]} | {r["B2"]} | {r["family"]} | {r["D"]} / {r["P"]} | {r["chain"]} | {r["baby"]} / {r["giant"]} | {r["values"]["scalar"]["seconds"]:.3f} | {r["values"]["auto"]["seconds"]:.3f} | {r["values"]["auto"]["n"]} |')
    (a.output/'table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');figures(result,a.figures)
    if a.publish:
        a.publish.mkdir(parents=True,exist_ok=True)
        for f in a.figures.glob('*.png'):shutil.copy2(f,a.publish/f.name)
    print({k:result[k] for k in ('complete','calls','gmp_checked','busy_clocks','totals','change_percent','regressions')})


if __name__=='__main__':main()
