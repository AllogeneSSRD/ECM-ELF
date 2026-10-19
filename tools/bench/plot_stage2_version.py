"""Publish version-labelled Stage2 figures and retained full-range references.

Original studies stay untouched. New outputs contain numeric program versions,
or explicitly state that a historical study did not record one.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/test'))
from check_stage2_version import audit
from check_stage2_runtime_matrix import audit as audit_runtime
from bench_stage2_version import load

OLD=ROOT/'data/benchmarks/edwards_stage2_reference_20261017/gpu_final_02/summary.json'
GW=ROOT/'data/benchmarks/cpu_gw_poly_stage2_20261017/final_01/summary.json'
COLORS=['#24699c','#d16a32','#67717c','#729e48']


def runtime_matrix(study,baseline):
    checked=audit_runtime(study);plan=load(study/'plan.json')
    if Path(plan['baseline']).resolve()!=baseline.resolve():
        raise ValueError('Runtime data uses a different default baseline')
    cases={c['id']:c for c in plan['cases']};cells={c['id']:c for c in plan['cells']}
    runs={r['id']:r for r in plan['runs']};rows=[]
    for decision in plan['decisions']:
        cell=cells[decision['cell']];case=cases[cell['case']];sample=decision['sample']
        selected=[runs[tag] for tag in decision['run_ids']];first=selected[0]
        times=[r['observation']['mean_seconds'] for r in selected]
        cell_runs=[r for r in plan['runs'] if r['cell']==cell['id']]
        searches=[s for r in cell_runs for s in r['selection'].get('runtime_search',[])]
        last_search={}
        for search in searches:last_search[str(search['carrier_exponent'])]=search
        rows.append(dict(cell=cell['id'],bits=case['bits'],exponent=case['exponent'],B1=case['b1'],sigma=case['sigma'],
            B2=cell['B2'],nominal_B2=cell['nominal_B2'],version=plan['program_version'],n=sample['repeats'],
            seconds=sample['mean_seconds'],stddev_seconds=sample['stddev_seconds'],min_seconds=min(times),max_seconds=max(times),
            statistic='best admitted repeated profile mean; selected on these same observations',
            D=sample['d'],P=decision['selection']['P'],carrier_exponent=sample['carrier_exponent'],
            G=int(first['shape']['num_poly_g']),giants=int(first['shape']['giant_points']),
            parent_seconds_mean=st.mean(r['parent_seconds'] for r in selected),run_ids=decision['run_ids'],
            prepare_seconds_mean=st.mean(r['selection']['runtime_prepare']['total_seconds'] for r in selected),
            ordinary_measured_D=sorted({r['D'] for r in cell_runs if not r['carrier_exponent']}),
            carrier_measured_D=sorted({r['D'] for r in cell_runs if r['carrier_exponent']}),
            learning_calls=len(cell_runs),queue_finish_reason=next(q['finish_reason'] for q in plan['queues'] if q['cell']==cell['id']),
            search_reasons=dict(Counter(s['reason'] for s in searches)),last_search=last_search,
            memory_rejection_events=sum(s.get('memory_rejected',0) for s in searches),
            ntt_peak_mib=st.median(int(r['stats']['ntt_workspace_stats']['full_peak_bytes'])/1048576 for r in selected),
            init_fraction=st.mean(float(r['phase_seconds']['init'])/r['engine_seconds'] for r in selected),
            component_fractions={k:st.mean(float(r['stats']['real_batched_split'][k])/r['engine_seconds'] for r in selected)
                for k in ['giant','gtrees','fold','descent','inv']}))
    busy=[]
    for queue in plan['queues']:
        if queue.get('telemetry_unavailable'):continue
        busy.extend(t for t in load(Path(queue['folder'])/'telemetry.json') if (t.get('gpu_percent') or 0)>=50)
    telemetry={k:distribution(t[k] for t in busy if t.get(k) is not None) for k in ['sm_clock_mhz','temperature_c','power_mw']}
    telemetry.update(busy_samples=len(busy),power_outside_0_200w=sum(t.get('power_mw') is not None and not 0<=t['power_mw']<=200000 for t in busy))
    return dict(complete=True,audit=checked,rows=rows,planned_curve_cap=plan['curves_per_condition'],
        stop_repeats=plan['stop_repeats'],imported_calls=plan.get('imported_calls',0),sample_limit=plan['sample_limit'],
        learning_calls=len(plan['runs']),learning_engine_seconds=sum(r['engine_seconds'] for r in plan['runs']),
        learning_queue_seconds=sum(q['parent_sum'] for q in plan['queues']),
        learning_process_seconds=sum(q['process_seconds'] for q in plan['queues'] if q['process_seconds'] is not None),
        process_time_missing_queues=sum(q['process_seconds'] is None for q in plan['queues']),
        telemetry_missing_queues=sum(bool(q.get('telemetry_unavailable')) for q in plan['queues']),
        busy_gpu_telemetry=telemetry,context=load(study/'context.json'),
        profile=str(study/'stage2_ecm_tune_frozen.toml'),trace=[dict(cell=r['cell'],index=r['index'],
            seconds=r['engine_seconds'],D=r['D'],P=r['P'],carrier=r['carrier_exponent'],
            probe=r['selection']['runtime_probe'],samples_before=r['selection']['runtime_samples']) for r in plan['runs']])


def distribution(values):
    values=list(values)
    if not values:
        return None
    return dict(n=len(values),median=st.median(values),min=min(values),max=max(values))


def save(fig,name,args,caption,scales=None):
    fig.text(.5,.02,caption,ha='center',va='bottom',fontsize=9)
    for scale in (['log','linear'] if scales else [None]):
        target=name+(f'-{scale}' if scale else '')
        if scale:
            for ax in fig.axes:
                ax.set_yscale(scale)
                ax.relim();ax.autoscale_view(scaley=True)
                if scale=='linear':ax.set_ylim(bottom=0)
                label='Full Stage2 seconds' if scales=='time' else 'NTT module full peak (MiB), median per cell'
                ax.set_ylabel(f'{label} ({scale} scale)')
        fig.tight_layout(rect=(0,.10,1,.94))
        for ext in ('png','svg'):
            fig.savefig(args.figures/f'{target}.{ext}')
        if args.publish:
            args.publish.mkdir(parents=True,exist_ok=True)
            shutil.copy2(args.figures/f'{target}.png',args.publish/f'{target}.png')
    plt.close(fig)


def errorline(ax,x,rows,label,color,field='seconds',style='-'):
    values=[r[field] for r in rows]
    err=None
    if rows and all('min_seconds' in r for r in rows):
        err=[[r[field]-r['min_seconds'] for r in rows],
             [r['max_seconds']-r[field] for r in rows]]
    ax.errorbar(x,values,yerr=err,label=label,color=color,marker='o',ms=4,
                capsize=2,lw=1.5,linestyle=style)


def axes_bits(ax,bits,bound,log=True,adjusted=True):
    ax.set_xticks(np.arange(len(bits)),[str(b) for b in bits],rotation=55)
    ax.set_xlabel('Actual cofactor N (bits)')
    ax.set_title(f'Request B2 {bound:.1e}; exact adjusted bounds per input' if adjusted
                 else f'Executed B2 = {bound:.1e}')
    if log:
        ax.set_yscale('log');ax.set_ylabel('Full Stage2 seconds (log scale)')


def references(plan,args):
    old=load(OLD);gw=load(GW)
    if not old['complete'] or not gw['complete']:
        raise ValueError('Historical reference is incomplete in its declared scope')
    bounds=sorted({r['b2_requested'] for r in plan['prime95_rows']})
    bits=[c['bits'] for c in plan['cases']]
    refs=plan['prime95_rows']
    fig,axes=plt.subplots(1,2,figsize=(16,5.8))
    for ax,bound in zip(axes,bounds):
        for i,curve in enumerate(['Edwards','Montgomery']):
            rows=sorted([r for r in refs if r['curve_type']==curve and r['b2_requested']==bound],key=lambda r:r['actual_input_bits'])
            if len(rows)!=13:
                raise ValueError('Historical Prime95 scope differs')
            ax.plot(np.arange(13),[r['s2_total_time'] for r in rows],'-o',color=COLORS[i],label=f'{curve}, B1={rows[0]["b1"]:,} (n=1)',ms=4)
        axes_bits(ax,bits,bound);ax.legend(fontsize=9)
    fig.suptitle('Prime95 31.4b05 | retained complete Edwards / Montgomery reference')
    save(fig,'stage2-prime95-full-reference',args,
         'Ryzen AI 9 HX 370 | 4 threads, declared 4 GHz | 4096 MB available\nInit + main + GCD. Different curve families and B1; context, not a matched algorithm A/B.',scales='time')

    fig,axes=plt.subplots(1,2,figsize=(16,5.8))
    cpu=[r for r in old['rows'] if r['engine'].startswith('cpu-')]
    for ax,bound in zip(axes,bounds):
        for i,engine in enumerate(['cpu-ntt','cpu-plain']):
            rows=sorted([r for r in cpu if r['nominal_B2']==bound and r['engine']==engine],key=lambda r:r['bits'])
            if not rows:continue
            positions=[bits.index(r['bits']) for r in rows]
            ax.plot(positions,[r['seconds']['median'] for r in rows],'o--',color=COLORS[i],label=f'libecm {engine[4:]} (n=1)',ms=4)
        prime=sorted([r for r in refs if r['curve_type']=='Edwards' and r['b2_requested']==bound],key=lambda r:r['actual_input_bits'])
        ax.plot(np.arange(13),[r['s2_total_time'] for r in prime],'o-',color=COLORS[2],label='Prime95 4T (n=1)',ms=4)
        axes_bits(ax,bits,bound);ax.legend(fontsize=9)
    fig.suptitle('Retained CPU reference | libecm 7.0.6-dev32; ECM-ELF version unrecorded')
    save(fig,'stage2-libecm-retained-reference',args,
         '32 retained single-thread calls; collection stopped. No interpolation of missing cells.\nDeclared 4 GHz, 4096 MiB planner budget. Prime95 is historical 4T; no thread normalization.',scales='time')

    cells=gw['cells'];exponents=sorted({r['exponent'] for r in cells});x=np.arange(len(exponents))
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    for ax,bound in zip(axes,bounds):
        for i,threads in enumerate([0,1,4]):
            rows=[next(r for r in cells if r['exponent']==e and r['nominal_B2']==bound and r['threads']==(threads or 4)) for e in exponents]
            color=[COLORS[2],COLORS[0],COLORS[1]][i]
            if threads:
                errorline(ax,x,rows,f'ECM-ELF 0.33.0 GW {threads}T (n=3)',color)
            else:
                ax.plot(x,[r['prime95_4t_seconds'] for r in rows],'o-',color=color,label='Prime95 31.4b05 4T (n=1)')
        ax.set_xticks(x,[f'M{r["exponent"]}\nN {r["bits"]} bits' for r in rows]);ax.set_yscale('log')
        ax.set_title(f'Request B2 {bound:.1e}');ax.set_ylabel('Full Stage2 seconds (log scale)');ax.legend(fontsize=8)
    fig.suptitle('CPU fast polynomial Stage2 | exact Edwards inputs and adjusted bounds')
    save(fig,'stage2-cpu-gwpoly-0.33.0',args,
         'Ryzen AI 9 HX 370 | declared 4 GHz | affinity 0xAA | 4096 MiB planner\nGW medians and min-max; first samples retained. Full native timer includes curve setup and cleanup.',scales='time')
    fig,axes=plt.subplots(1,3,figsize=(14,5.4))
    rows=[r for r in cells if r['threads']==4]
    labels=[f'{r["bits"]}\n{r["nominal_B2"]:.1e}' for r in rows]
    one={(r['bits'],r['B2']):r for r in cells if r['threads']==1}
    values=[[one[r['bits'],r['B2']]['seconds']/r['seconds'] for r in rows],
            [r['cpu_wall_ratio'] for r in rows],
            [r['peak_working_set_mib'] for r in rows]]
    for ax,v,title in zip(axes,values,['Measured 1T / 4T speed ratio','CPU time / process wall (4T)','Process peak working set (MiB, 4T)']):
        ax.bar(np.arange(len(rows)),v,color=COLORS[0]);ax.set_xticks(np.arange(len(rows)),labels,rotation=50)
        ax.set_title(title);ax.set_xlabel('N bits / requested B2')
    fig.suptitle('ECM-ELF 0.33.0 | CPU threading and measured process memory')
    save(fig,'stage2-cpu-gwpoly-0.33.0-resources',args,
         'Three samples per cell; medians shown. CPU occupancy is not speedup.\nProcess working set is not the planner budget; component peaks are not added.')

    mont=[]
    fig,axes=plt.subplots(1,3,figsize=(16,5.8))
    series=[(None,55,'20261008_reaudit','Oct 08: 55 W setting',COLORS[2]),
            (1800,None,'20261009_current','Oct 09: 1800 MHz setting',COLORS[0]),
            (2400,None,'20261009_2400','Oct 09: 2400 MHz setting',COLORS[1])]
    for clock,power,stem,label,color in series:
        source=ROOT/f'data/benchmarks/stage2_n_scaling_{stem}_summary.csv'
        with source.open(encoding='utf-8-sig') as f:
            selected=[r for r in csv.DictReader(f) if r['variant']=='cofactor']
        if len(selected)!=27 or any(int(r['samples'])!=3 for r in selected):
            raise ValueError('Montgomery full-range reference differs')
        for row in selected:
            mont.append(dict(series=stem,clock_setting_mhz=clock,power_setting_w=power,bits=int(row['bits']),B2=int(row['B2']),
                exponent=int(row['exponent']),B1=int(row['B1']),sigma=int(row['sigma']),D=int(row['D']),P=int(row['P']),
                mean_seconds=float(row['mean_seconds']),std_seconds=float(row['std_seconds']),n=3,version=None))
        for ax,bound in zip(axes,[26000000000,260000000000,2600000000000]):
            rows=sorted([r for r in mont if r['series']==stem and r['B2']==bound],key=lambda r:r['bits'])
            ax.errorbar(np.arange(9),[r['mean_seconds'] for r in rows],yerr=[r['std_seconds'] for r in rows],
                        label=label,color=color,marker='o',capsize=2,ms=4)
            axes_bits(ax,[r['bits'] for r in rows],bound,adjusted=False);ax.legend(fontsize=8)
    fig.suptitle('Montgomery PARAM0 | full 9 widths x 3 B2 reference (2026-10-08/09, versions unrecorded)')
    save(fig,'stage2-montgomery-full-reference',args,
         'RTX 4060 Laptop | B1=20 | 3 samples/cell, mean +/- sample SD | 6300/640/256 MiB budgets\nDifferent versions/settings; Oct 09 reuses one build. 1800 MHz dual-GPU / 2400 MHz single-GPU. No normalization.',scales='time')
    return dict(libecm_reference_invocations=32,gwpoly_version='0.33.0',gwpoly_cells=cells,montgomery=mont)


def current(plan,args):
    checked=audit(args.study)
    cases={c['id']:c for c in plan['cases']};old=load(OLD)
    grouped=defaultdict(list)
    for run in plan['runs']:
        grouped[run['case'],run['B2'],run['engine']].append(run)
    rows=[];busy=[]
    for (case_id,b2,engine),runs in grouped.items():
        case=cases[case_id];ref=next(r for r in plan['prime95_rows'] if r['curve_type']=='Edwards' and r['N_hex']==case['N_hex'] and r['b2']==b2)
        baseline=next(r for r in old['rows'] if r['case']==case_id and r['B2']==b2 and r['engine']==engine)
        if (baseline['B1'],baseline['sigma'],baseline['bits']) != (case['b1'],case['sigma'],case['bits']):
            raise ValueError('Historical comparison identity differs')
        times=[r['engine_seconds'] for r in runs];median=st.median(times)
        rows.append(dict(case=case_id,bits=case['bits'],exponent=case['exponent'],B1=case['b1'],sigma=case['sigma'],B2=b2,
            nominal_B2=ref['b2_requested'],engine=engine,version=plan['program_version'],n=3,seconds=median,
            min_seconds=min(times),max_seconds=max(times),process_seconds=st.median(r['process_seconds'] for r in runs),
            old_process_seconds=baseline['process_seconds']['median'],
            init_fraction=st.mean(float(r['phase_seconds']['init'])/r['engine_seconds'] for r in runs),
            old_seconds=baseline['seconds']['median'],old_min_seconds=baseline['seconds']['min'],old_max_seconds=baseline['seconds']['max'],
            old_D=baseline['D'],old_P=baseline['P'],geometry_changed=baseline['D']!=[runs[0]['D']] or baseline['P']!=[runs[0]['P']],
            time_reduction_percent=100*(1-median/baseline['seconds']['median']),prime95_seconds=ref['s2_total_time'],
            D=runs[0]['D'],P=runs[0]['P'],G=int(runs[0]['shape']['num_poly_g']),giants=int(runs[0]['shape']['giant_points']),
            ntt_peak_mib=st.median(int(r['stats']['ntt_workspace_stats']['full_peak_bytes'])/1048576 for r in runs),
            component_medians={k:st.median(float(r['stats']['real_batched_split'][k]) for r in runs)
                for k in ['giant','gtrees','fold','descent','inv']},
            component_fractions={k:st.mean(float(r['stats']['real_batched_split'][k])/r['engine_seconds'] for r in runs)
                for k in ['giant','gtrees','fold','descent','inv']}))
        for run in runs:
            busy.extend(t for t in load(Path(run['prefix']).with_suffix('.telemetry.json')) if (t.get('gpu_percent') or 0)>=50)
    bounds=sorted({r['nominal_B2'] for r in rows});bits=[c['bits'] for c in plan['cases']];x=np.arange(13)
    lookup={(r['bits'],r['nominal_B2'],r['engine']):r for r in rows}
    learned=runtime_matrix(args.runtime_study,args.study) if args.runtime_study else None
    runtime_lookup={(r['bits'],r['nominal_B2']):r for r in learned['rows']} if learned else {}
    title=f'ECM-ELF {plan["program_version"]} | Edwards GPU full Stage2'
    caption='RTX 4060 Laptop, GPU1 | ordinary / Mersenne carrier | n=3 medians, min-max | B1=50,000\nSame exact N / sigma / adjusted B2 as Prime95 31.4b05 4T (n=1). No warmup removal or normalization.'
    if learned:
        audit_counts=learned['audit']
        runtime_n=f"{audit_counts['best_sample_count_min']}..{audit_counts['best_sample_count_max']}"
        caption=f'RTX 4060 Laptop, GPU1 | defaults: n=3 medians | runtime: best profile means (actual n={runtime_n}), min-max\nSame exact N / sigma / adjusted B2; B1=50,000. Runtime selected on these observations; costs separate. Prime95 4T: historical n=1.'
    fig,axes=plt.subplots(1,2,figsize=(16,6.2))
    for ax,bound in zip(axes,bounds):
        for i,mode in enumerate(['gpu','gpu-carrier']):
            errorline(ax,x,[lookup[b,bound,mode] for b in bits],['Ordinary N','Mersenne carrier'][i],COLORS[i])
        ax.plot(x,[lookup[b,bound,'gpu']['prime95_seconds'] for b in bits],'s--',color=COLORS[2],label='Prime95 4T (n=1)',ms=4)
        if learned:
            errorline(ax,x,[runtime_lookup[b,bound] for b in bits],f'Runtime {learned["rows"][0]["version"]}: best profile mean',COLORS[3])
        axes_bits(ax,bits,bound);ax.legend(fontsize=9)
    prefix=f'stage2-{plan["program_version"]}'
    combined_prefix=f'stage2-runtime-{learned["rows"][0]["version"]}' if learned else prefix
    fig.suptitle(title if not learned else f'Edwards full Stage2 | defaults {plan["program_version"]} / runtime {learned["rows"][0]["version"]}');save(fig,combined_prefix+'-overall',args,caption,scales='time')

    fig,axes=plt.subplots(1,2,figsize=(16,6.2),sharey=True)
    for ax,bound in zip(axes,bounds):
        for i,mode in enumerate(['gpu','gpu-carrier']):
            cells=[lookup[b,bound,mode] for b in bits]
            ax.plot(x,[r['time_reduction_percent'] for r in cells],'-o',color=COLORS[i],label=['Ordinary N','Mersenne carrier'][i],ms=4)
            low=[100*(1-r['max_seconds']/r['old_min_seconds']) for r in cells]
            high=[100*(1-r['min_seconds']/r['old_max_seconds']) for r in cells]
            ax.fill_between(x,low,high,color=COLORS[i],alpha=.12)
        ax.axhline(0,color='#666',lw=1);axes_bits(ax,bits,bound,False)
        ax.set_ylabel('Full time reduction (%) | positive = faster');ax.legend(fontsize=9)
    fig.suptitle(f'ECM-ELF {plan["program_version"]} vs retained 2026-10-17 full matrix (version unrecorded)')
    save(fig,prefix+'-change',args,
         'Both studies: same 13 exact N, Edwards inputs, actual B2, memory budgets; 3 retained samples per cell\nShading combines observed min-max, not a confidence interval. Separate sessions; not isolated causality for one optimization.')

    modes=['gpu','gpu-carrier']+(['runtime'] if learned else [])
    mode_labels=['Default ordinary N','Default carrier']+(['Runtime-selected'] if learned else [])
    fig,axes=plt.subplots(len(modes),2,figsize=(16,4.5*len(modes)))
    for j,mode in enumerate(modes):
        for ax,bound in zip(axes[j],bounds):
            values=[(runtime_lookup[b,bound] if mode=='runtime' else lookup[b,bound,mode])['init_fraction']*100 for b in bits]
            ax.bar(x,values,color=COLORS[0],label='Init');ax.bar(x,100-np.array(values),bottom=values,color=COLORS[1],label='Main incl. GCD / checks')
            axes_bits(ax,bits,bound,False);ax.set_ylim(0,100);ax.set_ylabel(f'{mode_labels[j]} | full time (%)');ax.legend(fontsize=8)
    fig.suptitle(title+' | disjoint timing boundaries' if not learned else f'Full Stage2 phases | defaults {plan["program_version"]} / runtime {learned["rows"][0]["version"]}')
    save(fig,combined_prefix+'-phases',args,'Mean of per-call fractions (defaults n=3; runtime actual chosen observations). Init + main = full engine timer.\nNested NTT, S4 reduction and component timers are not added to these bars.' if learned else 'Mean of per-call fractions (n=3). Init + main = full engine timer.\nNested NTT, S4 reduction and component timers are not added to these bars.')

    # Diagnostic component timers are shown individually, never as a 100% stack.
    fig,axes=plt.subplots(len(modes),2,figsize=(16,4.5*len(modes)))
    components=['giant','gtrees','fold','descent','inv']
    for j,mode in enumerate(modes):
        for ax,bound in zip(axes[j],bounds):
            values=np.array([[(runtime_lookup[b,bound] if mode=='runtime' else lookup[b,bound,mode])['component_fractions'][k]*100 for b in bits] for k in components])
            ax.imshow(values,aspect='auto',cmap='Blues',vmin=0,vmax=100)
            for iy in range(len(components)):
                for ix in range(len(bits)):
                    ax.text(ix,iy,f'{values[iy,ix]:.0f}%',ha='center',va='center',fontsize=8,
                            color='white' if values[iy,ix]>60 else '#152c45')
            ax.grid(False);ax.set_yticks(np.arange(5),['Giant','G trees','Fold','Descent','Inverse'])
            axes_bits(ax,bits,bound,False);ax.set_ylabel(mode_labels[j])
    fig.suptitle(title+' | individual component timer / full engine time' if not learned else f'Full Stage2 components | defaults {plan["program_version"]} / runtime {learned["rows"][0]["version"]}')
    save(fig,combined_prefix+'-components',args,
         ('Mean of per-call fractions (defaults n=3; runtime actual chosen observations)' if learned else 'Mean of per-call fractions (n=3)')+'; diagnostic intervals may include waits and shared work.\nCells are not an additive timing ledger. NTT / S4 reduction remain nested; percentages must not be summed.')

    fig,axes=plt.subplots(1,2,figsize=(16,6.2))
    for ax,bound in zip(axes,bounds):
        for i,mode in enumerate(['gpu','gpu-carrier']):
            ax.plot(x,[lookup[b,bound,mode]['ntt_peak_mib'] for b in bits],'-o',color=COLORS[i],label=['Ordinary N','Carrier'][i],ms=4)
        if learned:
            ax.plot(x,[runtime_lookup[b,bound]['ntt_peak_mib'] for b in bits],'-o',color=COLORS[3],label='Runtime-selected',ms=4)
        axes_bits(ax,bits,bound,False);ax.set_ylabel('NTT module full peak (MiB), median per cell');ax.legend(fontsize=9)
    fig.suptitle(title+' | NTT module memory' if not learned else f'NTT module memory | defaults {plan["program_version"]} / runtime {learned["rows"][0]["version"]}')
    save(fig,combined_prefix+'-ntt-memory',args,'Module-owned peak includes workspace, tables and fused base; it is not the process or whole-GPU peak.\n6300 MiB arena / 640 MiB fold owner / 256 MiB batch budgets. Component peaks are not added.',scales='memory')

    totals={}
    for mode in ['gpu','gpu-carrier']:
        selected=[r for r in rows if r['engine']==mode]
        new=sum(r['seconds'] for r in selected);baseline=sum(r['old_seconds'] for r in selected)
        totals[mode]=dict(cells=26,median_seconds_sum=new,old_median_seconds_sum=baseline,
                          time_reduction_percent=100*(1-new/baseline),cell_reduction=distribution(r['time_reduction_percent'] for r in selected),
                          process_median_seconds_sum=sum(r['process_seconds'] for r in selected),
                          old_process_median_seconds_sum=sum(r['old_process_seconds'] for r in selected))
    telemetry={k:distribution(t[k] for t in busy if t.get(k) is not None) for k in ['sm_clock_mhz','temperature_c','power_mw']}
    telemetry['busy_samples']=len(busy)
    telemetry['power_outside_0_200w']=sum(t.get('power_mw') is not None and not 0<=t['power_mw']<=200000 for t in busy)
    if learned:
        for r in learned['rows']:
            ordinary=lookup[r['bits'],r['nominal_B2'],'gpu'];carrier=lookup[r['bits'],r['nominal_B2'],'gpu-carrier']
            same=carrier if r['carrier_exponent'] else ordinary
            r.update(default_ordinary_seconds=ordinary['seconds'],default_carrier_seconds=carrier['seconds'],
                default_ordinary_D=ordinary['D'],default_ordinary_P=ordinary['P'],
                default_carrier_D=carrier['D'],default_carrier_P=carrier['P'],
                best_default_seconds=min(ordinary['seconds'],carrier['seconds']),default_same_family_D=same['D'],
                default_same_family_P=same['P'],geometry_changed=(r['D'],r['P'])!=(same['D'],same['P']))
            r['reduction_vs_best_default_percent']=100*(1-r['seconds']/r['best_default_seconds'])
        total=sum(r['seconds'] for r in learned['rows'])
        best=sum(r['best_default_seconds'] for r in learned['rows'])
        learned['totals']=dict(runtime_mean_seconds_sum=total,best_default_median_seconds_sum=best,
            selected_parent_mean_seconds_sum=sum(r['parent_seconds_mean'] for r in learned['rows']),
            selected_prepare_mean_seconds_sum=sum(r['prepare_seconds_mean'] for r in learned['rows']),
            reduction_vs_best_default_percent=100*(1-total/best),
            reduction_vs_ordinary_percent=100*(1-total/totals['gpu']['median_seconds_sum']),
            reduction_vs_carrier_percent=100*(1-total/totals['gpu-carrier']['median_seconds_sum']),
            changed_geometry_cells=sum(r['geometry_changed'] for r in learned['rows']),
            carrier_cells=sum(bool(r['carrier_exponent']) for r in learned['rows']))
        fig,axes=plt.subplots(1,2,figsize=(16,6.2),sharey=True)
        for ax,bound in zip(axes,bounds):
            cells=[runtime_lookup[b,bound] for b in bits]
            ax.plot(x,[r['reduction_vs_best_default_percent'] for r in cells],'-o',color=COLORS[3],ms=4)
            ax.axhline(0,color='#666',lw=1);axes_bits(ax,bits,bound,False)
            ax.set_ylabel('Full time reduction vs faster default family (%)')
        fig.suptitle(f'Edwards full Stage2 | runtime {learned["rows"][0]["version"]} vs defaults {plan["program_version"]}')
        save(fig,combined_prefix+'-change',args,
            f'Runtime {learned["rows"][0]["version"]}: best admitted profile means, actual n={runtime_n}; same observations used for selection\nReference: faster default-family median ({plan["program_version"]}). Separate versions/sessions; no independent post-selection validation or global optimum claim.')
        fig,axes=plt.subplots(1,2,figsize=(16,8.5))
        by_cell=defaultdict(list)
        for r in learned['trace']:by_cell[r['cell']].append(r)
        for ax,bound in zip(axes,bounds):
            values=np.full((13,max(len(by_cell[r['cell']]) for r in learned['rows'] if r['nominal_B2']==bound)),np.nan)
            for iy,b in enumerate(bits):
                r=runtime_lookup[b,bound]
                for ix,call in enumerate(by_cell[r['cell']]):values[iy,ix]=call['seconds']/r['best_default_seconds']
            im=ax.imshow(values,aspect='auto',cmap='RdYlGn_r',vmin=.5,vmax=1.5)
            ax.set_yticks(np.arange(13),bits);ax.set_xlabel('Invocation within each exact condition (1-based)')
            ticks=np.arange(0,values.shape[1],5);ax.set_xticks(ticks,ticks+1)
            ax.set_ylabel('Actual cofactor N (bits)');ax.set_title(f'Request B2 {bound:.1e}');ax.grid(False)
            fig.colorbar(im,ax=ax,label='Engine seconds / faster default median; colors clipped at 0.5 / 1.5')
        fig.suptitle(f'ECM-ELF {learned["rows"][0]["version"]} | continuous runtime learning queues')
        save(fig,combined_prefix+'-learning',args,
            'All complete NF calls retained; blank cells follow early/manual stops and have no completed NF observation.\nEarlier observations retained; cap 48, sample cap 10. Timing replays, not new ECM coverage. No manual D/carrier choice.')
    context=args.study/'context.json'
    return dict(complete=True,program_version=plan['program_version'],rows=rows,audit=checked,totals=totals,
                busy_gpu_telemetry=telemetry,context=load(context) if context.exists() else None,runtime=learned)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True)
    p.add_argument('--publish',type=Path)
    p.add_argument('--references-only',action='store_true')
    p.add_argument('--runtime-study',type=Path,help='Complete queue-learning matrix; add best measured profile means as a fourth curve')
    a=p.parse_args();plan=load(a.study/'plan.json')
    if a.references_only and a.runtime_study:p.error('--runtime-study requires the complete comparison')
    # Refuse incomplete studies before writing any public attachment copies.
    if not a.references_only:audit(a.study)
    if a.runtime_study:runtime_matrix(a.runtime_study,a.study)
    a.output.mkdir(parents=True,exist_ok=False);a.figures.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.20,'axes.axisbelow':True,'figure.dpi':150})
    summary=references(plan,a)
    summary['plot_environment']=dict(python=sys.version.split()[0],matplotlib=matplotlib.__version__,numpy=np.__version__)
    if not a.references_only:
        summary.update(current(plan,a))
        with (a.output/'cells.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=summary['rows'][0].keys());writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in summary['rows'])
        if summary['runtime']:
            with (a.output/'runtime_cells.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=summary['runtime']['rows'][0].keys());writer.writeheader()
                writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in summary['runtime']['rows'])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(__file__,a.output/'plot_snapshot.py')
    print('Wrote',a.output,'and',a.figures)


if __name__=='__main__':main()
