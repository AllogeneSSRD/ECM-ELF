"""Audit and plot frozen matched worktodo preparation studies, without CUDA."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import statistics as st

from bench_stage2_d_prepare import log_rows,normalize_result,sha


def audit(study):
    load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    report=load(study/'report.json')
    if not report['complete'] or not report['frozen_control'] or report['failures']:
        raise ValueError('Need a complete frozen control: '+str(study))
    manifest=load(study/'bin/build_manifest.json')
    binary=study/'bin'/Path(manifest['exe']).name
    if sha(binary)!=manifest['sha256'].lower():raise ValueError('Frozen executable differs')
    for name,digest in manifest['source_hashes'].items():
        if sha(study/'sources'/name)!=digest.lower():raise ValueError('Compiled input differs: '+name)
    modes={};C=report['curves_per_queue'];checked=selftests=0;versions=set();busy_clocks=[]
    for run in report['runs']:
        folder=Path(run['folder'])
        command=load(folder/'command.json')
        for flag,value in {'--device':'1','--arena-mb':'6300','--owner-budget-mb':'640','--batch-mb':'256'}.items():
            if command[command.index(flag)+1]!=value:raise ValueError('Plot budget/device differs')
        busy_clocks.extend(s['sm_clock_mhz'] for s in load(folder/'telemetry.json') if s.get('gpu_percent',0)>=50 and s.get('sm_clock_mhz',0)>0)
        raw=[json.loads(line) for line in (folder/'result.jsonl').read_text(encoding='utf-8').splitlines()]
        driver=(folder/'driver.log').read_text(encoding='utf-8')
        parent=list(map(float,re.findall(r'curve_done:.*?\bwall=([\d.]+)',driver)))
        log=(folder/'engine.log').read_text(encoding='utf-8')
        phases=log_rows(log,'stage2_full_wall');checks=log_rows(log,'s4_multiply_stats')
        if not run['profile_unchanged'] or not len(raw)==len(parent)==len(phases)==len(checks)==len(run['curves'])==C:
            raise ValueError('Incomplete or changing queue')
        if run['mode'] in ('on','off','uncached') and sha(folder/'stage2_ecm_tune.toml') not in report['source_ids'].values():
            raise ValueError('Frozen profile differs from its initial identity')
        for rr,p,phase,check,c in zip(raw,parent,phases,checks,run['curves']):
            r=normalize_result(rr);n=int(r['N_hex'],16);versions.add(r['program']['version'])
            times=[p,c['engine_seconds'],c['parent_seconds'],*[float(phase[k]) for k in ('init','main','total')]]
            if any(not math.isfinite(v) or v<0 for v in times) or not p or not float(phase['total']):
                raise ValueError('Invalid elapsed time')
            if rr['status']!='NF' or rr['source-build']!=manifest['sha256'].lower() or r['bad_factors'] or r['factors']:
                raise ValueError('Input, binary or arithmetic status differs')
            if n.bit_length()!=report['input']['bits'] or ((1<<report['input']['exponent'])-1)%n or r['B1']!=report['input']['b1'] or r['B2']!=report['B2'] or r['sigma']!=report['input']['sigma'] or r['curve']!='ed' or r['carrier_exponent'] not in (0,report['input']['exponent']):
                raise ValueError('Task identity differs')
            if phase['clean']!='1' or abs(float(phase['total'])-float(phase['init'])-float(phase['main']))>.003:
                raise ValueError('Engine timing does not close')
            if abs(float(phase['total'])-c['engine_seconds'])>1e-8 or abs(p-c['parent_seconds'])>1e-8 or p<c['engine_seconds']:
                raise ValueError('Stored timing differs from the original ledger')
            if c['D']!=r['d'] or c['carrier']!=r['carrier_exponent'] or c['observation_added']:
                raise ValueError('Geometry or frozen observation status differs')
            if run['mode'] in ('on','off','uncached') and (r['tune_plan']['runtime_probe'] or r['tune_plan']['runtime_samples']<10):
                raise ValueError('Measured route is not stable')
            if min(int(check['gmp_checked']),int(check['gmp_selftest_cases']))<=0 or int(check['gmp_check_bad']) or int(check['gmp_selftest_bad']):
                raise ValueError('Missing or failed mandatory checks')
            checked+=int(check['gmp_checked']);selftests+=int(check['gmp_selftest_cases'])
        if any(not math.isfinite(run[k]) or run[k]<=0 for k in ('curve_wall_sum','engine_sum')) or abs(sum(parent)-run['curve_wall_sum'])>1e-8 or abs(sum(c['engine_seconds'] for c in run['curves'])-run['engine_sum'])>1e-8:
            raise ValueError('Queue totals differ')
        for suffix in ('before.json','after.json'):
            snapshot=load(folder/suffix)
            if snapshot['foreign'] or snapshot['shared_allowed']:raise ValueError('Shared-device control')
        modes.setdefault(run['mode'],[]).append(run)
    results=[]
    for mode,runs in modes.items():
        shapes=sorted({tuple(shape) for r in runs for shape in r['selected_shapes']})
        if len(shapes)!=1:raise ValueError('Changing geometry within a mode')
        engine=st.median(r['engine_sum'] for r in runs)/C
        wall=st.median(r['curve_wall_sum'] for r in runs)/C
        results.append(dict(mode=mode,queues=len(runs),curves_per_queue=C,D=shapes[0][0],carrier=shapes[0][1],
            engine_seconds_per_curve=engine,parent_seconds_per_curve=wall,outside_engine_seconds_per_curve=wall-engine,
            queue_parent_seconds_per_curve=[r['curve_wall_sum']/C for r in runs]))
    return dict(study=str(study.resolve()),report_sha256=sha(study/'report.json'),exe_sha256=manifest['sha256'].lower(),
        versions=sorted(versions),gpu=report['gpu'],input=report['input'],B2=report['B2'],curves_per_queue=C,modes=results,
        busy_sm_mhz=dict(min=min(busy_clocks),median=st.median(busy_clocks),max=max(busy_clocks)) if busy_clocks else None,
        gmp_checked=checked,selftest_cases=selftests)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True,help='Fresh summary directory under data/benchmarks')
    p.add_argument('--figures',type=Path,required=True,help='Fresh figure directory under data/figures')
    a=p.parse_args();studies=[audit(s.resolve()) for s in a.study]
    if a.output.exists() or a.figures.exists():raise ValueError('Preserve existing outputs; choose fresh directories')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=dict(carrier='Default carrier',normal='Default ordinary',uncached='Runtime, cache off',on='Runtime, cache on',off='Frozen selection')
    fig,axes=plt.subplots(1,len(studies),figsize=(7*len(studies),6),squeeze=False)
    for ax,study in zip(axes[0],studies):
        rows=study['modes'];x=list(range(len(rows)))
        repeats='/'.join(map(str,sorted({r['queues'] for r in rows})))
        engine=[r['engine_seconds_per_curve'] for r in rows];outside=[r['outside_engine_seconds_per_curve'] for r in rows]
        ax.bar(x,engine,label='Engine init + main',color='#2a6f9d',width=.62)
        ax.bar(x,outside,bottom=engine,label='Outside engine',color='#e9aa58',width=.62)
        for i,r in enumerate(rows):
            ax.scatter([i]*r['queues'],r['queue_parent_seconds_per_curve'],c='#222',s=22,zorder=3)
            ax.annotate(f'{r["parent_seconds_per_curve"]:.3f}',(i,r['parent_seconds_per_curve']),xytext=(0,8),textcoords='offset points',ha='center')
        ax.set_xticks(x,[labels[r['mode']]+'\nD='+str(r['D']) for r in rows])
        ax.set_title(f'N={study["input"]["bits"]} bits (M{study["input"]["exponent"]})\n'
            f'{study["curves_per_queue"]} curves/queue; {repeats} queues/mode\n'
            f'B1={study["input"]["b1"]}, B2={study["B2"]}')
        ax.set_ylabel('Seconds / curve; median complete queue total / curves')
        ax.set_ylim(0,max(r['parent_seconds_per_curve'] for r in rows)*1.18);ax.grid(axis='y',alpha=.2)
    versions=' / '.join(dict.fromkeys(v for study in studies for v in study['versions']))
    fig.suptitle(f'ECM-ELF {versions} | Stage2 runtime D search: matched queues, no warmup')
    names=' / '.join(dict.fromkeys(s['gpu']['name'] for s in studies))
    clocks=' / '.join(str(s['busy_sm_mhz']['median']) if s['busy_sm_mhz'] else 'N/A' for s in studies)
    fig.text(.5,.025,f'GPU1: {names} | busy SM medians {clocks} MHz (observed) | arena/fold/batch 6300/640/256 MiB\n'
        'Cold first curves and short-queue tails retained. Dots: complete queues. Replayed NF points, not independent ECM coverage.',ha='center',fontsize=9)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.085),ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.16,1,.94))
    a.output.mkdir(parents=True);a.figures.mkdir(parents=True)
    for ext in ('png','svg'):fig.savefig(a.figures/('queue-preparation.'+ext),dpi=160)
    plt.close(fig)
    summary=dict(complete=True,studies=studies,primary='Sum curve_done.wall; engine init+main shown separately',
        limitations='Medians of complete queues divided by their curve count; no variance or causal claim for all N/B2. Outside engine includes launch, selection, preflight and queue I/O.',
        plot_sha256=sha(Path(__file__)),figures={ext:sha(a.figures/('queue-preparation.'+ext)) for ext in ('png','svg')})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(__file__,a.output/'plot_snapshot.py')
    print(json.dumps(dict(complete=True,studies=len(studies),curves=sum(s['curves_per_queue']*sum(r['queues'] for r in s['modes']) for s in studies))))


if __name__=='__main__':main()
