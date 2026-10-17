"""Audit a complete exact-N build/family comparison and render exportable plots."""
import argparse
import csv
import math
import json
from pathlib import Path
import statistics

from bench_stage2_production import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True);a=p.parse_args()
    path=a.study/'plan.json';data=json.loads(path.read_text())
    if not data['complete']:raise ValueError('complete matrix required')
    audit=json.loads(a.audit.read_text())
    if not audit['complete'] or audit['study_sha256']!=sha(path):raise ValueError('matching independent audit required')
    for directory in (a.output,a.figures):
        directory.mkdir(parents=True,exist_ok=True)
        if any(directory.iterdir()):raise ValueError('use fresh output directories')
    for binary in data['binaries'].values():
        if sha(binary['path'])!=binary['sha256']:raise ValueError('binary changed')
    for row in data['runs']:
        for name,digest in row['files'].items():
            if sha(name)!=digest:raise ValueError('raw evidence changed')
    expected=len(data['cells'])*4*(data['repeats']+1)
    if len(data['runs'])!=expected or len({r['id'] for r in data['runs']})!=expected:
        raise ValueError('incomplete or duplicate run set')
    rows=[];active=[]
    for cell in data['cells']:
        case=data['cases'][str(cell['exponent'])]
        for model in ('ed','mont'):
            for build in ('baseline','candidate'):
                samples=[r for r in data['runs'] if r['cell']==cell['id'] and r['model']==model and r['build']==build and r['category']=='timing']
                if len(samples)!=data['repeats'] or {r['repeat'] for r in samples}!=set(range(1,data['repeats']+1)):
                    raise ValueError('incorrect formal repetitions')
                values=[r['engine_seconds'] for r in samples]
                process=[r['process_seconds'] for r in samples]
                rows.append(dict(cell=cell['id'],exponent=cell['exponent'],bits=case['bits'],carrier=cell['carrier'],
                    b1=case['b1'],b2=cell['b2'],sigma=case['sigma'],d=cell['d'],model=model,build=build,n=len(values),
                    version=samples[0]['version'],median=statistics.median(values),minimum=min(values),maximum=max(values),
                    mean=statistics.mean(values),process_median=statistics.median(process),
                    process_minimum=min(process),process_maximum=max(process),
                    init_mean=statistics.mean(float(r['stats']['stage2_full_wall']['init']) for r in samples),
                    main_mean=statistics.mean(float(r['stats']['stage2_full_wall']['main']) for r in samples)))
                for run in samples:
                    telemetry=next(Path(f) for f in run['files'] if Path(f).name=='telemetry.json')
                    active.extend(v for v in json.loads(telemetry.read_text()) if (v.get('gpu_percent') or 0)>=50)
    result=dict(complete=True,study_sha256=sha(path),audit_sha256=sha(a.audit),collector_sha256=data['collector_sha256'],
                plotter_sha256=sha(__file__),binaries=data['binaries'],rows=rows,formal_samples=len(rows)*data['repeats'],
                warmup_samples=4*len(data['cells']),device=data['device_identity'],protocol=data['protocol'],
                timing_scope=dict(engine='stage2_full_wall init+main; baby enumeration (shape, already in init), in-engine NTT/S4 checks and GCD included',
                    excluded='Stage1, startup, outer D selection, preflight Montgomery/xADD checks',
                    process='whole child process; includes at most one 0.2-second exit polling interval'))
    observations={}
    for key in ('sm_clock_mhz','power_mw','temperature_c'):
        v=[s[key] for s in active if s.get(key) is not None]
        if v:observations[key]=dict(n=len(v),minimum=min(v),median=statistics.median(v),maximum=max(v))
    result['active_device_observations']=observations
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (a.output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    cells=data['cells'];x=np.arange(len(cells));colors=dict(ed='#2468A0',mont='#C06A29')
    def group(cell,model,build):return next(r for r in rows if r['cell']==cell['id'] and r['model']==model and r['build']==build)
    labels=[f'M{c["exponent"]} / {data["cases"][str(c["exponent"])]["bits"]} bits\n'
            f'{c["mode"]}\nB2={c["b2"]:,}\nD={c["d"]:,}' for c in cells]
    clock=observations.get('sm_clock_mhz',{})
    observed=f'observed SM clocks {clock.get("minimum","?")}–{clock.get("maximum","?")} MHz; no setting changes'
    power=observations.get('power_mw',{})
    watts=f'{power["minimum"]/1000:.1f}–{power["maximum"]/1000:.1f} W' if power else 'unavailable'
    device=data['device_identity']
    caption=f'GPU{device["index"]} {device["name"]}; {observed}; sampled whole-device power {watts}.\n'
    bounds=','.join(map(str,sorted({r['b1'] for r in rows})))
    seeds=','.join(map(str,sorted({r['sigma'] for r in rows})))
    caption+=f'n={data["repeats"]} per arm; one predeclared warmup per arm excluded. B1={bounds}; sigma={seeds}.\n'
    caption+='Engine init+main includes baby enumeration, in-engine checks/GCD; preflight Mont/xADD checks, startup, outer D selection and Stage1 excluded.\n'
    caption+='Same exact N/B1/B2/D; separate curve families. '
    versions={build:next(r['version'] for r in rows if r['build']==build) for build in ('baseline','candidate')}
    caption+='Baseline '+versions['baseline']+' '+data['binaries']['baseline']['sha256'][:12]+'; candidate '+versions['candidate']+' '+data['binaries']['candidate']['sha256'][:12]+'. Study '+sha(path)[:12]+'.'
    artifacts=[]
    def save(fig,name):
        for extension in ('png','svg'):
            target=a.figures/(name+'.'+extension);fig.savefig(target,dpi=170)
            artifacts.append(dict(path=str(target.resolve()),sha256=sha(target)))
        plt.close(fig)
    fig,(ax,bx)=plt.subplots(2,1,figsize=(16,10),gridspec_kw={'height_ratios':[2,1]},sharex=True)
    width=.18
    for index,(model,build) in enumerate((('ed','baseline'),('ed','candidate'),('mont','baseline'),('mont','candidate'))):
        selected=[group(c,model,build) for c in cells];y=[r['median'] for r in selected]
        error=[[r['median']-r['minimum'] for r in selected],[r['maximum']-r['median'] for r in selected]]
        ax.bar(x+(index-1.5)*width,y,width,yerr=error,capsize=2,color=colors[model],alpha=.45 if build=='baseline' else 1,
               label=('Edwards' if model=='ed' else 'Montgomery PARAM0')+' '+selected[0]['version'])
    ax.set_yscale('log');ax.set_ylabel('Stage2 engine median (seconds, log scale)')
    fig.suptitle('Same-input GPU Stage2: Edwards and Montgomery PARAM0, before and after optimization',y=.98)
    handles,legend=ax.get_legend_handles_labels()
    fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.5,.95),ncol=2)
    ax.grid(axis='y',alpha=.2)
    for index,model in enumerate(('ed','mont')):
        ratios=[group(c,model,'baseline')['median']/group(c,model,'candidate')['median'] for c in cells]
        bars=bx.bar(x+(index-.5)*.3,ratios,.3,color=colors[model],label='Edwards' if model=='ed' else 'PARAM0')
        bx.bar_label(bars,fmt='%.2fx',fontsize=9,padding=2)
    bx.axhline(1,color='0.4',linestyle='--',linewidth=1);bx.set_ylabel('Baseline / candidate time')
    bx.set_xticks(x,labels);bx.tick_params(axis='x',labelsize=9)
    bx.set_xlabel('Original M exponent / actual cofactor width, arithmetic mode, exact B2 and D')
    bx.grid(axis='y',alpha=.2)
    fig.subplots_adjust(top=.84,bottom=.25,hspace=.12);fig.text(.07,.025,caption,fontsize=8)
    save(fig,'edwards_param0_complete')
    nrows=math.ceil(len(cells)/3)
    fig,axes=plt.subplots(nrows,3,figsize=(16,4*nrows+2),squeeze=False)
    fig.suptitle('GPU Stage2 init/main means',y=.98)
    for ax,cell in zip(axes.flat,cells):
        selected=[group(cell,m,b) for m,b in (('ed','baseline'),('ed','candidate'),('mont','baseline'),('mont','candidate'))]
        init=[r['init_mean'] for r in selected];main=[r['main_mean'] for r in selected]
        ax.bar(range(4),init,label='Init',color='#728AA4');ax.bar(range(4),main,bottom=init,label='Main',color='#334F6F')
        ax.set_xticks(range(4),['Ed old','Ed new','P0 old','P0 new']);ax.set_ylabel('Stage2 mean (seconds)')
        ax.set_title(f'M{cell["exponent"]} / {data["cases"][str(cell["exponent"])]["bits"]} bits — {cell["mode"]}\n'
                     f'B2={cell["b2"]:,}  D={cell["d"]:,}');ax.grid(axis='y',alpha=.2)
    for ax in list(axes.flat)[len(cells):]:ax.set_visible(False)
    handles,legend=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.5,.95),ncol=2)
    fig.subplots_adjust(top=.85,bottom=.18,hspace=.55)
    fig.text(.06,.025,'Mutually exclusive init/main means sum to the full mean at log precision. Nested timers are not added.\n'+caption,fontsize=8)
    save(fig,'edwards_param0_phases')
    (a.figures/'manifest.json').write_text(json.dumps(dict(study_sha256=sha(path),audit_sha256=sha(a.audit),plotter_sha256=sha(__file__),artifacts=artifacts),indent=2)+'\n')
    print('COMPLETE',len(rows),'statistics rows;',len(artifacts),'figure artifacts')


if __name__=='__main__':main()
