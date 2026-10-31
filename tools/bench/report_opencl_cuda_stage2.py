"""Validate and plot the complete fixed-D OpenCL/CUDA comparison."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics as st
import re
import sys
from bench_edwards_gpu_stage1 import integer_expression

if hasattr(sys,'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def counters(text,prefix):
    line=next((s for s in text.splitlines() if s.startswith(prefix+':')),'')
    return {k:float(v) for k,v in re.findall(r'(\w+)=([0-9]+(?:\.[0-9]+)?)',line)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--figure',type=Path,required=True)
    parser.add_argument('--linear',action='store_true',help='Linear engine-time axis; ratios always remain linear')
    a=parser.parse_args()
    phase_figure=a.figure.with_name(a.figure.stem+'_phases.png')
    if a.out.exists() or a.figure.exists() or phase_figure.exists():
        raise SystemExit('Choose new derived output paths')
    manifest=json.loads((a.run/'completed.json').read_text(encoding='utf-8'))
    if not manifest['complete']:
        raise RuntimeError('Incomplete matrix')
    for name,digest in manifest.get('frozen_files',{}).items():
        if sha(a.run/name)!=digest:
            raise RuntimeError('Modified frozen build: '+name)
    rows=[json.loads(s) for s in (a.run/'measurements.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=manifest['runs']:
        raise RuntimeError('Measurement count does not match completion manifest')
    if 'matrix_plan' in manifest:
        plan=manifest['matrix_plan']
        expected={(e,b,r,k,e if m=='mersenne' else 0) for e in plan['exponents'] for b in plan['B2']
                  for r in range(manifest['repeats']) for k in manifest['binaries'] for m in plan.get('carrier_modes',['ordinary'])}
        if {(r['exponent'],r['B2'],r['repeat'],r['backend'],r.get('carrier_exponent',0)) for r in rows}!=expected:
            raise RuntimeError('Completed matrix does not match its plan')
    groups=defaultdict(list)
    for row in rows:
        path=Path(row['log'])
        if sha(path)!=row['log_sha256'] or sha(path.parent/'results.jsonl')!=row['result_sha256']:
            raise RuntimeError('Modified raw evidence: '+str(path))
        if row['source_build'].lower()!=manifest['binaries'][row['backend']]['sha256'].lower():
            raise RuntimeError('Executed binary identity mismatch')
        results=[json.loads(s) for s in (path.parent/'results.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        if (len(results)!=1 or results[0]['status']!='NF' or results[0].get('factors',[])
            or results[0]['stage2-engine']['bad_factors'] or results[0]['errors']['invariant']
            or int(results[0]['N']).bit_length()!=row['bits'] or results[0]['d']!=row['D']
            or results[0]['b2']!=row['B2'] or results[0]['poly-size']!=row['P']
            or results[0]['stage2-engine']['carrier_exponent']!=row.get('carrier_exponent',0)
            or results[0]['source-build'].lower()!=row['source_build'].lower()):
            raise RuntimeError('Not a matching verified NF: '+str(path))
        frozen=a.run/'inputs'/Path(results[0]['stage2-engine']['save']).name
        if sha(frozen)!=row['save_sha256']:
            raise RuntimeError('Modified input: '+str(frozen))
        saved=next(s for s in frozen.read_text(encoding='utf-8-sig').splitlines() if 'SIGMA=' in s)
        point=dict(part.strip().split('=',1) for part in saved.split(';') if part.strip())
        if int(results[0]['N'])!=integer_expression(point['N']) or results[0]['b1']!=int(float(point['B1'])):
            raise RuntimeError('Result is not the exact saved N/B1')
        if point['METHOD']=='ECM-EDWARDS-AM' and results[0].get('Edwards',{}).get('sigma')!=int(point['SIGMA']):
            raise RuntimeError('Saved Edwards curve identity differs')
        carrier=row.get('carrier_exponent',0)
        if carrier and ((1<<carrier)-1)%int(results[0]['N']):
            raise RuntimeError('Target does not divide the claimed carrier')
        full=counters(path.read_text(encoding='utf-8-sig'),'stage2_full_wall')
        if (abs(full['total']-full['init']-full['main'])>.003
            or abs(full['total']-row['engine_seconds'])>1e-6):
            raise RuntimeError('Raw timing does not close or differs from the measurement')
        text=path.read_text(encoding='utf-8-sig')
        shape=counters(text,'real_shape')
        if shape['baby_points']!=row['P'] or shape['giant_points']!=row['B2']//row['D']+2:
            raise RuntimeError('Point counts do not cover the requested complete interval')
        if row['backend']=='opencl':
            checks=counters(text,'opencl_checks')
            if checks.get('startup_cases',0)<1 or checks.get('checked_words',0)<1 or checks.get('bad',1):
                raise RuntimeError('Missing/failed OpenCL verification')
        else:
            checks=counters(text,'s4_multiply_stats')
            if (checks.get('gmp_selftest_cases',0)<1 or checks.get('gmp_checked',0)<1
                or checks.get('gmp_selftest_bad',1) or checks.get('gmp_check_bad',1)):
                raise RuntimeError('Missing/failed CUDA verification')
        groups[(row['bits'],row['exponent'],row['B2'],row['backend'],row.get('mode','ordinary'))].append(row)
    cells=[];components=[];phase_shares=[]
    for (bits,exponent,b2,backend,mode),samples in sorted(groups.items()):
        if (len(samples)!=manifest['repeats'] or len({r['save_sha256'] for r in samples})!=1
            or {r['repeat'] for r in samples}!=set(range(manifest['repeats']))
            or len({(r['D'],r['P']) for r in samples})!=1):
            raise RuntimeError('Wrong sample count or changing input')
        cells.append(dict(bits=bits,exponent=exponent,B2=b2,backend=backend,mode=mode,carrier_exponent=samples[0].get('carrier_exponent',0),n=len(samples),D=samples[0]['D'],P=samples[0]['P'],save_sha256=samples[0]['save_sha256'],
            engine_seconds=st.median(r['engine_seconds'] for r in samples),
            min_engine_seconds=min(r['engine_seconds'] for r in samples),max_engine_seconds=max(r['engine_seconds'] for r in samples),
            process_seconds=st.median(r['process_seconds'] for r in samples),
            job_cpu_seconds=st.median(r['accounting']['cpu_seconds'] for r in samples) if all('accounting' in r for r in samples) else None,
            cpu_core_equivalents=st.median(r['accounting']['cpu_seconds']/r['process_seconds'] for r in samples) if all('accounting' in r for r in samples) else None,
            job_peak_commit_mib=st.median(r['accounting']['peak_job_commit_bytes']/2**20 for r in samples) if all('accounting' in r for r in samples) else None,
            init_seconds=st.median(r['init_seconds'] for r in samples),main_seconds=st.median(r['main_seconds'] for r in samples)))
        if backend=='opencl':
            metrics=[]
            for row in samples:
                text=Path(row['log']).read_text(encoding='utf-8-sig')
                point=counters(text,'opencl_points');split=counters(text,'opencl_split');tail=counters(text,'opencl_tail')
                stages=dict(baby=split['baby'],ftree_inverse=split['ftree_inverse'],giant=split['giant'],
                    gtree_fold=split['gtree_fold'],descent=tail['descent'],leaf_oracle=tail['leaf_oracle'],accum=tail['accum'])
                remainder=row['engine_seconds']-sum(stages.values())
                if remainder < -.005:
                    raise RuntimeError('Disjoint phase times exceed full engine time')
                stages['other']=max(0.,remainder)
                metrics.append(dict(point_kernel=point['kernel'],point_coordinates=point['coordinates'],
                    point_kernel_fraction=point['kernel']/row['engine_seconds'],giant=split['giant'],
                    ftree_inverse=split['ftree_inverse'],gtree_fold=split['gtree_fold'],descent=tail['descent'],
                    leaf_oracle=tail['leaf_oracle'],accum=tail['accum'],
                    owned_peak_bytes=counters(text,'opencl_payload')['peak_bytes'],
                    program_build=counters(text,'opencl_programs')['build_seconds'],
                    **{name+'_fraction':value/row['engine_seconds'] for name,value in stages.items()}))
            components.append(dict(bits=bits,exponent=exponent,B2=b2,mode=mode,n=len(samples),
                **{k:st.median(v[k] for v in metrics) for k in metrics[0]}))
            phase_shares.append(dict(bits=bits,exponent=exponent,B2=b2,mode=mode,n=len(samples),
                **{name:100*st.mean(v[name+'_fraction'] for v in metrics) for name in stages}))
    lookup={(r['bits'],r['exponent'],r['B2'],r['backend'],r['mode']):r for r in cells}
    paired=[]
    for r in cells:
        if r['backend']!='opencl':
            continue
        cuda=lookup[(r['bits'],r['exponent'],r['B2'],'cuda',r['mode'])]
        if (r['D'],r['P'],r['exponent'],r['save_sha256'])!=(cuda['D'],cuda['P'],cuda['exponent'],cuda['save_sha256']):
            raise RuntimeError('Backends do not use the same input and shape')
        paired.append(dict(bits=r['bits'],exponent=r['exponent'],B2=r['B2'],D=r['D'],P=r['P'],mode=r['mode'],carrier_exponent=r['carrier_exponent'],
            cuda_seconds=cuda['engine_seconds'],opencl_seconds=r['engine_seconds'],
            opencl_over_cuda=r['engine_seconds']/cuda['engine_seconds'],
            cuda_process_seconds=cuda['process_seconds'],opencl_process_seconds=r['process_seconds'],
            process_ratio=r['process_seconds']/cuda['process_seconds']))
    a.out.mkdir(parents=True)
    for name,data in [('cells',cells),('pairs',paired),('opencl_components',components),('phase_shares',phase_shares)]:
        with (a.out/(name+'.csv')).open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    telemetry=[]
    telemetry_files=sorted(a.run.glob('gpu*.csv'))
    for path in telemetry_files:
        with path.open(encoding='utf-8-sig',newline='') as file:
            telemetry.extend(csv.DictReader(file,skipinitialspace=True))
    clocks=defaultdict(int);active_clocks=defaultdict(int)
    for row in telemetry:
        clock=re.match(r'\d+',row.get('clocks.current.graphics [MHz]') or '')
        utilization=re.match(r'\d+',row.get('utilization.gpu [%]') or '')
        if clock:
            clocks[int(clock[0])]+=1
            if utilization and int(utilization[0])>0:
                active_clocks[int(clock[0])]+=1
    result={'manifest':manifest,'samples':len(rows),'cells':cells,'pairs':paired,'opencl_components':components,'phase_shares':phase_shares,
        'telemetry':{'samples':len(telemetry),'unparsed_clock_samples':len(telemetry)-sum(clocks.values()),'clock_mhz_counts':dict(clocks),'nonzero_utilization_clock_mhz_counts':dict(active_clocks),'files_sha256':{p.name:sha(p) for p in telemetry_files}},
        'sum_medians':{b:sum(r['engine_seconds'] for r in cells if r['backend']==b) for b in ['cuda','opencl']},
        'sum_medians_by_mode':{m:{b:sum(r['engine_seconds'] for r in cells if r['backend']==b and r['mode']==m) for b in ['cuda','opencl']} for m in sorted({r['mode'] for r in cells})},
        'sum_process_medians':{b:sum(r['process_seconds'] for r in cells if r['backend']==b) for b in ['cuda','opencl']}}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(a.out/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bounds=sorted({r['B2'] for r in cells})
    modes=sorted({r['mode'] for r in cells},key=lambda m:m!='ordinary')
    fig,axes=plt.subplots(2*len(modes),len(bounds),figsize=(6.5*len(bounds),7*len(modes)),squeeze=False,layout='constrained')
    for mi,col in ((i,j) for i in range(len(modes)) for j in range(len(bounds))):
        mode=modes[mi];b2=bounds[col];ax=axes[2*mi,col]
        for backend,color in [('cuda','#2563eb'),('opencl','#ea580c')]:
            line=sorted((r for r in cells if r['B2']==b2 and r['backend']==backend and r['mode']==mode),key=lambda x:x['bits'])
            x=[r['bits'] for r in line];y=[r['engine_seconds'] for r in line]
            version=manifest['versions'][backend].removeprefix('ECM-ELF ')
            ax.errorbar(x,y,yerr=[[r['engine_seconds']-r['min_engine_seconds'] for r in line],[r['max_engine_seconds']-r['engine_seconds'] for r in line]],marker='o',label=f'{backend.upper()} {version}',color=color,capsize=4)
        ax.set(title=f'{mode} | B2 = {b2:,}; D = {manifest["D"]:,}',ylabel='Full engine seconds'+('' if a.linear else ' (log scale)'),yscale='linear' if a.linear else 'log',xlabel='Actual cofactor bits')
        if a.linear:ax.set_ylim(bottom=0)
        ticks=range(0,max(r['bits'] for r in cells)+1000,1000)
        ax.set_xticks(ticks);ax.tick_params(axis='x',labelsize=8,rotation=45);ax.grid(alpha=.2);ax.legend()
        if manifest['versions']['opencl']=='ECM-ELF 0.54.3' and max(x)>8192:
            ax.axvline(8192,color='#777',ls=':',alpha=.6)
        ax=axes[2*mi+1,col]
        line=sorted((r for r in paired if r['B2']==b2 and r['mode']==mode),key=lambda x:x['bits'])
        x=[r['bits'] for r in line]
        ax.plot(x,[r['opencl_over_cuda'] for r in line],marker='o',label='Engine total')
        ax.plot(x,[r['process_ratio'] for r in line],marker='s',label='Full process')
        ax.axhline(1,color='#777',ls='--');ax.set(xlabel='Actual cofactor bits',ylabel='OpenCL / CUDA (higher = slower OpenCL)')
        ax.set_xticks(ticks);ax.tick_params(axis='x',labelsize=8,rotation=45);ax.grid(alpha=.2);ax.legend()
        if manifest['versions']['opencl']=='ECM-ELF 0.54.3' and max(x)>8192:
            ax.axvline(8192,color='#777',ls=':',alpha=.6)
    observed='/'.join(map(str,sorted(active_clocks)))+' MHz active samples'
    fig.suptitle(f'Native OpenCL vs frozen CUDA Stage2 | RTX 4060 Laptop | {observed} | n={manifest["repeats"]}\nIdentical save/D/B2; no extra warmup; initialization and mandatory checks included')
    a.figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.figure,dpi=170);plt.close(fig)
    # Means of per-run shares close to 100%; nested point/program counters are
    # deliberately absent. They must not be stacked on their parent phases.
    fig,axes=plt.subplots(len(modes),len(bounds),figsize=(7*len(bounds),4.5*len(modes)),squeeze=False)
    fig.subplots_adjust(top=.85,bottom=.09,hspace=.5,wspace=.16)
    labels={'baby':'Baby points','ftree_inverse':'F tree / inverse','giant':'Giant points',
        'gtree_fold':'G trees / fold','descent':'Descent','leaf_oracle':'Leaf oracle','accum':'Product / GCD','other':'Preparation / other'}
    colors=['#3684b9','#79b9d6','#e17c33','#57a66a','#9273b6','#dd9fb0','#c2ad50','#a6a6a6']
    for mi,mode in enumerate(modes):
        for col,b2 in enumerate(bounds):
            line=sorted((r for r in phase_shares if r['mode']==mode and r['B2']==b2),key=lambda r:r['bits'])
            x=list(range(len(line)));bottom=[0.]*len(line);ax=axes[mi,col]
            for (key,label),color in zip(labels.items(),colors):
                values=[r[key] for r in line]
                ax.bar(x,values,bottom=bottom,label=label,color=color,width=.85)
                bottom=[a+b for a,b in zip(bottom,values)]
            ax.set(ylim=(0,100),title=f'{mode} | B2 = {b2:,}',ylabel='OpenCL engine time (%)',xlabel='Actual cofactor bits')
            ax.set_xticks(x,[r['bits'] for r in line],rotation=45,ha='right');ax.grid(axis='y',alpha=.2)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=4)
    fig.suptitle(f'OpenCL {manifest["versions"]["opencl"].removeprefix("ECM-ELF ")} phase shares | mean of per-run percentages | n={manifest["repeats"]}',y=.99)
    fig.savefig(phase_figure,dpi=170);plt.close(fig)
    print(json.dumps({'samples':len(rows),'sum_medians':result['sum_medians'],'sum_process_medians':result['sum_process_medians'],'telemetry':result['telemetry'],'pairs':paired},ensure_ascii=False))

if __name__=='__main__':
    main()
