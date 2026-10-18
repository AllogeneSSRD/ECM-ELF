"""Render audited original-path Stage2 comparisons and explicit B2 expansion."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import statistics as st
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):return dict(n=len(v),median=st.median(v),minimum=min(v),maximum=max(v),mean=st.mean(v))
def percent(v):return '0.00%' if abs(v)<.005 else f'{v:+.2f}%'
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','output','figures'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();p=read(a.study/'plan.json');audit=read(a.audit)
    if not audit['complete'] or audit['study_sha256']!=sha(a.study/'plan.json'):raise ValueError('Audit differs')
    for folder in (a.output,a.figures):
        folder.mkdir(parents=True,exist_ok=True)
        if any(folder.iterdir()):raise ValueError('Use fresh output directories')
    summary=dict(study_sha256=sha(a.study/'plan.json'),audit_sha256=sha(a.audit),source_sha256=sha(__file__),
        versions=audit['versions'],binaries={k:b['sha256'] for k,b in p['binaries'].items()},cells=[])
    if 'context_sha256' in audit:
        if sha(a.study/'context.json')!=audit['context_sha256']:raise ValueError('Context differs')
        summary['context']=read(a.study/'context.json');summary['context_sha256']=audit['context_sha256']
    table=[];ntt=[];telemetry=[]
    for c in p['cells']:
        cell={k:v for k,v in c.items() if k not in ('save','save_sha256')};cell['arms']={}
        for arm in ('baseline','candidate','rounded'):
            rows=[r for r in p['runs'] if r['cell']==c['id'] and r['arm']==arm and not r['warmup']]
            if not rows:continue
            v={key:stats([r['timing'][key] for r in rows]) for key in ('init','main','total')}
            v['process']=stats([r['process_seconds'] for r in rows]);v['b2']=rows[0]['b2'];cell['arms'][arm]=v
            v['points']=v['b2']//c['d']+2
            v['giant_points_per_engine_second']=stats([v['points']/r['timing']['total'] for r in rows])
            for r in rows:
                telemetry+=read(Path(r['raw'])/'telemetry.json')
                text=(Path(r['raw'])/'engine.log').read_text()
                counts=re.findall(r'^ntt_count_row:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$',text,re.M)
                for L,k,conv,fwd,inv,total in counts:ntt.append([c['id'],arm,r['repeat'],L,conv,total])
            t=v['total'];table.append([c['id'],c['kind'],c['bits'],c['b1'],v['b2'],c['d'],arm,t['n'],t['median'],t['minimum'],t['maximum'],v['init']['median'],v['main']['median'],v['process']['median']])
        cell['reduction_percent']=100*(1-cell['arms']['candidate']['total']['median']/cell['arms']['baseline']['total']['median'])
        if 'rounded' in cell['arms']:
            cell['rounding_reduction_percent']=100*(1-cell['arms']['rounded']['total']['median']/cell['arms']['candidate']['total']['median'])
            cell['rounding_point_throughput_gain_percent']=100*(cell['arms']['rounded']['giant_points_per_engine_second']['median']/cell['arms']['candidate']['giant_points_per_engine_second']['median']-1)
        summary['cells'].append(cell)
    busy=[v for v in telemetry if (v.get('gpu_percent') or 0)>=50]
    summary['telemetry']=dict(busy_samples=len(busy),**{k:stats([v[k] for v in busy if v.get(k) is not None]) for k in ('sm_clock_mhz','power_mw','temperature_c')})
    for name,header,rows in [('timings.csv',['cell','kind','bits','B1','actual_B2','D','arm','n','total_median_s','min_s','max_s','init_median_s','main_median_s','process_median_s'],table),('ntt_counts.csv',['cell','arm','repeat','length','convolutions','transforms'],ntt)]:
        with (a.output/name).open('w',newline='',encoding='utf-8') as f:w=csv.writer(f);w.writerow(header);w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    summary['plot_environment']=dict(python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=importlib.metadata.version('numpy'))
    def plot(cells,arms,name,title):
        fig,axes=plt.subplots((len(cells)+1)//2,2,figsize=(11,8 if len(cells)>2 else 5.2),squeeze=False)
        for ax,c in zip(axes.flat,cells):
            for j,(key,label,color) in enumerate(arms):
                v=c['arms'][key]['total'];m=v['median']
                ax.bar(j,m,.62,color=color,yerr=[[m-v['minimum']],[v['maximum']-m]],capsize=4)
                ax.annotate(f'{m:.3f} s',(j,v['maximum']),xytext=(0,5),textcoords='offset points',ha='center')
            ax.set_xticks(range(len(arms)),[v[1] for v in arms]);ax.set_ylabel('Seconds: init + main')
            ax.set_ylim(0,max(c['arms'][v[0]]['total']['maximum'] for v in arms)*1.24)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if name=='full_g_boundary':
                extra=100*(c['g']*c['p']/c['points']-1)
                detail=f"B2: {c['b2']:,} → {c['rounded_b2']:,}\nExtra points: {extra:.2f}% | time reduction: {percent(c['rounding_reduction_percent'])}"
            else:
                detail=f"B2={c['b2']:,} | D={c['d']:,}\nTime reduction: {percent(c['reduction_percent'])}"
                if c['kind']=='F':detail+=f" | factor={int(c['known_factor']):,}"
            ax.set_title(f"{c['bits']:,} bits | {c['kind']} | sigma={c['sigma']}\n{detail}",fontsize=10)
        for ax in list(axes.flat)[len(cells):]:ax.set_visible(False)
        fig.suptitle(title,fontsize=14,y=.99)
        bounds=', '.join(f'{v:,}' for v in sorted({c['b1'] for c in cells}))
        fig.text(.5,.02,'RTX 4060 Laptop GPU (GPU1) | relocation off | B1='+bounds+' | fresh process per curve | median; whiskers: min–max\n'+str(p['repeats'])+' measured runs per arm, one warmup excluded and retained | actual B2 is expanded only in the full-G experiment',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.09,1,.94))
        for ext in ('png','svg'):fig.savefig(a.figures/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    normal=[('baseline',audit['versions']['baseline'],'#64798A'),('candidate',audit['versions']['candidate'],'#148C82')]
    plot([c for c in summary['cells'] if c['kind']=='NF'],normal,'original_path_nf','Edwards original path: complete Stage2, identical coverage')
    plot([c for c in summary['cells'] if c['kind']=='F'],normal,'original_path_factor','Edwards factor results: complete Stage2, identical coverage')
    rounded=[c for c in summary['cells'] if 'rounded' in c['arms']]
    if rounded:plot(rounded,[('candidate','Requested B2','#148C82'),('rounded','Full-G B2','#D58E3D')],'full_g_boundary','Edwards '+audit['versions']['candidate']+': explicit B2 expansion, additional coverage and cost')
    summary['figures']={f.name:sha(f) for f in a.figures.iterdir() if f.is_file()}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps([dict(cell=c['id'],reduction_percent=c['reduction_percent'],rounding=c.get('rounding_reduction_percent')) for c in summary['cells']],indent=2))
if __name__=='__main__':main()
