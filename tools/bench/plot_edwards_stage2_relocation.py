"""Generate audited relocation summaries and standalone scientific figures."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import statistics as st


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def stats(v):return dict(median=st.median(v),minimum=min(v),maximum=max(v),mean=st.mean(v),n=len(v))


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--study',type=Path,required=True)
    a.add_argument('--audit',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--figures',type=Path,required=True);args=a.parse_args()
    p=read(args.study/'plan.json');audit=read(args.audit)
    assert audit['complete'] and audit['study_sha256']==sha(args.study/'plan.json')
    args.output.mkdir(parents=True,exist_ok=True);args.figures.mkdir(parents=True,exist_ok=True)
    summary=dict(study_sha256=audit['study_sha256'],audit_sha256=sha(args.audit),
                 plot_source_sha256=sha(__file__),version=audit['version'],binary_sha256=p['binary']['sha256'],cells=[],telemetry={})
    samples=[];count_table=[]
    for c in p['cells']:
        cell={k:v for k,v in c.items() if k not in ('save','save_sha256','known_factor')};cell['arms']={}
        for mode,name in ((0,'original'),(1,'relocation')):
            rows=[r for r in p['runs'] if r['cell']==c['id'] and r['mode']==mode and not r['warmup']]
            arm={k:stats([r['timing'][k] for r in rows]) for k in ('init','main','total')}
            arm['process']=stats([r['process_seconds'] for r in rows])
            ntt=[];by_length={}
            for r in rows:
                text=(Path(r['raw'])/'engine.log').read_text()
                table=re.findall(r'^ntt_count_row:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$',text,re.M)
                ntt.append(sum(int(v[5]) for v in table))
                for L,k,conv,fwd,inv,total in table:by_length.setdefault(int(L),[]).append(int(conv))
            arm['ntt_transforms']=stats(ntt)
            for L,values in sorted(by_length.items()):
                assert len(values)==len(rows)
                count_table.append([c['id'],name,L,len(values),st.median(values),3*st.median(values)])
            for k in ('probe','replay'):arm[k]=stats([r['relocation']['relocation_'+k+'_seconds'] for r in rows])
            cell['arms'][name]=arm
            for r in rows:
                samples.extend(read(Path(r['raw'])/'telemetry.json'))
        cell['reduction_percent']=100*(1-cell['arms']['relocation']['total']['median']/cell['arms']['original']['total']['median'])
        cell['speedup']=cell['arms']['original']['total']['median']/cell['arms']['relocation']['total']['median']
        summary['cells'].append(cell)
    busy=[r for r in samples if (r.get('gpu_percent') or 0)>=50]
    summary['telemetry']['busy_samples']=len(busy)
    for k in ('sm_clock_mhz','power_mw','temperature_c'):
        summary['telemetry'][k]=stats([r[k] for r in busy if r[k] is not None])
    with (args.output/'ntt_counts.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['cell','mode','length','n','convolutions_median','transforms_median']);w.writerows(count_table)
    with (args.output/'timings.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['cell','kind','bits','B1','B2','D','mode','n','total_median_s','total_min_s',
                                  'total_max_s','init_median_s','main_median_s','probe_median_s','replay_median_s','process_median_s'])
        for c in summary['cells']:
            for mode,v in c['arms'].items():w.writerow([c['id'],c['kind'],c['bits'],c['b1'],c['b2'],c['d'],mode,v['total']['n'],
                v['total']['median'],v['total']['minimum'],v['total']['maximum'],v['init']['median'],v['main']['median'],
                v['probe']['median'],v['replay']['median'],v['process']['median']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary['plot_environment']=dict(python=platform.python_version(),matplotlib=matplotlib.__version__,
                                    numpy=importlib.metadata.version('numpy'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    for kind,title,name in (('NF','No-factor runs: complete Stage2','relocation_nf'),('F','Factor runs: complete Stage2 including replay','relocation_factor')):
        cells=[c for c in summary['cells'] if c['kind']==kind]
        fig,axes=plt.subplots(2 if kind=='NF' else 1,2,figsize=(11,8 if kind=='NF' else 5.6),squeeze=False)
        for ax,c in zip(axes.flat,cells):
            for j,(mode,color,label) in enumerate((('original','#617789','Original'),('relocation','#148C82','Relocation'))):
                v=c['arms'][mode]['total'];med=v['median']
                ax.bar([j],[med],.6,color=color,yerr=[[med-v['minimum']],[v['maximum']-med]],capsize=5)
                ax.annotate(f'{med:.3f}s',(j,v['maximum']),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
            ax.set_xticks([0,1],['Original','Relocation']);ax.set_ylabel('Seconds (init + main)')
            ax.set_ylim(0,max(c['arms'][m]['total']['maximum'] for m in ('original','relocation'))*1.2)
            case=f"{c['bits']:,} bits | B2={c['b2']:,} | D={c['d']:,}"
            if kind=='F':case=('Probe hit → replay' if c['id']=='wide_factor_unit' else 'Baby nonunit → original')+'\n'+case
            ax.set_title(case,fontsize=10,pad=22)
            note=f"{c['speedup']:.3f}× original/relocation"
            if kind=='F':note+=f"\nprobe {c['arms']['relocation']['probe']['median']:.3f}s; replay {c['arms']['relocation']['replay']['median']:.3f}s"
            ax.text(.5,.97,note,transform=ax.transAxes,ha='center',va='top',fontsize=9)
        fig.suptitle(title,fontsize=14)
        fig.text(.02,.015,f"Edwards | GPU1 RTX 4060 Laptop | ECM-ELF {audit['version']} ({p['binary']['sha256'][:12]}) | n={p['repeats']}, min/max\nStage1/startup excluded; mandatory checks and replay included. Fixed D, ordinary modulus; each panel has its own scale.",fontsize=8)
        fig.tight_layout(rect=(0,.08,1,.95))
        for ext in ('png','svg'):fig.savefig(args.figures/(name+'.'+ext),dpi=180)
        plt.close(fig)
    summary['figure_sha256']={name+'.'+ext:sha(args.figures/(name+'.'+ext))
                             for name in ('relocation_nf','relocation_factor') for ext in ('png','svg')}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
