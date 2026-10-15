"""Summarize isolated preparation and complete Stage2 pack/zero A/B evidence."""
import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator',type=Path,required=True)
    p.add_argument('--stage2',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True)
    p.add_argument('--systems-off',type=Path,help='Validated default-path Systems output directory')
    p.add_argument('--systems-on',type=Path,help='Validated fused-path Systems output directory')
    a=p.parse_args()
    if bool(a.systems_off)!=bool(a.systems_on):p.error('both Systems directories required')
    op=json.loads(a.operator.read_text());full=json.loads(a.stage2.read_text())
    if not op['complete'] or not full['complete'] or full['identity']['variant']!='pack-zero':
        raise ValueError('complete pack/zero studies required')
    a.output.mkdir(parents=True,exist_ok=True);a.figures.mkdir(parents=True,exist_ok=True)
    shapes=list(dict.fromkeys((s['bits'],s['P'],s['batch']) for s in op['summaries']))
    fig,axes=plt.subplots(2,1,figsize=(13,9),layout='constrained')
    x=np.arange(len(shapes));width=.25
    operator=[]
    for path,name in [(0,'contiguous'),(1,'gather'),(2,'reverse gather')]:
        values=[]
        for shape in shapes:
            s=next(s for s in op['summaries'] if (s['bits'],s['P'],s['batch'],s['path'])==(*shape,path))
            values.append(s['gain_percent'])
            operator.append(dict(bits=s['bits'],P=s['P'],batch=s['batch'],path=name,N=s['N'],
                off_seconds=s['off']['median_seconds'],on_seconds=s['on']['median_seconds'],
                off_mad=s['off']['mad_seconds'],on_mad=s['on']['mad_seconds'],gain_percent=s['gain_percent']))
        axes[0].bar(x+(path-1)*width,values,width,label=name)
    op_counts=sorted({len(s['samples'])//2 for s in op['summaries']})
    op_count_text='/'.join(map(str,op_counts))
    axes[0].set(xticks=x,xticklabels=[f'{b} bits\nP={P}, batch={batch}' for b,P,batch in shapes],
        ylabel='Preparation time reduction (%)',title=f'Isolated memset+pack vs complete pack/zero | {op_count_text} samples/side')
    axes[0].legend(ncols=3);axes[0].grid(axis='y',alpha=.2);axes[0].set_axisbelow(True)
    rows=[]
    for s in full['summaries']:
        off,on=s['off'],s['on']
        if off['samples']!=on['samples']:raise ValueError('unbalanced full curve samples')
        rows.append(dict(case=s['case'],off_seconds=off['engine_seconds'],on_seconds=on['engine_seconds'],
            off_mad=off['engine_seconds_mad'],on_mad=on['engine_seconds_mad'],gain_percent=s['gain_percent'],
            samples_per_side=off['samples'],off_process_seconds=off['process_seconds'],on_process_seconds=on['process_seconds'],
            off_full_peak_bytes=off['full_peak_bytes'],on_full_peak_bytes=on['full_peak_bytes']))
    off=sum(s['off_seconds'] for s in rows);on=sum(s['on_seconds'] for s in rows)
    gain=100*(1-on/off)
    errors=[100*s['on_seconds']/s['off_seconds']*math.hypot(s['on_mad']/s['on_seconds'],s['off_mad']/s['off_seconds']) for s in rows]
    axes[1].bar(range(len(rows)),[s['gain_percent'] for s in rows],yerr=errors,capsize=3,
        color=['#27836e' if s['gain_percent']>=0 else '#b95756' for s in rows])
    axes[1].axhline(0,color='#555',lw=.8)
    axes[1].set(xticks=range(len(rows)),xticklabels=[s['case'] for s in rows],ylabel='Engine time reduction (%)',
        title=f'Fixed-D full Stage2 | sum of case medians {off:.3f} -> {on:.3f}s ({gain:+.2f}%)\nPropagated MAD scale, not CI')
    axes[1].tick_params(axis='x',rotation=28,labelsize=8);axes[1].grid(axis='y',alpha=.2);axes[1].set_axisbelow(True)
    for i,r in enumerate(rows):axes[1].annotate(f"{r['gain_percent']:+.2f}",(i,r['gain_percent']+(errors[i] if r['gain_percent']>=0 else -errors[i])),
        xytext=(0,3 if r['gain_percent']>=0 else -3),textcoords='offset points',ha='center',va='bottom' if r['gain_percent']>=0 else 'top',fontsize=8)
    fig.suptitle('RTX 4060 Laptop | default 55W / 1800MHz cap; actual clocks vary | no discarded warmup')
    fig.savefig(a.figures/'stage2_pack_zero_20261015.png',dpi=170);plt.close(fig)
    for name,data in [('operator',operator),('stage2',rows)]:
        with (a.output/(name+'.csv')).open('w',newline='',encoding='utf-8') as stream:
            w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summary=dict(operator_samples=sum(len(s['samples']) for s in op['summaries']),stage2_samples=sum(2*s['samples_per_side'] for s in rows),
        matrix_off_seconds=off,matrix_on_seconds=on,matrix_gain_percent=gain,operator_binary=op['binary_sha256'],
        stage2_binary=full['identity']['identity']['binary_sha256'],rows=rows)
    if a.systems_off:
        profiles={};inputs=None
        for enabled,directory in [(0,a.systems_off),(1,a.systems_on)]:
            report=json.loads((directory/'summary.json').read_text())
            meta=json.loads((directory/'command.json').read_text())
            if not report['arithmetic_verified'] or meta['binary_sha256']!=summary['stage2_binary'] or int(meta['environment']['NTT_S4_PACK_ZERO'])!=enabled:
                raise ValueError('Systems capture does not match full curve binary/path')
            invariant=[report['application'][key] for key in ['correctness','histogram','convolutions','chunks']]+[meta['inputs']]
            if inputs is None:inputs=invariant
            if invariant!=inputs:raise ValueError('Systems input/arithmetic/workload changed')
            kinds={}
            for k in report['kernels']:
                n=k['name']
                kind='NTT transform' if any(x in n for x in ['outer_fwd_kernel','outer_inv_kernel','outer_coop_kernel','tile_kernel']) else 'pack' if 'pack_zero_kernel' in n or 's4_pack_' in n else 'S4 reduce' if 's4_reduce_kernel' in n or 's4_mersenne_kernel' in n else 'other'
                kinds[kind]=kinds.get(kind,0)+k['seconds']
            uri='file:'+(directory/'trace.sqlite').resolve().as_posix()+'?mode=ro'
            with sqlite3.connect(uri,uri=True) as db:
                count,byte_count,seconds=db.execute('select count(*),sum(bytes),sum(end-start)/1e9 from CUPTI_ACTIVITY_KIND_MEMSET where deviceId=?',(full['identity']['device'],)).fetchone()
            profiles['on' if enabled else 'off']=dict(kernel_seconds=kinds,
                kernel_total_seconds=sum(k['seconds'] for k in report['kernels']),
                memset=dict(count=count,bytes=byte_count,seconds=seconds),window=report['window'],copies=report['copies'],
                binary=meta['binary_sha256'],capture_sha256=report['report_sha256'])
        x,y=profiles['off'],profiles['on']
        profiles['preparation_gain_percent']=100*(1-(y['kernel_seconds']['pack']+y['memset']['seconds'])/(x['kernel_seconds']['pack']+x['memset']['seconds']))
        profiles['scope']='Diagnostic event sums include counter memset; excluded from formal A/B; do not add API waits.'
        summary['profiles']=profiles
        (a.output/'profile_comparison.json').write_text(json.dumps(profiles,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','profiles')},indent=2))
if __name__=='__main__':main()
