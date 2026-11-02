"""Plot full-curve Stage1 A/B alongside the fused-square operator throughput."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics


def load(path):
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    assert data['complete'] and not data['failures'],path
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=Path('data/experiments/cgbn_square_redc_20261102'))
    p.add_argument('--stage1',default='stage1_matrix01')
    p.add_argument('--output',type=Path,default=Path('data/benchmarks/cgbn_square_redc_20261102'))
    p.add_argument('--figures',type=Path,default=Path('data/figures/cgbn_square_redc_20261102'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.figures.mkdir(parents=True,exist_ok=True)
    report=dict(operators=[],qualification=[],clocks_mhz={})
    clock=collections.Counter();busy_clock=collections.Counter()
    folders=[]
    for bits,tpi in ((4096,16),(8192,16),(16384,32)):
        folder=a.evidence/f'perf{bits}_v3';data=load(folder/'summary.json');folders.append(folder)
        for count in sorted({r['count'] for r in data['runs'] if 'round'in r}):
            for mode in (0,7,16):
                rows=[r for r in data['runs'] if r['count']==count and r['mode']==mode and 'round'in r]
                assert len(rows)==3 and {r['round'] for r in rows}=={0,1,2}
                assert len({r['recurrence'] for r in rows})==1
                ms=statistics.median(r['median_ms'] for r in rows);repeats=rows[0]['recurrence']
                report['operators'].append(dict(bits=bits,tpi=tpi,count=count,native=count==data['native_count'],
                    mode=mode,repeats=repeats,median_ms=ms,squares_per_second=count*repeats/(ms*.001),
                    process_medians_ms=[r['median_ms'] for r in rows],exe_sha256=data['exe_sha256'],
                    registers=int(rows[0]['fields']['regs']),local_bytes=int(rows[0]['fields']['local'])))
    stage=load(a.evidence/a.stage1/'summary.json');folders.append(a.evidence/a.stage1)
    assert len(stage['runs'])==len(stage['aggregates'])*6
    assert all(r['matched_full_batch'] for r in stage['runs'])
    report['stage1']=stage
    for folder in sorted(a.evidence.glob('check*_v3')):
        data=load(folder/'summary.json')
        report['qualification'].append(dict(source=str(folder),calls=len(data['runs']),
            rejected_faults=sum(bool(r.get('fault_rejected')) for r in data['runs'])))
    for folder in folders:
        for path in folder.rglob('telemetry.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                if row.get('sm_clock_mhz') is not None:
                    clock[row['sm_clock_mhz']]+=1
                    if row.get('gpu_percent',0)>=20:busy_clock[row['sm_clock_mhz']]+=1
    report['clocks_mhz']=dict(sorted(clock.items()))
    report['busy_clocks_mhz']=dict(sorted(busy_clock.items()))
    resources=a.evidence/'stage1_resources.json'
    if resources.is_file():report['stage1_resources']=json.loads(resources.read_text())
    profile=a.evidence/'ncu_stage1_8192/summary.json'
    if profile.is_file() and json.loads(profile.read_text())['complete']:
        report['stage1_ncu']=[]
        for backend in ('baseline','carrysave'):
            with (profile.parent/backend/'metrics.csv').open(encoding='utf-8-sig') as file:
                rows=[r for r in csv.DictReader(file) if r.get('ID','').isdigit()]
            assert len(rows)==1
            keys=('smsp__inst_executed.sum','launch__registers_per_thread',
                'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum')
            report['stage1_ncu'].append(dict(backend=backend,metrics={k:float(rows[0][k]) for k in keys}))
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.output/'operators.csv').open('w',newline='') as file:
        cols=['bits','tpi','count','native','mode','repeats','median_ms','squares_per_second','registers','local_bytes']
        writer=csv.DictWriter(file,cols,extrasaction='ignore');writer.writeheader();writer.writerows(report['operators'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.9));fig.subplots_adjust(top=.78,bottom=.25,wspace=.32)
    colours={0:'#286b9c',16:'#bd5934'}
    for mode in (0,16):
        for native in (True,False):
            rows=[r for r in report['operators'] if r['mode']==mode and (r['native'] if native else r['count']==4096)]
            assert len(rows)==3
            axes[0].plot(range(3),[r['squares_per_second']/1e6 for r in rows],
                color=colours[mode],linestyle='-' if native else '--',marker='o',
                label=('CGBN' if mode==0 else 'Fused')+(' / native batch' if native else ' / 4096 batch'))
    axes[0].set_yscale('log');axes[0].set_ylabel('Million Montgomery squares / second')
    axes[0].set_title('Operator throughput');axes[0].legend(fontsize=8)
    for model,colour in [('mont0','#286b9c'),('ed','#22826d')]:
        rows=[r for r in stage['aggregates'] if r['model']==model]
        axes[1].plot(range(3),[r['gpu_ratio'] for r in rows],marker='o',color=colour,label=model+' GPU interval')
        axes[1].plot(range(3),[r['groups']['carrysave']['wall_seconds']/r['groups']['baseline']['wall_seconds'] for r in rows],
            marker='x',linestyle='--',color=colour,label=model+' process wall')
    axes[1].axhline(1,color='#555555',linestyle=':',linewidth=1)
    axes[1].set_ylabel('Fused / original time (lower is better)');axes[1].set_title('Complete Stage1');axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(3),['4096 / TPI16','8192 / TPI16','16384 / TPI32']);ax.set_xlabel('Container bits / threads per integer')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('RTX 4060 Laptop: carry-save square/REDC vs original CGBN',y=.96,fontsize=13)
    fig.text(.5,.025,f'3 process rounds, no warmup; Stage1 B1={stage["b1"]}, identical saved batches checked\n'
        'Mont GPU interval includes preparation; Edwards GPU interval is NAF loop only; CPU oracle excluded',
        ha='center',fontsize=8)
    for ext in ('png','svg'):fig.savefig(a.figures/f'cgbn-square-redc-4060lp.{ext}',dpi=170)
    for row in report['operators']:
        if row['native']:print('operator',row['bits'],row['mode'],row['median_ms'],row['squares_per_second'])
    for row in stage['aggregates']:print('Stage1',row['exponent'],row['model'],row['gpu_ratio'],row['groups'])
    print('clocks',report['clocks_mhz'])


if __name__=='__main__':main()
