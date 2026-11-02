"""Summarise qualified square cohorts without mixing legacy low-half timings.

Input: bench_cgbn_square.py result folders under --evidence.
Output: evidence identities, all process medians, ratios and a static figure.
"""
import argparse
import json
import statistics
from pathlib import Path


def summarise(source, reference, candidate, require_full_wide=False):
    report=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    assert report['complete'] and not report['failures'], source
    if require_full_wide:
        assert report['wide_feedback']=='xor_halves', source
    counts=sorted({r['count'] for r in report['runs'] if 'round' in r})
    rows=[]
    for count in counts:
        groups=[]
        for mode in (reference,candidate):
            runs=[r for r in report['runs'] if r['count']==count and r['mode']==mode and 'round' in r]
            assert len(runs)==3 and {r['round'] for r in runs}=={0,1,2}, (source,count,mode)
            assert len({r['recurrence'] for r in runs})==1
            assert all(int(r['fields']['bad'])==0 and int(r['fields']['guards'])==1 for r in runs)
            values=[r['median_ms'] for r in runs]
            groups.append(dict(mode=mode,median_ms=statistics.median(values),
                process_medians_ms=values,recurrence=runs[0]['recurrence'],
                registers=int(runs[0]['fields']['regs']),local_bytes=int(runs[0]['fields']['local']),
                shared_bytes=int(runs[0]['fields']['shared'])))
        rows.append(dict(count=count,native=count==report['native_count'],
            reference=groups[0],candidate=groups[1],
            time_ratio=groups[1]['median_ms']/groups[0]['median_ms']))
    telemetry=[]
    for path in source.glob('call*/telemetry.jsonl'):
        telemetry.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
    clocks=[r['sm_clock_mhz'] for r in telemetry if r.get('sm_clock_mhz') is not None]
    return dict(source=str(source),bits=report['bits'],tpi=report['tpi'],
        native_count=report['native_count'],exe_sha256=report['exe_sha256'],
        dll_sha256=report['dll_sha256'],wide_feedback=report.get('wide_feedback','legacy_low_half'),rows=rows,
        device_identity=report['runs'][0].get('device_identity'),
        telemetry_samples=len(telemetry),clock_range_mhz=[min(clocks),max(clocks)] if clocks else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=Path('data/experiments/cgbn_square_20261102'))
    p.add_argument('--output',type=Path,default=Path('data/benchmarks/cgbn_square_20261102'))
    p.add_argument('--figures',type=Path,default=Path('data/figures/cgbn_square_20261102'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.figures.mkdir(parents=True,exist_ok=True)
    result=dict(cohorts=[],qualification=[])
    for bits,tpi in ((2048,8),(4096,16),(8192,16),(16384,32)):
        for name,prefix,ref,cand,wide in (
            ('Montgomery','final_performance' if bits<8192 else 'final_perf',0,7,False),
            ('Mersenne','final_mersenne',13,14,False),
            ('Full wide + XOR feedback','raw_performance',3,8,True)):
            row=summarise(a.evidence/f'{prefix}{bits}_{tpi}',ref,cand,wide)
            row['operation']=name;result['cohorts'].append(row)
    patterns=('raw_correctness*','final_check*','final_correctness*','boundary_check*')
    for pattern in patterns:
        for file in sorted(a.evidence.glob(pattern+'/summary.json')):
            report=json.loads(file.read_text(encoding='utf-8'))
            assert report['complete'] and not report['failures'], file
            faults=sum(bool(r.get('fault_rejected')) for r in report['runs'])
            result['qualification'].append(dict(source=str(file.parent),
                bits=report['bits'],tpi=report['tpi'],calls=len(report['runs']),
                checked_calls=len(report['runs'])-faults,faults_rejected=faults,
                exe_sha256=report['exe_sha256']))
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5.4))
    fig.subplots_adjust(top=.78,bottom=.23,left=.10,right=.98)
    colours=('#b65333','#238066','#356fac')
    operations=('Montgomery','Mersenne','Full wide + XOR feedback')
    for j,(name,colour) in enumerate(zip(operations,colours)):
        cohort=[r for r in result['cohorts'] if r['operation']==name]
        values=[next(v for v in r['rows'] if v['native'])['time_ratio'] for r in cohort]
        ax.bar([i+(j-1)*.24 for i in range(4)],values,width=.23,label=name,color=colour)
        for i,v in enumerate(values):ax.text(i+(j-1)*.24,v+.012,f'{v:.3f}',ha='center',fontsize=8)
    ax.axhline(1,color='#444444',linewidth=1,linestyle='--')
    ax.set_xticks(range(4),['2048 / TPI8','4096 / TPI16','8192 / TPI16','16384 / TPI32'])
    ax.set_xlabel('Container bits / threads per integer (modular inputs: bits - 6)')
    ax.set_ylabel('Specialised / original time (lower is better)')
    fig.suptitle('RTX 4060 Laptop, 1500 MHz: finite-depth CGBN square',y=.97,fontsize=14)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.05),ncol=3,frameon=False,fontsize=8)
    ax.set_ylim(0,max(1.4,ax.get_ylim()[1]+.12))
    fig.text(.5,.025,'2 blocks/SM launch; 3 process rounds × 5 event samples; no warmup\n'
        'Operator microbenchmark only; full-wide includes XOR of both halves',
        ha='center',va='bottom',fontsize=8,color='#444444')
    for ext in ('png','svg'):fig.savefig(a.figures/f'cgbn-square-4060lp.{ext}',dpi=170)
    for r in result['cohorts']:
        n=next(v for v in r['rows'] if v['native'])
        print(r['bits'],r['tpi'],r['operation'],n['reference']['median_ms'],n['candidate']['median_ms'],round((n['time_ratio']-1)*100,2))


if __name__=='__main__':main()
