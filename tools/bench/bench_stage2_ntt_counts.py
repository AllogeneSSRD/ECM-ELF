"""Alternate optional NTT host counting on/off on the same frozen GPU binary."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('counter_checks',ROOT/'tools/test/test_stage2_ntt_counts.py')
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--save',type=Path,nargs='+',required=True)
    p.add_argument('--study',type=Path,help='Exact target metadata from the reference benchmark plan; required for large cofactors')
    p.add_argument('--b2',type=int,required=True);p.add_argument('--d',type=int,default=0)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--device',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.repeats<3:raise ValueError('At least three retained samples per setting are required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);exe=a.exe.resolve()
    report=dict(complete=False,started_utc=datetime.now(timezone.utc).isoformat(),exe_sha256=checks.sha(exe),
        script_sha256=checks.sha(__file__),helper_sha256=checks.sha(checks.__file__),device=a.device,
        b2=a.b2,d=a.d,repeats=a.repeats,warmups=0,first_invocation='retained',runs=[],comparisons=[])
    study=None
    if a.study:
        raw=(a.study/'plan.json').read_bytes();study=json.loads(raw)
        (out/'study.snapshot.json').write_bytes(raw)
        report['study_sha256']=checks.sha(out/'study.snapshot.json')
    for index,save in enumerate(a.save):
        reference=None;save=save.resolve()
        metadata=['--carrier-exponent',0]
        if study:
            case=next(c for c in study['cases'] if Path(c['save']).resolve()==save)
            assert checks.sha(save)==case['save_sha256']
            metadata+=['--exponent',case['exponent']]
            for factor in case['known']:metadata+=['--known-factor',factor]
        for rep in range(a.repeats):
            for enabled in ((0,1) if rep%2==0 else (1,0)):
                assert checks.sha(exe)==report['exe_sha256']
                row=checks.invoke(exe,save,out,f'case{index}_r{rep+1}_on{enabled}',f'stage2_ntt_counts={enabled}\n',
                    extra=metadata,b2=a.b2,d=a.d,device=a.device)
                signature=(row['leaf_hashes'],row['factors'])
                if reference is None:reference=signature
                assert reference==signature and not any(row['factors']) and bool(row['tables'])==bool(enabled)
                row.update(case=index,enabled=enabled,repeat=rep+1);report['runs'].append(row)
                (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
                print(row['name'],'engine',row['engine_seconds'][0],flush=True)
        samples=[r for r in report['runs'] if r['case']==index]
        values={on:[r['engine_seconds'][0] for r in samples if r['enabled']==on] for on in (0,1)}
        process={on:statistics.median(r['process_seconds'] for r in samples if r['enabled']==on) for on in (0,1)}
        ratios=[next(r for r in samples if r['repeat']==rep and r['enabled']==1)['engine_seconds'][0]/
                next(r for r in samples if r['repeat']==rep and r['enabled']==0)['engine_seconds'][0] for rep in range(1,a.repeats+1)]
        report['comparisons'].append(dict(save=str(save),save_sha256=checks.sha(save),
            off=dict(n=len(values[0]),median=statistics.median(values[0]),min=min(values[0]),max=max(values[0])),
            on=dict(n=len(values[1]),median=statistics.median(values[1]),min=min(values[1]),max=max(values[1])),
            median_change_percent=100*(statistics.median(values[1])/statistics.median(values[0])-1),
            off_process_median_seconds=process[0],on_process_median_seconds=process[1],
            process_median_change_percent=100*(process[1]/process[0]-1),
            paired_change_percent=[100*(r-1) for r in ratios]))
    report.update(complete=True,finished_utc=datetime.now(timezone.utc).isoformat())
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS retained paired NTT counting measurements')


if __name__=='__main__':main()
