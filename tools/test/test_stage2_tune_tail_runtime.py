"""Run adaptive full-curve tune and verify planned grid/receipt provenance on a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from stage2_tune_route_cost import route_work

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['exe','save','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--carrier',type=int,default=0)
    p.add_argument('--level',type=int,choices=range(3,11),default=3)
    p.add_argument('--ds',default='60060,120120')
    p.add_argument('--b2',help='Explicit comma separated grid; exact unless tail-samples provided')
    p.add_argument('--tail-samples',type=int,choices=range(17))
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-batches',type=int)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe,save=a.exe.resolve(),a.save.resolve();sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    identities={str(path):sha(path) for path in [exe,save,Path(__file__)]}
    ini=out/'ecm.ini';ini.write_text('verbose=false\nstage2_debug_log=false\n',encoding='utf-8')
    common=[str(exe),'--ini',str(ini),'--device',str(a.device),'--batch-mb','256','--arena-mb','6300','--owner-budget-mb','640']
    # All failures below are parser-only; no GPU benchmark or published file.
    refusals=[(['--tune','ecm','--tune-tail-samples','17'],'tune-tail-samples must be'),
        (['--tune','ecm','--tune-tail-samples','-1'],'invalid'),
        (['--tune','ntt','--tune-tail-samples','0'],'ECM tune grid/merge options'),
        (['--tune-tail-samples','0'],'tune options require'),
        (['--tune','ecm','--tune-merge',str(out/'missing.toml'),'--tune-tail-samples','0'],'independent of benchmark grid')]
    for index,(args,error) in enumerate(refusals):
        proc=subprocess.run(common+args,cwd=ROOT,capture_output=True,text=True,errors='replace',timeout=60)
        (out/f'cli_{index}.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
        assert proc.returncode and error in proc.stderr,(index,proc.stderr)
    profile=out/'profile.toml'
    args=common+['--tune','ecm','--tune-save',str(save),'--tune-level',str(a.level),
        '--tune-d',a.ds,'--tune-repeats',str(a.repeats),'--tune-file',str(profile)]
    if a.carrier:args+=['--tune-carrier-exponent',str(a.carrier)]
    if a.b2:args+=['--tune-b2',a.b2]
    if a.tail_samples is not None:args+=['--tune-tail-samples',str(a.tail_samples)]
    if a.max_batches is not None:args+=['--tune-max-batches',str(a.max_batches)]
    state=dict(complete=False,command=args,identities=identities)
    publish=lambda:(out/'state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    publish();rows=[]
    try:
        with (out/'tune.console.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
            state['child_pid']=proc.pid;publish()
            for row in proc.stdout:log.write(row);log.flush();print(row,end='',flush=True);rows.append(row)
            code=proc.wait()
        assert code==0,('tune',code)
        text=''.join(rows);evidence=Path(text.split(' evidence=')[-1].splitlines()[0].strip())
        data=tomllib.loads(profile.read_text(encoding='utf-8-sig'))
        expected=a.tail_samples if a.tail_samples is not None else 0 if a.b2 else 3
        assert data['profile']['sampling_model']=='giant_tail_grid_v1' and data['profile']['tail_samples']==expected
        ready=re.search(r'ecm_tune_grid_ready: base=(\d+) ladder_tail=(\d+) chain_anchor=(\d+) cases=(\d+)',text)
        assert ready
        base,tails,chains,total=map(int,ready.groups());assert total==base+tails+chains
        assert data['summary']['measured']+data['summary']['skipped']==total
        assert expected or (tails==chains==0)
        if expected and not a.b2:assert tails>0
        counts=dict(scopes=0,formal=0,warmup=0,selftest_cases=0,checked=0)
        samples=list(data['ecm'].values());receipts={}
        for path in evidence.glob('case_*_*.jsonl'):
            row=json.loads(path.read_text(encoding='utf-8-sig'));parts=path.stem.split('_');number,repeat=int(parts[1]),int(parts[2])
            assert row['clean']==row['fold_resident']==row['frontier_resident']==1 and row['hits']==row['bad']==0
            assert row['selftest_cases'] and row['checked'] and row['phase_accounting']=='exclusive_engine_v1'
            counts['selftest_cases']+=row['selftest_cases'];counts['checked']+=row['checked']
            counts['formal' if repeat else 'warmup']+=1
            receipts.setdefault(number,[]).append(row)
        groups={};seen=set()
        for s in samples:
            key=(s['target_bits'],s['carrier_exponent'],s['d']);groups.setdefault(key,[]).append(s)
            scope=(*key,s['b2']);assert scope not in seen;seen.add(scope)
            work=route_work(s['giant_points'],s['d'],s['giant_chunk_points'],s['giant_chain_min'],bool(s['giant_force_ladder']))
            assert all(s[k]==v for k,v in work.items())
            if s['sampling_source']!='base':assert bool(work['giant_ladder_steps'])==(s['sampling_source']=='ladder_tail')
        for number,rows in receipts.items():
            assert len(rows)==a.repeats+1
            plan=json.loads((evidence/f'case_{number}.plan.jsonl').read_text(encoding='utf-8-sig'))
            matches=[s for s in samples if s['d']==plan['D'] and s['b2']==plan['B2'] and s['carrier_exponent']==plan['carrier_exponent']]
            assert len(matches)==1 and all(r['d']==plan['D'] and r['giant_points']==plan['I'] for r in rows)
        counts['scopes']=len(samples)
        assert counts['scopes']==len(receipts)==data['summary']['measured']
        assert counts['formal']==counts['scopes']*a.repeats and counts['warmup']==counts['scopes']
        assert all(sha(Path(path))==value for path,value in identities.items())
        state.update(complete=True,evidence=str(evidence),profile=str(profile),counts=counts,
            planned=dict(base=base,ladder_tail=tails,chain_anchor=chains,total=total),
            groups=[dict(target_bits=k[0],carrier=k[1],d=k[2],scopes=len(v),
                sources={source:sum(s['sampling_source']==source for s in v) for source in ['base','ladder_tail','chain_anchor']}) for k,v in groups.items()])
        publish();print('tail_runtime_complete:',json.dumps(state['counts']),flush=True)
    except Exception as error:
        state['failure']=repr(error);publish();raise


if __name__=='__main__':main()
