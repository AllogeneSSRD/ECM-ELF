"""Stage1 GP-first fallback and preservation of real CPU/CUDA factor publication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('exe','fixture','gp','output'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--target',choices=('cpu','gpu'),default='cpu');ap.add_argument('--device',type=int,default=1)
    a=ap.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    fixture=a.fixture.resolve();gp=a.gp.resolve();exe=a.exe.resolve();checks=[]
    for mode in ('real','bad_curve','missing','timeout','protocol','exit','flood','invalid','math','native','exhausted','gp_only'):
        work=root/mode;work.mkdir();env=os.environ.copy();counter=work/'counter.txt'
        env.update(ECM_TEST_GP_MODE=mode,ECM_TEST_GP_COUNTER=str(counter))
        env.pop('ECM_TEST_BAD_CURVE_FIRST',None)
        if mode=='bad_curve':env['ECM_TEST_BAD_CURVE_FIRST']='1'
        backend='native' if mode in ('native','exhausted') else 'gp' if mode=='gp_only' else 'auto'
        path=gp if mode in ('real','bad_curve') else work/'missing.exe' if mode in ('missing','gp_only') else fixture
        run=subprocess.run([str(fixture),str(path),backend,'0' if mode=='exhausted' else '10000'],env=env,capture_output=True,text=True,timeout=12)
        (work/'run.log').write_text(run.stdout+run.stderr,encoding='utf-8');assert run.returncode==0
        rows=[json.loads(line) for line in run.stdout.splitlines()];assert len(rows)==2
        if mode in ('real','bad_curve'):assert all(r['backend']=='gp' and r['status']=='complete' for r in rows),rows
        elif mode in ('invalid','math'):assert all(r['backend']=='gp' and r['status']=='invalid_input' for r in rows),rows
        elif mode in ('exhausted','gp_only'):assert all(r['status']=='resource_limit' for r in rows),rows
        else:assert all(r['backend']=='native' and r['status']=='complete' for r in rows),rows
        if rows[0]['status']=='complete':assert all(r['group']=='59649589504338456' and r['point']=='14912397376084614' for r in rows)
        calls=len(counter.read_text().splitlines()) if counter.exists() else 0
        if mode in ('timeout','protocol','exit','flood'):assert calls==1,(mode,calls)
        if mode in ('native','exhausted'):assert calls==0
        checks.append(dict(case=mode,passed=True,gp_calls=calls,rows=rows))
    baseline=None;published_baseline=None;legacy=None
    for mode in ('off','real','missing','native','timeout','protocol','exit','flood','invalid','math','gp_only','large_native_limit'):
        work=root/('cli_'+mode);work.mkdir();ini=work/'ecm.toml'
        ini.write_text('[shared]\nuser=""\ncomputer=""\n',encoding='utf-8')
        env=os.environ.copy();env['ECM_TEST_GP_MODE']=mode
        large=mode=='large_native_limit'
        p,s,b1=('228898605233001999017',1048576,5783) if large else ('59649589127497217',26,1000000)
        n=int(p)*((1<<127)-1)
        cmd=[str(exe),'--curve','mont','--target',a.target,
             '--lcm','1','-sigma',str(s),'-gpucurves','1','--config',str(ini),'--tmp-dir',str(work),'--exp-cache','off']
        if a.target=='cpu':cmd+=['--impl','scalar','--stage1-threads','1']
        else:cmd+=['--gpu-param','0','-d',str(a.device)]
        if mode!='off':
            cmd+=['--go','--go-gp-timeout','1']
            if mode in ('missing','gp_only','large_native_limit'):cmd+=['--gp',str(work/'missing.exe')]
            elif mode=='real':cmd+=['--gp',str(gp)]
            else:cmd+=['--gp',str(fixture)]
            if mode=='native':cmd+=['--go-backend','native']
            if mode=='gp_only':cmd+=['--go-backend','gp']
        cmd += [str(b1),'0']
        run=subprocess.run(cmd,input=str(n)+'\n',capture_output=True,text=True,errors='replace',cwd=work,env=env,timeout=30)
        log=run.stdout+run.stderr;(work/'run.log').write_text(log,encoding='utf-8')
        assert run.returncode==0,log
        assert re.search(r'factor\[0\]='+p+r'\b',log),log
        result=work/'stage1_results.jsonl';assert result.is_file(),list(work.iterdir())
        rows=[json.loads(line) for line in result.read_text(encoding='utf-8').splitlines()]
        assert any(p in r.get('factors',[]) for r in rows),rows
        submission=work/'results.json.txt';assert submission.is_file()
        published=[json.loads(line) for line in submission.read_text(encoding='utf-8').splitlines()]
        assert len(published)==1,published
        record=published[0]
        assert record['status']=='F' and record['factors']==[p] and record['N']==n,record
        assert record['sigma']==s and record['b1']==b1 and record['stage']==1,record
        assert record['errors']['invariant']==0,record
        # CPU writes factor saves; a CUDA hit publishes results without a save.
        # Compare the entire submission apart from timestamp and its checksum.
        stable={k:v for k,v in record.items() if k not in ('timestamp','checksum')}
        saves=[f for f in work.rglob('*') if f.is_file() and f.suffix in ('.save','.factor')]
        saved=[f.read_bytes() for f in saves]
        if a.target=='cpu':assert saved,list(work.iterdir())
        else:assert not saved,saves
        if not large:
            if published_baseline is None:published_baseline=stable
            else:assert stable==published_baseline,(mode,stable,published_baseline)
            signature=sorted(hashlib.sha256(re.sub(rb'TIME=[^;]*;',b'TIME=<per-run>;',data)).hexdigest() for data in saved)
            if baseline is None:baseline=signature
            else:assert signature==baseline,(mode,signature,baseline)
        diagnostics=[re.sub(r'^\[[^\]]+\] ','',line) for line in log.splitlines()]
        old=[line for line in diagnostics if line.startswith(('  go[','  go_factor[','  go_min_bounds['))]
        if mode=='real':legacy=old;assert 'backend=gp status=complete' in log,log
        if mode in ('missing','native','timeout','protocol','exit','flood'):
            assert 'backend=native status=complete' in log,log
            assert old==legacy,(mode,old,legacy)
            assert 'go_point_order[0]=14912397376084614' in log
        if large:assert 'backend=native status=resource_limit' in log,log
        checks.append(dict(case='cli_'+mode,passed=True,returncode=run.returncode,saved_files=[f.name for f in saves]))
        print('passed',mode,flush=True)
    summary=dict(passed=len(checks),failed=0,target=a.target,device=a.device if a.target=='gpu' else None,
                 checks=checks,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(dict(passed=len(checks),failed=0)))


if __name__=='__main__':main()
