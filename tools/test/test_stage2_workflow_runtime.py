"""Compare actual Stage2 queue completion and failure boundaries in isolated folders."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
from test_result_output_consistency import crc_valid

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','exe','save','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',type=int,default=1);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    lines=[v for v in a.save.read_text(encoding='utf-8-sig').splitlines() if v.startswith('METHOD=')][:2];assert len(lines)==2
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')};snapshots={};checks=[]
    for label,exe in {'before':a.baseline.resolve(),'after':a.exe.resolve()}.items():
        snapshots[label]={}
        for case in ['complete','dry','bad_input','invalid_carrier','changed_save','publication_failure']:
            work=out/label/case;work.mkdir(parents=True);saved=work/'points.save';saved.write_text('\n'.join(lines)+'\n')
            queue=work/'worktodo.txt';row='ECMSTAGE2=1,2,127,-1,points.save,40000,0,2';queue.write_text(row+'\n')
            internal=work/'internal.jsonl';public=work/'results.json.txt';finished=work/'finished.txt'
            config=work/'ecm.toml';config.write_text('[shared]\nuser=""\ncomputer=""\n[stage1]\nverbose=true\n[stage2]\n'+
                f'stage2_save_dir={json.dumps(str(work))}\nstage2_finished={json.dumps(str(finished))}\n'+
                f'stage2_progress_dir={json.dumps(str(work/"progress"))}\n')
            args=[exe,'--config',config,'--device',a.device,'--once','--worktodo',queue,'--results',internal,'--submission',public,
                '--short-calibration','0','--tune-runtime','0','--auto-carrier','0','--d','210','--log',work/'engine.log']
            if case=='dry':args+=['--dry-run']
            if case=='bad_input':saved.write_text(re.sub(r'CHECKSUM=\d+','CHECKSUM=0',saved.read_text()))
            if case=='invalid_carrier':args+=['--carrier-exponent','128']
            if case=='changed_save':args+=['--curve-worker','--save',saved,'--record-index','1','--record-offset','0','--record-hash','0','--b2','40000']
            if case=='publication_failure':internal.mkdir()
            command=list(map(str,args));(work/'command.json').write_text(json.dumps(command,indent=2)+'\n')
            r=subprocess.run(command,cwd=work,env=env,capture_output=True,text=True,errors='replace',timeout=120)
            log=r.stdout+r.stderr;(work/'console.log').write_text(log,encoding='utf8')
            if case=='complete':
                assert r.returncode==0,log[-2000:]
                rows=[json.loads(v) for v in internal.read_text().splitlines()];submitted=[json.loads(v) for v in public.read_text().splitlines()]
                assert len(rows)==2 and [v['stage2-engine']['record'] for v in rows]==[1,2]
                assert len({v['queue_receipt'] for v in rows})==2 and all(v['queue_receipt'] for v in rows)
                assert len(submitted)==1 and submitted[0]['status']=='NF' and submitted[0]['curves']==2
                assert all(map(crc_valid,rows+submitted)) and not queue.read_text().strip()
                assert row in finished.read_text() and not list((work/'progress').iterdir())
                snapshots[label][case]=dict(code=0,receipts=2,public_batches=1,curves=2,crc=True)
            elif case=='bad_input':
                assert r.returncode==1 and 'save CHECKSUM mismatch' in log,log[-2000:]
                assert queue.read_text().lstrip().startswith('#') and row in queue.read_text()
                assert '# ERROR reason=' in finished.read_text() and not internal.exists() and not public.exists()
                snapshots[label][case]=dict(code=1,marked=True,reason_durable=True)
            else:
                expected=0 if case=='dry' else 2
                assert r.returncode==expected,(case,r.returncode,log[-2500:])
                assert queue.read_text()==row+'\n' and not finished.exists() and not public.exists()
                if case!='publication_failure':assert not internal.exists()
                if case=='invalid_carrier':assert 'carrier' in log.lower() or 'divid' in log.lower(),log
                if case=='changed_save':assert 'save changed after planning' in log,log
                if case=='publication_failure':
                    assert 'curve failed; queue and completed progress retained' in log,log[-2000:]
                    states=list((work/'progress').glob('.ecm_ckpt_stage2_*.dat'));assert len(states)==1
                    state=states[0].read_text();assert re.match(r'^2 0 2 "[^"]+" "[^"]+" "[0-9a-f]{64}" "[^"]+"\s*$',state),state
                snapshots[label][case]=dict(code=expected,queue_retained=True,no_finished=True,no_publication=True)
            checks.append(label+'_'+case)
    assert snapshots['before']==snapshots['after']
    report=dict(complete=True,checks=checks,semantic_snapshots=snapshots,performance_samples=0,
        binaries={label:hashlib.sha256(exe.read_bytes()).hexdigest() for label,exe in {'before':a.baseline,'after':a.exe}.items()})
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
