"""Run real CPU Stage1 tasks through local and routed automatic Stage2 handoff."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();exe=args.exe.resolve();out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith(('ECM_','NTT_'))}
    cases=[]
    def run(name,settings='',tail='',worker=2,seed='',task=None,good=True):
        work=out/name;work.mkdir()
        task=task or 'ECMSTAGE1=1,2,127,-1,"e_m127_100.save",2000,0,1'
        queue=work/'worktodo.txt'
        queue.write_text(f'# preserved\n[Worker #{worker}]\n{task}\n'+seed,encoding='utf-8')
        config=work/'ecm.toml'
        base='[shared]\nworktodo = "worktodo.txt"\n[stage1]\ncurve = "ed"\ntarget = "cpu"\nimpl = "scalar"\nsigma = 20260922\nlog_file = ""\nckpt_seconds = 0\n'
        base+='finished = '+json.dumps(str(work/'finished.txt'))+'\ntmp_dir = '+json.dumps(str(work/'saves'))+'\n'
        config.write_text(base+settings+'\n'+tail,encoding='utf-8')
        result=subprocess.run([str(exe),'--config',str(config),'--worker',str(worker)],
            cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (work/'run.log').write_bytes(result.stdout)
        assert (result.returncode==0)==good,(name,result.returncode,result.stdout[-1500:])
        cases.append(name)
        return work,queue,config,task,result.stdout.decode('utf-8',errors='replace')
    work,queue,_,task,_=run('in_place','auto_stage2 = 1',seed='[Worker #3]\n# receiver empty\n')
    text=queue.read_text();assert text.index('ECMSTAGE2=')<text.index('[Worker #3]')
    assert task in (work/'finished.txt').read_text() and '# preserved' in text
    assert not list(work.glob('*.stage2_pending_worker*'))

    work,queue,_,task,log=run('worker_override','auto_stage2 = 0\nauto_stage2_workers = "1"',
        '[workers.2.stage1]\nauto_stage2 = 1\nauto_stage2_workers = "3"',seed='[Worker #3]\n')
    assert task not in queue.read_text() and 'ECMSTAGE2=' in queue.read_text().split('[Worker #3]')[-1]
    assert 'stage2_transfer: ok worker=3' in log

    seed='[Worker #1]\nECMSTAGE2=1,2,127,-1,"existing.save",2000,0,1\n[Worker #3]\n'
    work,queue,_,_,log=run('shortest','auto_stage2 = 1\nauto_stage2_workers = "3,1"',seed=seed)
    assert 'stage2_transfer: ok worker=3' in log

    work,queue,_,task,log=run('external_same_worker',
        'auto_stage2 = 1\nauto_stage2_worktodo = "other_worktodo.txt"')
    dest=(work/'other_worktodo.txt').read_text()
    assert '[Worker #2]' in dest and 'ECMSTAGE2=' in dest and task not in queue.read_text()
    delivered=next(line for line in dest.splitlines() if line.startswith('ECMSTAGE2='))
    save=Path(next(csv.reader([delivered.split('=',1)[1]]))[4])
    assert save.is_absolute() and save==work/'e_m127_100.save' and save.is_file()

    absolute=out/'absolute_target.txt'
    work,queue,_,_,log=run('external_range','auto_stage2 = 1\nauto_stage2_workers = "5-4"\nauto_stage2_worktodo = '+json.dumps(str(absolute)))
    assert '[Worker #4]' in absolute.read_text() and 'stage2_transfer: ok worker=4' in log

    work,queue,_,_,log=run('auto_existing','auto_stage2 = 1\nauto_stage2_workers = "auto"',
        seed='[Worker #7]\n[Worker #3]\n')
    assert 'stage2_transfer: ok worker=3' in log

    work,queue,_,_,log=run('disabled','auto_stage2 = 0\nauto_stage2_workers = "3"\nauto_stage2_worktodo = "unused.txt"')
    assert 'ECMSTAGE2=' not in queue.read_text() and not (work/'unused.txt').exists()

    for spec in ('0','wrong','1000001'):
        work,queue,_,task,_=run('invalid_'+spec,'auto_stage2 = 1\nauto_stage2_workers = '+json.dumps(spec),good=False)
        assert task in queue.read_text() and not (work/'finished.txt').exists()

    # Target publication fails after Stage1 succeeds: outbox survives and startup
    # replay works even with auto_stage2 disabled, without rerunning Stage1.
    blocked=out/'blocked_target';blocked.mkdir()
    work,queue,config,task,log=run('pending_retry','auto_stage2 = 1\nauto_stage2_workers = "3"\nauto_stage2_worktodo = '+json.dumps(str(blocked)))
    pending=work/'worktodo.txt.stage2_pending_worker2'
    assert pending.read_text().strip() and 'stage2_transfer: pending' in log and task not in queue.read_text()
    saved=next(work.glob('e_*.save'));before=saved.read_bytes()
    blocked.rmdir()
    config.write_text('[shared]\nworktodo="worktodo.txt"\n[stage1]\nlog_file=""\nauto_stage2=0\n',encoding='utf-8')
    retry=subprocess.run([str(exe),'--config',str(config),'--worker','2'],cwd=work,env=env,capture_output=True,timeout=90)
    assert retry.returncode==0 and not pending.read_text().strip() and '[Worker #3]' in blocked.read_text()
    assert saved.read_bytes()==before

    work,queue,_,task,log=run('factor_no_handoff','auto_stage2 = 1\nauto_stage2_workers = "3"\nauto_stage2_worktodo = "must_not_exist.txt"',
        task='ECMSTAGE1=1,2,131,-1,"e_m131_1000.save",20000,0,1')
    assert 'FACTOR FOUND' in log and task not in queue.read_text() and not (work/'must_not_exist.txt').exists()
    assert not list(work.glob('*.stage2_pending_worker*'))

    work,queue,_,_,log=run('native_ecm2','auto_stage2 = 1\nauto_stage2_workers = "3"',
        task='ECM2=1,2,127,-1,100,2000,1,20260922')
    assert 'stage2_transfer: ok worker=3' in log and 'ECMSTAGE2=' in queue.read_text()
    report={'passed':True,'cases':cases,'count':len(cases)}
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
