"""Run real CPU Stage1 tasks through local and routed automatic Stage2 handoff."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage2',type=Path)
    parser.add_argument('--device',type=int,default=1)
    args=parser.parse_args();exe=args.exe.resolve();out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith(('ECM_','NTT_'))}
    cases=[]
    def run(name,settings='',tail='',worker=2,seed='',task=None,good=True,v3=False):
        work=out/name;work.mkdir()
        worker_exe=work/exe.name
        shutil.copy2(exe,worker_exe)
        runtime=exe.with_name('gmp-10.dll')
        if runtime.is_file():
            shutil.copy2(runtime,work/runtime.name)
        task=task or 'ECMSTAGE1=1,2,127,-1,"e_m127_100.save",2000,0,1'
        queue=work/'worktodo.txt'
        queue.write_text(f'# preserved\n[Worker #{worker}]\n{task}\n'+seed,encoding='utf-8')
        config=work/'ecm.toml'
        base='[shared]\nworktodo = "worktodo.txt"\n[stage1]\ncurve = "ed"\ntarget = "cpu"\nimpl = "scalar"\nsigma = 20260922\nlog_file = ""\nckpt_seconds = 0\n'
        base+='finished = '+json.dumps(str(work/'finished.txt'))+'\ntmp_dir = '+json.dumps(str(work/'saves'))+'\n'
        if v3:
            base=('config_version = 3\n[shared]\nworktodo_file = "worktodo.txt"\n'
                  'finished_file = "finished.txt"\nlog_file = ""\n[stage1]\n'
                  'target = "cpu"\ncpu_implementation = "scalar"\nsigma = 20260922\n'
                  'checkpoint_interval_seconds = 0\nsave_dir = "."\n')
        config.write_text(base+settings+'\n'+tail,encoding='utf-8')
        result=subprocess.run([str(worker_exe),'--config',str(config),'--worker',str(worker)],
            cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (work/'run.log').write_bytes(result.stdout)
        assert (result.returncode==0)==good,(name,result.returncode,result.stdout[-1500:])
        cases.append(name)
        return work,queue,config,task,result.stdout.decode('utf-8',errors='replace')
    work,queue,_,task,_=run('in_place','auto_stage2 = 1',seed='[Worker #3]\n# receiver empty\n')
    text=queue.read_text();assert text.index('ECMSTAGE2=')<text.index('[Worker #3]')
    completed=task.replace('"e_m127_100.save"',f'"{work / "e_m127_100.save"}"')
    assert completed in (work/'finished.txt').read_text() and '# preserved' in text
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
    retry=subprocess.run([str(work/exe.name),'--config',str(config),'--worker','2'],cwd=work,env=env,capture_output=True,timeout=90)
    assert retry.returncode==0 and not pending.read_text().strip() and '[Worker #3]' in blocked.read_text()
    assert saved.read_bytes()==before

    work,queue,_,task,log=run('factor_no_handoff','auto_stage2 = 1\nauto_stage2_workers = "3"\nauto_stage2_worktodo = "must_not_exist.txt"',
        task='ECMSTAGE1=1,2,131,-1,"e_m131_1000.save",20000,0,1')
    assert 'FACTOR FOUND' in log and task not in queue.read_text() and not (work/'must_not_exist.txt').exists()
    assert not list(work.glob('*.stage2_pending_worker*'))

    work,queue,_,_,log=run('native_ecm2','auto_stage2 = 1\nauto_stage2_workers = "3"',
        task='ECM2=1,2,127,-1,100,2000,1,20260922')
    assert 'stage2_transfer: ok worker=3' in log and 'ECMSTAGE2=' in queue.read_text()

    waiting='ECMSTAGE2N="earlier.save",2000\n'
    seed=waiting+'[Worker #3]\n# independent\n[Worker #2]\n# tail\n[Worker #4]\n'
    ed,queue,_,task,log=run('v3_default_ed',v3=True,seed=seed,task='ECM=N/A,1,2,127,-1,100,2000,1')
    assert 'Stage1 execution: curve=ed ' in log and 'invalid submission assignment id' not in log
    text=queue.read_text()
    delivered,=[row for row in text.splitlines() if row.startswith('ECMSTAGE2=')]
    assert waiting in text and text.index('# tail')<text.index(delivered)<text.index('[Worker #4]')
    fields=next(csv.reader([delivered.split('=',1)[1]]))
    assert fields[:4]==['1','2','127','-1'] and fields[5:]==['2000','0','1']
    save=Path(fields[4]);assert save.is_absolute() and save.name.startswith('e_') and save.is_file()
    assert '# ERROR' not in text and task not in text

    aid='01234567890123456789012345678901'
    factor=str((1<<127)+1)
    work,queue,_,_,log=run('v3_native_ecm2',v3=True,
        task=f'ECM2={aid},1,2,254,-1,100,2000,1,"{factor}"',
        settings='curve = ""\nhandoff.workers = "3"',seed='[Worker #3]\n')
    assert 'Stage1 execution: curve=ed ' in log and 'stage2_transfer: ok worker=3' in log
    delivered=queue.read_text().split('[Worker #3]\n')[1].strip()
    assert delivered.startswith(f'ECMSTAGE2={aid},1,2,254,-1,') and delivered.endswith(f',2000,0,1,"{factor}"')

    _,queue,_,_,log=run('v3_curve_override',v3=True,settings='curve = "mont"\nhandoff.mode = "off"',
        task='ECM=,1,2,127,-1,100,2000,1')
    assert 'Stage1 execution: curve=mont param=m0 ' in log and 'ECMSTAGE2=' not in queue.read_text()
    _,queue,_,_,log=run('v3_untyped_stage1',v3=True,
        task='ECMSTAGE1=1,2,127,-1,"points_100.save",2000,0,1')
    assert 'Stage1 execution: curve=ed ' in log and 'e_points_100.save' in queue.read_text()
    _,queue,_,_,log=run('v3_typed_stage1',v3=True,
        task='ECMSTAGE1=1,2,127,-1,"m0_m127_100.save",2000,0,1')
    assert 'Stage1 execution: curve=mont param=m0 ' in log and 'ECMSTAGE2=' in queue.read_text()

    if args.stage2:
        worker_exe=args.stage2.resolve(strict=True)
        case=out/'stage2_from_save';case.mkdir()
        cfg=case/'ecm.toml'
        cfg.write_text('config_version=3\n[shared]\nworktodo_file="worktodo.txt"\n'
                       'finished_file="finished.txt"\nlog_file=""\n[stage2]\nb2=2000\n',
                       encoding='utf-8')
        path=case/'worktodo.txt'
        joined=case/'points,copy.save'
        saved=save.read_text(encoding='utf-8')
        joined.write_text('METHOD=ECM; SIGMA=26; B1=100; N=91; X=2;\n'+saved,encoding='utf-8')
        prefix=f'ECMSTAGE2N={aid},"{joined.as_posix()}"'
        cmd=[str(worker_exe),'--config',str(cfg),'--device',str(args.device),'--once',
             '--tune-runtime','0','--short-calibration','0','--log-level','curve',
             '--results',str(case/'internal.jsonl'),'--submission',str(case/'submission.txt')]
        def stage2(name,row,*extra):
            path.write_text(row+'\n',encoding='utf-8')
            result=subprocess.run(cmd+list(extra),cwd=case,env=env,capture_output=True,timeout=90)
            log=(result.stdout+result.stderr).decode('utf-8',errors='replace')
            (case/(name+'.log')).write_text(log,encoding='utf-8')
            cases.append(name)
            return result.returncode,log
        code,log=stage2('mixed_N_rejected',prefix+',0,0,2','--dry-run')
        assert code and 'selected save records must have the same N' in log
        code,log=stage2('saved_N_selected_range',prefix+f',0,1,1,"{factor}"')
        assert code==0 and 'completed=1' in log and not path.read_text().strip(),log
        report,=[json.loads(line) for line in (case/'submission.txt').read_text().splitlines()]
        assert int(report['N'])==(1<<127)-1 and report['known-factors']==[factor] and report['aid']==aid
        code,log=stage2('saved_N_optional_count',f'ECMSTAGE2N=N/A,"{save.as_posix()}"','--dry-run')
        assert code==0 and 'requested=all available=1' in log,log

    report={'passed':True,'cases':cases,'count':len(cases)}
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
