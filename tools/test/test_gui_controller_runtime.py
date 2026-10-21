"""Drive the production Controller headlessly with fixtures and real CUDA workers."""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
from test_result_output_consistency import crc_valid
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['controller','stage1','stage2','save','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',type=int,default=1);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binaries={k:getattr(a,k).resolve() for k in ['controller','stage1','stage2']}
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    identity={k:sha(v) for k,v in binaries.items()}
    def run(name,args,work):
        command=list(map(str,args));(out/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
        env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_EDWARDS_'))}
        r=subprocess.run(command,cwd=work,env=env,capture_output=True,timeout=90,creationflags=subprocess.CREATE_NO_WINDOW)
        (out/(name+'.log')).write_bytes(r.stdout+r.stderr)
        assert r.returncode==0,(name,(r.stdout+r.stderr)[-3500:])
        results=[json.loads(v) for v in r.stdout.decode('utf8',errors='replace').splitlines() if v.startswith('{"passed":')]
        assert len(results)==1 and results[0]['passed'];return results[0]
    fixtures=run('fixtures',[binaries['controller'],out/'fixtures'],out)
    assert fixtures['scenarios']==10 and fixtures['checks']==44 and fixtures['gpu_devices']==0
    work=out/'actual';work.mkdir();paths={}
    for name in ['stage1','stage2']:
        directory=work/name;directory.mkdir();exe=binaries[name]
        for filename in [exe.name,'gmp-10.dll']:shutil.copy2(exe.parent/filename,directory/filename)
        paths[name]=directory/exe.name
    saved=work/'points.save';saved.write_bytes(a.save.read_bytes())
    queue=work/'worktodo.txt';queue.write_text('ECM2=1,2,127,-1,20,0,2,26\nECMSTAGE2=1,2,127,-1,points.save,40000,0,2\n')
    config=work/'ecm.toml'
    q=lambda p:json.dumps(str(p))
    config.write_text('[shared]\nuser=""\ncomputer=""\nworktodo='+q(queue)+'\n[stage1]\ncurve="mont"\ntarget="gpu"\n'
        f'device={a.device}\ngpu_param=0\nauto_stage2=0\nckpt_seconds=0\nexp_cache="off"\n'
        'finished='+q(work/'finished_stage1.txt')+'\ntmp_dir='+q(work)+'\nlog_file='+q(work/'stage1.log')+'\n'
        'stage1_results_file='+q(work/'stage1.jsonl')+'\nstage1_submission_file='+q(work/'public.json.txt')+'\n'
        '[stage2]\nstage2_device='+str(a.device)+'\nstage2_d=210\nstage2_short_calibration=false\nstage2_tune_runtime=false\n'
        'stage2_auto_carrier=false\nstage2_save_dir='+q(work)+'\nstage2_finished='+q(work/'finished_stage2.txt')+'\n'
        'stage2_progress_dir='+q(work/'progress')+'\nstage2_log_file='+q(work/'stage2.log')+'\nstage2_results_file='+q(work/'stage2.jsonl')+'\n'
        'stage2_submission_file='+q(work/'public.json.txt')+'\n[workers.1]\nautostart=0\ngpucurves=2\n',encoding='utf8')
    actual=run('actual',[binaries['controller'],'--real',config,paths['stage1'],paths['stage2']],work)
    assert actual['actual_workers']==2 and actual['queue_empty'] and actual['stage1_to_stage2']
    assert not queue.read_text().strip()
    rows={name:[json.loads(v) for v in (work/(name+'.jsonl')).read_text().splitlines()] for name in ['stage1','stage2']}
    assert all(len(v)==2 and all(map(crc_valid,v)) for v in rows.values())
    public=[json.loads(v) for v in (work/'public.json.txt').read_text().splitlines()]
    assert len(public)==2 and all(v['status']=='NF' and v['curves']==2 and crc_valid(v) for v in public)
    assert all(sha(v)==identity[k] for k,v in binaries.items())
    report=dict(complete=True,fixture=fixtures,actual=actual,binary_sha256=identity,device=a.device,
        actual_internal_rows={k:len(v) for k,v in rows.items()},actual_submission_batches=2,gui_window=False,performance_samples=0)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
