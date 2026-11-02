"""Exercise seed-policy CLI/TOML/environment precedence through a real queue child."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from bench_stage2_point_matrix import command, environment
from bench_stage2_seed_coop import sha
from bench_stage2_version import load, write, check_gpu_processes
from bench_stage2_n_scaling import fields
from bench_stage2_runtime_matrix import completed_records
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result
from stage2_tune_tables import summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=load(a.study/'plan.json')
    case=next(c for c in source['cases'] if c['bits']>8192)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'bin').mkdir();(out/'bin/.gitignore').write_text('*\n')
    for name in ('ecm_cuda_stage2.exe','gmp-10.dll'):shutil.copy2(a.exe.parent/name,out/'bin'/name)
    shutil.copy2(__file__,out/'test_snapshot.py')
    identity=sha(out/'bin/ecm_cuda_stage2.exe')
    report=dict(complete=False,exe_sha256=identity,calls=[])
    settings=(('worker','scalar','coop32','0',[],32),
              ('cli','scalar','coop32','32',['--seed-backend','scalar'],0),
              ('environment',None,None,'32',[],32),
              ('invalid','auto',None,'0',[],None))
    for name,global_mode,worker_mode,override,extra,expected in settings:
        folder=out/name;folder.mkdir()
        cell=dict(B2=300750450,D=30030,P=2880,carrier=0,layout=1)
        cmd=command(out,folder,case,cell,out/'profile.toml','32',False)
        cmd[cmd.index('--tune-runtime')+1]='1'
        save=folder/'points.save';queue=folder/'worktodo.txt'
        queue.write_text('[Worker #2]\nECMSTAGE2=1,2,'+str(case['exponent'])+',-1,'+
            json.dumps(str(save))+','+str(cell['B2'])+',0,1,'+json.dumps(','.join(case['known']))+'\n')
        for flag in ('--save','--curves'):
            at=cmd.index(flag);del cmd[at:at+2]
        cmd+=['--worktodo',str(queue),'--worker','2','--once','--baby-backend','coop32',
              '--giant-backend','coop32',*extra]
        config=folder/'ecm.toml'
        config.write_text('[stage2]\nstage2_short_calibration=false\n'+
            (('stage2_seed_backend='+json.dumps(global_mode)+'\n') if global_mode else '')+
            (('[workers.2.stage2]\nstage2_seed_backend='+json.dumps(worker_mode)+'\n') if worker_mode else ''))
        env=environment();env['NTT_GIANT_SEED_COOP']=override
        monitor=TelemetryRecorder(1,folder/'telemetry.jsonl',interval=.2)
        try:
            check_gpu_processes(monitor.monitor,out,folder/'before.json',allowed=source['allowed_processes'])
            result=subprocess.run(cmd,cwd=folder,env=env,capture_output=True,timeout=120)
            text=(result.stdout+result.stderr).decode('utf-8',errors='replace')
            (folder/'console.log').write_text(text,encoding='utf-8');write(folder/'command.json',cmd)
            check_gpu_processes(monitor.monitor,out,folder/'after.json',allowed=source['allowed_processes'])
        finally:monitor.close()
        if expected is None:
            if result.returncode==0 or 'stage2_seed_backend' not in text:raise ValueError('Invalid enum accepted')
        else:
            if result.returncode:raise ValueError('Worker failed: '+str(folder))
            records=completed_records(folder/'result.jsonl')
            if len(records)!=1:raise ValueError('Queue child did not complete exactly one curve')
            r=records[0];n=normalize_result(r)
            if r['status']!='NF' or r['source-build']!=identity or n['N_hex']!=case['N_hex'] or n['factors'] or n['bad_factors']:
                raise ValueError('Queue identity or arithmetic changed')
            pair=fields((folder/'debug.log').read_text(encoding='utf-8'),'real_giant_seed_pair')
            if {int(pair[k]) for k in ('seed_backend','base_backend')}!={expected}:raise ValueError('Policy lost in child')
        report['calls'].append(dict(name=name,returncode=result.returncode,backend=expected))
        write(out/'summary.json',report);print('PASS seed_config:',name,flush=True)
    if sha(a.exe)!=identity:raise ValueError('Binary changed')
    profile=tomllib.loads((out/'profile.toml').read_text(encoding='utf-8-sig'))
    counts={0:0,32:0}
    for row in summaries(profile):
        if row['source']=='measured':
            policy=int(profile['condition'][str(row['condition'])].get('env_giant_seed_coop',0))
            counts[policy]+=row['repeats']
    if counts!={0:1,32:2}:raise ValueError('ECM observations crossed seed policies: '+str(counts))
    report['tune_repeats_by_seed_policy']=counts
    report['complete']=True;write(out/'summary.json',report)


if __name__=='__main__':main()
