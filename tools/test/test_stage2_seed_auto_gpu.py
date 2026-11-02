"""Useful curves establish seed auto, repeat thresholds and read-only protection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from bench_stage2_point_matrix import command, environment
from bench_stage2_point_routes import sha
from bench_stage2_version import check_gpu_processes, load, write
from bench_stage2_runtime_matrix import completed_records
from gpu_telemetry import TelemetryRecorder
from stage2_tune_tables import summaries
from result_format import normalize_result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('study','exe','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--reduce-backend',choices=('auto','scalar','coop32'),default='scalar')
    a=p.parse_args();source=load(a.study/'plan.json');case=next(c for c in source['cases'] if c['bits']>8192)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    import shutil
    (out/'bin').mkdir();(out/'bin/.gitignore').write_text('*\n')
    for name in ('ecm_cuda_stage2.exe','gmp-10.dll','build_manifest.json'):shutil.copy2(a.exe.parent/name,out/'bin'/name)
    shutil.copy2(__file__,out/'test_snapshot.py')
    report=dict(complete=False,exe_sha256=sha(out/'bin/ecm_cuda_stage2.exe'),calls=[],failures=[])
    def run(name,profile,curves,minimum,backend,learn,carrier=0,all_auto=False,chain='32'):
        folder=out/name;folder.mkdir();cell=dict(B2=300750450,D=30030,P=2880,carrier=carrier,layout=1)
        cmd=command(out,folder,case,cell,profile,chain,False)
        point=(folder/'points.save').read_text();(folder/'points.save').write_text(point*curves)
        for flag,value in (('--curves',curves),('--tune-min-repeats',minimum),('--tune-runtime',int(learn))):cmd[cmd.index(flag)+1]=str(value)
        cmd+=['--reduce-backend',a.reduce_backend]
        cmd+=['--baby-backend','auto' if all_auto else 'coop32','--giant-backend','auto' if all_auto else 'coop32','--seed-backend',backend]
        m=TelemetryRecorder(1,folder/'telemetry.jsonl',interval=.2)
        try:
            check_gpu_processes(m.monitor,out,folder/'before.json',allowed=source['allowed_processes'])
            with (folder/'console.log').open('wb') as log:r=subprocess.run(cmd,cwd=folder,env=environment(),stdout=log,stderr=subprocess.STDOUT,timeout=600)
            check_gpu_processes(m.monitor,out,folder/'after.json',allowed=source['allowed_processes'])
        finally:m.close()
        write(folder/'command.json',cmd)
        if r.returncode:raise ValueError('Curve process failed: '+str(folder))
        records=completed_records(folder/'result.jsonl')
        if len(records)!=curves or any(r['status']!='NF' or r['source-build']!=report['exe_sha256'] for r in records):raise ValueError('Incomplete NF curves')
        for r in records:
            value=normalize_result(r)
            if value['N_hex']!=case['N_hex'] or value['B1']!=case['b1'] or value['B2']!=cell['B2'] or \
               value['sigma']!=case['sigma'] or value['carrier_exponent']!=carrier or r['d']!=cell['D']:
                raise ValueError('Input or geometry changed')
        seeds=[r['stage2-engine']['seed_coop'] for r in records]
        report['calls'].append(dict(name=name,seeds=seeds,folder=str(folder)))
        write(out/'summary.json',report);print('seed_auto:',name,seeds,flush=True)
        return records
    try:
        profile=out/'paired.toml';records=run('minimum2',profile,8,2,'auto',True)
        seeds=[r['stage2-engine']['seed_coop'] for r in records]
        if seeds[:4]!=[0,0,32,32]:raise ValueError('Minimum repeats not respected: '+str(seeds))
        rows=[r for r in summaries(tomllib.loads(profile.read_text(encoding='utf-8-sig'))) if r['source']=='measured']
        best=min(rows,key=lambda r:r['mean_seconds']+r.get('point_check_seconds',0))
        if seeds[4:]!=[best['seed_coop']]*4:raise ValueError('Fastest qualified seed not retained')
        before=sha(profile);read=run('readonly',profile,1,2,'auto',False)
        if sha(profile)!=before or read[0]['stage2-engine']['seed_coop']!=best['seed_coop']:raise ValueError('Read-only reuse failed')
        lone=out/'unpaired.toml';run('forced',lone,1,1,'coop32',True)
        before=sha(lone);read=run('unpaired_readonly',lone,1,1,'auto',False)
        if sha(lone)!=before or read[0]['stage2-engine']['seed_coop']!=0:raise ValueError('Forced alone qualified auto')
        run('minimum1_carrier',out/'carrier.toml',4,1,'auto',True,case['exponent'])
        all_profile=out/'all_auto.toml'
        all_auto=run('all_auto',all_profile,12,1,'auto',True,all_auto=True,chain='auto')
        if not any(r['stage2-engine']['seed_coop']==32 for r in all_auto):raise ValueError('Other automatic backends prevented bounded seed trial')
        all_rows=[r for r in summaries(tomllib.loads(all_profile.read_text(encoding='utf-8-sig'))) if r['source']=='measured']
        best=min(all_rows,key=lambda r:r['mean_seconds']+r.get('point_check_seconds',0))
        before=sha(all_profile)
        frozen=run('all_readonly',all_profile,1,1,'auto',False,all_auto=True,chain='auto')[0]['stage2-engine']
        if sha(all_profile)!=before or (frozen['baby_coop'],frozen['giant_coop'],frozen['seed_coop'],frozen['giant_chain_block'])!=\
            tuple(best[k] for k in ('baby_coop','giant_coop','seed_coop','giant_chain_block')):
            raise ValueError('Read-only lost the qualified chain/backend winner')
        report['complete']=True
    except Exception as e:report['failures'].append(str(e));raise
    finally:write(out/'summary.json',report)


if __name__=='__main__':main()
