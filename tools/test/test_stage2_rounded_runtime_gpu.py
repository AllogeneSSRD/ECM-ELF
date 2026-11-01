"""Useful-curve learning with rounded bounds and independent worker profiles.

The workers test uses one idle test GPU for both workers sequentially; it tests
file routing and learning, not two-GPU performance. No production queue is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from bench_stage2_version import check_gpu_processes,write
from bench_stage2_runtime_matrix import completed_records
from bench_stage2_n_scaling import fields
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result
from stage2_tune_tables import summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('stage2','save','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--case',choices=('wide','workers','default'),required=True)
    p.add_argument('--baseline',type=Path,help='Frozen pre-change executable for the unrounded control')
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--allow-gpu-pid',type=int,action='append',default=[])
    a=p.parse_args()
    if a.case=='default' and not a.baseline:p.error('--baseline is required for the default control')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'bin';binary.mkdir();(binary/'.gitignore').write_text('*\n')
    for name in (a.stage2.name,'gmp-10.dll','build_manifest.json'):
        f=a.stage2.resolve().parent/name
        if f.exists():shutil.copy2(f,binary/name)
    exe=binary/a.stage2.name;saved=out/'points.save';saved.write_bytes(a.save.read_bytes()*10)
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    identity=sha(exe);source_hash=sha(a.save);shutil.copy2(__file__,out/'test_snapshot.py')
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')}
    env['NTT_NO_PROGRESS']='1'
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    allowed=[]
    for pid in a.allow_gpu_pid:
        names={s.split(',',2)[2].strip() for s in processes.splitlines() if s.split(',')[1].strip()==str(pid)}
        if len(names)!=1 or not Path(next(iter(names))).is_absolute():raise ValueError('Cannot bind allowed process')
        allowed.append(dict(pid=pid,process_name=next(iter(names))))
    profile1=out/'profiles/gpu0.toml';profile2=out/'profiles/gpu1.toml';global_file=out/'profiles/global.toml'
    config=out/'ecm.toml'
    config.write_text('[stage2]\nstage2_short_calibration=false\nstage2_tune_runtime=true\n'
        'stage2_auto_carrier=false\nstage2_tune_min_repeats=2\nstage2_b2_round_chain='+str(a.case!='default').lower()+'\n'
        'stage2_giant_chain="auto"\nstage2_groot_balanced=false\n'
        'stage2_baby_backend="'+('auto' if a.case=='wide' else 'scalar')+'"\n'
        'stage2_giant_backend="'+('coop32' if a.case=='wide' else 'scalar')+'"\n'
        'stage2_tune_profile="profiles/global.toml"\n'
        '[workers.1.stage2]\nstage2_tune_profile="profiles/gpu0.toml"\n'
        '[workers.2.stage2]\nstage2_tune_profile="profiles/gpu1.toml"\n')
    report=dict(complete=False,case=a.case,exe_sha256=identity,source_sha256=source_hash,calls=[],failures=[])
    write(out/'summary.json',report);telemetry=TelemetryRecorder(a.device,out/'telemetry.jsonl',interval=.2)

    def run(name,worker,curves,d,b2,extra=()):
        directory=out/name;directory.mkdir()
        command=[str(exe),'--config',str(config),'--device',str(a.device),'--worker',str(worker),'--save',str(saved),
            '--curves',str(curves),'--b2',str(b2),'--d',str(d),'--carrier-exponent','0','--ntt-counts','1',
            '--results',str(directory/'result.jsonl'),'--submission',str(directory/'result.txt'),
            '--log',str(directory/'engine.log'),'--debug-log-file',str(directory/'debug.log'),*extra]
        write(directory/'command.json',command)
        check_gpu_processes(telemetry.monitor,out,directory/'before.json',allowed=allowed)
        with (directory/'console.log').open('wb') as log:
            result=subprocess.run(command,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        check_gpu_processes(telemetry.monitor,out,directory/'after.json',allowed=allowed)
        console=(directory/'console.log').read_text(encoding='utf-8');assert result.returncode==0,console
        records=completed_records(directory/'result.jsonl');assert len(records)==curves
        for row in records:
            normal=normalize_result(row);engine=row['stage2-engine'];plan=engine['tune_plan']
            assert row['status']=='NF' and row['source-build'].lower()==identity and not normal['factors'] and not normal['bad_factors']
            assert engine['requested_B2']==b2 and normal['B2']>=b2
            if a.case!='default':
                assert plan['B2']==normal['B2'] and plan['requested_B2']==b2
                assert plan['runtime_candidates'] and all(c['B2']>=b2 for c in plan['runtime_candidates'])
            else:assert normal['B2']==b2
        trace=(directory/'engine.log').read_text(encoding='utf-8')
        assert 'runtime tune sample skipped' not in console and 'gmp_check_bad=0' in trace and 'gmp_selftest_bad=0' in trace
        assert not global_file.exists() and sha(a.save)==source_hash and sha(exe)==identity
        modes=[(r['stage2-engine']['baby_coop'],r['stage2-engine']['giant_chain_block']) for r in records]
        report['calls'].append(dict(name=name,directory=str(directory),modes=modes,
            decisions=[r['stage2-engine']['tune_plan'] for r in records]))
        write(out/'summary.json',report);print('PASS:',name,modes,flush=True)
        return records

    try:
        if a.case=='wide':
            records=run('learning',1,8,6006,26000000)
            modes=[r['stage2-engine']['baby_coop'] for r in records]
            assert modes[:4]==[0,0,32,32],modes
            rows=[r for r in summaries(tomllib.loads(profile1.read_text())) if r['source']=='measured']
            assert sum(r['repeats'] for r in rows)==8
            assert {r['baby_coop'] for r in rows}=={0,32} and all(r['b2']>=26000000 for r in rows)
            before=sha(profile1);run('readonly',1,1,6006,26000000,('--tune-runtime','0'))
            assert sha(profile1)==before and not profile2.exists()
        elif a.case=='workers':
            first=run('worker1',1,10,0,27015978990)
            assert profile1.is_file() and not profile2.exists();before=sha(profile1)
            second=run('worker2',2,10,0,27015978990)
            assert profile2.is_file() and sha(profile1)==before
            for profile,records in ((profile1,first),(profile2,second)):
                rows=[r for r in summaries(tomllib.loads(profile.read_text())) if r['source']=='measured']
                assert sum(r['repeats'] for r in rows)==10
                assert any(r['stage2-engine']['tune_plan']['runtime_probe'] for r in records[2:])
                assert any(r['stage2-engine']['tune_plan']['runtime_search'] for r in records[2:]),'D search never resumed'
        else:
            control=out/'baseline_bin';control.mkdir();(control/'.gitignore').write_text('*\n')
            for name in (a.baseline.name,'gmp-10.dll'):shutil.copy2(a.baseline.resolve().parent/name,control/name)
            candidate_exe,candidate_identity=exe,identity
            signatures=[]
            for name in ('baseline','candidate'):
                exe=control/a.baseline.name if name=='baseline' else candidate_exe;identity=sha(exe)
                run(name,1,1,6006,26000000,('--tune-runtime','0','--b2-round-chain','0','--giant-chain','64'))
                trace=(out/name/'engine.log').read_text(encoding='utf-8')
                digest=fields(trace,'real_batched_rootfold');stats=fields(trace,'s4_multiply_stats')
                counts=[line for line in trace.splitlines() if line.startswith('ntt_length_counts:')]
                signatures.append(([digest[k] for k in ('digest_words','digest_sum','digest_xor')],stats['coeffs_reduced'],counts))
            assert signatures[0]==signatures[1],'Unrounded workload or arithmetic changed'
            exe,identity=candidate_exe,candidate_identity
            report['default_workload_unchanged']=True
        report['complete']=True
    except Exception as error:
        report['failures'].append(str(error));raise
    finally:
        telemetry.close();write(out/'summary.json',report)


if __name__=='__main__':main()
