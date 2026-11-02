"""Real worker configuration, actual point routes and bounded automatic learning.

Uses a verified wide Stage1 save, a fresh profile and useful no-factor curves.
These calls validate behavior; they are not the fixed-D performance A/B.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
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
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage2_tune_tables import summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('exe','save','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scope',choices=('configuration','formal','giant','giant-auto','giant-small','giant-formal','unpaired','fallback'),default='configuration')
    p.add_argument('--allow-gpu-pid',type=int,action='append',default=[])
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'bin';binary.mkdir();(binary/'.gitignore').write_text('*\n')
    for name in (a.exe.name,'gmp-10.dll','build_manifest.json'):shutil.copy2(a.exe.resolve().parent/name,binary/name)
    exe=binary/a.exe.name;saved=out/'point.save';shutil.copy2(a.save,saved)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();identity=sha(exe)
    shutil.copy2(__file__,out/'test_snapshot.py')
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')}
    env['NTT_NO_PROGRESS']='1';allowed=[]
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    for pid in a.allow_gpu_pid:
        names={s.split(',',2)[2].strip() for s in text.splitlines() if s.split(',')[1].strip()==str(pid)}
        if len(names)!=1 or not Path(next(iter(names))).is_absolute():raise ValueError('Cannot bind allowed process')
        allowed.append(dict(pid=pid,process_name=next(iter(names))))
    report=dict(complete=False,diagnostic=True,exe_sha256=identity,save_sha256=sha(saved),cases=[],failures=[])
    write(out/'summary.json',report);telemetry=TelemetryRecorder(1,out/'telemetry.jsonl',interval=.2)
    signatures={}

    def call(name,body='',extra=(),overrides=None,expected=0,profile=None,runtime=False,minimum=2,budget_fallback=False,expected_giant=0):
        directory=out/name;directory.mkdir();config=directory/'ecm.toml'
        config.write_text('[stage2]\nstage2_short_calibration=false\nstage2_tune_runtime='+str(runtime).lower()+'\n'+body,encoding='utf-8')
        expected_modes=list(expected) if isinstance(expected,(list,tuple)) else [expected]
        command=[str(exe),'--config',str(config),'--save',str(saved),'--curves',str(len(expected_modes)),'--device','1',
                 '--b2','26000000','--d','6006','--carrier-exponent','0','--auto-carrier','0','--groot-layout','0',
                 '--giant-chain','64','--ntt-counts','1','--tune-min-repeats',str(minimum),'--tune-profile',str(profile or directory/'absent.toml'),
                 '--results',str(directory/'result.jsonl'),'--submission',str(directory/'submission.txt'),
                 '--log',str(directory/'engine.log'),'--debug-log-file',str(directory/'debug.log'),*extra]
        runenv=env|(overrides or {});write(directory/'command.json',dict(argv=command,overrides={k:v for k,v in runenv.items() if k.startswith('NTT_')}))
        check_gpu_processes(telemetry.monitor,out,directory/'before.json',allowed=allowed)
        with (directory/'console.log').open('wb') as log:
            result=subprocess.run(command,cwd=directory,env=runenv,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        check_gpu_processes(telemetry.monitor,out,directory/'after.json',allowed=allowed)
        console=(directory/'console.log').read_text(encoding='utf-8');assert sha(exe)==identity
        if expected is None:
            assert result.returncode and ('backend' in console),console
            row=dict(name=name,rejected=True,returncode=result.returncode)
        else:
            assert result.returncode==0,console
            records=completed_records(directory/'result.jsonl');assert len(records)==len(expected_modes)
            giant_modes=list(expected_giant) if isinstance(expected_giant,(list,tuple)) else [expected_giant]*len(records)
            assert len(giant_modes)==len(records)
            for raw,mode,giant in zip(records,expected_modes,giant_modes):
                normal=normalize_result(raw);engine=raw['stage2-engine']
                assert raw['status']=='NF' and raw['source-build'].lower()==identity and not normal['factors'] and not normal['bad_factors']
                assert engine['baby_coop']==mode and (giant is None or engine['giant_coop']==giant),engine
            trace=(directory/'engine.log').read_text(encoding='utf-8')+(directory/'debug.log').read_text(encoding='utf-8')
            check=fields(trace,'s4_multiply_stats');assert not int(check['gmp_check_bad']) and not int(check['gmp_selftest_bad'])
            digest=fields(trace,'real_batched_rootfold');counts=re.findall(r'^ntt_length_counts: log2_length:num_of_ntt (.*)$',trace,re.M)
            assert len(counts)==len(records) and len(set(counts))==1
            current=(normal['N_hex'],normal['B1'],normal['B2'],normal['sigma'],digest['digest_words'],digest['digest_sum'],digest['digest_xor'],counts[0])
            key=current[:4]
            if key in signatures:assert signatures[key]==current,'Point output/workload changed'
            signatures[key]=current
            if budget_fallback:assert 'baby_device_fallback:' in trace
            row=dict(name=name,baby_coop=engine['baby_coop'],giant_coop=engine['giant_coop'],returncode=result.returncode,
                     tune_plan=engine.get('tune_plan'),directory=str(directory))
        report['cases'].append(row);write(out/'summary.json',report);print('PASS:',name,flush=True)
        return row

    try:
        if a.scope in ('giant-auto','giant-small'):
            queue=out/'ten_curves.save';queue.write_bytes(saved.read_bytes()*10);saved=queue
            for name,extra in ([] if a.scope=='giant-small' else [('fixed',('--b2','51879828')),('rounded',('--b2-round-chain','1'))]):
                profile=out/(name+'.toml')
                call(name,'stage2_baby_backend="auto"\nstage2_giant_backend="auto"\n',extra=extra,
                    profile=profile,runtime=True,expected=[0,0,32,32,32,32,32,32,32,32],expected_giant=[0,0,0,0,32,32,16,16,None,None])
                measured=[r for r in summaries(tomllib.loads(profile.read_text())) if r['source']=='measured']
                assert len(measured)==4 and all(r['repeats']>=2 for r in measured) and sum(r['repeats'] for r in measured)==10,measured
                best=min(measured,key=lambda r:float(r['mean_seconds'])+float(r.get('point_check_seconds',0)))
                before=sha(profile)
                call(name+'_readonly','stage2_baby_backend="auto"\nstage2_giant_backend="auto"\n',extra=extra,
                    profile=profile,expected=int(best['baby_coop']),expected_giant=int(best['giant_coop']))
                assert sha(profile)==before
            # Small baby degree must not disable an independently eligible
            # giant chain when the user explicitly fixes D.
            call('small_baby_degree','stage2_baby_backend="auto"\nstage2_giant_backend="auto"\n',
                extra=('--d','2310'),profile=out/'small_P.toml',runtime=True,
                expected=[0]*8,expected_giant=[0,0,32,32,16,16,None,None])
            if a.scope=='giant-small':report['complete']=True;return
            profile=out/'giant_unpaired.toml'
            for i in range(2):
                call('forced_giant_'+str(i),'stage2_baby_backend="coop32"\nstage2_giant_backend="coop32"\n',
                    extra=('--b2','51879828'),profile=profile,runtime=True,expected=32,expected_giant=32)
            before=sha(profile)
            call('unpaired_auto','stage2_baby_backend="auto"\nstage2_giant_backend="auto"\n',
                extra=('--b2','51879828'),profile=profile)
            assert sha(profile)==before
            report['complete']=True;return
        if a.scope=='fallback':
            for mode in ('auto','scalar','coop32'):
                call('low_'+mode,'stage2_baby_backend="'+mode+'"\nstage2_giant_backend="coop32"\n')
            report['complete']=True
            return
        if a.scope=='unpaired':
            profile=out/'unpaired.toml'
            for i in range(2):
                call('forced_'+str(i),'stage2_baby_backend="coop32"\n',profile=profile,runtime=True,expected=32)
            before=sha(profile)
            call('auto_needs_scalar','stage2_baby_backend="auto"\n',profile=profile)
            assert sha(profile)==before,'read-only auto changed forced measurement evidence'
            report['complete']=True
            return
        if a.scope=='giant':
            for mode in (0,16,32):
                call('chain_'+str(mode),'stage2_baby_backend="scalar"\nstage2_giant_backend="'+
                    ('scalar' if not mode else 'coop'+str(mode))+'"\n',extra=('--b2','260000000'),expected_giant=mode)
            call('short_fallback','stage2_giant_backend="coop32"\n')
            report['complete']=True
            return
        if a.scope in ('formal','giant-formal'):
            giant_auto=a.scope=='giant-formal';neighbor=5460 if giant_auto else 6930
            profile=out/'formal.toml'
            config=out/'formal_config.toml';config.write_text('[stage2]\nstage2_short_calibration=false\n')
            snapshots=[]
            for name,d in [('paired',6006),('neighbor',neighbor),('incremental',6006)]:
                directory=out/name;directory.mkdir()
                command=[str(exe),'--config',str(config),'--device','1','--tune','ecm','--tune-save',str(saved),
                    '--tune-d',str(d),'--tune-b2',('51879828' if giant_auto else '26000000'),'--tune-repeats','1','--tune-min-repeats','1',
                    '--tune-file',str(profile),'--tune-budget-seconds','90','--baby-backend','auto',
                    '--giant-backend',('auto' if giant_auto else 'scalar'),'--groot-layout','0','--auto-carrier','0','--tune-arithmetic','generic']
                write(directory/'command.json',command)
                check_gpu_processes(telemetry.monitor,out,directory/'before.json',allowed=allowed)
                with (directory/'console.log').open('wb') as log:
                    result=subprocess.run(command,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                check_gpu_processes(telemetry.monitor,out,directory/'after.json',allowed=allowed)
                console=(directory/'console.log').read_text(encoding='utf-8');assert result.returncode==0,console
                assert not any(s in console for s in ('real_baby:','s4_multiply_stats:','stage2_phase:','batched_progress:'))
                rows=summaries(tomllib.loads(profile.read_text(encoding='utf-8')))
                measured=[r for r in rows if r['source']=='measured']
                assert all(r['repeats']==1 and float(r['point_check_seconds'])>0 for r in measured)
                modes={(int(r['d']),int(r['baby_coop'])) for r in measured}
                assert {(6006,0),(6006,32)}<=modes,modes
                if giant_auto:
                    triples={(int(r['d']),int(r['baby_coop']),int(r['giant_coop'])) for r in measured}
                    assert {(6006,0,0),(6006,32,0),(6006,32,32),(6006,32,16)}<=triples,triples
                    if name!='paired':assert sum(d==neighbor for d,b,g in triples)==1,triples
                elif name!='paired':assert (neighbor,32) in modes and (neighbor,0) not in modes,modes
                snapshots.append(measured);report['cases'].append(dict(name=name,modes=sorted(modes)))
                write(out/'summary.json',report);print('PASS:',name,flush=True)
            assert snapshots[1]==snapshots[2],'Completed formal measurements repeated'
            report['complete']=True
            return
        call('default')
        call('toml_scalar','stage2_baby_backend="scalar"\n')
        call('toml_coop','stage2_baby_backend="coop32"\n',expected=32)
        call('cli_over_toml','stage2_baby_backend="coop32"\n',extra=('--baby-backend','scalar'))
        call('toml_over_environment','stage2_baby_backend="coop32"\n',overrides={'NTT_BABY_COOP':'0'},expected=32)
        call('legacy_environment',overrides={'NTT_BABY_COOP':'32'},expected=32)
        call('worker_override','stage2_baby_backend="scalar"\n[workers.2.stage2]\nstage2_baby_backend="coop32"\n',extra=('--worker','2'),expected=32)
        call('bad_toml','stage2_baby_backend="coop16"\n',expected=None)
        call('bad_cli',extra=('--baby-backend','wrong'),expected=None)
        call('budget_fallback','stage2_baby_backend="coop32"\n',overrides={'NTT_BABY_DEVICE_MAX_MB':'0'},budget_fallback=True)
        profile=out/'learned.toml';queue=out/'six_curves.save'
        # Create a separate fixture exclusively; retain the frozen source input.
        with queue.open('xb') as target:target.write(saved.read_bytes()*6)
        report['queue_save_sha256']=sha(queue);saved=queue
        modes=[0,0,32,32,32,32]
        call('learning_queue','stage2_baby_backend="auto"\n',profile=profile,runtime=True,expected=modes)
        data=tomllib.loads(profile.read_text(encoding='utf-8'));rows=summaries(data)
        assert len(rows)==2 and {int(r['baby_coop']) for r in rows}=={0,32},rows
        assert {int(r['baby_coop']):int(r['repeats']) for r in rows}=={0:2,32:4},rows
        assert all(float(r['point_check_seconds'])>0 for r in rows)
        before=sha(profile)
        call('reuse_without_learning','stage2_baby_backend="auto"\n',profile=profile,expected=32)
        assert sha(profile)==before,'disabled runtime tuning wrote the profile'
        call('force_scalar_over_profile','stage2_baby_backend="scalar"\n',profile=profile)
        report['learning_modes']=modes;report['complete']=True
    except Exception as error:
        report['failures'].append(str(error));raise
    finally:
        telemetry.close();write(out/'summary.json',report)


if __name__=='__main__':
    main()
