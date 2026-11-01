"""Exact calibration bounds must ignore the production B2-rounding request.

All runs use a frozen save and fresh profiles; plan-only never consumes a queue.
--expect-bug retains a before-fix reproduction, including passing controls.
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
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage2_tune_tables import summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('stage2','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--save',type=Path,required=True,help='Frozen no-factor cost-probe save with B1=20')
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--allow-gpu-pid',action='append',type=int,default=[])
    p.add_argument('--expect-bug',action='store_true')
    p.add_argument('--run-curve',action='store_true',help='Also finish short_on as useful Stage2 work, not only plan it')
    a=p.parse_args();a.stage2=a.stage2.resolve()
    if a.expect_bug and a.run_curve:p.error('--run-curve requires the fixed implementation')
    for source in (a.stage2,a.stage2.parent/'gmp-10.dll',a.save):
        if not source.is_file():raise FileNotFoundError(source)
    b1=re.search(r'\bB1\s*=\s*(\d+)',a.save.read_text(encoding='utf-8-sig'))
    if not b1 or int(b1[1])!=20:p.error('--save must be a no-factor cost probe with B1=20')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'bin';binary.mkdir();(binary/'.gitignore').write_text('*\n')
    for name in (a.stage2.name,'gmp-10.dll'):
        source=a.stage2.resolve().parent/name
        if source.exists():shutil.copy2(source,binary/name)
    exe=binary/a.stage2.name;saved=out/'point.save';shutil.copy2(a.save,saved)
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    original_save=sha(a.save);identity=sha(exe)
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')}
    env['NTT_NO_PROGRESS']='1'
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    allowed=[]
    for pid in a.allow_gpu_pid:
        names={line.split(',',2)[2].strip() for line in processes.splitlines() if line.split(',')[1].strip()==str(pid)}
        if len(names)!=1 or not Path(next(iter(names))).is_absolute():raise ValueError('Cannot bind allowed process')
        allowed.append(dict(pid=pid,process_name=next(iter(names))))
    report=dict(complete=False,expect_bug=a.expect_bug,exe_sha256=identity,save_sha256=original_save,cases=[],failures=[])
    write(out/'summary.json',report);shutil.copy2(__file__,out/'test_snapshot.py')
    telemetry=TelemetryRecorder(a.device,out/'telemetry.jsonl',interval=.2)
    try:
        for name,rounding,calibration,formal in [('short_on',True,True,False),('short_off',False,True,False),
                ('short_disabled',True,False,False),('formal_rejected',True,True,True),('formal_fixed',True,True,True)]:
            directory=out/name;directory.mkdir();config=directory/'ecm.toml';profile=directory/'tune.toml'
            config.write_text('[stage2]\nstage2_b2_round_chain='+str(rounding).lower()+
                '\nstage2_short_calibration='+str(calibration).lower()+
                '\nstage2_tune_runtime=false\nstage2_giant_chain="auto"\nstage2_groot_balanced="auto"\n'
                'stage2_tune_min_repeats=1\nstage2_auto_carrier=false\n')
            common=[str(exe),'--config',str(config),'--device',str(a.device)]
            if formal:
                command=common+['--tune','ecm','--tune-save',str(saved),'--tune-d','210','--tune-b2','42020',
                    '--tune-repeats','1','--tune-budget-seconds','20','--tune-file',str(profile)]
                if name=='formal_fixed':command+=['--b2-round-chain','0']
            else:
                action=['--curves','1'] if a.run_curve and name=='short_on' else ['--plan-only']
                command=common+['--save',str(saved),*action,'--b2','26000000','--d','6006','--carrier-exponent','0',
                    '--tune-profile',str(profile),'--results',str(directory/'result.jsonl'),
                    '--submission',str(directory/'result.txt'),'--log',str(directory/'engine.log')]
            write(directory/'command.json',command)
            check_gpu_processes(telemetry.monitor,out,directory/'before.json',allowed=allowed)
            with (directory/'console.log').open('wb') as log:
                result=subprocess.run(command,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            check_gpu_processes(telemetry.monitor,out,directory/'after.json',allowed=allowed)
            console=(directory/'console.log').read_text(encoding='utf-8')
            rejected=name=='formal_rejected'
            fails=a.expect_bug and rounding and calibration and not formal
            assert (result.returncode!=0)==(fails or rejected),console
            if rejected:assert 'B2 rounding is not an ECM calibration grid' in console
            elif fails:assert 'invalid completed calibration shape' in console
            else:
                assert 'invalid completed calibration shape' not in console
                if calibration:
                    measured=[r for r in summaries(tomllib.loads(profile.read_text(encoding='utf-8'))) if r['source']=='measured']
                    assert measured and all(r['b1']==20 and r['b2']==42020 and r['d']==210 for r in measured),measured
                if rounding and not formal:
                    engine=directory/'engine.log'
                    trace=console+(engine.read_text(encoding='utf-8') if engine.exists() else '')
                    bounds=re.findall(r'stage2_bounds: requested_B2=(\d+) actual_B2=(\d+)',trace)
                    assert bounds and all(int(requested)==26000000 and int(actual)>int(requested) for requested,actual in bounds),bounds
                if a.run_curve and name=='short_on':
                    records=completed_records(directory/'result.jsonl');assert len(records)==1
                    normal=normalize_result(records[0])
                    assert records[0]['status']=='NF' and not normal['factors'] and not normal['bad_factors']
                    assert normal['B2']==int(bounds[-1][1]) and normal['B1']==20
                    trace=(directory/'engine.log').read_text(encoding='utf-8')
                    assert 'gmp_check_bad=0' in trace and 'gmp_selftest_bad=0' in trace
            receipts=[]
            for path in directory.glob('data/experiments/**/measure*_[0-9].jsonl'):
                receipt=json.loads(path.read_text(encoding='utf-8'))
                assert receipt['d']==210 and receipt['clean']==1 and not receipt['bad'] and receipt['checked']>0
                assert receipt['giant_points']==(8208 if fails else 202),receipt
                receipts.append(dict(file=str(path),giant_points=receipt['giant_points']))
            if calibration and not rejected:assert receipts,'Missing calibration execution evidence'
            assert sha(a.save)==original_save and sha(exe)==identity,'Frozen inputs changed'
            report['cases'].append(dict(name=name,returncode=result.returncode,receipts=receipts))
            write(out/'summary.json',report);print('PASS:',name,flush=True)
        report['complete']=True
    except Exception as error:
        report['failures'].append(str(error));raise
    finally:
        telemetry.close();write(out/'summary.json',report)


if __name__=='__main__':main()
