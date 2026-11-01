"""Check independent baby opt-in, mixed TPIs and device-budget fallback.

Reference must be a completed wide-carrier full-leaf diagnostic from the
fixed-D collector. These are correctness calls, never performance samples.
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from bench_stage2_version import check_gpu_processes,load,write
from bench_stage2_runtime_matrix import completed_records
from bench_stage2_n_scaling import fields
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-gpu-pid',type=int,action='append',default=[])
    a=p.parse_args();ref=a.reference.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    meta=load(ref/'command.json');original=completed_records(ref/'result.jsonl')[0]
    normalized=normalize_result(original)
    carrier_bits=int(normalized['carrier_exponent'])
    if not 8192<carrier_bits<=16384:raise ValueError('Need wide carrier reference')
    binary=out/'bin';binary.mkdir();(binary/'.gitignore').write_text('*\n')
    for name in (Path(meta['argv'][0]).name,'gmp-10.dll','build_manifest.json'):
        shutil.copy2(Path(meta['argv'][0]).parent/name,binary/name)
    exe=binary/Path(meta['argv'][0]).name;sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    identity=sha(exe);prior=(ref/'engine.log').read_text(encoding='utf-8')+(ref/'debug.log').read_text(encoding='utf-8')
    baseline=fields(prior,'target_descent_values')
    allowed=[]
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    for pid in a.allow_gpu_pid:
        names={line.split(',',2)[2].strip() for line in text.splitlines() if line.split(',')[1].strip()==str(pid)}
        if len(names)!=1 or not Path(next(iter(names))).is_absolute():raise ValueError('Cannot bind allowed process')
        allowed.append(dict(pid=pid,process_name=next(iter(names))))
    report=dict(complete=False,diagnostic=True,exe_sha256=identity,dll_sha256=sha(binary/'gmp-10.dll'),
                reference=str(ref),allowed_processes=allowed,cases=[],failures=[])
    write(out/'summary.json',report);shutil.copy2(__file__,out/'test_snapshot.py')
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')}
    env.update(meta['overrides'])
    cases=[('scalar',0,0,1,False),('baby_only',32,0,1,False),('mixed_tpis',32,16,1,False),
           ('baby_budget_fallback',32,0,1,True),('generic_reduction_rejected',32,0,0,False)]
    telemetry=TelemetryRecorder(1,out/'telemetry.jsonl',interval=.2)
    try:
        for name,baby,giant,mersenne,fallback in cases:
            directory=out/name;directory.mkdir()
            for file in ('ecm.toml','points.save'):shutil.copy2(ref/file,directory/file)
            command=[str(exe),*[str(directory)+arg[len(str(ref)):] if arg.startswith(str(ref)) else arg for arg in meta['argv'][1:]]]
            runenv=env|dict(NTT_BABY_COOP=str(baby),NTT_GIANT_COOP=str(giant),NTT_POINT_MERSENNE=str(mersenne))
            if fallback:runenv['NTT_BABY_DEVICE_MAX_MB']='0'
            write(directory/'command.json',dict(argv=command,overrides={k:v for k,v in runenv.items() if k.startswith('NTT_')}))
            check_gpu_processes(telemetry.monitor,out,directory/'before.json',allowed=allowed)
            with (directory/'console.log').open('wb') as log:
                call=subprocess.run(command,cwd=directory,env=runenv,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            check_gpu_processes(telemetry.monitor,out,directory/'after.json',allowed=allowed)
            if not mersenne:
                # Production deliberately fixes this algorithm flag; ordinary
                # modulus and generic-field arithmetic have separate probes.
                console=(directory/'console.log').read_text(encoding='utf-8')
                if not call.returncode or 'production Stage2 requires NTT_POINT_MERSENNE=1' not in console:
                    raise ValueError('Missing production configuration guard')
                if (directory/'result.jsonl').exists():raise ValueError('Rejected run produced a result')
                report['cases'].append(dict(name=name,returncode=call.returncode,rejected=True,directory=str(directory)))
                write(out/'summary.json',report);print('PASS:',name,flush=True);continue
            if call.returncode:raise ValueError('Curve failed: '+name)
            raw=completed_records(directory/'result.jsonl')[0];actual=normalize_result(raw)
            if raw['source-build'].lower()!=identity or raw['status']!='NF':raise ValueError('Wrong binary/status')
            for k in ('N_hex','B1','B2','sigma','carrier_exponent','curve','factors','bad_factors'):
                if actual[k]!=normalized[k]:raise ValueError('Result differs: '+k)
            trace=(directory/'engine.log').read_text(encoding='utf-8')+(directory/'debug.log').read_text(encoding='utf-8')
            digest=fields(trace,'target_descent_values')
            if any(digest[k]!=baseline[k] for k in ('words','hash')):raise ValueError('Full target leaves differ')
            leaf=fields(trace,'device_gleaf')
            if leaf['checked_groups']!=leaf['groups'] or leaf['checked_leaf_words']!=leaf['device_leaf_words']:
                raise ValueError('Incomplete giant leaf checks')
            s4=fields(trace,'s4_multiply_stats')
            if int(s4['gmp_check_bad']) or int(s4['gmp_selftest_bad']):raise ValueError('Arithmetic check failed')
            tests=re.findall(r'^coop_selftest: tpi=(\d+) multiplications=64 squares=64 mismatches=(\d+)$',trace,re.M)
            expected=[] if not baby and not giant else [('16','0'),('32','0')] if giant==16 else [('32','0')]
            if tests!=expected:raise ValueError('Wrong per-TPI selftests: '+str(tests))
            launches=re.findall(r'^baby_coop: tpi=32 .*$',trace,re.M)
            if bool(launches)!=bool(baby and not fallback):raise ValueError('Wrong baby route')
            if launches and int(fields(trace,'baby_coop')['mersenne_bits'])!=(carrier_bits if mersenne else 0):
                raise ValueError('Arithmetic modulus classification differs')
            if fallback and 'baby_device_fallback:' not in trace:raise ValueError('Budget fallback was not exercised')
            report['cases'].append(dict(name=name,returncode=call.returncode,coop_selftests=tests,directory=str(directory)))
            write(out/'summary.json',report);print('PASS:',name,flush=True)
        report['complete']=True
    except Exception as error:
        report['failures'].append(str(error));raise
    finally:
        telemetry.close();write(out/'summary.json',report)


if __name__=='__main__':
    main()
