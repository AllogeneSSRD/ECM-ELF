"""Fixed-D complete Stage2 A/B using the frozen small-B2 point matrix.

Distinct process per call, no discarded warmup, no learning or clock writes.
Optional stock executable is a performance guard against the pre-change build.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

from bench_stage2_version import load,write,stamp,check_gpu_processes
from bench_stage2_runtime_matrix import completed_records
from bench_stage2_point_routes import validate_clock
from bench_stage2_n_scaling import fields
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result
from stage2_d_catalog import phi


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def option(command,key,value):
    command[command.index(key)+1]=str(value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--stock',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bits',default='10232,12285,16272')
    p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--reverse-first',action='store_true',help='Reverse treatments on the first pass for independent retests')
    p.add_argument('--allow-gpu-pid',type=int,action='append',default=[])
    p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    desired=set(map(int,a.bits.split(',')))
    cells=[r for r in load(a.summary)['rows'] if not r['tail_permission'] and r['target_bits'] in desired]
    if len(cells)!=2*len(desired):raise ValueError('Need both families for every width')
    variants={}
    for name,path in [('candidate',a.exe),('stock',a.stock)]:
        if path is None:continue
        directory=out/name;directory.mkdir();(directory/'.gitignore').write_text('*\n')
        for f in (path.resolve(),path.resolve().parent/'gmp-10.dll'):shutil.copy2(f,directory/f.name)
        manifest=path.resolve().parent/'build_manifest.json'
        if manifest.exists():shutil.copy2(manifest,directory/manifest.name)
        variants[name]=dict(exe=str(directory/path.name),sha256=sha(directory/path.name))
    shutil.copy2(__file__,out/'collector_snapshot.py')
    allowed=[]
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    for pid in a.allow_gpu_pid:
        names={line.split(',',2)[2].strip() for line in text.splitlines() if line.split(',')[1].strip()==str(pid)}
        if len(names)!=1 or not Path(next(iter(names))).is_absolute():raise ValueError('Cannot bind allowed process')
        allowed.append(dict(pid=pid,process_name=next(iter(names))))
    report=dict(complete=False,started_utc=stamp(),variants=variants,diagnostic=a.diagnostic,
                source_summary=str(a.summary),allowed_processes=allowed,rows=[],failures=[])
    write(out/'results.json',report)
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and
         k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')}
    env['NTT_NO_PROGRESS']='1'
    if a.diagnostic:
        env.update(NTT_TARGET_LEAF_HASH='1',NTT_DEVICE_GLEAF_CHECK='1',NTT_S4_CARRY_TRACE='1',NTT_XADD6_TEST='1')
    signatures={}
    try:
        for round in range(a.rounds):
            for cell in cells:
                source=Path(cell['evidence']);key=f"n{cell['target_bits']}_{cell['family']}"
                treatments=['off','on']
                if a.stock and round==0:treatments=['stock',*treatments]
                if (round+int(a.reverse_first))%2:treatments.reverse()
                for treatment in treatments:
                    directory=out/f'{len(report["rows"]):03d}_{key}_{treatment}';directory.mkdir()
                    name='stock' if treatment=='stock' else 'candidate'
                    command=list(load(source/'command.json'));command[0]=variants[name]['exe']
                    for file in ('ecm.toml','points.save'):shutil.copy2(source/file,directory/file)
                    for flag,value in [('--config',directory/'ecm.toml'),('--save',directory/'points.save'),
                        ('--results',directory/'result.jsonl'),('--submission',directory/'result.txt'),
                        ('--log',directory/'engine.log'),('--debug-log-file',directory/'debug.log'),
                        ('--tune-profile',directory/'absent_tune.toml')]:option(command,flag,value)
                    if a.diagnostic:
                        option(command,'--d',2310);option(command,'--b2',2600000)
                    expected_d=2310 if a.diagnostic else cell['D']
                    expected_b2=2600000 if a.diagnostic else cell['B2']
                    runenv=env|{'NTT_POINT_SQUARE':'1' if treatment=='on' else '0'}
                    write(directory/'command.json',dict(argv=command,overrides={k:v for k,v in runenv.items() if k.startswith('NTT_')}))
                    telemetry=TelemetryRecorder(1,directory/'telemetry.jsonl',interval=.2)
                    started=time.perf_counter()
                    try:
                        check_gpu_processes(telemetry.monitor,out,directory/'before.json',allowed=allowed)
                        with (directory/'console.log').open('wb') as log:
                            result=subprocess.run(command,cwd=directory,env=runenv,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                        check_gpu_processes(telemetry.monitor,out,directory/'after.json',allowed=allowed)
                    finally:telemetry.close()
                    if result.returncode:raise RuntimeError('Curve failed: '+str(directory))
                    raw=completed_records(directory/'result.jsonl')
                    if len(raw)!=1:raise ValueError('Need one complete curve')
                    raw=raw[0];normalized=normalize_result(raw)
                    old=normalize_result(completed_records(source/'result.jsonl')[0])
                    expected={k:old[k] for k in ('N_hex','B1','sigma','carrier_exponent','curve')};expected['B2']=expected_b2
                    if any(normalized[k]!=v for k,v in expected.items()) or raw['status']!='NF' or normalized['factors'] or normalized['bad_factors']:
                        raise ValueError('Input/status differs')
                    if raw['source-build'].lower()!=variants[name]['sha256'] or (raw['d'],raw['poly-size'])!=(expected_d,phi(expected_d)//2):
                        raise ValueError('Build/geometry differs')
                    text=(directory/'engine.log').read_text(encoding='utf-8')+'\n'+(directory/'debug.log').read_text(encoding='utf-8')
                    clock=fields(text,'stage2_full_wall');check=fields(text,'s4_multiply_stats');validate_clock(clock,a.diagnostic)
                    if any(int(check[k]) for k in ('gmp_check_bad','gmp_selftest_bad')) or min(int(check[k]) for k in ('gmp_checked','gmp_selftest_cases'))<=0:
                        raise ValueError('Arithmetic checks failed/missing')
                    mode=fields(text,'point_square_mode',False)
                    if name=='candidate' and int(mode['enabled'])!=int(treatment=='on'):raise ValueError('Wrong square mode')
                    if treatment=='on' and int(fields(text,'square_selftest')['mismatches']):raise ValueError('Square selftest failed')
                    if a.diagnostic:
                        digest=fields(text,'target_descent_values');signature=(digest['words'],digest['hash'])
                        leaf=fields(text,'device_gleaf')
                        if int(leaf['checked_groups'])!=int(leaf['groups']) or int(leaf['checked_leaf_words'])!=int(leaf['device_leaf_words']):
                            raise ValueError('Incomplete leaf checks')
                    else:
                        digest=fields(text,'real_batched_rootfold');signature=tuple(digest[k] for k in ('digest_words','digest_sum','digest_xor'))
                        if int(digest['digest_words'])<=0:raise ValueError('Missing root digest')
                    counts=re.findall(r'^ntt_length_counts: log2_length:num_of_ntt (.*)$',text,re.M)
                    if len(counts)!=1:raise ValueError('Missing NTT distribution')
                    signature=(*signature,counts[0],check['coeffs_reduced'])
                    if key in signatures and signatures[key]!=signature:raise ValueError('Output/workload differs')
                    signatures[key]=signature
                    row=dict(key=key,family=cell['family'],bits=cell['target_bits'],round=round,treatment=treatment,
                             directory=str(directory),seconds=float(clock['total']),clock=clock,s4=check,
                             wall_seconds=time.perf_counter()-started,signature=signature,
                             stats={k:fields(text,k) for k in ('real_baby','real_batched_split','real_batched_wall')})
                    report['rows'].append(row);write(out/'results.json',report)
                    print(f"square_curve_progress: calls={len(report['rows'])} {key} {treatment} seconds={row['seconds']:.6f}",flush=True)
        report['complete']=True;report['finished_utc']=stamp();write(out/'results.json',report)
    except Exception as e:
        report['failures'].append(dict(utc=stamp(),error=str(e)));write(out/'results.json',report);raise


if __name__=='__main__':
    main()
