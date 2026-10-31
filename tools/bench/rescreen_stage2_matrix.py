"""Recheck explicitly flagged default-screen cells; retain their first sample."""
import argparse
import os
from pathlib import Path
import statistics
import subprocess
import time
from bench_stage2_version import load,write,stamp,shape_fields,check_gpu_processes
from bench_stage2_n_scaling import fields,number
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result
from stage2_d_catalog import phi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--threshold',type=float,default=.05)
    p.add_argument('--minimum-seconds',type=float,default=.02)
    p.add_argument('--legacy-controls',action='store_true',help='Fixed-D same-binary old-layout control only where automatic policy changes layout')
    a=p.parse_args();study=a.study.resolve();plan=load(study/'plan.json')
    if not plan['complete']:raise ValueError('Wait for learning and validation to finish')
    baseline=load(Path(plan['defaults'])/'plan.json');old=load(a.reference/'plan.json')
    out=study/('legacy_controls' if a.legacy_controls else 'rescreens');out.mkdir(exist_ok=False)
    candidates=[]
    for r in baseline['runs']:
        if a.legacy_controls:
            words=(int(r['shape']['S_bits'])+63)//64
            legacy=int(words>128 and (r['B2']//r['D']+2)//r['P']>=4)
            if legacy!=int(r['stats']['groot_layout']['balanced']):candidates.append(r)
            continue
        reference=[s['engine_seconds'] for s in old['runs'] if s['case']==r['case'] and
                   s['engine']==r['engine'] and s['B2']==r['B2']]
        median=statistics.median(reference)
        if r['engine_seconds']>median*(1+a.threshold) and r['engine_seconds']-median>a.minimum_seconds:
            candidates.append(r)
    report=dict(complete=False,threshold=a.threshold,minimum_seconds=a.minimum_seconds,
        reference=str(a.reference.resolve()),legacy_controls=a.legacy_controls,candidates=[r['id'] for r in candidates],runs=[])
    write(out/'summary.json',report)
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and
         k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')};env['NTT_NO_PROGRESS']='1'
    cases={c['id']:c for c in baseline['cases']}
    for original in candidates:
        case=cases[original['case']]
        for repeat in ((1,) if a.legacy_controls else (2,3)):
            folder=out/(original['id']+f'_recheck{repeat}');folder.mkdir()
            prefix=folder/'run'
            command=[s.replace(original['prefix'],str(prefix)) for s in original['command']]
            if a.legacy_controls:
                command[command.index('--d')+1]=str(original['D'])
                words=(int(original['shape']['S_bits'])+63)//64
                legacy=int(words>128 and (original['B2']//original['D']+2)//original['P']>=4)
                command+=['--groot-layout',str(legacy)]
            write(prefix.with_suffix('.command.json'),command)
            started_utc=stamp()
            monitor=TelemetryRecorder(1,prefix.with_suffix('.telemetry.jsonl'));start=time.perf_counter()
            try:
                check_gpu_processes(monitor.monitor,Path(original['prefix']).parents[1],prefix.with_suffix('.before.json'),allowed=plan['allowed_processes'])
                with prefix.with_suffix('.driver.log').open('wb') as log:
                    result=subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                check_gpu_processes(monitor.monitor,Path(original['prefix']).parents[1],prefix.with_suffix('.after.json'),allowed=plan['allowed_processes'])
            finally:monitor.close()
            if result.returncode:raise ValueError('Recheck failed; evidence retained')
            raw=load(prefix.with_suffix('.jsonl'));r=normalize_result(raw)
            identity=dict(N_hex=case['N_hex'],B1=case['b1'],B2=original['B2'],sigma=case['sigma'],
                          curve='ed',carrier_exponent=original['carrier_exponent'])
            if any(r[k]!=v for k,v in identity.items()) or raw['status']!='NF' or r['factors'] or r['bad_factors'] or raw['source-build']!=plan['exe_sha256']:
                raise ValueError('Recheck result/input identity differs')
            text=prefix.with_suffix('.engine.log').read_text(encoding='utf-8')+'\n'+prefix.with_suffix('.debug.log').read_text(encoding='utf-8')
            clock=fields(text,'stage2_full_wall');check=fields(text,'s4_multiply_stats');shape=shape_fields(text)
            if clock['clean']!='1' or any(int(check[k]) for k in ('gmp_selftest_bad','gmp_check_bad')) or min(int(check[k]) for k in ('gmp_selftest_cases','gmp_checked'))<=0:
                raise ValueError('Recheck arithmetic gates failed')
            if (number(shape['D']),number(shape['P']))!=(original['D'],original['P']) or phi(original['D'])//2!=original['P']:
                raise ValueError('Default geometry changed during recheck')
            expected=legacy if a.legacy_controls else int(original['stats']['groot_layout']['balanced'])
            if int(fields(text,'groot_layout')['balanced'])!=expected:raise ValueError('Layout differs from control')
            if a.legacy_controls:
                source_text=Path(original['prefix']).with_suffix('.engine.log').read_text(encoding='utf-8')+'\n'+Path(original['prefix']).with_suffix('.debug.log').read_text(encoding='utf-8')
                before,after=fields(source_text,'real_batched_rootfold',False),fields(text,'real_batched_rootfold',False)
                if not before or not after or int(before.get('digest_words',0))<=0:
                    raise ValueError('Same-shape G-root digest is absent')
                if any(before[k]!=after[k] for k in ('digest_words','digest_sum','digest_xor','digest_kind')):
                    raise ValueError('Same-shape G-root digest changed with layout')
            row=original|dict(id=original['id'].rsplit('_r',1)[0]+f'_r{repeat}',repeat=repeat,
                prefix=str(prefix),command=command,engine_seconds=float(clock['total']),
                process_seconds=time.perf_counter()-start,phase_seconds=clock,shape=shape,s4=check,
                stats={k:fields(text,k,False) for k in original['stats']},started_utc=started_utc,finished_utc=stamp())
            report['runs'].append(row);write(out/'summary.json',report)
            print(row['id'],row['engine_seconds'],flush=True)
    report['complete']=True;write(out/'summary.json',report)


if __name__=='__main__':main()
