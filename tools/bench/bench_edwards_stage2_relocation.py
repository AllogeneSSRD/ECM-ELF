"""Freeze and alternate Edwards-only original/relocation full-engine timings.

prepare freezes one production binary, checked gates and exact Stage1 inputs.
run accepts only that immutable study and keeps every warmup/formal invocation.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"ecm_report"))
from result_format import normalize_result
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/test'))
from test_edwards_stage2_relocation import save,fields
from test_edwards_gpu_pipeline import stage1
from bench_edwards_stage2_optimization import freeze_binary
from bench_stage2_n_scaling import Monitor


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return normalize_result(json.loads(Path(p).read_text(encoding='utf-8-sig')))
def write(p,r):
    temp=Path(str(p)+'.new');temp.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def prime(n):return n>=2 and all(n%d for d in range(2,math.isqrt(n)+1))


def point_order(p,sigma,b1):
    """Independent affine point order via small-field BSGS, not PARI or GPU."""
    a,x,z=stage1(p,sigma,b1)
    if not z or not x:raise ValueError('nonunit point')
    x=x*pow(z,-1,p)%p;A=(4*a-2)%p;B=4*a%p
    y2=(x*x*x+A*x*x+x)*pow(B,-1,p)%p;y=pow(y2,(p+1)//4,p)
    if y*y%p!=y2:raise ValueError('point is not on the expected twist')
    Q=(B*x%p,B*B*y%p);a2=A*B%p;a4=B*B%p
    def add(P,Q):
        if P is None:return Q
        if Q is None:return P
        x,y=P;u,v=Q
        if x==u:
            if (y+v)%p==0:return None
            slope=(3*x*x+2*a2*x+a4)*pow(2*y,-1,p)%p
        else:slope=(v-y)*pow(u-x,-1,p)%p
        nx=(slope*slope-a2-x-u)%p
        return nx,(slope*(x-nx)-y)%p
    upper=p+1+2*math.isqrt(p);m=math.isqrt(upper)+1
    table={None:0};R=None
    for j in range(1,m+1):
        R=add(R,Q)
        if R is None:return j
        if j<m:table[R]=j
    step=R;R=None
    for k in range(1,m+1):
        R=add(R,step)
        if R in table:
            order=k*m-table[R]
            if order>0:return order
    raise ValueError('No point order within Hasse bound')


def prepare(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh study directory')
    gates=read(a.gates)
    if not gates['complete'] or gates['engine_sha256']!=sha(a.exe):raise ValueError('Gates/binary mismatch')
    ref=read(a.reference)
    if not ref['complete']:raise ValueError('Incomplete reference inputs')
    identity=freeze_binary(a.exe,out/'bin')
    build=read(a.exe.parent/'build_manifest.json')
    if build['engine']!='production' or build['architecture']!='sm_89':raise ValueError('Expected production sm89 binary')
    inputs=out/'inputs';inputs.mkdir();cells=[]
    wanted={'m1009_low_normal','m10267_low_normal','m10267_high_normal','m16319_low_normal'}
    for cell in ref['cells']:
        if cell['id'] not in wanted:continue
        source=ref['cases'][str(cell['exponent'])];original=Path(source['saves']['ed'])
        if sha(original)!=source['save_hashes']['ed']:raise ValueError('Changed Edwards save')
        dest=inputs/(cell['id']+'.save');shutil.copy2(original,dest)
        cells.append(dict(id=cell['id'],kind='NF',bits=source['bits'],b1=source['b1'],b2=cell['b2'],
                          d=cell['d'],carrier=0,sigma=source['sigma'],save=str(dest),save_sha256=sha(dest)))
    base=int(ref['cases']['10267']['N_hex'],16)
    selected=None
    for p in range(1000003,1100000,4):
        if not prime(p):continue
        try:r=point_order(p,26,20)
        except ValueError:continue
        if r>15015 and prime(r) and math.gcd(r,30030)==1:
            try:case=save(inputs/'wide_factor.save',base*p,26,20)
            except ValueError:continue
            selected=(p,r,case);break
    if selected is None:raise ValueError('No verified wide factor fixture')
    p,r,case=selected
    cells.append(dict(id='wide_factor_unit',kind='F',bits=case['n'].bit_length(),b1=20,b2=100000000,
                      d=30030,carrier=0,sigma=26,save=case['save'],save_sha256=case['save_sha256'],
                      known_factor=str(p),residual_order=r,order_proof='independent_affine_BSGS'))
    # Same width, a small residual order appears in baby generation, so it must
    # bypass the suffix without spending a speculative giant pass.
    small=None
    for p2 in (10007,10009,10037,10039,10061,10067,10069,10079):
        try:other=save(inputs/'wide_baby_nonunit.save',base*p2,17,20)
        except ValueError:continue
        small=(p2,other);break
    if small is None:raise ValueError('No wide nonunit fixture')
    p2,other=small
    cells.append(dict(id='wide_factor_baby_nonunit',kind='F',bits=other['n'].bit_length(),b1=20,
                      b2=100000000,d=30030,carrier=0,sigma=17,save=other['save'],
                      save_sha256=other['save_sha256'],known_factor=str(p2)))
    ini=out/'bench.ini';ini.write_text('user=\ncomputer=\nstage2_tune_runtime=0\nstage2_relocation=0\n')
    driver=subprocess.check_output(['nvidia-smi','--id='+str(a.device),'--query-gpu=driver_version',
                                    '--format=csv,noheader'],text=True).strip()
    monitor=Monitor(a.device)
    try:device=monitor.identity
    finally:monitor.close()
    helpers={name:sha(ROOT/name) for name in (
        'tools/test/test_edwards_stage2_relocation.py','tools/test/test_stage2_relocation_baby_zero.py',
        'tools/test/test_edwards_gpu_pipeline.py',
        'tools/stat/suyama_mont_ref.py','tools/bench/bench_edwards_stage2_optimization.py',
        'tools/bench/bench_stage2_n_scaling.py')}
    for name in helpers:
        target=out/'helpers'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    shutil.copy2(__file__,out/'collector_snapshot.py')
    plan=dict(schema=1,complete=False,created_utc=datetime.now(timezone.utc).isoformat(),device=device,driver=driver,
              binary=identity,collector_sha256=sha(__file__),gates=dict(path=str(a.gates.resolve()),sha256=sha(a.gates)),
              helpers=helpers,
              source_reference=dict(path=str(a.reference.resolve()),sha256=sha(a.reference)),
              cells=cells,runs=[],ini=str(ini),ini_sha256=sha(ini),repeats=a.repeats,warmups=1,
              protocol=dict(arms=['original','relocation'],order='alternating pairs, every warmup retained',
                            timing='stage2_full_wall init+main includes probe/replay/oracle drain; process wall separate',
                            arena_mb=6300,owner_mb=640,batch_mb=256,short_calibration=0,tune_runtime=0,ntt_counts=1))
    write(out/'plan.json',plan)
    print('PREPARED',len(cells),'Edwards cells; factor',p,'residual order',r,flush=True)


def run(a):
    folder=a.study.resolve();path=folder/'plan.json';plan=read(path)
    if plan['collector_sha256']!=sha(__file__):raise ValueError('Collector changed; prepare a fresh study')
    for name,h in plan['helpers'].items():
        if sha(ROOT/name)!=h or sha(folder/'helpers'/name)!=h:raise ValueError('Helper changed: '+name)
    if sha(plan['gates']['path'])!=plan['gates']['sha256']:raise ValueError('Gates changed')
    exe=Path(plan['binary']['path'])
    for name,h in plan['binary']['runtime_files'].items():
        if sha(exe.parent/name)!=h:raise ValueError('Runtime changed')
    if sha(exe)!=plan['binary']['sha256'] or sha(plan['ini'])!=plan['ini_sha256']:raise ValueError('Binary/INI changed')
    for name,h in plan['binary']['identity']['sources'].items():
        if sha(exe.parent/'sources'/name)!=h:raise ValueError('Frozen source changed: '+name)
    monitor=Monitor(plan['device']['index'])
    if monitor.identity!=plan['device']:raise ValueError('GPU identity changed')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(CUDA_LAUNCH_BLOCKING='0')
    completed={r['id'] for r in plan['runs']}
    total=(plan['repeats']+1)*2*len(plan['cells']);done=len(completed)
    try:
        for repeat in range(plan['repeats']+1):
            for ci,cell in enumerate(plan['cells']):
                for mode in ((0,1) if (repeat+ci)%2==0 else (1,0)):
                    tag=f'{cell["id"]}_{"warmup" if repeat==0 else "timing"}{repeat}_{mode}'
                    if tag in completed:continue
                    if sha(cell['save'])!=cell['save_sha256']:raise ValueError('Input changed')
                    out=folder/'runs'/tag
                    if out.exists():raise ValueError('Unrecorded invocation exists; inspect it before any continuation: '+str(out))
                    out.mkdir(parents=True)
                    command=list(map(str,[exe,'--ini',plan['ini'],'--save',cell['save'],'--b2',cell['b2'],'--d',cell['d'],
                        '--device',plan['device']['index'],'--curves',1,'--relocation',mode,'--carrier-exponent',0,
                        '--arena-mb',6300,'--owner-budget-mb',640,'--batch-mb',256,'--factor-only',
                        '--short-calibration',0,'--tune-runtime',0,'--ntt-counts',1,'--log-level','debug',
                        '--results',out/'internal.jsonl','--submission',out/'results.json.txt',
                        '--log',out/'engine.log','--debug-log-file',out/'debug.log']))
                    write(out/'command.json',dict(command=command,environment={'CUDA_LAUNCH_BLOCKING':env['CUDA_LAUNCH_BLOCKING']}))
                    telemetry=[];start=time.perf_counter()
                    with (out/'driver.log').open('wb') as log:
                        proc=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
                        while proc.poll() is None:
                            telemetry.append(monitor.sample());time.sleep(.2)
                            if time.perf_counter()-start>1800:
                                proc.kill();proc.wait();raise TimeoutError(tag)
                    seconds=time.perf_counter()-start
                    write(out/'telemetry.json',telemetry)
                    if proc.returncode:raise ValueError(f'{tag}: exit={proc.returncode}; inspect {out}')
                    row=read(out/'internal.jsonl');text=(out/'engine.log').read_text()+'\n'+(out/'debug.log').read_text()
                    timing=fields(text,'stage2_full_wall');coverage=fields(text,'s4_multiply_stats')
                    if row['curve']!='ed' or row['param'] is not None or row['bad_factors'] or timing['clean']!='1':raise ValueError('Invalid curve/check scope')
                    if not int(coverage['gmp_checked']) or int(coverage['gmp_check_bad']) or int(coverage['gmp_selftest_bad']):raise ValueError('Arithmetic gate failed')
                    if cell['kind']=='NF' and (row['factors'] or row['relocation_replayed']):raise ValueError('NF did not stay on the clean path')
                    if cell['kind']=='F' and (not row['factors'] or (mode and not row['relocation_replayed'])):raise ValueError('Missing factor/replay')
                    if cell['kind']=='F' and not any(int(f)%int(cell['known_factor'])==0 for f in row['factors']):raise ValueError('Known factor absent')
                    peer=[r for r in plan['runs'] if r['cell']==cell['id']]
                    if any(r['factors']!=sorted(row['factors']) for r in peer):raise ValueError('Cross-algorithm factor mismatch')
                    leaf=fields(text,'descent_values')
                    same=[r for r in peer if r['mode']==mode]
                    if any(r['leaf']!=leaf for r in same):raise ValueError('Same-algorithm complete target leaf mismatch')
                    hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
                    record=dict(id=tag,cell=cell['id'],mode=mode,warmup=repeat==0,repeat=repeat,
                                timing={k:float(timing[k]) for k in ('shape','init','main','total')},process_seconds=seconds,
                                factors=sorted(row['factors']),leaf=leaf,coverage=coverage,
                                relocation={k:v for k,v in row.items() if k.startswith('relocation') or k.startswith('giant_')},
                                raw=str(out),raw_hashes=hashes)
                    plan['runs'].append(record);write(path,plan);done+=1
                    print(f'{done}/{total} {tag} total={timing["total"]} replay={row["relocation_replay_seconds"]}',flush=True)
        plan['complete']=True;write(path,plan)
    finally:monitor.close()
    print('COMPLETE',len(plan['runs']),'Edwards timings, including warmups',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--exe',type=Path,required=True);p.add_argument('--gates',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
    p=sub.add_parser('run');p.add_argument('--study',type=Path,required=True)
    a=parser.parse_args();prepare(a) if a.command=='prepare' else run(a)


if __name__=='__main__':main()
