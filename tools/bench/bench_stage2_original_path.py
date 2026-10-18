"""Freeze and compare the original Stage2 path and explicit full-G B2 trials.

All timings use fresh native processes, relocation=0, fixed D, exact saved
Edwards inputs and mandatory native arithmetic checks. No online submission.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/test'))
from test_edwards_stage2_relocation import fields
from bench_edwards_stage2_optimization import freeze_binary
from bench_stage2_n_scaling import Monitor
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    t=Path(str(p)+'.new');t.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');t.replace(p)
def degree(d):
    phi,left,q=d,d,2
    while q*q<=left:
        if left%q==0:
            phi=phi//q*(q-1)
            while left%q==0:left//=q
        q+=1
    if left>1:phi=phi//left*(left-1)
    return phi//2

def prepare(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh study directory')
    ref=read(a.reference)
    if not ref['complete']:raise ValueError('Incomplete reference')
    gates=read(a.gates)
    if not gates['complete'] or gates['engine_sha256']!=sha(a.candidate):raise ValueError('Candidate gates differ')
    binaries={key:freeze_binary(exe,out/'bin'/key) for key,exe in [('baseline',a.baseline),('candidate',a.candidate)]}
    ini=out/'bench.ini';ini.write_text('user=\ncomputer=\nstage2_relocation=0\n',encoding='utf-8')
    cells=[];(out/'inputs').mkdir()
    for original in ref['cells']:
        c=original.copy();p=Path(c['save'])
        if sha(p)!=c['save_sha256']:raise ValueError('Reference save changed')
        dest=out/'inputs'/p.name;shutil.copy2(p,dest);c['save']=str(dest)
        psize=degree(c['d'])
        imax=c['b2']//c['d']+2;g=(imax+psize-1)//psize
        c.update(p=psize,points=imax,g=g,rounded_b2=max(c['b2'],(g*psize-2)*c['d']))
        cells.append(c)
    monitor=Monitor(a.device)
    try:device=monitor.identity
    finally:monitor.close()
    helpers={n:sha(ROOT/n) for n in ['tools/bench/bench_edwards_stage2_optimization.py','tools/bench/bench_stage2_production.py','tools/bench/bench_stage2_n_scaling.py','tools/test/test_edwards_stage2_relocation.py']}
    for name in helpers:
        dest=out/'helpers'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(__file__,out/'collector_snapshot.py')
    plan=dict(schema=1,complete=False,created_utc=datetime.now(timezone.utc).isoformat(),binaries=binaries,
        reference=dict(path=str(a.reference.resolve()),sha256=sha(a.reference)),
        gates=dict(path=str(a.gates.resolve()),sha256=sha(a.gates)),collector_sha256=sha(__file__),helpers=helpers,
        ini=str(ini),ini_sha256=sha(ini),device=device,cells=cells,runs=[],repeats=a.repeats,
        protocol=dict(warmups=1,order='alternating pairs; explicit rounding is a separate arm',
            arena_mb=6300,owner_mb=640,batch_mb=256,relocation=0,ntt_counts=1,
            timing='stage2_full_wall total=init+main; startup/Stage1/prechecks excluded'),rounding=a.rounding)
    write(out/'plan.json',plan);print('PREPARED',len(cells),'cells',flush=True)

def verify(p,folder):
    if sha(__file__)!=p['collector_sha256'] or sha(p['ini'])!=p['ini_sha256']:raise ValueError('Collector/INI changed')
    if sha(p['gates']['path'])!=p['gates']['sha256']:raise ValueError('Gates changed')
    for name,h in p['helpers'].items():
        if sha(ROOT/name)!=h or sha(folder/'helpers'/name)!=h:raise ValueError('Helper changed: '+name)
    for binary in p['binaries'].values():
        exe=Path(binary['path'])
        if sha(exe)!=binary['sha256']:raise ValueError('Binary changed')
        for name,h in binary['runtime_files'].items():
            if sha(exe.parent/name)!=h:raise ValueError('Runtime changed')
        for name,h in binary['identity']['sources'].items():
            if sha(exe.parent/'sources'/name)!=h:raise ValueError('Frozen source changed')

def run(a):
    folder=a.study.resolve()
    if (folder/'stop_reason.json').exists():raise ValueError('Selection study was stopped; prepare a new study')
    p=read(folder/'plan.json');verify(p,folder)
    monitor=Monitor(p['device']['index'])
    if monitor.identity!=p['device']:raise ValueError('GPU changed')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')};env['CUDA_LAUNCH_BLOCKING']='0'
    done={r['id'] for r in p['runs']}
    try:
        for repeat in range(p['repeats']+1):
            for ci,cell in enumerate(p['cells']):
                arms=['baseline','candidate']
                if p['rounding'] and cell['id'] in ('m10267_low_normal','m10267_high_normal'):arms+=['rounded']
                if (repeat+ci)%2:arms.reverse()
                for arm in arms:
                    tag=f'{cell["id"]}_{"warmup" if repeat==0 else "timing"}{repeat}_{arm}'
                    if tag in done:continue
                    if sha(cell['save'])!=cell['save_sha256']:raise ValueError('Input changed')
                    out=folder/'runs'/tag
                    if out.exists():raise ValueError('Unrecorded invocation needs inspection: '+str(out))
                    out.mkdir(parents=True)
                    b2=cell['rounded_b2'] if arm=='rounded' else cell['b2']
                    exe=p['binaries']['baseline' if arm=='baseline' else 'candidate']['path']
                    command=list(map(str,[exe,'--ini',p['ini'],'--save',cell['save'],'--b2',b2,'--d',cell['d'],
                        '--device',p['device']['index'],'--curves',1,'--carrier-exponent',0,'--relocation',0,
                        '--arena-mb',6300,'--owner-budget-mb',640,'--batch-mb',256,'--factor-only',
                        '--short-calibration',0,'--tune-runtime',0,'--ntt-counts',1,'--log-level','debug',
                        '--results',out/'internal.jsonl','--submission',out/'results.json.txt',
                        '--log',out/'engine.log','--debug-log-file',out/'debug.log']))
                    write(out/'command.json',dict(command=command,environment=dict(CUDA_LAUNCH_BLOCKING='0')))
                    telemetry=[];start=time.perf_counter()
                    with (out/'driver.log').open('wb') as log:
                        proc=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
                        while proc.poll() is None:
                            telemetry.append(monitor.sample());time.sleep(.2)
                            if time.perf_counter()-start>1800:proc.kill();proc.wait();raise TimeoutError(tag)
                    seconds=time.perf_counter()-start;write(out/'telemetry.json',telemetry)
                    if proc.returncode:raise ValueError(f'{tag}: exit={proc.returncode}')
                    row=json.loads((out/'internal.jsonl').read_text().splitlines()[-1])
                    text=(out/'debug.log').read_text()+(out/'engine.log').read_text()
                    timer=fields(text,'stage2_full_wall');oracle=fields(text,'s4_multiply_stats')
                    if any(int(oracle[k]) for k in ('gmp_check_bad','gmp_selftest_bad')) or not int(oracle['gmp_checked']):raise ValueError('Arithmetic gate failed')
                    if row['relocation'] or row['relocation_replayed'] or row['bad_factors'] or row['B2']!=b2:raise ValueError('Work/result mismatch')
                    for f in row['factors']:
                        if int(row['N_hex'],16)%int(f) or not 1<int(f)<int(row['N_hex'],16):raise ValueError('Invalid factor')
                    if cell['kind']=='NF' and row['factors']:raise ValueError('NF changed; inspect coverage')
                    if cell['kind']=='F' and not any(int(f)%int(cell['known_factor'])==0 for f in row['factors']):raise ValueError('Missing factor')
                    peer=[r for r in p['runs'] if r['cell']==cell['id'] and r['b2']==b2]
                    leaf=fields(text,'descent_values')
                    if any(r['leaf']!=leaf or r['factors']!=sorted(row['factors']) for r in peer):raise ValueError('Complete leaf/factors differ')
                    coverage=fields(text,'real_batched_projective')
                    if coverage['leaves']!=coverage['gamma_points'] or int(row['giant_points'])!=b2//cell['d']+2:raise ValueError('Coverage differs')
                    p['runs'].append(dict(id=tag,cell=cell['id'],arm=arm,b2=b2,warmup=repeat==0,repeat=repeat,
                        timing={k:float(timer[k]) for k in ('shape','init','main','total')},process_seconds=seconds,
                        factors=sorted(row['factors']),leaf=leaf,coverage=coverage,
                        raw=str(out),files={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}))
                    write(folder/'plan.json',p);print('PASS',tag,timer['total'],flush=True)
        verify(p,folder);p['complete']=True;write(folder/'plan.json',p);print('COMPLETE',len(p['runs']),flush=True)
    finally:monitor.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    for n in ('baseline','candidate','reference','gates','output'):prep.add_argument('--'+n,type=Path,required=True)
    prep.add_argument('--device',type=int,default=1);prep.add_argument('--repeats',type=int,default=3)
    prep.add_argument('--rounding',action='store_true')
    runp=sub.add_parser('run');runp.add_argument('--study',type=Path,required=True)
    a=parser.parse_args();(prepare if a.action=='prepare' else run)(a)
if __name__=='__main__':main()
