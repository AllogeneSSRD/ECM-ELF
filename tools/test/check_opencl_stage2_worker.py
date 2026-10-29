"""Real native OpenCL worker/save/queue qualification in an isolated directory."""
import argparse
import ctypes as ct
from ctypes import wintypes as wt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

if hasattr(sys,'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0) # Supported save/result N can be 16384 bits.

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def curve(n,sigma,b1):
    u=(sigma*sigma-5)%n
    v=4*sigma%n
    a24=pow(v-u,3,n)*(3*u+v)*pow(16*pow(u,3,n)*v,-1,n)%n
    base=(pow(u,3,n),pow(v,3,n))
    def twice(p):
        a=(p[0]+p[1])**2%n
        b=(p[0]-p[1])**2%n
        e=(a-b)%n
        return a*b%n,e*(b+a24*e)%n
    def add(p,q):
        a=(q[0]-q[1])*(p[0]+p[1])%n
        b=(q[0]+q[1])*(p[0]-p[1])%n
        return base[1]*(a+b)**2%n,base[0]*(a-b)**2%n
    scalar=math.lcm(*range(1,b1+1))
    r,s=base,twice(base)
    for bit in bin(scalar)[3:]:
        t=add(r,s)
        r,s=(t,twice(s)) if bit=='1' else (twice(r),t)
    return r[0]*pow(r[1],-1,n)%n

def save_line(n,sigma=26,b1=20):
    x=curve(n,sigma,b1)
    return f'METHOD=ECM; PARAM=0; SIGMA={sigma}; B1={b1}; N=0x{n:x}; X=0x{x:x}; Z=1; CHECKSUM={b1*sigma*n*x%4294967291};\n'

def config(folder,device,arena=512):
    folder.mkdir(parents=True,exist_ok=False)
    path=folder/'ecm.toml'
    path.write_text(f'''[shared]
worktodo="worktodo.txt"
[stage2]
stage2_device={device}
stage2_save_dir="."
stage2_log_level="phases"
stage2_arena_mb={arena}
stage2_auto_carrier=false
stage2_tune_runtime=false
stage2_results_file="results.jsonl"
stage2_log_file="worker.log"
stage2_finished="finished.txt"
stage2_progress_dir="."
''',encoding='utf-8')
    return path

def records(folder):
    path=folder/'results.jsonl'
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()] if path.exists() else []

def run(exe,folder,args,label='run',expect=0,environment=None):
    cmd=[str(exe),'--config',str((folder/'ecm.toml').resolve()),*map(str,args)]
    (folder/(label+'.command.json')).write_text(json.dumps(cmd,indent=2),encoding='utf-8')
    with (folder/(label+'.log')).open('w',encoding='utf-8') as log:
        env=os.environ.copy()
        env.update(environment or {})
        p=subprocess.run(cmd,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
    if (expect==0 and p.returncode) or (expect!=0 and not p.returncode):
        raise AssertionError(f'{label}: exit {p.returncode}, expected {expect}; {folder}')
    return (folder/(label+'.log')).read_text(encoding='utf-8')

def stop_after_first_start(exe,folder):
    # Signal the same inherited event used by the GUI, during the first curve.
    # The driver must finish that curve and commit exactly one receipt.
    kernel=ct.WinDLL('kernel32',use_last_error=True)
    kernel.CreateEventW.argtypes=[ct.c_void_p,wt.BOOL,wt.BOOL,wt.LPCWSTR]
    kernel.CreateEventW.restype=wt.HANDLE
    kernel.SetHandleInformation.argtypes=[wt.HANDLE,wt.DWORD,wt.DWORD]
    kernel.SetEvent.argtypes=[wt.HANDLE]
    kernel.CloseHandle.argtypes=[wt.HANDLE]
    event=kernel.CreateEventW(None,True,False,None)
    if not event or not kernel.SetHandleInformation(event,1,1):
        raise ct.WinError(ct.get_last_error())
    startup=subprocess.STARTUPINFO()
    startup.lpAttributeList={'handle_list':[event]}
    cmd=[str(exe),'--config',str((folder/'ecm.toml').resolve()),'--stage2-stop-handle',str(event)]
    lines=[]
    try:
        p=subprocess.Popen(cmd,cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',startupinfo=startup)
        for line in p.stdout:
            lines.append(line)
            if 'curve_start:' in line:
                if not kernel.SetEvent(event):
                    raise ct.WinError(ct.get_last_error())
        code=p.wait(timeout=240)
    finally:
        kernel.CloseHandle(event)
        (folder/'stop.log').write_text(''.join(lines),encoding='utf-8')
    if code or len(records(folder))!=1:
        raise AssertionError('graceful stop must commit one curve')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',type=int,default=1)
    a=parser.parse_args()
    if a.out.exists():
        raise SystemExit('Choose a new evidence directory')
    a.out.mkdir(parents=True)
    shutil.copy2(__file__,a.out/Path(__file__).name)
    frozen=a.out/'bin'
    frozen.mkdir()
    for name in ['ecm_opencl_stage2.exe','gmp-10.dll','build_manifest.json']:
        shutil.copy2(a.build/name,frozen/name)
    shutil.copytree(a.build/'opencl_resources',frozen/'opencl_resources')
    exe=(frozen/'ecm_opencl_stage2.exe').resolve()
    tests=[]
    start=time.perf_counter()
    def case(name,line,b2=300,d=30,extra=(),expect=0,arena=512,environment=None):
        folder=a.out/name
        config(folder,a.device,arena)
        (folder/'input.save').write_text(line,encoding='utf-8')
        log=run(exe,folder,['--save',str((folder/'input.save').resolve()),'--b2',b2,'--d',d,'--curves',1,'--factor-only','--ntt-counts',1,*extra],expect=expect,environment=environment)
        rows=records(folder)
        if expect:
            assert not rows,'failure published result'
        else:
            assert len(rows)==1
            assert rows[0]['stage2-engine']['name']=='opencl-ntt'
        tests.append(name)
        return rows,log
    rows,_=case('stage2_hit',save_line(257000771,6))
    assert rows[0].get('factors')==['257']
    # A factor may first appear while constructing [D]Q. Rebasing must publish
    # that proper factor, rather than dropping it or inventing an affine root.
    rows,log=case('giant_seed_hit',save_line(257000771,6),d=46)
    assert rows[0].get('factors')==['257']
    assert 'Giant points, G trees and fold' in log and 'Normalize fold and descend' not in log
    prime=(1<<127)-1
    line=save_line(prime)
    ordinary,_=case('no_factor',line)
    # Widen point generation over several G batches, with a final partial
    # window and a point chunk not divisible by the G degree. NTT work and
    # independent point-check coverage must remain identical to the baseline.
    case('point_window_reference',line,b2=960,
        environment={'ECM_OPENCL_POINT_WINDOW':'0','ECM_OPENCL_POINT_CHUNK':'7'})
    case('point_window_partial',line,b2=960,
        environment={'ECM_OPENCL_POINT_WINDOW':'18','ECM_OPENCL_POINT_CHUNK':'7'})
    # Diagnostic summaries belong to the file log, not the readable console.
    reference=(a.out/'point_window_reference/worker.log').read_text(encoding='utf-8-sig')
    window=(a.out/'point_window_partial/worker.log').read_text(encoding='utf-8-sig')
    def statistic(text,prefix):
        return next(s for s in text.splitlines() if s.startswith(prefix+':'))
    assert statistic(reference,'ntt_lengths')==statistic(window,'ntt_lengths')
    assert statistic(reference,'opencl_checks')==statistic(window,'opencl_checks')
    assert 'requested=18 effective=16' in window
    case('point_window_budget_cap',line,b2=960,
        environment={'ECM_OPENCL_POINT_WINDOW':'65536','ECM_OPENCL_POINT_CHUNK':'7','NTT_S4_BATCH_MB':'1'})
    capped=(a.out/'point_window_budget_cap/worker.log').read_text(encoding='utf-8-sig')
    values=dict(re.findall(r'(requested|effective|cap|G_batch)=(\d+)',statistic(capped,'opencl_point_window')))
    assert int(values['effective'])<=int(values['cap']) and int(values['effective'])%int(values['G_batch'])==0
    assert statistic(reference,'ntt_lengths')==statistic(capped,'ntt_lengths')
    case('invalid_point_window',line,expect=2,environment={'ECM_OPENCL_POINT_WINDOW':'65537'})
    # Real wider-than-512-bit points exercise the adaptive call site, not only
    # its arithmetic policy. The same G batches must retain all their checks.
    adaptive_line=save_line((1<<607)-1)
    case('point_auto_reference',adaptive_line,b2=2600000,d=2310,
        environment={'ECM_OPENCL_POINT_AUTO':'0'})
    case('point_auto',adaptive_line,b2=2600000,d=2310,
        environment={'ECM_OPENCL_POINT_AUTO':'1'})
    ref_auto=(a.out/'point_auto_reference/worker.log').read_text(encoding='utf-8-sig')
    auto=(a.out/'point_auto/worker.log').read_text(encoding='utf-8-sig')
    assert statistic(ref_auto,'ntt_lengths')==statistic(auto,'ntt_lengths')
    assert statistic(ref_auto,'opencl_checks')==statistic(auto,'opencl_checks')
    eligible='name=NVIDIA' in auto and 'point_backend=opencl' in auto
    assert ('enabled=1' if eligible else 'enabled=0') in statistic(auto,'opencl_point_plan')
    case('point_auto_carrier',adaptive_line,b2=2600000,d=2310,
        extra=['--carrier-exponent',1214],environment={'ECM_OPENCL_POINT_AUTO':'1'})
    case('point_auto_manual_lock',adaptive_line,b2=2600000,d=2310,
        environment={'ECM_OPENCL_POINT_AUTO':'1','ECM_OPENCL_POINT_CHUNK':'64'})
    locked=(a.out/'point_auto_manual_lock/worker.log').read_text(encoding='utf-8-sig')
    assert 'enabled=0' in statistic(locked,'opencl_point_plan')
    case('invalid_point_auto',line,expect=2,environment={'ECM_OPENCL_POINT_AUTO':'2'})
    lifted,_=case('mersenne',line,extra=['--carrier-exponent',127])
    assert ordinary[0].get('factors',[])==lifted[0].get('factors',[])==[]
    target=((1<<41)-1)//13367
    cofactor=save_line(target,6)
    case('cofactor_ordinary',cofactor)
    case('cofactor_carrier',cofactor,extra=['--carrier-exponent',41])
    case('invalid_carrier',line,extra=['--carrier-exponent',128],expect=2)
    case('budget_failure',line,expect=2,arena=1)
    case('bad_checksum',line.replace('CHECKSUM=','CHECKSUM=1'),expect=2)
    case('basic_d_planner',line,d=0)
    case('unsupported_auto_carrier',line,extra=['--auto-carrier',1],expect=2)
    failure=a.out/'queue_budget_failure'
    config(failure,a.device,arena=1)
    (failure/'input.save').write_text(line,encoding='utf-8')
    queue_line='ECMSTAGE2=1,2,127,-1,"input.save",300,0,1\n'
    (failure/'worktodo.txt').write_text(queue_line,encoding='utf-8')
    run(exe,failure,['--once'],'failed_curve',expect=2)
    assert not records(failure) and (failure/'worktodo.txt').read_text(encoding='utf-8')==queue_line
    tests.append('queue_failure_does_not_advance')
    plan=a.out/'plan_only'
    config(plan,a.device)
    (plan/'input.save').write_text(line,encoding='utf-8')
    log=run(exe,plan,['--save',str((plan/'input.save').resolve()),'--b2',300,'--d',30,'--plan-only'])
    assert not records(plan) and 'estimated_payload_bytes' in log
    tests.append('plan_only')
    queue=a.out/'queue_resume'
    config(queue,a.device)
    (queue/'a.save').write_text(line*3,encoding='utf-8')
    (queue/'b.save').write_text(line,encoding='utf-8')
    first='ECMSTAGE2=1,2,127,-1,"a.save",300,0,5'
    second='ECMSTAGE2=1,2,127,-1,"b.save",400,0,1'
    (queue/'worktodo.txt').write_text(first+'\n'+second+'\n',encoding='utf-8')
    # Pin D in TOML so --once and an inherited stop retain identical identity.
    with (queue/'ecm.toml').open('a',encoding='utf-8') as f:f.write('stage2_d=30\n')
    stop_after_first_start(exe,queue)
    (queue/'worktodo.txt').write_text(second+'\n'+first+'\n',encoding='utf-8')
    run(exe,queue,['--once'],'reordered_first')
    run(exe,queue,['--once'],'resume_a')
    rows=records(queue)
    assert len(rows)==4
    arows=[r for r in rows if r['stage2-engine']['save'].endswith('a.save')]
    assert sorted(r['stage2-engine']['record'] for r in arows)==[1,2,3]
    tests.append('queue_stop_reorder_resume')
    out={'status':'passed','tests':tests,'seconds':time.perf_counter()-start,
         'device':a.device,'binary_sha256':sha(exe),'files':{str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.is_file()}}
    (a.out/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','tests':len(tests),'seconds':out['seconds']},ensure_ascii=False))

if __name__=='__main__':
    main()
