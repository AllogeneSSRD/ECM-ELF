"""Same-binary GPU1 ABBA diagnosis of process scheduling and oracle waits.

Uses verified sigma26 saves. Only owned process trees are queried; only the
launched process's priority is set through its creation flags. Production queues
and device settings are not changed. Samples are partial process observations,
not exact CPU attribution to an engine phase or a complete resource peak.
"""
import argparse
import ctypes
from ctypes import wintypes as w
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from bench_stage2_concurrency import OwnMemory,Counters
from diagnose_ecm_cost_drift import State
from bench_stage2_budget_scaling import fields
from calibrate_stage2_d import parse
from ecm_cost_model import features


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OwnProcessStats(OwnMemory):
    def __init__(self):
        super().__init__()
        self.k.GetPriorityClass.argtypes=[w.HANDLE];self.k.GetPriorityClass.restype=w.DWORD
        self.k.GetProcessTimes.argtypes=[w.HANDLE]+[ctypes.POINTER(w.FILETIME)]*4
        self.k.GetProcessTimes.restype=w.BOOL
        self.k.QueryProcessCycleTime.argtypes=[w.HANDLE,ctypes.POINTER(ctypes.c_ulonglong)]
        self.k.QueryProcessCycleTime.restype=w.BOOL

    def sample(self,roots):
        result=super().sample(roots)
        def ticks(f):return (f.dwHighDateTime<<32)|f.dwLowDateTime
        for row in result['processes']:
            handle=self.k.OpenProcess(0x410,False,row['pid'])
            if not handle:continue
            try:
                row['priority_class']=self.k.GetPriorityClass(handle)
                creation,exit,kernel,user=[w.FILETIME() for _ in range(4)]
                if self.k.GetProcessTimes(handle,*[ctypes.byref(f) for f in (creation,exit,kernel,user)]):
                    row.update(creation_100ns=ticks(creation),kernel_100ns=ticks(kernel),user_100ns=ticks(user))
                cycles=ctypes.c_ulonglong()
                if self.k.QueryProcessCycleTime(handle,ctypes.byref(cycles)):row['cycles']=cycles.value
                m=Counters();m.cb=ctypes.sizeof(m)
                if self.ps.GetProcessMemoryInfo(handle,ctypes.byref(m),m.cb):row['page_fault_count']=m.PageFaultCount
            finally:self.k.CloseHandle(handle)
        return result

    def completed(self,child):
        # Popen retains the process handle after communicate; PID reuse cannot
        # make this query observe another process. Windows reports CPU time in
        # 100ns units with scheduler accounting granularity.
        handle=w.HANDLE(int(child._handle))
        values=[w.FILETIME() for _ in range(4)]
        if not self.k.GetProcessTimes(handle,*[ctypes.byref(f) for f in values]):
            raise OSError(ctypes.get_last_error(),'Completed process times unavailable')
        ticks=[(f.dwHighDateTime<<32)|f.dwLowDateTime for f in values]
        if not ticks[1]:raise ValueError('Process has not exited')
        cycles=ctypes.c_ulonglong();has_cycles=self.k.QueryProcessCycleTime(handle,ctypes.byref(cycles))
        return dict(pid=child.pid,creation_100ns=ticks[0],exit_100ns=ticks[1],
            lifecycle_wall_seconds=(ticks[1]-ticks[0])/1e7,kernel_seconds=ticks[2]/1e7,user_seconds=ticks[3]/1e7,
            cpu_seconds=(ticks[2]+ticks[3])/1e7,cycles=cycles.value if has_cycles else None)


def process_intervals(samples):
    grouped={}
    for sample in samples:
        for row in sample['own']['processes']:
            grouped.setdefault((row['pid'],row.get('creation_100ns')),[]).append((sample['monotonic'],row))
    out=[]
    for (pid,creation),rows in grouped.items():
        first,last=rows[0][1],rows[-1][1]
        deltas={key:last[key]-first[key] for key in ('user_100ns','kernel_100ns','cycles','page_fault_count') if key in first and key in last}
        out.append(dict(pid=pid,creation_100ns=creation,observations=len(rows),sampled_wall_seconds=rows[-1][0]-rows[0][0],
            observed_priority_classes=sorted({r['priority_class'] for _,r in rows if r.get('priority_class')}),
            sampled_deltas=deltas))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bits',type=int,nargs='+',default=[8191])
    p.add_argument('--d',type=int,default=120120);p.add_argument('--b2',type=int,nargs='+',default=[1383422040,5250000000])
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--variant',choices=('priority','oracle','cuda_wait','replay'),default='priority')
    p.add_argument('--tail-seconds',type=float,help='Explicit diagnostic full-wall threshold for identical replay runs; never a cost release gate')
    p.add_argument('--replay-order',choices=('blocked','interleaved'),help='Replay defaults to alternating input shapes each repetition')
    p.add_argument('--baseline-measurements',type=Path,help='Completed identical replay for fixed per-shape tail thresholds')
    p.add_argument('--tail-ratio',type=float,help='Tail threshold = frozen baseline median times this ratio; diagnostic only')
    p.add_argument('--compare-entry',action='store_true',help='Explicitly permit driver/worker entry as the sole changed baseline control')
    p.add_argument('--wait-mode',type=int,choices=(2,4),default=4,help='cuda_wait compares Auto with Yield(2) or BlockingSync(4)')
    p.add_argument('--entry',choices=('driver','worker'),default='worker',
        help='Direct worker minimises the protocol; driver priority is not inherited by its child on this host')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh directory')
    if a.repeats<1 or not set(a.bits)<={2203,4423,8191}:raise ValueError('Invalid corpus')
    if a.tail_seconds is not None and (a.variant!='replay' or not math.isfinite(a.tail_seconds) or a.tail_seconds<=0):
        raise ValueError('A positive finite tail threshold requires --variant replay')
    if a.replay_order is not None and a.variant!='replay':raise ValueError('Replay order requires --variant replay')
    if bool(a.baseline_measurements)!=(a.tail_ratio is not None):raise ValueError('Use baseline measurements and tail ratio together')
    if a.compare_entry and not a.baseline_measurements:raise ValueError('Entry comparison requires a frozen baseline')
    if a.tail_ratio is not None and (a.variant!='replay' or a.tail_seconds is not None or not math.isfinite(a.tail_ratio) or a.tail_ratio<=1):
        raise ValueError('A finite tail ratio >1 requires replay and conflicts with tail seconds')
    exe=a.exe.resolve();study=json.loads(a.study.read_text(encoding='utf-8'))
    build=json.loads((exe.parent/'frozen_sources_manifest.json').read_text(encoding='utf-8'))
    if not study['complete'] or build['binary_sha256']!=sha(exe):raise ValueError('Unverified study/build')
    saves={bits:Path(study['saves'][str(bits)]['path']) for bits in a.bits}
    names=('bench_stage2_host_wait.py','bench_stage2_concurrency.py','diagnose_ecm_cost_drift.py',
        'bench_stage2_budget_scaling.py','calibrate_stage2_d.py','ecm_cost_model.py')
    identity=dict(binary_sha256=sha(exe),calibration_binary_sha256=study['identity']['stage2_sha256'],study_sha256=sha(a.study),tools={n:sha(Path(__file__).with_name(n)) for n in names},
        save_sha256={str(bits):study['saves'][str(bits)]['sha256'] for bits in a.bits})
    thresholds={}
    if a.baseline_measurements:
        baseline=json.loads(a.baseline_measurements.read_text(encoding='utf-8'))
        if (not baseline.get('complete') or baseline['variant']!='replay' or
            baseline['identity']['binary_sha256']!=identity['binary_sha256'] or
            baseline['identity']['study_sha256']!=identity['study_sha256'] or
            any(baseline['controls'][k]!=v for k,v in dict(D=a.d,arena_mb=4096,owner_mb=0,chain_min=8192).items()) or
            (baseline['controls']['entry']!=a.entry and not a.compare_entry)):
            raise ValueError('Baseline must match the actual replay input/execution contract')
        identity['baseline_sha256']=sha(a.baseline_measurements)
        for bits in a.bits:
            for b2 in a.b2:
                rows=[r for r in baseline['runs'] if (r['case']['bits'],r['case']['B2'])==(bits,b2)]
                if not rows:raise ValueError('Baseline is missing a requested shape')
                for r in rows:
                    if sha(r['log'])!=r['log_sha256']:raise ValueError('Baseline log changed')
                thresholds[bits,b2]=statistics.median(r['phases']['full'] for r in rows)*a.tail_ratio
    def verify():
        if sha(exe)!=identity['binary_sha256'] or sha(a.study)!=identity['study_sha256']:raise ValueError('Input changed')
        if a.baseline_measurements and sha(a.baseline_measurements)!=identity['baseline_sha256']:raise ValueError('Frozen baseline changed')
        for bits,saved in saves.items():
            if sha(saved)!=identity['save_sha256'][str(bits)]:raise ValueError('Verified save changed')
        for n,digest in identity['tools'].items():
            if sha(Path(__file__).with_name(n))!=digest:raise ValueError('Tool changed')
        for n,digest in build['sources'].items():
            if sha(exe.parent/'sources'/n)!=digest.lower():raise ValueError('Frozen source changed')
    verify();memory=OwnProcessStats();state=State()
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_D_MODEL='0',NTT_STAGE1_Q_DUMP='1',NTT_CARRY_CHECK_FUSED='0',
        NTT_POINT_MERSENNE='1',NTT_GIANT_CHAIN_MIN='8192',NTT_FOLD_DEVICE_MAX_MB='0',NTT_ARENA_CAP_KB=str(4096*1024),CUDA_LAUNCH_BLOCKING='0')
    info=json.loads(subprocess.run([str(exe),'--cost-device-info','--device','1'],env=env,capture_output=True,check=True,timeout=60).stdout)
    if info['uuid_hex']!='8a67b1f8ef1c3177a822813a7ac2224d':raise ValueError('Wrong device')
    # Both oracle configurations preserve the same selected coefficient checks.
    # Priority trials retain the native async default through an explicit value.
    cases=[dict(bits=bits,B2=b2,rep=rep,value=value) for bits in a.bits for b2 in a.b2
        for rep in range(a.repeats) for value in ((0,) if a.variant=='replay' else (0,1,1,0))]
    if a.variant=='replay' and a.replay_order!='blocked':
        cases=[dict(bits=bits,B2=b2,rep=rep,value=0) for rep in range(a.repeats) for bits in a.bits for b2 in a.b2]
    data=dict(schema=1,kind='host_wait_diagnostic',identity=identity,device=info,variant=a.variant,
        controls=dict(D=a.d,arena_mb=4096,owner_mb=0,chain_min=8192,entry=a.entry,sequence=[0] if a.variant=='replay' else [0,1,1,0],repeats=a.repeats,wait_mode=a.wait_mode,tail_seconds=a.tail_seconds,
            replay_order=(a.replay_order or 'interleaved') if a.variant=='replay' else None,tail_ratio=a.tail_ratio,
            baseline_measurements=str(a.baseline_measurements.resolve()) if a.baseline_measurements else None,
            entry_is_controlled_variable=a.compare_entry,
            frozen_shape_thresholds=[dict(bits=k[0],B2=k[1],seconds=v) for k,v in thresholds.items()]),cases=cases,runs=[])
    dest=out/'measurements.json'
    def persist():dest.write_text(json.dumps(data,indent=2),encoding='utf-8')
    persist();reference={};ini=out/'manual.ini';ini.write_text('[gpu]\ndevice=1\n',encoding='utf-8')
    for index,c in enumerate(cases):
        name=f'{index:03d}_m{c["bits"]}_b{c["B2"]}_{a.variant}{c["value"]}'
        log=out/(name+'.log');result=out/(name+'.jsonl');samples=[];errors=[];stop=threading.Event()
        priority=0x8000 if a.variant=='priority' and c['value'] else 0x20
        async_oracle=c['value'] if a.variant=='oracle' else 1
        cmd=[str(exe),'--ini',str(ini),'--save',str(saves[c['bits']]),'--device','1','--b2',str(c['B2']),
            '--d',str(a.d),'--arena-mb','4096','--factor-only','--log',str(log),'--results',str(result)]
        if a.entry=='worker':
            line=saves[c['bits']].read_bytes().partition(b'\n')[0]
            if not line.startswith(b'METHOD=ECM;'):raise ValueError('Need the verified first raw record')
            fingerprint=14695981039346656037
            for byte in line:fingerprint=((fingerprint^byte)*1099511628211)&((1<<64)-1)
            cmd+=['--curve-worker','--record-offset','0','--record-index','1','--record-hash',str(fingerprint)]
        engine=log.open('wb') if a.entry=='worker' else None
        curve_env=env|{'NTT_S4_ORACLE_ASYNC':str(async_oracle)}
        if a.variant=='cuda_wait':curve_env['NTT_CUDA_WAIT_MODE']=str(a.wait_mode if c['value'] else 0)
        start=time.perf_counter();child=subprocess.Popen(cmd,env=curve_env,
            stdout=engine if engine else subprocess.PIPE,stderr=subprocess.STDOUT if engine else subprocess.PIPE,creationflags=priority)
        def observe():
            try:
                while not stop.is_set():
                    sample=state.sample();sample['own']=memory.sample([child.pid]);samples.append(sample);stop.wait(.2)
            except Exception as e:errors.append(str(e))
        sampler=threading.Thread(target=observe);sampler.start()
        try:stdout,stderr=child.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.communicate(timeout=30)
            raise RuntimeError('Owned diagnostic tree timed out')
        finally:
            elapsed=time.perf_counter()-start;stop.set();sampler.join()
            if engine:engine.close()
        (out/(name+'_driver.log')).write_bytes((stdout or b'')+(stderr or b''))
        if child.returncode or errors:raise RuntimeError('Curve or sampler failed: '+str(errors))
        cpu=memory.completed(child)
        text=log.read_text(encoding='utf-8');r=json.loads(result.read_text(encoding='utf-8'))
        if not all(t in text for t in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')):raise ValueError('Missing arithmetic checks')
        n=(1<<c['bits'])-1
        if r['B1']!=1000 or r['B2']!=c['B2'] or r['sigma']!=26 or int(r['N_hex'],16)!=n or r['bad_factors']:
            raise ValueError('Actual input/output differs')
        if any(not 1<int(f)<n or n%int(f) for f in r['factors']):raise ValueError('Invalid factor')
        f=features(a.d,c['B2'],c['bits'],8192);gt=fields(text,'real_batched_gdevice');oracle=fields(text,'s4_oracle_stats')
        if [int(gt[k]) for k in ('pairs','groups','copies')]!=[f[k] for k in ('g_tree_pairs','gtrees_groups','g_tree_copies')]:raise ValueError('Tree route differs')
        if int(oracle['async'])!=async_oracle or oracle['selected']!=oracle['compared']:raise ValueError('Oracle route/check coverage differs')
        wait=fields(text,'stage2_cuda_wait') if a.variant=='cuda_wait' else None
        if wait and (int(wait['device'])!=1 or int(wait['requested'])!=(a.wait_mode if c['value'] else 0) or int(wait['after'])&7!=int(wait['requested'])):
            raise ValueError('CUDA wait context differs')
        leaf=fields(text,'descent_values')['hash'];key=(c['bits'],c['B2']);value=(leaf,r['factors'],oracle['signature'],oracle['samples'])
        if key in reference and reference[key]!=value:raise ValueError('Output/check selection changed')
        reference[key]=value;intervals=process_intervals(samples)
        classes={v for row in intervals for v in row['observed_priority_classes']}
        if classes!={priority}:raise ValueError('Driver/worker priority did not match requested class: '+str(classes))
        verify();phases=parse(text)
        data['runs'].append(dict(name=name,case=c,command=cmd,driver_pid=child.pid,worker_pid=child.pid if a.entry=='worker' else r.get('process_id'),
            process_seconds=elapsed,completed_process=cpu,phases=phases,leaf_hash=leaf,factors=r['factors'],oracle=oracle,cuda_wait=wait,
            carry=fields(text,'real_batched_carrytime'),multiply=fields(text,'s4_multiply_stats'),
            log=str(log),log_sha256=sha(log),samples=samples,process_intervals=intervals))
        persist();print(name,'full',phases['full'],'G',phases['gtrees'],'descent',phases['descent'],
            'GMP',oracle['t_gmp'],'priority',sorted(classes),flush=True)
    data['comparisons']=[]
    for bits in a.bits:
        for b2 in a.b2:
            rows=[r for r in data['runs'] if (r['case']['bits'],r['case']['B2'])==(bits,b2)]
            if a.variant=='replay':
                threshold=thresholds.get((bits,b2),a.tail_seconds)
                data['comparisons'].append(dict(bits=bits,B2=b2,identical_configurations=True,
                    samples=len(rows),median_full_seconds=statistics.median(r['phases']['full'] for r in rows),
                    full_range=[min(r['phases']['full'] for r in rows),max(r['phases']['full'] for r in rows)],
                    tail_seconds=threshold,
                    tail_runs=[r['name'] for r in rows if threshold is not None and r['phases']['full']>threshold]))
                continue
            med={str(v):statistics.median(r['phases']['full'] for r in rows if r['case']['value']==v) for v in (0,1)}
            data['comparisons'].append(dict(bits=bits,B2=b2,median_full_seconds=med,change_percent=100*(med['1']/med['0']-1)))
    data['complete']=True;persist();print(json.dumps(data['comparisons'],indent=2))


if __name__=='__main__':main()
