"""GPU1 serial/two-worker ABBA trial with sampled own-process RAM and VRAM.

This bounded experiment uses two independently verified Stage1 points, D30030,
and a 512MiB arena. The conservative trial admission is not a production lease
or a proof of whole-process peak memory. Production queues are not touched.
"""
import argparse
import ctypes
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
from diagnose_ecm_cost_drift import State
from calibrate_stage2_d import parse
from bench_stage2_budget_scaling import fields


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProcessEntry(ctypes.Structure):
    _fields_=[('dwSize',w.DWORD),('cntUsage',w.DWORD),('th32ProcessID',w.DWORD),
              ('th32DefaultHeapID',ctypes.c_size_t),('th32ModuleID',w.DWORD),
              ('cntThreads',w.DWORD),('th32ParentProcessID',w.DWORD),
              ('pcPriClassBase',w.LONG),('dwFlags',w.DWORD),('szExeFile',w.WCHAR*260)]


class Counters(ctypes.Structure):
    _fields_=[('cb',w.DWORD),('PageFaultCount',w.DWORD)]+[(n,ctypes.c_size_t) for n in (
        'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
        'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]


class MemoryStatus(ctypes.Structure):
    _fields_=[('dwLength',w.DWORD),('dwMemoryLoad',w.DWORD)]+[(n,ctypes.c_ulonglong) for n in (
        'ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile',
        'ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]


class OwnMemory:
    def __init__(self):
        self.k=ctypes.WinDLL('kernel32',use_last_error=True);self.ps=ctypes.WinDLL('psapi',use_last_error=True)
        self.k.CreateToolhelp32Snapshot.argtypes=[w.DWORD,w.DWORD];self.k.CreateToolhelp32Snapshot.restype=w.HANDLE
        for name in ('Process32FirstW','Process32NextW'):
            fn=getattr(self.k,name);fn.argtypes=[w.HANDLE,ctypes.POINTER(ProcessEntry)];fn.restype=w.BOOL
        self.k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];self.k.OpenProcess.restype=w.HANDLE
        self.k.CloseHandle.argtypes=[w.HANDLE];self.k.CloseHandle.restype=w.BOOL
        self.ps.GetProcessMemoryInfo.argtypes=[w.HANDLE,ctypes.POINTER(Counters),w.DWORD];self.ps.GetProcessMemoryInfo.restype=w.BOOL
        self.k.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(MemoryStatus)];self.k.GlobalMemoryStatusEx.restype=w.BOOL
    def available(self):
        m=MemoryStatus();m.dwLength=ctypes.sizeof(m)
        if not self.k.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError(ctypes.get_last_error(),'RAM query failed')
        return m.ullAvailPhys
    def sample(self,roots):
        h=self.k.CreateToolhelp32Snapshot(2,0)
        if h==ctypes.c_void_p(-1).value:raise OSError(ctypes.get_last_error(),'Process tree snapshot failed')
        parents={};entry=ProcessEntry();entry.dwSize=ctypes.sizeof(entry)
        try:
            ok=self.k.Process32FirstW(h,ctypes.byref(entry))
            while ok:
                parents[entry.th32ProcessID]=entry.th32ParentProcessID
                ok=self.k.Process32NextW(h,ctypes.byref(entry))
        finally:self.k.CloseHandle(h)
        own=set(roots)
        while True:
            extra={pid for pid,parent in parents.items() if parent in own}
            if extra<=own:break
            own|=extra
        rows=[]
        for pid in sorted(own):
            handle=self.k.OpenProcess(0x410,False,pid)
            if not handle:continue # A process may exit between snapshot and open.
            try:
                m=Counters();m.cb=ctypes.sizeof(m)
                if self.ps.GetProcessMemoryInfo(handle,ctypes.byref(m),m.cb):
                    rows.append(dict(pid=pid,private_bytes=m.PrivateUsage,working_set_bytes=m.WorkingSetSize))
            finally:self.k.CloseHandle(handle)
        return dict(processes=rows,private_bytes=sum(r['private_bytes'] for r in rows),
                    working_set_bytes=sum(r['working_set_bytes'] for r in rows),available_ram_bytes=self.available())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bits',type=int,default=8191)
    p.add_argument('--b2',type=int,default=5250000000);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--vram-budget-mb',type=int,default=4096);p.add_argument('--host-budget-mb',type=int,default=4096)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    if a.bits not in (2203,4423,8191) or not 3000000000<=a.b2<=6000000000 or a.repeats<1:raise ValueError('Outside this bounded trial matrix')
    exe=a.exe.resolve();binary=sha(exe);study=json.loads(a.study.read_text(encoding='utf-8'))
    build=json.loads((exe.parent/'frozen_sources_manifest.json').read_text(encoding='utf-8'))
    if binary!=build['binary_sha256'] or not study['complete']:raise ValueError('Candidate/input identity mismatch')
    row=next(r for r in study['stage1'] if r['bits']==a.bits and r['batch']==12 and not r['warmup'])
    batch=Path(row['command'][row['command'].index('-save')+1]);raw=batch.read_bytes()
    if sha(batch)!=row['save_sha256']:raise ValueError('Verified Stage1 corpus changed')
    records=[line for line in raw.splitlines() if b'METHOD=ECM;' in line][:2]
    if len(records)!=2:raise ValueError('Need two independently verified points')
    saves=[]
    for i,line in enumerate(records):
        save=out/f'curve_{i}.save';save.write_bytes(line+b'\n');saves.append(save)
    helpers=('bench_stage2_concurrency.py','diagnose_ecm_cost_drift.py','bench_stage2_budget_scaling.py','calibrate_stage2_d.py')
    identity=dict(binary_sha256=binary,study_sha256=sha(a.study),batch_save=str(batch),batch_save_sha256=sha(batch),
        save_sha256=[sha(s) for s in saves],tools={n:sha(Path(__file__).with_name(n)) for n in helpers},sources=build['sources'])
    def verify():
        if sha(exe)!=binary or sha(batch)!=identity['batch_save_sha256'] or sha(a.study)!=identity['study_sha256']:raise ValueError('Input changed')
        if [sha(s) for s in saves]!=identity['save_sha256']:raise ValueError('Point changed')
        for name,digest in build['sources'].items():
            if sha(exe.parent/'sources'/name)!=digest.lower():raise ValueError('Frozen source changed')
        for name,digest in identity['tools'].items():
            if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Tool changed')
    verify();memory=OwnMemory()
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_D_MODEL='0',NTT_STAGE1_Q_DUMP='1',NTT_CARRY_CHECK_FUSED='0',
               NTT_POINT_MERSENNE='1',NTT_GIANT_CHAIN_MIN='8192',NTT_FOLD_DEVICE_MAX_MB='0',CUDA_LAUNCH_BLOCKING='0')
    query=subprocess.run([str(exe),'--cost-device-info','--device','1'],capture_output=True,env=env,check=True,timeout=60)
    device=json.loads(query.stdout)
    if device['uuid_hex']!='8a67b1f8ef1c3177a822813a7ac2224d' or device['fixed_mode']!=3:raise ValueError('Wrong device/backend')
    # Trial safeguards reserve 2GiB VRAM and RAM per active worker for these
    # measured small-D shapes, and leave 768MiB VRAM / 2GiB available RAM outside.
    vram_available=max(0,device['free_bytes']-(768<<20));ram_available=max(0,memory.available()-(2<<30))
    admission=dict(vram_available_bytes=vram_available,ram_available_bytes=ram_available,
        per_worker_vram_allowance_bytes=2<<30,per_worker_ram_allowance_bytes=2<<30,
        vram_budget_bytes=a.vram_budget_mb<<20,host_budget_bytes=a.host_budget_mb<<20,
        process_peak_guaranteed=False,scope='Empirical trial safeguard for D30030/three measured widths, not an engine allocation lease')
    (out/'admission.json').write_text(json.dumps(admission,indent=2),encoding='utf-8')
    if min(vram_available,admission['vram_budget_bytes'])<4<<30 or min(ram_available,admission['host_budget_bytes'])<4<<30:
        raise ValueError('Two-worker trial exceeds the conservative memory allowance; no curves launched')
    ini=out/'manual.ini';ini.write_text('[gpu]\ndevice=1\n',encoding='utf-8')
    data=dict(schema=1,kind='same_corpus_serial_parallel_abba',identity=identity,device=device,admission=admission,
        controls=dict(bits=a.bits,B2=a.b2,D=30030,arena_mb=512,owner_mb=0,chain_min=8192,
                      sigmas=[int(re.search(rb'SIGMA=([^;]+)',r)[1]) for r in records],sequence=[1,2,2,1],repeats=a.repeats),batches=[])
    dest=out/'measurements.json'
    def persist():dest.write_text(json.dumps(data,indent=2),encoding='utf-8')
    persist();state=State();reference={}
    for rep in range(a.repeats):
        for pos,parallel in enumerate((1,2,2,1)):
            name=f'r{rep}_p{pos}_workers{parallel}';roots=[];children=[];files=[];samples=[];errors=[];stop=threading.Event()
            def observe():
                try:
                    while not stop.is_set():
                        s=state.sample();s['own_memory']=memory.sample(list(roots));samples.append(s);stop.wait(0.2)
                except Exception as e:errors.append(str(e))
            sampler=threading.Thread(target=observe);sampler.start();start=time.perf_counter();curve_rows=[]
            try:
                for i,save in enumerate(saves):
                    tag=name+f'_curve{i}';log=out/(tag+'.log');result=out/(tag+'.jsonl');driver=out/(tag+'_driver.log')
                    cmd=[str(exe),'--ini',str(ini),'--save',str(save),'--device','1','--b2',str(a.b2),
                         '--d','30030','--arena-mb','512','--factor-only','--results',str(result),'--log',str(log)]
                    f=driver.open('wb');files.append(f);child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
                    roots.append(child.pid);children.append(child);curve_rows.append(dict(index=i,command=cmd,log=str(log),result=str(result)))
                    if parallel==1:child.wait(timeout=300)
                for child in children:child.wait(timeout=300)
                elapsed=time.perf_counter()-start
            except BaseException:
                for child in children:
                    if child.poll() is None:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                raise
            finally:
                stop.set();sampler.join()
                for f in files:f.close()
            if errors:raise ValueError('Memory/state monitor failed: '+str(errors))
            for child,r in zip(children,curve_rows):
                if child.returncode:raise RuntimeError('Curve failed; own batch retained')
                text=Path(r['log']).read_text(encoding='utf-8');result=json.loads(Path(r['result']).read_text(encoding='utf-8'));n=(1<<a.bits)-1
                if (result['sigma']!=data['controls']['sigmas'][r['index']] or result['B1']!=1000 or
                    result['B2']!=a.b2 or int(result['N_hex'],16)!=n):raise ValueError('Actual input differs from verified corpus')
                if result['bad_factors'] or any(not 1<int(f)<n or n%int(f) for f in result['factors']):raise ValueError('Invalid factor')
                if not all(t in text for t in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')):raise ValueError('Missing arithmetic checks')
                if int(fields(text,'real_batched_breakdown')['arena_overflow']):raise ValueError('Arena fallback changed the comparison')
                leaf=fields(text,'descent_values')['hash'];key=(leaf,result['factors'])
                if r['index'] in reference and reference[r['index']]!=key:raise ValueError('Concurrency changed output')
                reference[r['index']]=key;r.update(leaf_hash=leaf,factors=result['factors'],phases=parse(text),log_sha256=sha(r['log']),
                                                 workspace=fields(text,'ntt_workspace_stats'))
            verify()
            if not samples or not any(len(s['own_memory']['processes'])>=2*parallel for s in samples):raise ValueError('Own worker RAM was not observed')
            gpu_peak=max(s['used'] for s in samples);ram_peak=max(s['own_memory']['private_bytes'] for s in samples)
            working_peak=max(s['own_memory']['working_set_bytes'] for s in samples)
            data['batches'].append(dict(name=name,rep=rep,workers=parallel,curves=curve_rows,process_seconds=elapsed,
                curves_per_second=2/elapsed,sampled_gpu_peak_bytes=gpu_peak,sampled_own_private_peak_bytes=ram_peak,
                sampled_own_working_set_peak_bytes=working_peak,
                samples=samples,monitor_errors=errors));persist()
            print(name,'wall',round(elapsed,6),'curves/s',round(2/elapsed,6),'GPU MiB',round(gpu_peak/(1<<20),2),
                  'own private MiB',round(ram_peak/(1<<20),2),flush=True)
    med={str(k):statistics.median(r['curves_per_second'] for r in data['batches'] if r['workers']==k) for k in (1,2)}
    data['summary']=dict(median_curves_per_second=med,throughput_increase_percent=100*(med['2']/med['1']-1),
                        arithmetic_curves=sum(len(r['curves']) for r in data['batches']),memory_is_sampled=True)
    data['complete']=True;persist();print(json.dumps(data['summary'],indent=2))


if __name__=='__main__':main()
