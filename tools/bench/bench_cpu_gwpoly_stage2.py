"""Measure full saved-point GWNUM Stage2 against exact-N Prime95 references.

Freeze binaries and inputs; retain first samples. No bound/thread normalization.
"""
import argparse
from collections import defaultdict
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--study',type=Path,required=True,help='Frozen exact-N Edwards reference matrix')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exponents',type=int,nargs='+',default=[1009,10267,16319])
    p.add_argument('--threads',type=int,nargs='+',default=[1,4])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--bounds',choices=['low','high','all'],default='all')
    p.add_argument('--memory-mib',type=int,default=4096)
    p.add_argument('--affinity',type=lambda x:int(x,0),default=0xAA)
    p.add_argument('--cpu-ghz',type=float,required=True,help='Fixed clock declared by the operator; not telemetry')
    a=p.parse_args()
    if a.repeats<1 or len(a.threads)!=len(set(a.threads)) or any(t<1 for t in a.threads):p.error('invalid repetition/thread list')
    source=a.study/'plan.json';plan=json.loads(source.read_bytes())
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'bin';binary.mkdir();exe=binary/a.exe.name
    shutil.copy2(a.exe,exe);shutil.copy2(a.exe.parent/'gmp-10.dll',binary/'gmp-10.dll')
    shutil.copy2(source,out/'reference.snapshot.json')
    shutil.copy2(__file__,out/'collector.snapshot.py')
    root=Path(__file__).resolve().parents[2]
    closure=out/'source';closure.mkdir()
    manifest={}
    for path in [*sorted((root/'src/core').glob('*.h')),*sorted((root/'src/cpu').glob('*.h')),
                 root/'src/core/ecm_cpu_stage2_main.cpp',root/'src/core/ecm_stage1_checkpoint_runtime.cpp',
                 root/'src/cpu/ecm_stage2_gwpoly.cpp',*sorted((root/'tools/build/internal').glob('ecm*stage2.cmake')),
                 root/'tools/build/internal/ecm_libecm.cmake',root/'tools/build/test/build_cpu_stage2.ps1',
                 *sorted((root/'.refactor/p95v3106b01.source/gwnum').glob('*.h')),
                 root/'.refactor/p95v3106b01.source/gwnum/gwnum64.lib',
                 root/'.refactor/p95v3106b01.source/gwnum/polymult64.lib']:
        relative=path.relative_to(root);dest=closure/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);manifest[str(relative)]=sha(dest)
    write(out/'source_manifest.json',manifest)
    compile_commands=a.exe.parent/'compile_commands.json'
    if compile_commands.exists():shutil.copy2(compile_commands,out/'compile_commands.snapshot.json')
    static_library=a.exe.parent/'ecm_cpu_library.lib'
    if static_library.exists():
        shutil.copy2(static_library,binary/static_library.name)
        manifest['linked/ecm_cpu_library.lib']=sha(binary/static_library.name)
        write(out/'source_manifest.json',manifest)
    ini=out/'empty.ini';ini.write_text('user=\ncomputer=\n',encoding='utf-8')
    cases=[c.copy() for c in plan['cases'] if c['exponent'] in a.exponents]
    if {c['exponent'] for c in cases}!=set(a.exponents):raise ValueError('Missing requested reference inputs')
    inputs=out/'inputs';inputs.mkdir()
    for c in cases:
        if sha(c['save'])!=c['save_sha256']:raise ValueError('Verified save changed')
        saved=inputs/Path(c['save']).name;shutil.copy2(c['save'],saved);c['save']=str(saved)
    refs=[r for r in plan['prime95_rows'] if r['curve_type']=='Edwards' and r['benchmark_eligible'] and r['exponent'] in a.exponents]
    if a.bounds!='all':
        selected={e:sorted({r['b2_requested'] for r in refs if r['exponent']==e})[0 if a.bounds=='low' else -1] for e in a.exponents}
        refs=[r for r in refs if r['b2_requested']==selected[r['exponent']]]
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.SetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    api.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    api.GetProcessTimes.restype=wintypes.BOOL
    class Memory(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    memfn=ctypes.WinDLL('psapi').GetProcessMemoryInfo
    memfn.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
    report=dict(complete=False,exe_sha256=sha(exe),dll_sha256=sha(binary/'gmp-10.dll'),
                reference_sha256=sha(source),script_sha256=sha(__file__),cases=cases,
                memory_mib=a.memory_mib,affinity_mask=a.affinity,threads=a.threads,repeats=a.repeats,
                conditions=dict(cpu='AMD Ryzen AI 9 HX 370',os='Windows 11',clock_ghz=a.cpu_ghz,
                    clock_source='operator declaration; effective clock not sampled',power_watts=None,
                    temperature=None,parallel_workloads='no other benchmark launched by this collector',
                    previous_ecm_elf_prototypes_ghz=3.1,prime95_historical_ghz=4,
                    clock_scope='2026-10-17 operator confirmed only ECM-ELF prototype runs used 3.1 GHz'),
                source_manifest_sha256=sha(out/'source_manifest.json'),
                expected_invocations=len(refs)*len(a.threads)*a.repeats,runs=[],failures=[],
                timing='native curve construction + complete Stage2 including cleanup; separate process wall and CPU occupancy',
                prime95='historical 31.4b05, 4 threads, user-declared 4 GHz, init+main+GCD, n=1; no normalization')
    write(out/'summary.json',report)
    try:
        for rep in range(a.repeats):
            for r in refs:
                c=next(c for c in cases if c['N_hex']==r['N_hex'] and c['sigma']==int(r['s']))
                for threads in (a.threads if rep%2==0 else list(reversed(a.threads))):
                    name=f'm{c["exponent"]}_b{r["b2"]}_t{threads}_r{rep+1}'
                    prefix=out/name
                    cmd=list(map(str,[exe,'--save',c['save'],'--curves',1,'--backend','gwpoly','--b2',r['b2'],
                        '--threads',threads,'--memory-mib',a.memory_mib,'--ini',ini,'--results',prefix.with_suffix('.jsonl'),
                        '--submission',prefix.with_suffix('.json.txt'),'--exponent',c['exponent']]))
                    for factor in c['known']:cmd+=['--known-factor',str(factor)]
                    print('RUN',name,flush=True);start=time.perf_counter()
                    with prefix.with_suffix('.log').open('wb') as log:
                        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=out)
                        if a.affinity and not api.SetProcessAffinityMask(proc._handle,a.affinity):
                            proc.kill();proc.wait();raise ctypes.WinError(ctypes.get_last_error())
                        try:rc=proc.wait(timeout=1800)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait();raise
                        wall=time.perf_counter()-start;ft=[wintypes.FILETIME() for _ in range(4)]
                        if not api.GetProcessTimes(proc._handle,*map(ctypes.byref,ft)):raise ctypes.WinError(ctypes.get_last_error())
                        cpu=sum((t.dwHighDateTime<<32)+t.dwLowDateTime for t in ft[2:])/1e7
                        memory=Memory();memory.cb=ctypes.sizeof(memory)
                        if not memfn(proc._handle,ctypes.byref(memory),memory.cb):raise ctypes.WinError(ctypes.get_last_error())
                    if rc:raise RuntimeError(name+' exit '+str(rc))
                    result=json.loads(prefix.with_suffix('.jsonl').read_bytes())
                    assert result['b1']==c['b1'] and result['b2']==r['b2'] and result['Edwards']['sigma']==c['sigma']
                    assert result['stage2-engine']['backend']=='gwpoly'
                    assert result['exponent']==c['exponent'] and result.get('known-factors',[])==list(map(str,c['known']))
                    n=int(c['N_hex'],16)
                    assert all(1<int(f)<n and n%int(f)==0 for f in result.get('factors',[]))
                    text=prefix.with_suffix('.log').read_text(encoding='utf-8')
                    line=next(line for line in text.splitlines() if line.startswith('gwpoly:'))
                    metrics=dict(re.findall(r'([\w_]+)=([\d.eE+-]+)',line))
                    assert float(metrics['maxerr'])<=.4
                    row=dict(id=name,exponent=c['exponent'],bits=c['bits'],B1=c['b1'],B2=r['b2'],nominal_B2=r['b2_requested'],
                        threads=threads,repeat=rep+1,native_seconds=result['stage2-engine']['seconds'],process_seconds=wall,
                        cpu_seconds=cpu,cpu_wall_ratio=cpu/wall,peak_working_set_bytes=memory.peak_ws,
                        full_sample=result['status']=='NF',prime95_4t_seconds=r['s2_total_time'],metrics=metrics,
                        result_sha256=sha(prefix.with_suffix('.jsonl')),prefix=str(prefix))
                    report['runs'].append(row);write(out/'summary.json',report)
                    print('DONE',name,'native',round(row['native_seconds'],6),'CPU/wall',round(cpu/wall,3),flush=True)
        grouped=defaultdict(list)
        for row in report['runs']:
            if row['full_sample']:grouped[(row['exponent'],row['B2'],row['threads'])].append(row)
        report['groups']=[dict(exponent=k[0],B2=k[1],threads=k[2],n=len(v),
            median_seconds=statistics.median(r['native_seconds'] for r in v),min_seconds=min(r['native_seconds'] for r in v),
            max_seconds=max(r['native_seconds'] for r in v),cpu_wall_median=statistics.median(r['cpu_wall_ratio'] for r in v),
            peak_working_set_median_bytes=statistics.median(r['peak_working_set_bytes'] for r in v),
            prime95_4t_seconds=v[0]['prime95_4t_seconds']) for k,v in grouped.items()]
        report['complete']=len(report['runs'])==report['expected_invocations']
        write(out/'summary.json',report)
    except Exception as exc:
        report['failures'].append(str(exc));write(out/'summary.json',report);raise


if __name__=='__main__':main()
