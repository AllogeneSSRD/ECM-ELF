"""Measure actual libecm CPU use with 1/4 thread limits on identical saved input.

Process CPU/wall is aggregate occupancy, not proof of useful parallel speedup.
Keep elapsed time as the primary speed metric; retain every raw invocation.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--save',type=Path,required=True)
    p.add_argument('--b2',type=int,default=10**9)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False,parents=True)
    exe=a.exe.resolve();save=a.save.resolve()
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    report=dict(exe_sha256=sha(exe),save_sha256=sha(save),script_sha256=sha(__file__),b2=a.b2,runs=[],complete=False)
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    api.GetProcessTimes.restype=wintypes.BOOL
    ini=out/'empty.ini';ini.write_text('user=\ncomputer=\n',encoding='utf-8')
    for rep in range(a.repeats):
        for backend,threads in ([('ntt',1),('ntt',4),('plain',4),('plain',1)] if rep%2==0 else [('plain',1),('plain',4),('ntt',4),('ntt',1)]):
            name=f'{backend}_t{threads}_r{rep+1}';result=out/(name+'.jsonl')
            cmd=list(map(str,[exe,'--ini',ini,'--save',save,'--curves',1,'--b2',a.b2,'--backend',backend,'--threads',threads,
                             '--memory-mib',4096,'--results',result,'--submission',out/(name+'.json.txt')]))
            start=time.perf_counter()
            with (out/(name+'.log')).open('wb') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=out)
                rc=proc.wait(timeout=1800);wall=time.perf_counter()-start
                times=[wintypes.FILETIME() for _ in range(4)]
                if not api.GetProcessTimes(proc._handle,*map(ctypes.byref,times)):raise ctypes.WinError(ctypes.get_last_error())
            cpu=sum((t.dwHighDateTime<<32)+t.dwLowDateTime for t in times[2:])/1e7
            assert rc==0,name
            r=json.loads(result.read_text(encoding='utf-8'));assert r['status']=='NF'
            row=dict(backend=backend,thread_limit=threads,repeat=rep+1,process_seconds=wall,cpu_seconds=cpu,
                     cpu_wall_ratio=cpu/wall,engine_seconds=r['stage2-engine']['seconds'],log=name+'.log')
            report['runs'].append(row)
            (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            print(name,'wall',round(wall,3),'cpu/wall',round(cpu/wall,3),flush=True)
    report['complete']=True;(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
