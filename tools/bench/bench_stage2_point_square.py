"""Frozen, serial GPU probe of Stage2 square vs multiply(a,a).

No GPU clock or power writes. Reuse the Stage2 matrix process gate and record
all event samples, commands, failures, checksums, telemetry and DLL identity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from bench_stage2_version import check_gpu_processes, write, stamp
from gpu_telemetry import TelemetryRecorder


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--allow-gpu-pid',type=int,action='append',default=[])
    p.add_argument('--suite',choices=['correctness','performance'],required=True)
    p.add_argument('--bits',default='4096,8192,8193,16319')
    p.add_argument('--count',type=int,default=4096)
    p.add_argument('--recurrence',type=int,default=4)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--threads',type=int,default=64)
    p.add_argument('--candidate-mode',type=int,choices=(1,2),default=2,
                   help='1: dedicated primitive; 2: production runtime wrapper')
    a=p.parse_args();bits=[int(x) for x in a.bits.split(',')]
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'bin';binary.mkdir();(binary/'.gitignore').write_text('*\n')
    for source in (a.exe.resolve(),a.exe.resolve().parent/'gmp-10.dll',a.exe.resolve().parent/'manifest.json'):
        shutil.copy2(source,binary/source.name)
    shutil.copy2(__file__,out/'collector_snapshot.py')
    exe=binary/a.exe.name
    allowed=[]
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    for pid in a.allow_gpu_pid:
        names={line.split(',',2)[2].strip() for line in text.splitlines() if line.split(',')[1].strip()==str(pid)}
        if len(names)!=1 or not Path(next(iter(names))).is_absolute():raise ValueError('Cannot bind allowed process identity')
        allowed.append(dict(pid=pid,process_name=next(iter(names))))
    state=dict(complete=False,started_utc=stamp(),suite=a.suite,exe_sha256=sha(exe),dll_sha256=sha(binary/'gmp-10.dll'),
               allowed_processes=allowed,conditions=vars(a).copy(),rows=[],failures=[])
    state['conditions']={k:str(v) if isinstance(v,Path) else v for k,v in state['conditions'].items()}
    write(out/'results.json',state)
    telemetry=TelemetryRecorder(a.device,out/'telemetry.jsonl',interval=.2)
    state['device']=telemetry.identity;write(out/'results.json',state)
    def call(bits,count,recur,mode,alias,kind,fault=0):
        index=len(state['rows']);directory=out/f'call_{index:04d}';directory.mkdir()
        check_gpu_processes(telemetry.monitor,binary,directory/'before.json',allowed=allowed)
        command=[str(exe),*map(str,(bits,count,recur,mode,alias,a.threads,a.device,kind,fault))]
        write(directory/'command.json',command);started=time.perf_counter()
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        (directory/'console.log').write_text(result.stdout,encoding='utf-8')
        check_gpu_processes(telemetry.monitor,binary,directory/'after.json',allowed=allowed)
        fields=dict(re.findall(r'(\w+)=([^\s]+)',result.stdout))
        row=dict(index=index,command=command,returncode=result.returncode,wall_seconds=time.perf_counter()-started,
                 fields=fields,fault=fault,log=str(directory/'console.log'))
        state['rows'].append(row)
        expected=1 if fault else 0
        if result.returncode!=expected or 'bad' not in fields or bool(int(fields['bad']))!=bool(fault):
            state['failures'].append(row);write(out/'results.json',state);raise RuntimeError('Probe arithmetic/process failure: '+str(directory))
        write(out/'results.json',state)
        print(f"point_square_progress: calls={len(state['rows'])} bits={bits} kind={kind} mode={mode} alias={alias} ms={fields['median_ms']}",flush=True)
    try:
        if a.suite=='correctness':
            for bit in bits:
                for kind in (0,1,2):
                    for alias in (0,1):
                        for mode in (1,2):call(bit,64,a.recurrence,mode,alias,kind)
            call(4096,32,1,1,1,0,fault=1)
        else:
            for round in range(a.rounds):
                for bit in bits:
                    for kind in (0,1):
                        for mode in ((0,a.candidate_mode) if round%2==0 else (a.candidate_mode,0)):
                            call(bit,a.count,a.recurrence,mode,1,kind)
        state['complete']=True;state['finished_utc']=stamp();write(out/'results.json',state)
    finally:
        telemetry.close();state['telemetry_samples']=telemetry.samples;write(out/'results.json',state)


if __name__=='__main__':
    main()
