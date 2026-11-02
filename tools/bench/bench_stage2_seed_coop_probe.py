"""Serial production-header seed-pair/GMP qualification and CUDA-event comparison."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from bench_stage2_point_routes import sha
from bench_stage2_version import check_gpu_processes,stamp,write
from gpu_telemetry import TelemetryRecorder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bits',type=int,nargs='+',required=True)
    p.add_argument('--suite',choices=('correctness','performance'),required=True)
    p.add_argument('--chains',type=int,default=2048)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--allow-gpu-pid',type=int,action='append',default=[])
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'bin';binary.mkdir();(binary/'.gitignore').write_text('*\n')
    for f in (a.exe.resolve(),a.exe.resolve().parent/'gmp-10.dll',a.exe.resolve().parent/'manifest.json'):
        shutil.copy2(f,binary/f.name)
    shutil.copy2(__file__,out/'collector_snapshot.py');exe=binary/a.exe.name
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    allowed=[]
    for pid in a.allow_gpu_pid:
        names={row.split(',',2)[2].strip() for row in processes.splitlines() if row.split(',')[1].strip()==str(pid)}
        if len(names)!=1 or not Path(next(iter(names))).is_absolute():raise ValueError('Cannot bind allowed process')
        allowed.append(dict(pid=pid,process_name=next(iter(names))))
    report=dict(complete=False,started_utc=stamp(),suite=a.suite,exe_sha256=sha(exe),dll_sha256=sha(binary/'gmp-10.dll'),
        collector_sha256=sha(Path(__file__)),allowed_processes=allowed,rows=[],failures=[])
    monitor=TelemetryRecorder(a.device,out/'telemetry.jsonl',interval=.2);report['device']=monitor.identity
    environment={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')}
    signatures={}
    def call(bits,chains,per,first,tail,kind,tpi,samples=1,chunk=7,fault=0):
        npts=(chains-1)*per+tail;folder=out/('call_%04d'%len(report['rows']));folder.mkdir()
        command=[str(exe),*map(str,(bits,chains,per,first,npts,kind,tpi,samples,a.device,fault,chunk))]
        write(folder/'command.json',command);check_gpu_processes(monitor.monitor,out,folder/'before.json',allowed=allowed)
        r=subprocess.run(command,cwd=binary,env=environment,capture_output=True,text=True,timeout=180)
        log=r.stdout+r.stderr;(folder/'console.log').write_text(log,encoding='utf-8')
        write(folder/'exit.json',dict(returncode=r.returncode))
        check_gpu_processes(monitor.monitor,out,folder/'after.json',allowed=allowed)
        line=next((s for s in log.splitlines() if s.startswith('point_coop_seed:')),None)
        if line is None:raise ValueError('Missing seed result: '+str(folder))
        row=dict(re.findall(r'(\w+)=([^\s]+)',line));row.update(folder=str(folder),returncode=r.returncode,fault=fault)
        report['rows'].append(row);write(out/'results.json',report)
        if (r.returncode!=0)!=bool(fault) or (int(row['bad'])!=0)!=bool(fault):raise ValueError('Arithmetic gate failed: '+str(folder))
        if not fault:
            if a.suite=='correctness' and int(row['checked_words'])!=int(row['output_words']):raise ValueError('Incomplete GMP coverage')
            key=(bits,chains,per,first,npts,kind)
            if key in signatures and signatures[key]!=row['sha256']:raise ValueError('Scalar/cooperative coordinate mismatch')
            signatures[key]=row['sha256']
        if len(report['rows'])%12==0:print('seed_coop_progress:',len(report['rows']),bits,flush=True)
    try:
        if a.suite=='correctness':
            for bits in a.bits:
                for kind in (0,1,2):
                    for chains,per,first,tail in ((37,32,0,1),(2,64,1<<63,2),(1,1,(1<<64)-1,1)):
                        for tpi in (0,32):call(bits,chains,per,first,tail,kind,tpi)
            call(a.bits[0],1,1,0,1,1,32,fault=1)
        else:
            for bits in a.bits:
                for kind in (0,1):
                    for order in ((0,32),(32,0)):
                        for tpi in order:call(bits,a.chains,32,65536,32,kind,tpi,samples=3,chunk=8192)
        report['complete']=True;report['finished_utc']=stamp()
    except Exception as e:report['failures'].append(str(e));raise
    finally:
        monitor.close();write(out/'results.json',report)
    clocks=Counter()
    for line in (out/'telemetry.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row.get('gpu_percent',0)>=50:clocks[row['sm_clock_mhz']]+=1
    print(dict(complete=True,calls=len(report['rows']),busy_clocks=dict(clocks)),flush=True)


if __name__=='__main__':main()
