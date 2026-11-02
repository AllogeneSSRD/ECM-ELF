"""Serial exact S4 GMP comparisons: scalar, cache windows and shared warp division."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from bench_stage2_point_routes import sha
from bench_stage2_version import check_gpu_processes, write, stamp
from gpu_telemetry import TelemetryRecorder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bits',type=int,nargs='+',required=True)
    p.add_argument('--coeffs',type=int,nargs='+',default=[35])
    p.add_argument('--bpws',type=int,nargs='+',default=[26])
    p.add_argument('--kinds',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--backends',type=int,nargs='+',default=[0,32])
    p.add_argument('--threads',type=int,default=128)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--reverse',action='store_true',help='Retain a second call in reverse backend order')
    p.add_argument('--batches',type=int,default=1)
    p.add_argument('--fault',type=int,choices=(0,1,2),default=0)
    p.add_argument('--modulus-hex',type=Path)
    p.add_argument('--allow-gpu-pid',type=int,action='append',default=[])
    p.add_argument('--background-snapshot',type=Path)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'bin';binary.mkdir();(binary/'.gitignore').write_text('*\n')
    manifest=json.loads((a.exe.parent/'manifest.json').read_text(encoding='utf-8-sig'))
    if manifest['sha256'].lower()!=sha(a.exe):raise ValueError('Unverified probe executable')
    root=Path(__file__).resolve().parents[2]
    for name,digest in manifest['identity']['sources'].items():
        source=root/name
        if sha(source)!=digest.lower():raise ValueError('Probe source changed: '+name)
        target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for f in (a.exe.resolve(),a.exe.parent/'manifest.json',a.exe.parent/'gmp-10.dll'):shutil.copy2(f,binary/f.name)
    shutil.copy2(__file__,out/'collector_snapshot.py');exe=binary/a.exe.name
    modulus=None
    if a.modulus_hex:modulus=out/'modulus.hex';shutil.copy2(a.modulus_hex,modulus)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    allowed=[]
    for pid in a.allow_gpu_pid:
        paths={s.split(',',2)[2].strip() for s in processes.splitlines() if s.split(',')[1].strip()==str(pid)}
        if len(paths)!=1 or not Path(next(iter(paths))).is_absolute():raise ValueError('Cannot bind allowed GPU process')
        allowed.append(dict(pid=pid,process_name=next(iter(paths))))
    report=dict(complete=False,exe_sha256=sha(exe),started_utc=stamp(),rows=[],failures=[],allowed_processes=allowed,
                modulus_sha256=sha(modulus) if modulus else None)
    monitor=TelemetryRecorder(a.device,out/'telemetry.jsonl',interval=.2);report['device']=monitor.identity
    if a.background_snapshot:
        background=json.loads(a.background_snapshot.read_text(encoding='utf-8-sig'))
        if not background.get('approved_background_by_user') or background['device']!=a.device or background['uuid']!=monitor.identity['uuid']:
            monitor.close();raise ValueError('Background authorization is for a different device')
        allowed.extend(background['allowed_processes']);shutil.copy2(a.background_snapshot,out/'background_snapshot.json')
        report['approved_background_snapshot']=background
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')}
    hashes={}
    try:
        for bits in a.bits:
            for count in a.coeffs:
                for kind in a.kinds:
                    for bpw in a.bpws:
                        orders=[a.backends]+([list(reversed(a.backends))] if a.reverse else [])
                        for order in orders:
                            for backend in order:
                                folder=out/('call_%04d'%len(report['rows']));folder.mkdir()
                                cmd=[str(exe),*map(str,(bits,count,kind,bpw,backend,a.repeats,a.threads,a.device))]
                                if modulus or a.fault or a.batches!=1:cmd.append(str(modulus) if modulus else '-')
                                if a.fault or a.batches!=1:cmd.append(str(a.fault))
                                if a.batches!=1:cmd.append(str(a.batches))
                                write(folder/'command.json',cmd)
                                check_gpu_processes(monitor.monitor,out,folder/'before.json',allowed=allowed)
                                r=subprocess.run(cmd,cwd=binary,env=env,capture_output=True,text=True,timeout=180)
                                (folder/'console.log').write_text(r.stdout+r.stderr,encoding='utf-8');write(folder/'exit.json',dict(returncode=r.returncode))
                                check_gpu_processes(monitor.monitor,out,folder/'after.json',allowed=allowed)
                                line=next((s for s in r.stdout.splitlines() if s.startswith('s4_warp_probe:')),None)
                                if line is None:raise ValueError('Missing probe result: '+str(folder))
                                row=dict(re.findall(r'(\w+)=([^\s]+)',line));row['folder']=str(folder);report['rows'].append(row)
                                write(out/'summary.json',report)
                                valid=(r.returncode==0 and not int(row['bad']) and not int(row['gpu_bad']))
                                if a.fault==1:valid=r.returncode==1 and not int(row['bad']) and int(row['gpu_bad'])>0
                                if a.fault==2:valid=r.returncode==1 and int(row['bad'])>0 and not int(row['gpu_bad'])
                                if not valid:raise ValueError('S4 arithmetic/bounds gate failed: '+str(folder))
                                key=(bits,count,kind,bpw)
                                if not a.fault and key in hashes and hashes[key]!=row['hash']:raise ValueError('Cross-backend output differs')
                                hashes[key]=row['hash']
                                print('s4_probe:',bits,count,kind,bpw,backend,row['median_ms'],flush=True)
        report.update(complete=True,finished_utc=stamp())
    except Exception as e:report['failures'].append(str(e));raise
    finally:monitor.close();write(out/'summary.json',report)


if __name__=='__main__':main()
