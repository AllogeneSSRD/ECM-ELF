"""Frozen GPU1-style CGBN square gates and serial same-binary comparisons.

Modes 0/1/2: integrated CGBN / generic wide+REDC / symmetric wide+REDC.
Modes 3/4: generic / symmetric exact wide products. No hardware writes.
Modes 5/7: finite-depth split Montgomery square, 6/8: its exact wide product.
Modes 9/11: finite-depth Karatsuba Montgomery square, 10/12: its wide product.
Modes 13/14: existing Stage2 Mersenne reducer with generic / split wide product.
Modes 15/16: carry-save WMAD REDC, streamed / precombined high half.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
from pathlib import Path

from bench_stage2_point_matrix import environment
from bench_stage2_version import check_gpu_processes, load, write, stamp
from gpu_telemetry import TelemetryRecorder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--process-policy',type=Path,help='JSON containing the already approved allowed_processes list')
    p.add_argument('--suite',choices=('smoke','correctness','performance','sanitizer'),required=True)
    p.add_argument('--sanitizer',type=Path,help='Explicit compute-sanitizer executable for the sanitizer suite')
    p.add_argument('--sanitizer-tool',choices=('memcheck','synccheck'),default='memcheck')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--counts',default='1,native,4096')
    p.add_argument('--modes',default='0,1,2,3,4')
    p.add_argument('--kind',type=int,choices=(0,1,2,3),help='1: full Mersenne, 3: six spare container bits')
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    if a.suite=='sanitizer' and (a.sanitizer is None or not a.sanitizer.is_file()):
        p.error('--suite sanitizer requires --sanitizer PATH')
    modes=[int(s) for s in a.modes.split(',')];assert modes and all(0<=m<=16 for m in modes)
    build=a.build.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'bin';binary.mkdir();(binary/'.gitignore').write_text('*\n')
    m=load(build/'manifest.json');assert sha(build/'cgbn_square_probe.exe')==m['sha256'].lower()
    for name,digest in m['sources'].items():
        source=(build/'sources'/name) if (build/'sources').is_dir() else repo/name
        assert sha(source)==digest.lower(),name
        dest=binary/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    for name,digest in m['objects'].items():assert sha(name)==digest.lower(),name
    for name in ('cgbn_square_probe.exe','gmp-10.dll','manifest.json'):shutil.copy2(build/name,binary/name)
    shutil.copy2(__file__,out/'collector_snapshot.py')
    allowed=load(a.process_policy)['allowed_processes'] if a.process_policy else []
    bits=int(re.search(r'ECM_SQUARE_BITS=(\d+)',m['identity']['commands'])[1])
    tpi=int(re.search(r'ECM_SQUARE_TPI=(\d+)',m['identity']['commands'])[1])
    mask=re.search(r'ECM_SQUARE_MODE_MASK=(\d+)',m['identity']['commands'])
    if mask:assert all((1<<mode)&int(mask[1]) for mode in modes),'Mode not compiled'
    report=dict(complete=False,suite=a.suite,bits=bits,tpi=tpi,device=a.device,
        exe_sha256=sha(binary/'cgbn_square_probe.exe'),dll_sha256=sha(binary/'gmp-10.dll'),
        allowed_processes=allowed,started_utc=stamp(),warmups=0,runs=[],failures=[])
    xor_feedback='xor_halves' in (binary/'sources/tools/test/cgbn_square_probe.cu').read_text(encoding='utf-8-sig')
    report['wide_feedback']='xor_halves' if xor_feedback else 'legacy_low_half'
    if a.suite=='sanitizer':
        report['sanitizer']=dict(path=str(a.sanitizer.resolve()),sha256=sha(a.sanitizer),
            tool=a.sanitizer_tool,local_connection='named-pipes' if os.name=='nt' else 'default')
    write(out/'summary.json',report)

    def run(count,rec,mode,alias,kind,samples=1,fault=0):
        folder=out/f'call{len(report["runs"]):04d}';folder.mkdir()
        cmd=[str(binary/'cgbn_square_probe.exe'),*map(str,(a.device,count,rec,mode,alias,kind,samples,fault))]
        child_env=environment()
        if a.suite=='sanitizer':
            cmd=[str(a.sanitizer.resolve()),'--tool',a.sanitizer_tool,'--error-exitcode','87',*cmd]
            if os.name=='nt':
                # NVIDIA documents this local transport for Windows. Keep
                # instrumentation IPC off TCP; do not edit firewall/global env.
                child_env['NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE']='named-pipes'
        write(folder/'command.json',cmd);mon=TelemetryRecorder(a.device,folder/'telemetry.jsonl',interval=.05)
        try:
            check_gpu_processes(mon.monitor,out,folder/'before.json',allowed=allowed)
            r=subprocess.run(cmd,cwd=folder,env=child_env,capture_output=True,text=True,timeout=180)
            text=r.stdout+r.stderr;(folder/'console.log').write_text(text,encoding='utf-8')
            write(folder/'exit.json',dict(returncode=r.returncode))
            check_gpu_processes(mon.monitor,out,folder/'after.json',allowed=allowed)
            identity=mon.identity
        finally:mon.close()
        if fault:
            assert r.returncode==3 and 'cgbn_square_bad:' in text,folder
            row=dict(fault_rejected=True,fault=fault,mode=mode,count=count,recurrence=rec,folder=str(folder))
        else:
            assert r.returncode==0,folder
            if a.suite=='sanitizer':assert 'ERROR SUMMARY: 0 errors' in text,folder
            lines=[s for s in text.splitlines() if s.startswith('cgbn_square:')];assert len(lines)==1,folder
            fields=dict(re.findall(r'(\w+)=([^\s]+)',lines[0]))
            for key,value in dict(bits=bits,tpi=tpi,mode=mode,count=count,repeats=rec,alias=alias,kind=kind,bad=0).items():
                assert int(fields[key])==value,(folder,key)
            values=[float(x) for x in fields.pop('samples_ms').split(',')]
            assert len(values)==samples and all(v>0 for v in values)
            row=dict(fields=fields,samples_ms=values,median_ms=statistics.median(values),
                mode=mode,count=count,recurrence=rec,alias=alias,kind=kind,folder=str(folder),device_identity=identity)
            row['feedback']=fields.get('feedback','modular' if mode in (0,1,2,5,7,9,11,13,14,15,16) else 'legacy_low_half')
        report['runs'].append(row);write(out/'summary.json',report)
        print(a.suite,bits,tpi,count,rec,mode,row.get('median_ms','fault-rejected'),flush=True)
        return row

    try:
        if a.suite=='smoke':
            for mode in modes:run(33,1,mode,1,a.kind if a.kind is not None else 1)
        elif a.suite=='sanitizer':
            for mode in modes:run(67,3,mode,1,a.kind if a.kind is not None else 1)
        elif a.suite=='correctness':
            for rec in (1,3):
                for mode in modes:
                    for kind in ((a.kind,) if a.kind is not None else (1,3) if mode in (13,14) else (0,1,2)):
                        for alias in (0,1):run(67,rec,mode,alias,kind)
            for count in (1,33):
                for mode in modes:run(count,3,mode,1,1 if mode in (13,14) else 0)
            for mode in modes:
                if mode not in (0,1,3):run(33,1,mode,1,1,fault=1)
            if xor_feedback:
                for mode in modes:
                    if mode in (3,4,6,8,10,12):run(33,3,mode,1,1,fault=2)
        else:
            assert a.rounds>=2
            # Query the real SM count from a checked native control, not a
            # hard-coded laptop identifier. This call is separate from timing.
            kind=a.kind if a.kind is not None else 2
            query=run(1,1,0 if 0 in modes else modes[0],1,kind)
            native=2*int(query['fields']['sm_count'])*128//tpi
            counts=[native if s=='native' else int(s) for s in a.counts.split(',')]
            report['native_count']=native;report['timing_begin']=len(report['runs'])
            for count in counts:
                rec=128 if count==1 else 32 if count==native else 4
                for repeat in range(a.rounds):
                    order=modes if repeat%2==0 else list(reversed(modes))
                    for mode in order:
                        row=run(count,rec,mode,1,kind,samples=5);row['round']=repeat
                        write(out/'summary.json',report)
        report.update(complete=True,finished_utc=stamp())
    except Exception as e:
        report['failures'].append(str(e));raise
    finally:write(out/'summary.json',report)


if __name__=='__main__':main()
