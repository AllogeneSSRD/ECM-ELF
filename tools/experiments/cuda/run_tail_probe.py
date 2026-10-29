"""Freeze and run the small CUDA/GMP tail comparison; never changes GPU settings."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    if a.out.exists():
        raise SystemExit('Choose a new evidence directory')
    manifest=json.loads((a.build/'build_manifest.json').read_text(encoding='utf-8-sig'))
    repo=Path(__file__).resolve().parents[3]
    for name,digest in manifest['sources'].items():
        if sha(repo/name).lower()!=digest.lower():
            raise SystemExit('Build/source mismatch: '+name)
    if sha(a.build/'cuda_stage2_tail_probe.exe').lower()!=manifest['executable_sha256'].lower():
        raise SystemExit('Executable hash mismatch')
    a.out.mkdir(parents=True)
    for name in ['cuda_stage2_tail_probe.exe','gmp-10.dll','build_manifest.json','build.log','build.cmd']:
        shutil.copy2(a.build/name,a.out/name)
    for name in manifest['sources']:
        dest=a.out/'sources'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/name,dest)
    shutil.copy2(__file__,a.out/Path(__file__).name)
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    meta={'date_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'platform':platform.platform(),'processor':platform.processor(),
          'device':a.device,'repeats':a.repeats,'priming_each_implementation':1,
          'gpu_settings_changed':False,'input':'synthetic random projective coordinates / residues',
          'output':'host ordinary negative affine coordinates / full-width gcd',
          'excluded':'context setup, input preparation, allocation, verification, cleanup; initial upload for normalize/gcd',
          'included':'required transfers, CPU GMP imports/arithmetic/exports, CUDA kernels/events/waits',
          'boundaries':{'normalize':'resident X/Z -> host negative affine coefficients',
                        'gcd':'resident residues -> host full-width GCDs; device guard read included',
                        'accumulate':'host leaves -> host block GCDs; GPU route includes leaf H2D and product D2H'},
          'build':manifest}
    meta['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,clocks.current.graphics,power.draw,utilization.gpu','--format=csv'],text=True)
    with (a.out/'gpu.csv').open('w',encoding='utf-8') as gpu,(a.out/'samples.csv').open('w',encoding='utf-8') as log,(a.out/'stderr.log').open('w',encoding='utf-8') as err:
        poll=subprocess.Popen(['nvidia-smi',f'--id={a.device}','--query-gpu=timestamp,clocks.current.graphics,power.draw,utilization.gpu','--format=csv','-lms','500'],stdout=gpu,stderr=subprocess.DEVNULL,creationflags=flags)
        begin=time.perf_counter()
        try:
            result=subprocess.run([str((a.out/'cuda_stage2_tail_probe.exe').resolve()),str(a.device),str(a.repeats)],cwd=a.out,stdout=log,stderr=err,creationflags=flags)
        finally:
            poll.terminate()
            poll.wait()
        meta['process_seconds']=time.perf_counter()-begin
        meta['exit_code']=result.returncode
    meta['passed']=meta['exit_code']==0 and '# status=passed' in (a.out/'samples.csv').read_text(encoding='utf-8')
    meta['artifacts']={name:sha(a.out/name) for name in ['samples.csv','stderr.log','gpu.csv','cuda_stage2_tail_probe.exe','gmp-10.dll','run_tail_probe.py']}
    (a.out/'run_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':meta['passed'],'seconds':meta['process_seconds'],'out':str(a.out)},ensure_ascii=False))
    return 0 if meta['passed'] else 1

if __name__=='__main__':
    sys.exit(main())
