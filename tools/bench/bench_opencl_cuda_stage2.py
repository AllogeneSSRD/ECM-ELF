"""Fixed-save, fixed-D full Stage2 A/B; includes every measured invocation."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

# A supported 16384-bit N has more than Python's default 4300 decimal digits.
if hasattr(sys,'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opencl-build',type=Path,required=True)
    parser.add_argument('--cuda-build',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,default=Path('data/experiments/stage2_v0_38_4_20261018/full_matrix/inputs'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--exponents',default='503,2003,5003,8011')
    parser.add_argument('--b2',default='260000,2600000')
    parser.add_argument('--d',type=int,default=2310)
    parser.add_argument('--cuda-device',type=int,default=1)
    parser.add_argument('--opencl-device',type=int,default=1)
    parser.add_argument('--allow-version-difference',action='store_true',
                        help='Explicitly compare an identified frozen version with a newer implementation')
    a=parser.parse_args()
    if a.out.exists() or not 1<=a.repeats<=20:
        raise SystemExit('Choose a new directory and repeats=1..20')
    a.out.mkdir(parents=True)
    binaries={}
    for backend,build,name in [('opencl',a.opencl_build,'ecm_opencl_stage2.exe'),('cuda',a.cuda_build,'ecm_cuda_stage2.exe')]:
        dest=a.out/'bin'/backend
        dest.mkdir(parents=True)
        for file in [name,'gmp-10.dll','build_manifest.json']:
            shutil.copy2(build/file,dest/file)
        if backend=='opencl':
            shutil.copytree(build/'opencl_resources',dest/'opencl_resources')
        binaries[backend]=(dest/name).resolve()
    versions={}
    for backend,exe in binaries.items():
        version=subprocess.check_output([str(exe),'--version'],text=True).strip()
        if not version.startswith('ECM-ELF '):
            raise RuntimeError(f'{backend}: unexpected version {version}')
        versions[backend]=version
    if len(set(versions.values()))!=1 and not a.allow_version_difference:
        raise RuntimeError('A/B requires the same public version: '+str(versions))
    runs=[]
    manifest={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'kind':'production_save_fixed_D_AB','warmup':0,'repeats':a.repeats,
              'gpu_settings_changed':False,'D':a.d,
              'versions':versions,
              'allow_version_difference':a.allow_version_difference,
              'inherited_ntt_environment':{k:v for k,v in os.environ.items() if k.startswith(('ECM_OPENCL_','NTT_'))},
              'boundaries':'engine total and full parent process; Stage1 excluded',
              'binaries':{k:{'path':str(v),'sha256':sha(v)} for k,v in binaries.items()},
              'telemetry_before':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,clocks.current.graphics,power.draw,utilization.gpu','--format=csv'],text=True)}
    # OpenCL indices can be duplicated across ICD platforms. Keep the list and
    # require the test operator to select the physical card matching CUDA.
    manifest['opencl_devices']=subprocess.check_output([str(binaries['opencl']),'--list-devices'],text=True)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    with (a.out/'gpu.csv').open('w',encoding='utf-8') as telemetry:
        poll=subprocess.Popen(['nvidia-smi',f'--id={a.cuda_device}','--query-gpu=timestamp,clocks.current.graphics,power.draw,utilization.gpu','--format=csv','-lms','500'],stdout=telemetry,stderr=subprocess.DEVNULL,creationflags=flags)
        try:
            for exponent in map(int,a.exponents.split(',')):
                saves=list(a.inputs.glob(f'm{exponent}_n*_s*.save'))
                if len(saves)!=1:
                    raise RuntimeError(f'Need exactly one save for M{exponent}')
                bits=int(re.search(r'_n(\d+)_',saves[0].name)[1])
                frozen=a.out/'inputs'/saves[0].name
                frozen.parent.mkdir(exist_ok=True)
                shutil.copy2(saves[0],frozen)
                for b2 in map(int,a.b2.split(',')):
                    for repeat in range(a.repeats):
                        order=['cuda','opencl'] if repeat%2==0 else ['opencl','cuda']
                        for backend in order:
                            folder=a.out/f'm{exponent}_n{bits}_b{b2}_r{repeat}_{backend}'
                            folder.mkdir()
                            device=a.cuda_device if backend=='cuda' else a.opencl_device
                            config=folder/'ecm.toml'
                            config.write_text(f'''[stage2]
stage2_device={device}
stage2_log_level="phases"
stage2_auto_carrier=false
stage2_tune_runtime=false
stage2_short_calibration=false
stage2_arena_mb=512
stage2_batch_mb=64
stage2_fold_mb=640
stage2_results_file="results.jsonl"
stage2_log_file="worker.log"
''',encoding='utf-8')
                            cmd=[str(binaries[backend]),'--config',str(config.resolve()),'--save',str(frozen.resolve()),
                                 '--device',str(device),'--b2',str(b2),'--d',str(a.d),'--curves','1','--factor-only',
                                 '--carrier-exponent','0','--auto-carrier','0','--tune-runtime','0','--short-calibration','0','--ntt-counts','1']
                            (folder/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
                            start=time.perf_counter()
                            with (folder/'stdout.log').open('w',encoding='utf-8') as log:
                                result=subprocess.run(cmd,cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=300,creationflags=flags)
                            process=time.perf_counter()-start
                            if result.returncode:
                                raise RuntimeError(f'{folder}: exit {result.returncode}')
                            text=(folder/'worker.log').read_text(encoding='utf-8-sig')
                            full=re.search(r'stage2_full_wall:.*?init=([\d.]+).*?main=([\d.]+).*?total=([\d.]+)',text)
                            shape=re.search(r'real_shape: D=(\d+) P=phi\(D\)/2=(\d+)',text)
                            rows=[json.loads(x) for x in (folder/'results.jsonl').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
                            if not full or not shape or int(shape[1])!=a.d or len(rows)!=1 or rows[0].get('factors',[]):
                                raise RuntimeError('Not a matching clean NF full curve: '+str(folder))
                            row={'backend':backend,'exponent':exponent,'bits':bits,'B2':b2,'D':a.d,'P':int(shape[2]),'repeat':repeat,
                                 'init_seconds':float(full[1]),'main_seconds':float(full[2]),'engine_seconds':float(full[3]),
                                 'process_seconds':process,'source_build':rows[0]['source-build'],'save_sha256':sha(frozen),
                                 'log':str(folder/'worker.log'),'log_sha256':sha(folder/'worker.log'),'result_sha256':sha(folder/'results.jsonl')}
                            runs.append(row)
                            with (a.out/'measurements.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                            print(f'{backend} M{exponent} N={bits} B2={b2} repeat={repeat}: engine={row["engine_seconds"]:.6f}s process={process:.3f}s',flush=True)
        finally:
            poll.terminate()
            poll.wait()
    manifest['complete']=True
    manifest['runs']=len(runs)
    (a.out/'completed.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with (a.out/'measurements.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(runs[0]));writer.writeheader();writer.writerows(runs)

if __name__=='__main__':
    main()
