"""Fixed-save, fixed-D full Stage2 A/B; retains every invocation as evidence."""
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
from stage2_windows_job import WindowsJob

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
    parser.add_argument('--resume',action='store_true',help='Continue an interrupted matrix without replacing evidence')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--exponents',default='503,2003,5003,8011')
    parser.add_argument('--b2',default='260000,2600000')
    parser.add_argument('--d',type=int,default=2310)
    parser.add_argument('--arena-mb',type=int,default=512)
    parser.add_argument('--carrier-modes',default='ordinary',help='ordinary,mersenne; both use the same real target/save')
    parser.add_argument('--timeout',type=float,default=3600,help='Per-curve wall limit; the isolated process tree is drained on failure')
    parser.add_argument('--cuda-device',type=int,default=1)
    parser.add_argument('--opencl-device',type=int,default=1)
    parser.add_argument('--allow-version-difference',action='store_true',
                        help='Explicitly compare an identified frozen version with a newer implementation')
    a=parser.parse_args()
    exponents=list(map(int,a.exponents.split(',')))
    bounds=list(map(int,a.b2.split(',')))
    modes=a.carrier_modes.split(',')
    if not modes or len(set(modes))!=len(modes) or not set(modes)<={'ordinary','mersenne'} or a.timeout<=0:
        raise SystemExit('Need unique carrier modes ordinary/mersenne and a positive timeout')
    if (not exponents or not bounds or min(exponents+bounds)<=0 or a.d<=0
        or len(set(exponents))!=len(exponents) or len(set(bounds))!=len(bounds)):
        raise SystemExit('Need distinct positive exponents/B2 and a positive D')
    if (a.out.exists() and not a.resume) or not 1<=a.repeats<=20 or a.arena_mb<=0:
        raise SystemExit('Choose a new directory, repeats=1..20 and a positive arena budget')
    if a.resume and ((a.out/'completed.json').exists() or not (a.out/'manifest.json').exists()):
        raise SystemExit('Resume requires an interrupted matrix with a manifest')
    a.out.mkdir(parents=True,exist_ok=a.resume)
    binaries={}
    for backend,build,name in [('opencl',a.opencl_build,'ecm_opencl_stage2.exe'),('cuda',a.cuda_build,'ecm_cuda_stage2.exe')]:
        dest=a.out/'bin'/backend
        if not a.resume:
            dest.mkdir(parents=True)
            (dest/'.gitignore').write_text('*\n',encoding='utf-8')
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
              'gpu_settings_changed':False,'D':a.d,'arena_mb':a.arena_mb,
              'matrix_plan':{'exponents':exponents,'B2':bounds,'carrier_modes':modes,'timeout':a.timeout,'cuda_device':a.cuda_device,'opencl_device':a.opencl_device},
              'versions':versions,
              'allow_version_difference':a.allow_version_difference,
              'inherited_ntt_environment':{k:v for k,v in os.environ.items() if k.startswith(('ECM_OPENCL_','NTT_'))},
              'boundaries':'engine total and full parent process; Stage1 excluded',
              'binaries':{k:{'path':str(v),'sha256':sha(v)} for k,v in binaries.items()},
              'telemetry_before':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,clocks.current.graphics,power.draw,utilization.gpu','--format=csv'],text=True)}
    # Include the DLL, resource source and build identity, not just the EXE.
    manifest['frozen_files']={str(p.relative_to(a.out)):sha(p)
                             for p in sorted((a.out/'bin').rglob('*')) if p.is_file()}
    attempt=0
    if a.resume:
        previous=json.loads((a.out/'manifest.json').read_text(encoding='utf-8'))
        for key in ('D','arena_mb','repeats','versions','inherited_ntt_environment','frozen_files','matrix_plan'):
            if manifest[key]!=previous.get(key):
                raise RuntimeError('Resume configuration/build mismatch: '+key)
        manifest=previous
        runs=[json.loads(s) for s in (a.out/'measurements.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in runs:
            log=Path(row['log'])
            if sha(log)!=row['log_sha256'] or sha(log.parent/'results.jsonl')!=row['result_sha256']:
                raise RuntimeError('Modified completed measurement: '+str(log))
        attempt=1
        while (a.out/f'gpu_resume{attempt}.csv').exists():
            attempt+=1
    shutil.copy2(Path(__file__),a.out/(f'bench_resume{attempt}.py' if attempt else 'bench_script.py'))
    existing={(r['exponent'],r['B2'],r['repeat'],r['backend'],r.get('carrier_exponent',0)) for r in runs}
    expected={(e,b,r,k,e if m=='mersenne' else 0) for e in exponents for b in bounds for r in range(a.repeats) for k in binaries for m in modes}
    if len(existing)!=len(runs) or not existing<=expected:
        raise RuntimeError('Duplicate or out-of-plan completed measurements')
    # OpenCL indices can be duplicated across ICD platforms. Keep the list and
    # require the test operator to select the physical card matching CUDA.
    manifest['opencl_devices']=subprocess.check_output([str(binaries['opencl']),'--list-devices'],text=True)
    if not a.resume:
        (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    with (a.out/(f'gpu_resume{attempt}.csv' if attempt else 'gpu.csv')).open('x',encoding='utf-8') as telemetry:
        poll=subprocess.Popen(['nvidia-smi',f'--id={a.cuda_device}','--query-gpu=timestamp,clocks.current.graphics,power.draw,utilization.gpu','--format=csv','-lms','500'],stdout=telemetry,stderr=subprocess.DEVNULL,creationflags=flags)
        try:
            for exponent in exponents:
                saves=list(a.inputs.glob(f'm{exponent}_n*_s*.save'))
                if len(saves)!=1:
                    raise RuntimeError(f'Need exactly one save for M{exponent}')
                bits=int(re.search(r'_n(\d+)_',saves[0].name)[1])
                frozen=a.out/'inputs'/saves[0].name
                frozen.parent.mkdir(exist_ok=True)
                if frozen.exists():
                    if sha(frozen)!=sha(saves[0]):
                        raise RuntimeError('Resume input changed: '+str(frozen))
                else:
                    shutil.copy2(saves[0],frozen)
                for b2 in bounds:
                    for repeat in range(a.repeats):
                        backends=['cuda','opencl'] if repeat%2==0 else ['opencl','cuda']
                        order=modes if repeat%2==0 else list(reversed(modes))
                        for mode,backend in [(m,k) for m in order for k in backends]:
                            carrier=exponent if mode=='mersenne' else 0
                            if (exponent,b2,repeat,backend,carrier) in existing:
                                continue
                            folder=a.out/f'm{exponent}_n{bits}_b{b2}_r{repeat}_{backend}_{mode}'
                            if folder.exists():
                                folder=a.out/f'{folder.name}_resume{attempt}'
                            folder.mkdir()
                            device=a.cuda_device if backend=='cuda' else a.opencl_device
                            config=folder/'ecm.toml'
                            config.write_text(f'''[stage2]
stage2_device={device}
stage2_log_level="phases"
stage2_auto_carrier=false
stage2_tune_runtime=false
stage2_short_calibration=false
stage2_arena_mb={a.arena_mb}
stage2_batch_mb=64
stage2_fold_mb=640
stage2_results_file="results.jsonl"
stage2_log_file="worker.log"
''',encoding='utf-8')
                            cmd=[str(binaries[backend]),'--config',str(config.resolve()),'--save',str(frozen.resolve()),
                                 '--device',str(device),'--b2',str(b2),'--d',str(a.d),'--curves','1','--factor-only',
                                 '--carrier-exponent',str(carrier),'--auto-carrier','0','--tune-runtime','0','--short-calibration','0','--ntt-counts','1']
                            (folder/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
                            start=time.perf_counter()
                            job=WindowsJob()
                            try:
                                with (folder/'stdout.log').open('w',encoding='utf-8') as log:
                                    result=subprocess.Popen(cmd,cwd=folder,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
                                    try:
                                        job.assign(result)
                                        result.wait(timeout=a.timeout)
                                        account=job.drain()
                                    except BaseException:
                                        # Own only this invocation and its descendants; never kill unrelated GPU workers.
                                        job.terminate();result.wait();job.drain();raise
                            finally:
                                job.close()
                            process=time.perf_counter()-start
                            if result.returncode:
                                raise RuntimeError(f'{folder}: exit {result.returncode}')
                            text=(folder/'worker.log').read_text(encoding='utf-8-sig')
                            full=re.search(r'stage2_full_wall:.*?init=([\d.]+).*?main=([\d.]+).*?total=([\d.]+)',text)
                            shape=re.search(r'real_shape: D=(\d+) P=phi\(D\)/2=(\d+)',text)
                            rows=[json.loads(x) for x in (folder/'results.jsonl').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
                            actual_bits=re.search(r'(?:N_bits|target_bits)=(\d+)',text)
                            if (not full or not shape or int(shape[1])!=a.d or not actual_bits or int(actual_bits[1])!=bits
                                or len(rows)!=1 or rows[0].get('factors',[]) or rows[0]['stage2-engine']['bad_factors']
                                or rows[0].get('status')!='NF' or rows[0]['errors']['invariant']
                                or int(rows[0]['N']).bit_length()!=bits or rows[0]['d']!=a.d or rows[0]['b2']!=b2
                                or rows[0]['poly-size']!=int(shape[2])
                                or rows[0]['stage2-engine']['carrier_exponent']!=carrier
                                or rows[0]['source-build'].lower()!=manifest['binaries'][backend]['sha256'].lower()):
                                raise RuntimeError('Not a matching clean NF full curve: '+str(folder))
                            row={'backend':backend,'exponent':exponent,'bits':bits,'B2':b2,'D':a.d,'P':int(shape[2]),'repeat':repeat,'mode':mode,'carrier_exponent':carrier,
                                 'init_seconds':float(full[1]),'main_seconds':float(full[2]),'engine_seconds':float(full[3]),
                                 'process_seconds':process,'source_build':rows[0]['source-build'],'save_sha256':sha(frozen),
                                 'log':str(folder/'worker.log'),'log_sha256':sha(folder/'worker.log'),'result_sha256':sha(folder/'results.jsonl'),'accounting':account}
                            runs.append(row)
                            with (a.out/'measurements.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                            print(f'{backend} {mode} M{exponent} N={bits} B2={b2} repeat={repeat}: engine={row["engine_seconds"]:.6f}s process={process:.3f}s',flush=True)
        finally:
            poll.terminate()
            poll.wait()
    manifest['complete']=True
    if {(r['exponent'],r['B2'],r['repeat'],r['backend'],r.get('carrier_exponent',0)) for r in runs}!=expected:
        raise RuntimeError('Matrix is missing planned measurements')
    manifest['runs']=len(runs)
    (a.out/'completed.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with (a.out/'measurements.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(runs[0]));writer.writeheader();writer.writerows(runs)

if __name__=='__main__':
    main()
