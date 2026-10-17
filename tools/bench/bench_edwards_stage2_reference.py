"""Pair Edwards Stage2 against frozen Prime95 screen/result evidence.

Prepare verified exact-N inputs, then run NTT/plain CPU and ordinary/carrier GPU.
Private identifiers/checksums are retained only in ignored source snapshots.
All formal samples, including first invocations, are retained; no extrapolation.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
sys.path.insert(0,str(ROOT/'tools/log_parser'))
sys.path.insert(0,str(ROOT/'tools/bench'))
from analyze_prime95_ecm import rich_runs
from compare_prime95_gpu import recover
from bench_stage2_n_scaling import Monitor,fields,number


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,data):
    # One collector owns this file. Raw invocation logs are never overwritten.
    temp=Path(str(path)+'.new');temp.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n');temp.replace(path)


def prepare(a,out):
    if a.resume_prepare:
        report=read(out/'plan.json')
        if report['prepared'] or report['runs']:raise ValueError('Preparation is already complete or measured')
        for b in report['binaries'].values():
            if sha(b['path'])!=b['sha256']:raise ValueError('Frozen binary changed')
        if a.ladder_checker:
            target=out/'bin'/a.ladder_checker.name
            if target.exists() and sha(target)!=sha(a.ladder_checker):raise ValueError('Verifier changed')
            if not target.exists():shutil.copy2(a.ladder_checker,target)
            report['ladder_checker']=dict(path=str(target),sha256=sha(target))
            report.setdefault('preparation_continuations',[]).append(dict(utc=stamp(),script_sha256=sha(__file__),
                reason='Use independently cross-checked GMP x-only ladder; retain all Python-verified cases and source snapshots'))
            write(out/'plan.json',report)
        prepare_inputs(report,out);return
    out.mkdir(exist_ok=False,parents=True)
    sources=out/'prime95';sources.mkdir()
    record_path=sources/'results.snapshot.json.txt';shutil.copy2(a.results,record_path)
    records=[]
    for i,s in enumerate(record_path.read_text(encoding='utf-8-sig').splitlines(),1):
        if s.strip():r=json.loads(s);r['_line']=i;records.append(r)
    report=dict(created_utc=stamp(),git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf-8').strip(),
                source_files={},prime95_rows=[],cases=[],runs=[],failures=[],complete=False,prepared=False,
                engines=a.engines,repeats=a.repeats,thread_limit=a.threads,cpu_memory_mib=a.memory_mib,
                device=a.device,arena_mb=a.arena_mb,fold_mb=a.fold_mb,batch_mb=a.batch_mb,
                cpu_affinity_mask=a.cpu_affinity,
                protocol=dict(warmups=0,first_invocation='retained',order='rotate engine order for each repetition',
                              cpu='curve setup plus full libecm return; process timer also retained',
                              gpu='stage2_full_wall init+main, final arithmetic and GCD included',
                              prime95='init+main+GCD; nominal and adjusted B2 retained',
                              clock_note='Prime95 4 cores at 4 GHz per user; new CPU frequency not locked by collector',
                              power_note='No power/clock changes; NVML observations retained; other GPU may be busy'))
    for log in sorted(a.prime95.glob('screen_*.log')):
        target=sources/log.name;shutil.copy2(log,target);text=target.read_text(encoding='utf-8-sig')
        report['source_files'][str(log.resolve())]=sha(target)
        rows=rich_runs(text,[],[26_000_000_000,260_000_000_000],.10)
        model='Edwards' if '_ed_' in log.name else 'Montgomery'
        matching=[r for r in records if (('Edwards' in r)==(model=='Edwards'))]
        rows,tasks,warnings,clocks=recover(dict(runs=rows),matching,text.splitlines())
        if warnings:raise ValueError('Prime95 target recovery is ambiguous: '+str(warnings))
        for r in rows:
            if r['benchmark_eligible'] and not r.get('N_hex'):raise ValueError('Complete curve has no verified N')
            r['source_log']=log.name
        report['prime95_rows'].extend(rows)
        write(sources/(log.stem+'.analysis.json'),dict(runs=rows,tasks=tasks,clocks=clocks))
    report['source_files'][str(a.results.resolve())]=sha(record_path)
    if a.prime95_exe:
        report['prime95_exe_sha256']=sha(a.prime95_exe)
    binary=out/'bin';binary.mkdir()
    report['binaries']={}
    for name,source in [('stage1',a.stage1),('cpu',a.cpu),('gpu',a.gpu)]:
        target=binary/source.name;shutil.copy2(source,target)
        report['binaries'][name]=dict(path=str(target),sha256=sha(target),original=str(source.resolve()))
        dll=source.parent/'gmp-10.dll'
        if dll.exists():
            if (binary/dll.name).exists() and sha(binary/dll.name)!=sha(dll):raise ValueError('GMP dependency mismatch')
            shutil.copy2(dll,binary/dll.name)
        if name=='gpu' and (source.parent/'build_manifest.json').exists():shutil.copy2(source.parent/'build_manifest.json',binary/'gpu_build_manifest.json')
    report['dll_sha256']=sha(binary/'gmp-10.dll')
    write(out/'plan.json',report)
    prepare_inputs(report,out)


def prepare_inputs(report,out):
    inputs=out/'inputs';inputs.mkdir(exist_ok=True)
    unique={(c['N_hex'],c['b1'],c['sigma']):c for c in report['cases']}
    for c in unique.values():
        if sha(c['save'])!=c['save_sha256']:raise ValueError('Verified Stage1 save changed')
    for r in report['prime95_rows']:
        if r['curve_type']!='Edwards' or not r['benchmark_eligible']:continue
        key=(r['N_hex'],r['b1'],int(r['s']))
        if key in unique:continue
        tag=f'm{r["exponent"]}_n{r["actual_input_bits"]}_s{r["s"]}'
        save=inputs/(tag+'.save');ini=inputs/(tag+'.ini')
        ini.write_text('user=\ncomputer=\n',encoding='utf-8')
        cmd=[report['binaries']['stage1']['path'],'--curve','ed','--target','cpu','--impl','scalar',
             '-ini',str(ini),'-sigma',r['s'],'-gpucurves','1','--tmp-dir',str(inputs),'-save',str(save),str(r['b1']),str(r['b1'])]
        print('prepare',tag,flush=True)
        if save.exists():
            if not (inputs/(tag+'.log')).exists():raise ValueError('Existing save without preparation evidence')
        else:
            proc=subprocess.run(cmd,input=str(int(r['N_hex'],16)).encode(),capture_output=True,cwd=out,timeout=1800)
            (inputs/(tag+'.log')).write_bytes(proc.stdout+proc.stderr)
            if proc.returncode:raise RuntimeError('Stage1 preparation failed: '+tag)
        spec=importlib.util.spec_from_file_location('edwards_oracle',ROOT/'tools/test/test_edwards_gpu_pipeline.py')
        oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
        if report.get('ladder_checker'):
            checker=report['ladder_checker']
            if sha(checker['path'])!=checker['sha256']:raise ValueError('Independent verifier changed')
            checked=subprocess.run([checker['path'],str(save)],capture_output=True,cwd=out,timeout=1800)
            (inputs/(tag+'.ladder.log')).write_bytes(checked.stdout+checked.stderr)
            if checked.returncode:raise ValueError('Independent ladder failed: '+tag)
            saved=[]
            for line in save.read_text().splitlines():
                f=dict(token.strip().split('=',1) for token in line.split(';') if token.strip())
                saved.append(tuple(int(f[k],0) for k in ('N','X','SIGMA','B1'))+(f['STATUS'],))
        else:saved=oracle.records(save)
        if len(saved)!=1 or saved[0][0]!=int(r['N_hex'],16) or saved[0][2]!=int(r['s']) or saved[0][3]!=r['b1'] or saved[0][-1]!='POINT':
            raise ValueError('No complete POINT for '+tag)
        unique[key]=dict(id=tag,exponent=r['exponent'],bits=r['actual_input_bits'],N_hex=r['N_hex'],sigma=int(r['s']),b1=r['b1'],
                         known=r['known_factors_before_curve'],save=str(save),save_sha256=sha(save),
                         input_verification='GMP x-only ladder' if report.get('ladder_checker') else 'Python x-only ladder')
        report['cases']=list(unique.values());write(out/'plan.json',report)
    report['prepared']=True;write(out/'plan.json',report)
    print('PREPARED',len(unique),'exact-N Edwards inputs; historical rows',len(report['prime95_rows']),flush=True)


def run(a,out):
    report=read(out/'plan.json');assert report['prepared']
    if report['complete']:raise ValueError('Study already complete')
    if a.threads is not None and a.threads!=report['thread_limit']:
        if report['runs']:raise ValueError('Thread limit cannot change after measurement starts')
        report.setdefault('protocol_amendments',[]).append(dict(utc=stamp(),old_thread_limit=report['thread_limit'],
            new_thread_limit=a.threads,reason='Use single-thread CPU baseline; historical Prime95 four-thread data remains separate'))
        report['thread_limit']=a.threads;write(out/'plan.json',report)
    report.setdefault('cpu_affinity_mask',a.cpu_affinity)
    for b in report['binaries'].values():
        if sha(b['path'])!=b['sha256']:raise ValueError('Frozen binary changed')
    if sha(out/'bin/gmp-10.dll')!=report['dll_sha256']:raise ValueError('GMP changed')
    collector=out/('collector_'+sha(__file__)+'.py')
    if not collector.exists():shutil.copy2(__file__,collector)
    ini=out/'bench.ini'
    settings='user=\ncomputer=\nstage2_tune_runtime=0\nstage2_auto_carrier=0\n'
    if ini.exists() and ini.read_text()!=settings:raise ValueError('Benchmark INI changed')
    ini.write_text(settings,encoding='utf-8')
    runs=out/'runs';runs.mkdir(exist_ok=True)
    monitor=Monitor(report['device']);report['gpu_identity']=monitor.identity
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    api.SetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    api.SetProcessAffinityMask.restype=wintypes.BOOL
    class Memory(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    memfn=ctypes.WinDLL('psapi').GetProcessMemoryInfo;memfn.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_') and k!='CUDA_VISIBLE_DEVICES'}
    env.update(NTT_NO_PROGRESS='1',CUDA_LAUNCH_BLOCKING='0',OMP_WAIT_POLICY='PASSIVE')
    cases={c['id']:c for c in report['cases']}
    cells=[]
    for r in report['prime95_rows']:
        if r['curve_type']!='Edwards' or not r['benchmark_eligible']:continue
        case=next(c for c in cases.values() if c['N_hex']==r['N_hex'] and c['sigma']==int(r['s']) and c['b1']==r['b1'])
        cells.append((case,r))
    if a.exponents:cells=[(c,r) for c,r in cells if c['exponent'] in a.exponents]
    engines=a.engines or report['engines'];repeats=a.repeats or report['repeats']
    report.setdefault('execution_batches',[]).append(dict(started_utc=stamp(),engines=engines,repeats=repeats,
        exponents=a.exponents,thread_limit=report['thread_limit'],script_sha256=sha(__file__)))
    report['protocol']['process']='launch through observed exit; polling interval 0.25s, detection delay at most one polling interval'
    write(out/'plan.json',report)
    try:
        for rep in range(repeats):
            for case,reference in cells:
                for engine in engines[rep%len(engines):]+engines[:rep%len(engines)]:
                    tag=f'{case["id"]}_b{reference["b2"]}_{engine}_r{rep+1}'
                    if any(r['id']==tag for r in report['runs']):continue
                    prefix=runs/tag
                    if prefix.with_suffix('.driver.log').exists():raise ValueError('Unrecorded invocation exists; preserve evidence: '+tag)
                    if sha(case['save'])!=case['save_sha256']:raise ValueError('Input save changed')
                    gpu=engine.startswith('gpu');exe=report['binaries']['gpu' if gpu else 'cpu']['path']
                    cmd=[exe,'--ini',str(ini),'--save',case['save'],'--b2',str(reference['b2']),'--curves','1',
                         '--results',str(prefix.with_suffix('.jsonl')),'--submission',str(prefix.with_suffix('.json.txt')),
                         '--exponent',str(case['exponent'])]
                    for f in case['known']:cmd+=['--known-factor',str(f)]
                    if gpu:
                        cmd+=['--device',str(report['device']),'--d','0','--short-calibration','0','--tune-runtime','0',
                              '--arena-mb',str(report['arena_mb']),'--owner-budget-mb',str(report['fold_mb']),
                              '--batch-mb',str(report['batch_mb']),'--factor-only','--log-level','curve',
                              '--log',str(prefix.with_suffix('.engine.log')),'--debug-log-file',str(prefix.with_suffix('.debug.log')),
                              '--carrier-exponent',str(case['exponent'] if engine=='gpu-carrier' else 0)]
                    else:cmd+=['--backend',engine.removeprefix('cpu-'),'--threads',str(report['thread_limit']),'--memory-mib',str(report['cpu_memory_mib'])]
                    print('RUN',tag,flush=True);started=stamp();observations=[];start=time.perf_counter()
                    with prefix.with_suffix('.driver.log').open('wb') as log:
                        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=out,env=env)
                        affinity=report.get('cpu_affinity_mask',a.cpu_affinity)
                        if not gpu and affinity and not api.SetProcessAffinityMask(proc._handle,affinity):
                            proc.kill();proc.wait();raise ctypes.WinError(ctypes.get_last_error())
                        while proc.poll() is None:
                            if gpu:observations.append(monitor.sample())
                            if time.perf_counter()-start>a.timeout:proc.kill();proc.wait();raise TimeoutError(tag)
                            time.sleep(.25)
                        wall=time.perf_counter()-start
                        ft=[wintypes.FILETIME() for _ in range(4)]
                        if not api.GetProcessTimes(proc._handle,*map(ctypes.byref,ft)):raise ctypes.WinError(ctypes.get_last_error())
                        cpu=sum((t.dwHighDateTime<<32)+t.dwLowDateTime for t in ft[2:])/1e7
                        mem=Memory();mem.cb=ctypes.sizeof(mem)
                        if not memfn(proc._handle,ctypes.byref(mem),mem.cb):raise ctypes.WinError(ctypes.get_last_error())
                    write(prefix.with_suffix('.telemetry.json'),observations)
                    if proc.returncode:raise RuntimeError(tag+' exit '+str(proc.returncode))
                    result=read(prefix.with_suffix('.jsonl'))
                    n=int(case['N_hex'],16);factors=result.get('factors',[])
                    if any(not 1<int(f)<n or n%int(f) for f in factors):raise ValueError('Invalid factor')
                    row=dict(id=tag,case=case['id'],engine=engine,repeat=rep+1,B2=reference['b2'],nominal_B2=reference['b2_requested'],
                             started_utc=started,finished_utc=stamp(),process_seconds=wall,process_cpu_seconds=cpu,
                             parent_cpu_wall_ratio=cpu/wall,factors=factors,
                             full_sample=not factors,result_sha256=sha(prefix.with_suffix('.jsonl')),command=cmd,
                             prefix=str(prefix),script_sha256=sha(__file__))
                    row['parent_peak_ws_bytes' if gpu else 'process_peak_ws_bytes']=mem.peak_ws
                    if not gpu:row['cpu_affinity_mask']=affinity
                    if gpu:
                        if result['N_hex']!=case['N_hex'] or result['B1']!=case['b1'] or result['B2']!=reference['b2'] or result['bad_factors']:raise ValueError('GPU result mismatch')
                        normal=prefix.with_suffix('.engine.log').read_text(encoding='utf-8')
                        debug=prefix.with_suffix('.debug.log').read_text(encoding='utf-8') if prefix.with_suffix('.debug.log').exists() else ''
                        text=normal+'\n'+debug
                        ledger=fields(text,'stage2_full_wall');shape=fields(text,'real_shape');s4=fields(text,'s4_multiply_stats')
                        if ledger['clean']!='1' or abs(float(ledger['total'])-float(ledger['init'])-float(ledger['main']))>.003:raise ValueError('Unclosed GPU wall ledger')
                        if int(s4['gmp_selftest_bad']) or int(s4['gmp_check_bad']) or not int(s4['gmp_checked']):raise ValueError('Missing arithmetic verification')
                        row.update(engine_seconds=float(ledger['total']),phase_seconds=ledger,shape=shape,s4=s4,
                                   carrier_exponent=result['carrier_exponent'],D=number(shape['D']),P=number(shape['P']))
                        labels=('real_baby','real_giant_base','real_giant_seed_pair','real_batched_prepare','real_batched_wall',
                                'real_batched_split','real_batched_folddevice','scaled_frontier_device','scaled_root_device','ntt_workspace_stats')
                        row['stats']={label:fields(text,label,False) for label in labels}
                    else:
                        if result['b1']!=case['b1'] or result['b2']!=reference['b2'] or result['Edwards']['sigma']!=case['sigma']:raise ValueError('CPU result mismatch')
                        row.update(engine_seconds=result['stage2-engine']['seconds'],backend=result['stage2-engine']['backend'])
                    report['runs'].append(row);write(out/'plan.json',report)
                    print('DONE',tag,'engine',round(row['engine_seconds'],6),'wall',round(wall,3),'full',row['full_sample'],flush=True)
        # Subset pilot passes do not mark an unmeasured full study complete.
        expected=len([r for r in report['prime95_rows'] if r['curve_type']=='Edwards' and r['benchmark_eligible']])*len(report['engines'])*report['repeats']
        report['complete']=len(report['runs'])==expected;write(out/'plan.json',report)
    except Exception as exc:
        report['failures'].append(dict(utc=stamp(),error=str(exc)));write(out/'plan.json',report);raise
    finally:monitor.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','run'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prime95',type=Path)
    p.add_argument('--results',type=Path)
    p.add_argument('--prime95-exe',type=Path)
    p.add_argument('--stage1',type=Path)
    p.add_argument('--cpu',type=Path)
    p.add_argument('--gpu',type=Path)
    p.add_argument('--engines',nargs='+',choices=('gpu','gpu-carrier','cpu-ntt','cpu-plain'))
    p.add_argument('--repeats',type=int)
    p.add_argument('--exponents',type=int,nargs='+',help='Run only a declared subset, keeping the full plan incomplete')
    p.add_argument('--threads',type=int,help='CPU OpenMP thread limit; prepare defaults to 1, run preserves the stored plan unless explicitly amended before any measurement')
    p.add_argument('--cpu-affinity',type=lambda v:int(v,0),default=0xAA,help='CPU process affinity: default logical CPUs 1,3,5,7 as in the reference logs; 0 leaves scheduling unchanged')
    p.add_argument('--memory-mib',type=int,default=4096)
    p.add_argument('--arena-mb',type=int,default=6300)
    p.add_argument('--fold-mb',type=int,default=640)
    p.add_argument('--batch-mb',type=int,default=256)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--timeout',type=int,default=7200)
    p.add_argument('--resume-prepare',action='store_true',help='Continue an incomplete preparation without rewriting source snapshots or verified saves')
    p.add_argument('--ladder-checker',type=Path,help='Independent GMP ladder executable for resumed large-input verification')
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='prepare':
        if not a.resume_prepare and not all((a.prime95,a.results,a.stage1,a.cpu,a.gpu)):p.error('prepare requires source logs/results and all three executables')
        a.engines=a.engines or ['gpu','gpu-carrier','cpu-ntt','cpu-plain'];a.repeats=a.repeats or 3
        a.threads=a.threads or 1
        prepare(a,out)
    else:run(a,out)


if __name__=='__main__':main()
