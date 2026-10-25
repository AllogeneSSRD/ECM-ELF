"""Real-worker intake, live-queue, completion and P95 delivery regression matrix.

Use fresh evidence and portable binary copies. Optional timings compare whole
process samples and explicit save/result/queue boundaries; no component peaks
are added. Run outside restricted Windows path-canonicalization sandboxes.
"""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

def quoted(value):return json.dumps(str(value),ensure_ascii=False)
def sigma(row):return row.get('Edwards',row.get('GMP-ECM-param3',{})).get('sigma',row.get('sigma'))
def rows(path):return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=0)
    p.add_argument('--baseline',type=Path);p.add_argument('--samples',type=int,default=2)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,device=a.device,checks=[],runs=[],performance=[])
    def portable(source,name):
        folder=out/name;folder.mkdir();exe=folder/'ecm_cuda.exe';shutil.copy2(source,exe);shutil.copy2(a.runtime,folder/'gmp-10.dll')
        return exe
    exe=portable(a.exe.resolve(),'binary')
    report['binary_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
    def save_report():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def passed(name):report['checks'].append(name);save_report();print('PASS',name,flush=True)
    def run(name,args,case,text=None,success=True,edit=None,binary=exe):
        start=time.perf_counter();proc=subprocess.Popen([str(binary),*map(str,args)],cwd=case,
            stdin=subprocess.PIPE if text is not None else subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace')
        if text is not None:proc.stdin.write(text);proc.stdin.close()
        lines=[];events={};peak=[0];stop=threading.Event()
        def memory():
            if not hasattr(ctypes,'windll'):return
            class Counters(ctypes.Structure):
                _fields_=[('cb',ctypes.c_uint32),('faults',ctypes.c_uint32)]+[(v,ctypes.c_size_t) for v in
                    ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
            kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p
            handle=kernel.OpenProcess(0x410,False,proc.pid)
            if not handle:return
            try:
                while not stop.is_set():
                    c=Counters();c.cb=ctypes.sizeof(c)
                    if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle),ctypes.byref(c),c.cb):peak[0]=max(peak[0],c.PeakWorkingSetSize)
                    stop.wait(.02)
            finally:kernel.CloseHandle(ctypes.c_void_p(handle))
        monitor=threading.Thread(target=memory,daemon=True);monitor.start()
        timer=threading.Timer(180,proc.kill);timer.start()
        try:
            for line in proc.stdout:
                lines.append(line);now=time.perf_counter()-start
                if 'Saved ' in line and 'curve line(s)' in line or 'Edwards Stage1 save:' in line:events['save']=now
                if 'Stage1 results:' in line:events['result']=now
                if 'Stage1 save range:' in line:events['range']=now
                if 'queue done' in line:events['queue_done']=now
                if edit is not None and 'START:' in line:edit(line);edit=None
            code=proc.wait(timeout=10)
        finally:timer.cancel();stop.set();monitor.join(timeout=2)
        log=''.join(lines);(out/(name+'.log')).write_text(log,encoding='utf-8')
        result=dict(name=name,returncode=code,seconds=time.perf_counter()-start,events=events,sampled_peak_working_set_bytes=peak[0],args=list(map(str,args)))
        report['runs'].append(result);save_report()
        assert (code==0)==success,(name,code,log[-3000:])
        return log,result
    def setup(name,task,*,model='mont',target='cpu',extra='',bits=127):
        case=out/name;case.mkdir();queue=case/'worktodo.txt';queue.write_text(task,encoding='utf-8')
        ini=case/'ecm.toml';ini.write_text('[shared]\nworktodo='+quoted(queue)+'\n[stage1]\ncurve='+quoted(model)+'\ntarget='+quoted(target)+
            '\nimpl='+quoted('scalar' if target=='cpu' else 'auto')+'\ngpu_param=0\ndevice='+str(a.device)+
            '\nexp_cache="off"\nlog_file=""\nckpt_seconds=0\ntmp_dir='+quoted(case)+'\nfinished='+quoted(case/'finished.txt')+'\n'+extra,encoding='utf-8')
        return case,queue,ini
    for index,(args,key,value) in enumerate([
        (['--stage1-threads','-1','100'],'--stage1-threads','-1'),
        (['-gpucurves','4294967296','100'],'-gpucurves','4294967296'),
        (['--ckpt','nan','100'],'--ckpt','nan'),(['--curve','unknown','100'],'--curve','unknown'),
        (['--sigma','10junk','100'],'--sigma','10junk'),(['--config','--worker','2'],'--config','--worker')]):
        case=out/f'cli_invalid_{index}';case.mkdir();log,_=run(case.name,args,case,success=False)
        assert 'INPUT[cli]' in log and key in log and value in log and 'expected' in log
    passed('actionable_cli_diagnostics')
    case,q,ini=setup('forbidden_queue','ECM=1,2,127,-1,100,0,2,1000\n')
    original=q.read_bytes();log,_=run(case.name,['--config',ini,'--curve','mont'],case,success=False)
    assert 'INPUT[conflict]' in log and q.read_bytes()==original and not list(case.glob('*.save'))
    passed('task_cli_rejected_before_execution')
    case,q,ini=setup('masked_invalid_config','# empty\n')
    ini.write_text(ini.read_text().replace('device='+str(a.device),'device=-1')+'\n[workers.1.stage1]\ndevice=0\n')
    log,_=run(case.name,['--config',ini,'-d','0'],case,success=False)
    assert str(ini) in log and 'key=device' in log and "value='-1'" in log and 'expected' in log
    passed('invalid_explicit_config_not_masked')
    case,q,ini=setup('masked_invalid_affinity','# empty\n',extra='affinity="0,,1"\n')
    ini.write_text(ini.read_text()+'\n[workers.1.stage1]\naffinity="0"\n')
    log,_=run(case.name,['--config',ini,'--affinity','1'],case,success=False)
    assert 'key=affinity' in log and '0,,1' in log and 'expected' in log
    passed('invalid_affinity_not_masked')
    case,q,ini=setup('invalid_execution_combination','ECM=1,2,127,-1,100,0,2,1000\n')
    original=q.read_bytes();ini.write_text(ini.read_text().replace('target="cpu"','target="gpu"'))
    log,_=run(case.name,['--config',ini],case,success=False)
    assert 'keys=target,impl,reduction' in log and q.read_bytes()==original and '# ERROR' not in q.read_text()
    passed('invalid_config_combination_preserves_queue')
    case,q,ini=setup('precedence','',extra='sigma=2000\n')
    save=case/'m0_m127_100.save'
    native='ECM=1,2,127,-1,100,10000,4,1000'
    stage1=f'ECMSTAGE1=1,2,127,-1,"{save}",10000,0,3'
    foreign=stage1.replace(',0,3',',0,9')
    q.write_text('[Worker #1]\n'+native+'\n'+stage1+'\n[Worker #2]\n'+foreign+'\n')
    log,_=run(case.name,['--config',ini,'--target','cpu','--stage1-threads','1','--sigma','3000'],case)
    actual=rows(case/'stage1_results.jsonl');assert [sigma(r) for r in actual]==[1000,1001,1002,1003,3000,3001,3002]
    assert foreign in q.read_text() and '[Worker #1]\n[Worker #2]' in q.read_text()
    finished=(case/'finished.txt').read_text();assert ',10000,0,4' in finished and ',10000,4,3' in finished
    assert not list(case.glob('*.ckpt')) and not list(case.glob('*.stage1_completed*'))
    passed('task_cli_config_sigma_and_fresh_claim_skip')
    case,q,ini=setup('bad_task_skip','ECM=1,2,127,-1,100,0,bad,1000\nECM=1,2,127,-1,100,0,1,4000\n')
    log,_=run(case.name,['--config',ini],case)
    assert q.read_text().startswith('# ERROR ECM=') and 'key=curves_to_run' in log and len(rows(case/'stage1_results.jsonl'))==1
    passed('invalid_selected_row_marked_and_skipped')
    for name,delete in [('live_reorder',False),('active_deleted',True)]:
        first='ECM=1,2,521,-1,50000,0,1,5000'
        future='ECM=1,2,521,-1,100,0,1,6000'
        case,q,ini=setup(name,first+'\n'+future+'\n')
        def edit(line):q.write_text('# external edit\n'+future+'\n'+('' if delete else first+'\n')+'[Worker #2]\n'+future+'\n')
        log,_=run(name,['--config',ini],case,edit=edit,success=not delete)
        if delete:
            assert future in q.read_text() and len(rows(case/'stage1_results.jsonl'))==1
            assert list(case.glob('*.stage1_completed_worker1'))
            # Restore the deleted frozen ticket and stop at a Stage2 row. Recovery
            # commits the receipt without repeating arithmetic.
            q.write_text(first+'\nECMSTAGE2=1,2,521,-1,unused_100.save,0,0,1\n')
            resumed,_=run(name+'_recover',['--config',ini],case)
            assert 'START:' not in resumed and len(rows(case/'stage1_results.jsonl'))==1
        else:
            assert len(rows(case/'stage1_results.jsonl'))==2 and '[Worker #2]\n'+future in q.read_text()
        passed(name)
    case,q,ini=setup('local_stage2','ECM=1,2,127,-1,100,10000,2,7000\n',extra='auto_stage2=1\n')
    run(case.name,['--config',ini],case)
    assert q.read_text().startswith('ECMSTAGE2=') and ',10000,0,2' in q.read_text()
    passed('normalized_local_stage2_handoff')
    case,q,ini=setup('p95_failure','ECM=1,2,127,-1,100,10000,2,8000\n')
    external=case/'prime95';destination=external/'saves';destination.mkdir(parents=True)
    blocked=destination/'m0_m127_100.save';blocked.mkdir()
    ini.write_text(ini.read_text()+'p95_worktodo_path='+quoted(external/'worktodo.txt')+'\np95_sync_dir='+quoted(destination)+'\n')
    log,_=run(case.name,['--config',ini],case)
    assert not q.read_text().strip() and not (external/'worktodo.add').exists() and 'p95_add: pending' in log
    blocked.rmdir();log,_=run(case.name+'_retry',['--config',ini],case)
    assert 'START:' not in log and (external/'worktodo.add').exists() and blocked.is_file()
    assert str(blocked) in (external/'worktodo.add').read_text()
    passed('p95_durable_failure_and_idle_retry')
    # Unsupported constructions ignore Prime95 paths, including an unset sync
    # destination, and must not create transfer files in the external directory.
    case,q,ini=setup('edwards_ignores_p95','ECM=1,2,127,-1,100,0,2,9000\n',model='ed',extra='p95_worktodo_path="absent-p95/worktodo.txt"\n')
    run(case.name,['--config',ini],case);assert not (exe.parent/'absent-p95').exists();passed('unsupported_handoff_remains_local')
    for target in ['cpu','gpu']:
        case,q,ini=setup('edwards_no_points_'+target,'ECM=1,2,7,-1,100,10000,4,2026\n',model='ed',target=target,extra='auto_stage2=1\n')
        log,_=run(case.name,['--config',ini],case)
        assert 'no Stage2 point' in log and 'curves=0' in log
        assert not q.read_text().strip() and len(rows(case/'stage1_results.jsonl'))==4
        assert ',0,0' in (case/'finished.txt').read_text()
        passed('completed_without_stage2_points_'+target)
    if a.baseline:
        baseline=portable(a.baseline.resolve(),'baseline_binary')
        report['baseline_sha256']=hashlib.sha256(baseline.read_bytes()).hexdigest()
        report['timing_boundary']='parent receipt of save completion line -> Stage1 results line; result -> queue done; sampled OS peak working set (20ms), each sample is a fresh process'
        for bits in [521,10267]:
            for model in ['mont','ed']:
                for label,binary in [('baseline',baseline),('current',exe)]:
                    for sample in range(a.samples):
                        name=f'timing_{label}_{model}_{bits}_{sample}'
                        case,q,ini=setup(name,f'ECM=1,2,{bits},-1,100,0,1000,100000\n',model=model,target='gpu')
                        _,result=run(name,['--config',ini],case,binary=binary)
                        e=result['events'];assert 'save' in e and 'result' in e and 'queue_done' in e,(name,e)
                        report['performance'].append(dict(binary=label,curve=model,bits=bits,b1=100,curves=1000,sample=sample,
                            save_to_result_seconds=e['result']-e['save'],result_to_queue_done_seconds=e['queue_done']-e['result'],
                            process_seconds=result['seconds'],sampled_peak_working_set_bytes=result['sampled_peak_working_set_bytes']))
                        print('TIMING',name,report['performance'][-1],flush=True);save_report()
    report['complete']=True;save_report()

if __name__=='__main__':main()
