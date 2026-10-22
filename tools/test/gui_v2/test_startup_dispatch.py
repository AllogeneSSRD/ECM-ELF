"""Portable workflow discovery and stage-specific capability checks; no ECM work."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for name in ['exe','worker','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir();worker=a.worker.resolve()
    portable=out/'portable-中文';portable.mkdir();exe=portable/'ecm_gui_v2.exe';shutil.copy2(a.exe.resolve(),exe)
    cwd=out/'unrelated-cwd';cwd.mkdir();(cwd/'ecm.toml').write_text('[gui]\nNumWorkers=7\n[workers.1]\nautostart=0\n')
    legacy=Path(os.environ['SystemRoot'])/'System32/where.exe'
    queue=portable/'worktodo.txt';queue.write_text('[Worker #1]\nECMSTAGE2=1,2,127,-1,"absent.save",0,0,10\n')
    def config(path,first,second):
        path.write_text('[shared]\nworktodo='+json.dumps(str(queue))+'\n[gui]\nNumWorkers=1\nexit_confirm="stop"\ngraceful_stop_ms=1000\nexe='+json.dumps(str(first))+'\nstage2_exe='+json.dumps(str(second))+'\n[workers.1]\nautostart=0\nextra_args="--scenario ok --lines 0 --delay-ms 0"\n',encoding='utf-8')
    adjacent=portable/'ecm.toml';config(adjacent,legacy,worker)
    explicit=out/'explicit.toml';config(explicit,worker,legacy)
    startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    results={}
    def run(name,args):
        log=out/(name+'.log');cmd=[str(exe),'--frames','10','--no-nvml','--no-pet','--ui-state',str(out/(name+'.ui.toml')),'--diagnostics',str(log),*args]
        proc=subprocess.run(cmd,cwd=cwd,startupinfo=startup,timeout=30);assert proc.returncode==0,(name,proc.returncode)
        return log.read_text(encoding='utf-8')
    text=run('stage2-only',['--config',str(adjacent),'--start-worker','1'])
    results['stage2_ignores_unused_legacy_stage1']='started pid=' in text and 'lacks queue ownership' not in text and 'state Restarting' not in text
    text=run('adjacent-default',[])
    results['adjacent_config_found']=('config='+str(adjacent)) in text
    results['cwd_decoy_ignored']=('config='+str(cwd/'ecm.toml')) not in text
    text=run('explicit-override',['--config',str(explicit),'--start-worker','1'])
    results['explicit_config_overrides_adjacent']=('config='+str(explicit)) in text
    results['selected_legacy_stage2_refused']='lacks queue ownership' in text and str(legacy).casefold() in text.casefold() and 'started pid=' not in text
    text=run('explicit-missing',['--config',str(out/'missing.toml')])
    results['missing_explicit_does_not_fallback']='config=' not in text
    config(adjacent,legacy,worker);marker=out/'switch.marker'
    adjacent.write_text(adjacent.read_text(encoding='utf-8').replace('--scenario ok --lines 0 --delay-ms 0','--scenario v2-queue-switch --marker '+str(marker).replace('\\','/')),encoding='utf-8')
    log=out/'automatic-switch.log';proc=subprocess.Popen([str(exe),'--config',str(adjacent),'--start-worker','1','--frames','50','--no-nvml','--no-pet','--ui-state',str(out/'switch.ui.toml'),'--diagnostics',str(log)],cwd=cwd,startupinfo=startup)
    end=time.monotonic()+12;started=False
    try:
        while proc.poll() is None and time.monotonic()<end:
            if log.exists() and 'started pid=' in log.read_text(encoding='utf-8'):started=True;break
            time.sleep(.03)
        if started:
            queue.write_text('[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n');marker.write_text('switch')
        proc.wait(timeout=20)
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
    text=log.read_text(encoding='utf-8')
    results['automatic_switch_checks_next_engine']=started and 'dispatch ECMSTAGE1' in text and 'lacks queue ownership' in text and 'dispatched pid=' not in text and proc.returncode==0
    adjacent.unlink();text=run('missing-adjacent',[])
    results['absent_config_keeps_gui_open']='config=' not in text and 'frame: 1' in text
    report=dict(passed=all(results.values()),cases=results,scope='Isolated fake protocol-aware worker and inert Windows where.exe as legacy probe; no real queue or compute worker modified')
    (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report));assert report['passed'],results

if __name__=='__main__':main()
