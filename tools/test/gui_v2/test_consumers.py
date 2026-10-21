"""Actual compute entry refusal and both startup orders of unchanged v1/v2."""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path
from win32_probe import window,close,user
def main():
    p=argparse.ArgumentParser()
    for name in ['exe','old','worker','probe','stage1','stage2','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir();paths={k:getattr(a,k).resolve() for k in ['exe','old','worker','probe','stage1','stage2']}
    startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    old=out/'ecm_gui.exe';shutil.copy2(paths['old'],old);shutil.copytree(Path('src/ecm/application/gui/localization'),out/'localization')
    queue=out/'worktodo.txt';queue.write_text('[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n')
    cfg=out/'ecm.toml';cfg.write_text(f'[shared]\nworktodo={json.dumps(str(queue))}\n[gui]\nNumWorkers=1\nexit_confirm="stop"\ngraceful_stop_ms=1200\nexe={json.dumps(str(paths["worker"]))}\n[workers.1]\nautostart=0\nextra_args="--scenario v2-checkpoint-stop"\n')
    compute={}
    holder=subprocess.Popen([str(paths['probe']),str(queue),'1','60000'],stdout=subprocess.PIPE,startupinfo=startup);assert holder.stdout.readline().strip()==b'ACQUIRED';before=queue.read_bytes()
    try:
        for name in ['stage1','stage2']:
            engine=paths[name];cap=subprocess.run([str(engine),'--gui-queue-protocol'],capture_output=True,startupinfo=startup,timeout=10);assert cap.returncode==0 and cap.stdout.strip()==b'ECM_QUEUE_OWNER=1'
            run=subprocess.run([str(engine),'--config',str(cfg),'--worker','1'],cwd=out,capture_output=True,startupinfo=startup,timeout=30);(out/(name+'.log')).write_bytes(run.stdout+run.stderr)
            assert run.returncode!=0 and b'already owned' in run.stdout+run.stderr,(name,run.returncode,run.stdout,run.stderr)
            assert queue.read_bytes()==before;compute[name]=dict(passed=True,sha256=hashlib.sha256(engine.read_bytes()).hexdigest(),exit=run.returncode)
    finally:holder.kill();holder.wait()
    def wait(log,pattern,proc):
        end=time.monotonic()+15
        while time.monotonic()<end:
            if log.exists() and pattern in log.read_text(encoding='utf-8',errors='replace'):return
            assert proc.poll() is None;time.sleep(.04)
        raise AssertionError(f'{pattern}: '+log.read_text(encoding='utf-8',errors='replace'))
    orders=[]
    for old_first in [True,False]:
        run=out/('old-first' if old_first else 'v2-first');run.mkdir();oldcfg=run/'old.toml';oldcfg.write_text(cfg.read_text().replace('autostart=0','autostart=1'));newlog=run/'v2.log';oldlog=out/'ecm_gui_trace.log';processes=[]
        def launch_old():
            proc=subprocess.Popen([str(old),'--config',str(oldcfg),'--trace'],cwd=run,startupinfo=startup);handle=window(proc.pid,'ecm_gui');processes.append((proc,handle));return proc,handle
        def launch_new():
            cmd=[str(paths['exe']),'--config',str(cfg),'--ui-state',str(run/'ui.toml'),'--diagnostics',str(newlog),'--no-nvml','--audit-app']
            if not old_first:cmd+=['--start-worker','1']
            proc=subprocess.Popen(cmd,cwd=run,startupinfo=startup);handle=window(proc.pid,'ecm_gui_v2');processes.append((proc,handle));return proc,handle
        try:
            if old_first:
                legacy,legacy_hwnd=launch_old();wait(oldlog,'autostart pid=',legacy);time.sleep(.3)
                newer,new_hwnd=launch_new();wait(newlog,'config=',newer);user.PostMessageW(new_hwnd,0x8022,1,1);wait(newlog,'already owned',newer)
                assert 'started pid=' not in newlog.read_text(encoding='utf-8')
            else:
                newer,new_hwnd=launch_new();wait(newlog,'started pid=',newer);legacy,legacy_hwnd=launch_old();wait(oldlog,'exit code 23',legacy)
            check=subprocess.run([str(paths['probe']),str(queue),'1','0'],capture_output=True,startupinfo=startup,timeout=5);assert check.returncode==23
            orders.append('old-first' if old_first else 'v2-first')
        finally:
            for proc,handle in reversed(processes):close(proc,handle)
    result=dict(passed=True,actual_compute_entries=compute,startup_orders=orders,old_gui_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),scope='Updated protocol-aware workers only; isolated queue, no actual ECM task was executed')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
