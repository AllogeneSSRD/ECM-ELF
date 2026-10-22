"""Renderer failures must still deliver the safe-stop event to an active worker."""
import argparse,ctypes as c,json,os,re,subprocess
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    for key in ['exe','worker','out','layers']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir();cfg=out/'ecm.toml';queue=out/'worktodo.txt';queue.write_text('[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n')
    cfg.write_text('[shared]\nworktodo='+json.dumps(str(queue))+'\n[gui]\nNumWorkers=1\ngraceful_stop_ms=1200\nexe='+json.dumps(str(a.worker.resolve()))+'\n[workers.1]\nautostart=0\nextra_args="--scenario v2-checkpoint-stop"\n');before=cfg.read_bytes()
    env=os.environ.copy();env['VK_LAYER_PATH']=str(a.layers.resolve());startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    kernel=c.WinDLL('kernel32');kernel.OpenProcess.argtypes=[c.c_ulong,c.c_int,c.c_ulong];kernel.OpenProcess.restype=c.c_void_p;kernel.GetExitCodeProcess.argtypes=[c.c_void_p,c.POINTER(c.c_ulong)];kernel.CloseHandle.argtypes=[c.c_void_p]
    cases={}
    for fault in ['device-lost','surface-lost']:
        log=out/(fault+'.log');run=subprocess.run([str(a.exe.resolve()),'--config',str(cfg),'--ui-state',str(out/(fault+'.toml')),'--diagnostics',str(log),'--no-nvml','--renderer','vulkan','--validation','--start-worker','1','--inject-failure',fault],cwd=out,env=env,startupinfo=startup,timeout=30)
        text=log.read_text(encoding='utf-8');assert run.returncode==1 and 'injected renderer failure' in text and 'graceful stop requested' in text and 'shutdown: terminated' not in text
        assert 'validation_summary: errors=0 warnings=0' in text
        pid=int(re.search(r'started pid=(\d+)',text)[1]);handle=kernel.OpenProcess(0x1000,False,pid)
        if handle:
            code=c.c_ulong();assert kernel.GetExitCodeProcess(handle,c.byref(code));kernel.CloseHandle(handle);assert code.value!=259
        cases[fault]=dict(worker_pid=pid,safely_stopped=True)
    assert cfg.read_bytes()==before
    result=dict(passed=True,cases=cases,business_unchanged=True);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
