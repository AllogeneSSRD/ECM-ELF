"""Minimized CPU diagnosis: main thread versus auxiliary threads, NVML on/off."""
import argparse,ctypes as c,json,statistics,subprocess,time
from ctypes import wintypes as w
from pathlib import Path
from win32_probe import window,close,process_sample,user,kernel
def main():
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir();samples=[]
    kernel.OpenThread.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenThread.restype=w.HANDLE
    kernel.GetThreadTimes.argtypes=[w.HANDLE]+[c.POINTER(w.FILETIME)]*4;kernel.CloseHandle.argtypes=[w.HANDLE]
    def thread_seconds(handle):
        times=[w.FILETIME() for _ in range(4)];assert kernel.GetThreadTimes(handle,*(c.byref(v) for v in times));return sum((v.dwHighDateTime<<32)|v.dwLowDateTime for v in times[2:])/1e7
    startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    for nvml in [True,False]:
        for index in range(5):
            base=out/f'{nvml}-{index}';base.mkdir();cfg=base/'ecm.toml';cfg.write_text('[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n')
            cmd=[str(a.exe.resolve()),'--config',str(cfg),'--ui-state',str(base/'ui.toml'),'--diagnostics',str(base/'gui.log')]+([] if nvml else ['--no-nvml'])
            proc=subprocess.Popen(cmd,cwd=base,startupinfo=startup);hwnd=window(proc.pid,'ecm_gui_v2');tid=user.GetWindowThreadProcessId(hwnd,None);handle=kernel.OpenThread(0x0800,False,tid);assert handle
            try:
                user.ShowWindow(hwnd,6);time.sleep(2);before=process_sample(proc);thread_before=thread_seconds(handle);start=time.monotonic();time.sleep(3);duration=time.monotonic()-start;after=process_sample(proc);main_cpu=thread_seconds(handle)-thread_before;cpu=after['cpu_seconds']-before['cpu_seconds']
                samples.append(dict(nvml=nvml,index=index,one_core_cpu_percent=100*cpu/duration,main_thread_cpu_percent=100*main_cpu/duration,auxiliary_cpu_percent=100*(cpu-main_cpu)/duration));print(json.dumps(samples[-1]),flush=True)
            finally:kernel.CloseHandle(handle);close(proc,hwnd)
    result=dict(samples=samples,medians={str(nvml):{key:statistics.median(v[key] for v in samples if v['nvml']==nvml) for key in ['one_core_cpu_percent','main_thread_cpu_percent','auxiliary_cpu_percent']} for nvml in [True,False]},boundary='5 runs each, 2s warmup +3s interval; one-core percent, GetProcessTimes/GetThreadTimes; unchanged 500ms NVML when enabled')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['medians']))
if __name__=='__main__':main()
