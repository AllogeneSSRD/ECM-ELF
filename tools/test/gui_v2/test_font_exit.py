"""Live font adjustment and bounded native exit branches; no ECM computation."""
import argparse,ctypes as c,hashlib,json,re,subprocess,time,tomllib
from ctypes import wintypes as w
from pathlib import Path
from win32_probe import window,close,capture,user
from renderer_probe import renderer_clean

class Run:
    def __init__(self,root,exe,worker,name,policy='ask',count=0,delay=1800,deadline=8000,page=0):
        self.out=root/name;self.out.mkdir();self.log=self.out/'ui.log';self.state=self.out/'ui.toml'
        queue=self.out/'worktodo.txt';queue.write_text('[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n[Worker #2]\nECMSTAGE2=1,2,137,-1,"ed_m137.save",0,0,10\n')
        self.cfg=self.out/'ecm.toml'
        self.cfg.write_text('[shared]\nworktodo='+json.dumps(str(queue))+'\n[gui]\nNumWorkers='+str(max(1,count))+'\nexit_confirm='+json.dumps(policy)+'\ngraceful_stop_ms='+str(deadline)+'\nexe='+json.dumps(str(worker))+'\nstage2_exe='+json.dumps(str(worker))+'\n'+''.join(f'[workers.{i}]\nautostart={int(i<=count)}\nextra_args="--scenario v2-exit-wait --stop-delay {delay}"\n' for i in range(1,max(1,count)+1)),encoding='utf-8')
        self.before=self.cfg.read_bytes();self.state.write_text(f'[ui]\npage={page}\nlanguage="zh"\nlogs=false\ninspector=false\nscale=1.25\n')
        start=subprocess.STARTUPINFO();start.dwFlags=subprocess.STARTF_USESHOWWINDOW;start.wShowWindow=0
        self.proc=subprocess.Popen([str(exe),'--config',str(self.cfg),'--ui-state',str(self.state),'--diagnostics',str(self.log),'--audit-app','--no-nvml','--no-pet'],cwd=self.out,startupinfo=start)
        self.hwnd=window(self.proc.pid,'ecm_gui_v2');user.SetWindowPos(self.hwnd,None,40,40,1920,1200,0x14);user.ShowWindow(self.hwnd,4)
        self.wait(lambda v,t:t.count(': started pid=')>=count)
    def wait(self,predicate,seconds=12):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            text=self.log.read_text(encoding='utf-8');rows=[dict(re.findall(r'(\w+)=([-\d.,]+)',line)) for line in text.splitlines() if line.startswith('presentation_sample:')]
            if rows and predicate(rows[-1],text):return rows[-1]
            assert self.proc.poll() is None,text[-2500:];time.sleep(.05)
        raise AssertionError(self.log.read_text(encoding='utf-8')[-3500:])
    def click(self,value):
        x,y=map(int,value.split(','));assert x>0 and y>0
        for action in [0,1,2]:user.PostMessageW(self.hwnd,0x8021,action,(y<<16)|(x&65535));time.sleep(.12)
    def control(self,key,phase):
        def stable(v,text):
            rows=[dict(re.findall(r'(\w+)=([-\d.,]+)',line)) for line in text.splitlines() if line.startswith('presentation_sample:')]
            return v['exit_phase']==str(phase) and int(v[key].split(',')[0])>0 and len(rows)>=3 and all(row.get(key)==v[key] and row.get('exit_phase')==str(phase) for row in rows[-3:])
        return self.wait(stable)
    def key(self,value):
        user.PostMessageW(self.hwnd,0x100,value,1);time.sleep(.2);user.PostMessageW(self.hwnd,0x101,value,0xC0000001);time.sleep(.2)
    def request(self):user.PostMessageW(self.hwnd,0x10,0,0)
    def finish(self):
        self.proc.wait(timeout=15);assert self.proc.returncode==0
        assert self.cfg.read_bytes()==self.before
        text=self.log.read_text(encoding='utf-8');assert renderer_clean(text,'dx11')
        kernel=c.WinDLL('kernel32');kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
        kernel.GetExitCodeProcess.argtypes=[w.HANDLE,c.POINTER(w.DWORD)];kernel.CloseHandle.argtypes=[w.HANDLE]
        for pid in re.findall(r': started pid=(\d+)',text):
            handle=kernel.OpenProcess(0x1000,False,int(pid))
            if handle:
                code=w.DWORD();assert kernel.GetExitCodeProcess(handle,c.byref(code));kernel.CloseHandle(handle);assert code.value!=259
        return text
    def cleanup(self):
        if self.proc.poll() is None:
            capture(self.hwnd,self.out/'failure.png')
            user.PostMessageW(self.hwnd,0x8022,3,0)
            close(self.proc,self.hwnd)

def main():
    p=argparse.ArgumentParser()
    for name in ['exe','worker','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir();exe=a.exe.resolve();worker=a.worker.resolve()
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    user.IsIconic.argtypes=[w.HWND]
    results={};r=Run(out,exe,worker,'font',page=5)
    try:
        s=r.wait(lambda v,t:v['font_size']=='15' and int(v['font_control'].split(',')[1])>0);r.click(s['font_control'])
        s=r.wait(lambda v,t:19<=int(v['font_size'])<=21 and float(v['font_base'])==int(v['font_size']))
        size=int(s['font_size']);capture(r.hwnd,r.out/'font-adjusted.png');r.request();text=r.finish()
        assert text.count('fonts: language=')==1
        assert tomllib.loads(r.state.read_text(encoding='utf-8'))['ui']['font_size']==size
        log=r.out/'restart.log';restart=subprocess.run([str(exe),'--config',str(r.cfg),'--ui-state',str(r.state),'--diagnostics',str(log),'--audit-app','--frames','8','--no-nvml','--no-pet'],cwd=r.out,timeout=20)
        assert restart.returncode==0 and f'font_size={size} font_base={size:.1f}' in log.read_text(encoding='utf-8')
        results['font']=dict(live_size=size,persisted=True,no_font_reload=True,idle_exit=True)
    finally:r.cleanup()
    print('Font live adjustment and restart passed',flush=True)
    r=Run(out,exe,worker,'ask-safe',count=2)
    try:
        user.ShowWindow(r.hwnd,6);time.sleep(.2);r.request()
        s=r.control('exit1',1);assert not user.IsIconic(r.hwnd);capture(r.hwnd,r.out/'confirm.png')
        r.click(s['exit1']);r.wait(lambda v,t:v['exit_phase']=='0')
        r.request();r.control('exit1',1);r.key(27);r.wait(lambda v,t:v['exit_phase']=='0')
        r.request();r.control('exit0',1);r.key(13)
        s=r.control('exit2',2);deadline=s['deadline'];capture(r.hwnd,r.out/'stopping-mixed.png')
        r.request();time.sleep(.2);s=r.wait(lambda v,t:v['exit_phase']=='2');assert s['deadline']==deadline
        text=r.finish()
        assert 'worker 1: state Stopped' in text and 'worker 2: state Stopped' in text and 'exit: all workers stopped, closing' in text
        assert 'stop timeout' not in text and 'forced by user' not in text
        results['ask-safe']=dict(cancel_button=True,cancel_escape=True,accept_enter=True,mixed_stage_wait=True,restore_minimized=True,repeated_close_keeps_deadline=True)
    finally:r.cleanup()
    print('Ask, cancel and safe mixed-stage exit passed',flush=True)
    r=Run(out,exe,worker,'ask-force',count=2,delay=15000)
    try:
        r.request();s=r.control('exit0',1);r.click(s['exit0'])
        s=r.control('exit2',2);r.click(s['exit2'])
        text=r.finish();assert 'forced by user' in text;results['ask-force']=True
    finally:r.cleanup()
    print('Explicit force exit passed',flush=True)
    r=Run(out,exe,worker,'stop-timeout',policy='stop',count=1,delay=15000,deadline=1800)
    try:
        r.request();s=r.wait(lambda v,t:v['exit_phase']=='2');deadline=s['deadline'];r.request();time.sleep(.2)
        s=r.wait(lambda v,t:v['exit_phase']=='2');assert s['deadline']==deadline
        text=r.finish();assert 'stop timeout; terminating worker' in text;results['stop-timeout']=True
    finally:r.cleanup()
    r=Run(out,exe,worker,'kill-policy',policy='kill',count=1)
    try:
        r.request();text=r.finish();assert 'exit_confirm=kill' in text;results['kill-policy']=True
    finally:r.cleanup()
    report=dict(passed=True,cases=results,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),business_unchanged=True,scope='Synthetic stop handles; no ECM computation or Vulkan rendering')
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
