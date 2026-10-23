"""Native idle movement and click response, isolated config, no workers or NVML."""
import argparse,ctypes as c,json,subprocess,time,hashlib,math
from ctypes import wintypes as w
from pathlib import Path
from win32_probe import user,window,close,capture
from renderer_probe import renderer_clean

def main():
    p=argparse.ArgumentParser()
    for key in ['exe','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir();exe=a.exe.resolve()
    config=out/'ecm.toml';config.write_text('[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n')
    original=config.read_bytes();state=out/'ui.toml';log=out/'idle.log'
    state.write_text('[ui]\nlogs=false\ninspector=false\n[pet]\nenabled=true\nanimate=true\n')
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p
    user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    proc=subprocess.Popen([str(exe),'--config',str(config),'--ui-state',str(state),'--diagnostics',str(log),'--no-nvml','--pet'])
    samples=[]
    hwnd=window(proc.pid,'ecm_gui_v2')
    try:
        user.SetWindowPos(hwnd,None,200,350,1000,800,0x14);user.ShowWindow(hwnd,4)
        pet=window(proc.pid,'ecm_gui_v2_pet');time.sleep(.5);capture(pet,out/'idle.png')
        start=time.monotonic()
        for index in range(260):
            assert proc.poll() is None
            box=w.RECT();assert user.GetWindowRect(pet,c.byref(box))
            samples.append([round(time.monotonic()-start,3),box.left,box.top])
            if index in [120,125,130,135]:capture(pet,out/f'motion-{index}.png')
            time.sleep(.1)
        ys=[s[2] for s in samples];limit=math.ceil(3*user.GetDpiForWindow(hwnd)/96)+1
        assert 0<max(ys)-min(ys)<=limit,(min(ys),max(ys),limit)
        before=len(log.read_text(encoding='utf-8'))
        user.PostMessageW(pet,0x201,1,0);user.PostMessageW(pet,0x202,0,0)
        end=time.monotonic()+3;wink=False
        while time.monotonic()<end:
            text=log.read_text(encoding='utf-8')[before:]
            wink=wink or 'pet: mood=6 ' in text
            if wink and 'pet: mood=0 ' in text:break
            time.sleep(.05)
        assert wink and 'pet: mood=0 ' in text
        assert log.read_text(encoding='utf-8').count('pet: mood=0 ')<=3,'Floating should not spam transition logs'
    finally:close(proc,hwnd)
    assert config.read_bytes()==original and renderer_clean(log.read_text(encoding='utf-8'),'dx11')
    report=dict(passed=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),samples=samples,
                movement_pixels=max(ys)-min(ys),click_response=True,no_compute_or_clock_writes=True)
    (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
if __name__=='__main__':main()
