"""Floating pet must stay in contact with an external top edge, including native moves."""
import argparse,ctypes as c,hashlib,json,math,subprocess,time
from ctypes import wintypes as w
from pathlib import Path
from win32_probe import user,window,close,capture
from renderer_probe import renderer_clean


def main():
    p=argparse.ArgumentParser()
    for key in ['exe','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir();exe=a.exe.resolve()
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    user.ClientToScreen.argtypes=[w.HWND,c.POINTER(w.POINT)];user.ClientToScreen.restype=w.BOOL
    cfg=out/'ecm.toml';cfg.write_text('[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n');original=cfg.read_bytes()
    state=out/'ui.toml';state.write_text('[ui]\nlogs=false\n[window]\ntitle_bar=false\n[pet]\nenabled=true\nanimate=true\nleft=72\n')
    log=out/'pet.log';proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--no-nvml','--pet'],cwd=out)
    hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,200,400,1000,800,0x14);user.ShowWindow(hwnd,4);pet=window(proc.pid,'ecm_gui_v2_pet')
    def box(target):
        r=w.RECT();assert user.GetWindowRect(target,c.byref(r));return [r.left,r.top,r.right,r.bottom]
    def client():
        r=w.RECT();o=w.POINT();assert user.GetClientRect(hwnd,c.byref(r)) and user.ClientToScreen(hwnd,c.byref(o));return [o.x,o.y,o.x+r.right,o.y+r.bottom]
    def observe(edge,seconds):
        samples=[];end=time.monotonic()+seconds
        while time.monotonic()<end:
            assert proc.poll() is None
            owner,spot=client(),box(pet)
            offset=spot[3]-owner[1] if edge=='top' else owner[3]-spot[3]
            samples.append(dict(offset=offset,owner=owner,pet=spot));time.sleep(.04)
        capture(pet,out/(edge+'.png'));return samples
    cases={};samples={}
    try:
        time.sleep(.3);samples['top']=observe('top',7.4)
        limit=math.ceil(3*user.GetDpiForWindow(hwnd)/96)+1
        offsets=[s['offset'] for s in samples['top']]
        cases['external_float_never_separates']=min(offsets)>=0 and max(offsets)<=limit
        cases['external_float_preserved']=max(offsets)>min(offsets)
        # Reuse the frame-paused OS move path that updates pet placement directly.
        user.PostMessageW(hwnd,0x112,0xF010,0);time.sleep(.3)
        paused_frames=log.read_text(encoding='utf-8').count('\nframe:')
        before_owner,before_pet=client(),box(pet);moves=[]
        try:
            for x,y in [(300,430),(400,470)]:
                user.SetWindowPos(hwnd,None,x,y,1000,800,0x14);owner,spot=client(),box(pet)
                moves.append(dict(owner=owner,pet=spot))
        finally:
            user.PostMessageW(hwnd,0x100,0x1B,1);user.PostMessageW(hwnd,0x101,0x1B,0xC0000001)
        cases['native_move_keeps_contact']=all(0<=m['pet'][3]-m['owner'][1]<=limit for m in moves)
        cases['native_move_preserves_offset']=all(m['pet'][3]-m['owner'][1]==before_pet[3]-before_owner[1] for m in moves)
        # Escape ends the OS modal loop asynchronously and restores its original
        # owner position. Wait for normal rendering before starting the next case.
        deadline=time.monotonic()+4
        while log.read_text(encoding='utf-8').count('\nframe:')<=paused_frames and time.monotonic()<deadline:time.sleep(.03)
        assert log.read_text(encoding='utf-8').count('\nframe:')>paused_frames
        user.SetWindowPos(hwnd,None,200,0,1000,800,0x14);time.sleep(.3);assert client()[1]==0
        samples['bottom']=observe('bottom',4)
        offsets=[s['offset'] for s in samples['bottom']];cases['internal_float_stays_inside']=min(offsets)>=0 and max(offsets)<=limit
    finally:close(proc,hwnd)
    assert cfg.read_bytes()==original and renderer_clean(log.read_text(encoding='utf-8'),'dx11')
    report=dict(passed=all(cases.values()),cases=cases,samples=samples,moves=moves,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),scope='DX11, isolated empty workflow; no ECM or NVML')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ['samples','moves']}));assert report['passed'],cases

if __name__=='__main__':main()
