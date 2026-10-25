"""Native frameless chrome: client coverage, borderless controls and window actions."""
import argparse,ctypes as c,hashlib,json,re,subprocess,time
from ctypes import wintypes as w
from pathlib import Path
from win32_probe import window,close,capture,user


def main():
    p=argparse.ArgumentParser()
    for key in ['exe','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--light',action='store_true');a=p.parse_args();out=a.out.resolve();out.mkdir();exe=a.exe.resolve()
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    for name,args,result in [('ClientToScreen',[w.HWND,c.POINTER(w.POINT)],w.BOOL),('SendMessageW',[w.HWND,w.UINT,w.WPARAM,w.LPARAM],c.c_ssize_t),('IsZoomed',[w.HWND],w.BOOL),('IsIconic',[w.HWND],w.BOOL)]:
        getattr(user,name).argtypes=args;getattr(user,name).restype=result
    cfg=out/'ecm.toml';cfg.write_text('[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n')
    state=out/'ui.toml';state.write_text('[ui]\ntheme="'+('light' if a.light else 'dark')+'"\n[window]\ntitle_bar=false\n')
    log=out/'ui.log';proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-nvml','--no-pet','--ui-scale','1.25'],cwd=out)
    hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,80,80,1260,850,0x14);user.ShowWindow(hwnd,4)
    def sample(predicate=lambda s:True):
        end=time.monotonic()+8
        while time.monotonic()<end:
            rows=[dict(re.findall(r'(\w+)=([-\d.,]+)',line)) for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('presentation_sample:')]
            if rows and 'chrome2' in rows[-1] and predicate(rows[-1]):return rows[-1]
            assert proc.poll() is None;time.sleep(.04)
        raise AssertionError('No matching chrome sample')
    def click(value):
        x,y=map(int,value.split(','));packed=((y&65535)<<16)|(x&65535)
        for action in [0,1,2]:user.PostMessageW(hwnd,0x8021,action,packed);time.sleep(.12)
    def rect():
        r=w.RECT();assert user.GetWindowRect(hwnd,c.byref(r));return r
    def origin():
        point=w.POINT(0,0);assert user.ClientToScreen(hwnd,c.byref(point));return point
    def hit(x,y):return user.SendMessageW(hwnd,0x84,0,((y&65535)<<16)|(x&65535))
    results={}
    try:
        s=sample();time.sleep(.4);r=rect();o=origin();capture(hwnd,out/'normal.png')
        results['client_covers_window_top']=o.y==r.top
        from PIL import Image
        image=Image.open(out/'normal.png');ox,oy=o.x-r.left,o.y-r.top
        dx,dy=map(int,s['drag'].split(','));background=image.getpixel((ox+dx,oy+dy))
        # At 125%, buttons are ~34 px wide. This point lies outside each glyph.
        results['controls_have_no_resting_frame']=all(image.getpixel((ox+int(s[f'chrome{i}'].split(',')[0])+13,oy+int(s[f'chrome{i}'].split(',')[1])))==background for i in range(3))
        top=r.top+2;results['top_resize_hit']=hit((r.left+r.right)//2,top)==12
        point=w.POINT(dx,dy);user.ClientToScreen(hwnd,c.byref(point));results['caption_drag_hit']=hit(point.x,point.y)==2
        old_x=int(s['chrome1'].split(',')[0]);click(s['chrome1']);s=sample(lambda s:user.IsZoomed(hwnd) and int(s['chrome1'].split(',')[0])>old_x+50);time.sleep(.3)
        capture(hwnd,out/'maximized.png');r=rect();o=origin();client=w.RECT();user.GetClientRect(hwnd,c.byref(client))
        # Maximized client must fit the monitor work area, including taskbar.
        class MI(c.Structure):_fields_=[('cbSize',w.DWORD),('monitor',w.RECT),('work',w.RECT),('flags',w.DWORD)]
        user.MonitorFromWindow.argtypes=[w.HWND,w.DWORD];user.MonitorFromWindow.restype=w.HMONITOR
        user.GetMonitorInfoW.argtypes=[w.HMONITOR,c.POINTER(MI)]
        mi=MI();mi.cbSize=c.sizeof(mi);assert user.GetMonitorInfoW(user.MonitorFromWindow(hwnd,2),c.byref(mi))
        results['maximized_fits_work_area']=o.y==mi.work.top and o.x==mi.work.left and client.right==mi.work.right-mi.work.left and client.bottom==mi.work.bottom-mi.work.top
        click(s['chrome1']);s=sample(lambda s:not user.IsZoomed(hwnd) and abs(int(s['chrome1'].split(',')[0])-old_x)<10)
        click(s['chrome0']);end=time.monotonic()+3
        while not user.IsIconic(hwnd) and time.monotonic()<end:time.sleep(.03)
        results['minimize']=bool(user.IsIconic(hwnd));user.ShowWindow(hwnd,9);s=sample(lambda s:not user.IsIconic(hwnd))
        user.SetWindowPos(hwnd,None,100,100,1120,760,0x14);s=sample(lambda s:int(s['chrome1'].split(',')[0])<old_x-50)
        results['resize']=rect().right-rect().left==1120
        click(s['chrome2']);proc.wait(timeout=12);results['close']=proc.returncode==0
    finally:
        if proc.poll() is None:close(proc,hwnd)
    report=dict(passed=all(results.values()),cases=results,theme='light' if a.light else 'dark',exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),scope='DX11, empty isolated workflow, no NVML or ECM work')
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));assert report['passed'],results

if __name__=='__main__':main()
