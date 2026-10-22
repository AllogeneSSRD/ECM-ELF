"""Closing native right tabs must preserve central selection and custom split sizes."""
import argparse,ctypes as c,hashlib,json,re,subprocess,time,tomllib
from pathlib import Path
from win32_probe import window,close,capture,user

def main():
    p=argparse.ArgumentParser()
    for key in ['exe','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expect-bug',action='store_true');a=p.parse_args()
    out=a.out.resolve();out.mkdir();exe=a.exe.resolve()
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    cfg=out/'ecm.toml';cfg.write_text('[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n')
    state=out/'ui.toml';state.write_text('[ui]\npage=1\nright_pages=24\nright_page=4\nlogs=false\ninspector=false\nscale=1\ntoasts=false\n[window]\ntitle_bar=false\n')
    log=out/'ui.log';si=subprocess.STARTUPINFO();si.dwFlags=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
    proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--audit-app','--diagnostics',str(log),'--no-nvml','--no-pet'],cwd=out,startupinfo=si)
    hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,50,50,1500,1000,0x14);user.ShowWindow(hwnd,4)
    def sample():
        text=log.read_text(encoding='utf-8');v={}
        for prefix in ['usability_sample:','navigation_sample:']:
            rows=[s for s in text.splitlines() if s.startswith(prefix)]
            if rows:v.update(dict(re.findall(r'(\w+)=([^\s]+)',rows[-1])))
        return v
    def wait(predicate):
        end=time.monotonic()+10
        while time.monotonic()<end:
            s=sample()
            if predicate(s):return s
            assert proc.poll() is None;time.sleep(.08)
        raise AssertionError(sample())
    def pointer(x,y,action):user.PostMessageW(hwnd,0x8021,action,(round(y)<<16)|(round(x)&65535))
    def click(pos):
        x,y=map(int,pos.split(','))
        for action in [0,1,2]:pointer(x,y,action);time.sleep(.15)
    try:
        s=wait(lambda v:v.get('page')=='1' and v.get('mask')=='24' and int(v.get('close4','0,0').split(',')[0])>0)
        initial=float(s['nav_size'].split(',')[0]);custom=initial
        for offset in [1,-2,3]:
            x=custom+offset;pointer(x,320,0);time.sleep(.12);pointer(x,320,1);time.sleep(.2)
            for step in range(1,7):pointer(x+step*12,320,0);time.sleep(.08)
            pointer(x+72,320,2);time.sleep(.5);custom=float(sample()['nav_size'].split(',')[0])
            if abs(custom-initial)>20:break
        assert abs(custom-initial)>20,dict(initial=initial,sample=sample())
        click(sample()['tool4']);wait(lambda v:v.get('right')=='4');before=sample();capture(hwnd,out/'before-close.png')
        click(before['close4']);after=wait(lambda v:v.get('mask')=='8' and v.get('right_count')=='1');time.sleep(.5);after=sample();capture(hwnd,out/'after-close.png')
        changed_page=after['page']!=before['page'];changed_width=abs(float(after['nav_size'].split(',')[0])-custom)>1
        result=dict(reproduced=changed_page or changed_width,changed_page=changed_page,changed_width=changed_width,before=before,after=after,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
        if a.expect_bug:assert result['reproduced'],result
        else:
            assert not result['reproduced'],result
            click(after['tool3']);wait(lambda v:v.get('right')=='3');click(sample()['close3']);last=wait(lambda v:v.get('mask')=='0' and v.get('right_count')=='0')
            assert last['page']==before['page'] and abs(float(last['nav_size'].split(',')[0])-custom)<1,last
            result['last_close_preserved']=True
    finally:close(proc,hwnd)
    if not a.expect_bug:
        saved=tomllib.loads(state.read_text(encoding='utf-8'));assert saved['ui']['right_pages']==0 and saved['ui']['page']==int(before['page']),saved['ui']
        assert 'UI state not saved' not in log.read_text(encoding='utf-8')
        result['saved_after_last_close']=True
    result['passed']=True;(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['passed','reproduced','changed_page','changed_width']}))
if __name__=='__main__':main()
