"""Native View-menu log visibility must preserve custom docking dimensions."""
import argparse,ctypes as c,hashlib,json,re,subprocess,time,tomllib
from pathlib import Path
from win32_probe import window,close,capture,user

def main():
    p=argparse.ArgumentParser()
    for key in ['exe','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expect-bug',action='store_true');p.add_argument('--start-hidden',action='store_true')
    p.add_argument('--save-hidden',action='store_true');p.add_argument('--language',choices=['en','zh'],default='en');a=p.parse_args()
    out=a.out.resolve();out.mkdir();exe=a.exe.resolve()
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    cfg=out/'ecm.toml';cfg.write_text('[gui]\nNumWorkers=2\n[workers.1]\nautostart=0\n[workers.2]\nautostart=0\n');original=cfg.read_bytes()
    state=out/'ui.toml';state.write_text('[ui]\npage=1\nright_pages=24\nright_page=4\nlogs='+str(not a.start_hidden).lower()+'\ninspector=false\nscale=1\nfont_size=15\nlanguage="'+a.language+'"\ntoasts=false\n[window]\ntitle_bar=false\n')
    def launch(name):
        log=out/(name+'.log');si=subprocess.STARTUPINFO();si.dwFlags=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
        proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--audit-app','--diagnostics',str(log),'--no-nvml','--no-pet','--language',a.language],cwd=out,startupinfo=si)
        hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,30,30,1500,1000,0x14);user.ShowWindow(hwnd,4)
        return proc,hwnd,log
    proc,hwnd,log=launch('view')
    def sample():
        v={}
        latest={}
        for line in log.read_text(encoding='utf-8').splitlines():
            if line.startswith(('usability_sample:','navigation_sample:','presentation_sample:','view_sample:')):latest[line.split(':',1)[0]]=line
        for line in latest.values():v.update(dict(re.findall(r'(\w+)=([^\s]+)',line)))
        return v
    def wait(predicate):
        end=time.monotonic()+12
        while time.monotonic()<end:
            s=sample()
            if s and predicate(s):return s
            assert proc.poll() is None;time.sleep(.08)
        raise AssertionError(sample())
    def pointer(x,y,action):user.PostMessageW(hwnd,0x8021,action,(round(y)<<16)|(round(x)&65535))
    def click(pos):
        x,y=map(float,pos.split(','))
        for action in [0,1,2]:pointer(x,y,action);time.sleep(.15)
    def toggle_logs(visible):
        s=sample();click(s.get('view','149,13'));time.sleep(.2)
        s=sample();click(s.get('logs_control','150,104'))
        wait(lambda s:('logdock0' in s)==visible);time.sleep(.5)
    try:
        s=wait(lambda s:s.get('page')=='1' and s.get('right_count')=='2')
        initial=float(s['nav_size'].split(',')[0]);custom=initial
        for offset in [1,-2,3]:
            x=custom+offset;pointer(x,320,0);time.sleep(.12);pointer(x,320,1);time.sleep(.2)
            for step in range(1,7):pointer(x+step*12,320,0);time.sleep(.08)
            pointer(x+72,320,2);time.sleep(.5);custom=float(sample()['nav_size'].split(',')[0])
            if abs(custom-initial)>20:break
        assert abs(custom-initial)>20,sample()
        before=sample();capture(hwnd,out/'before.png');widths=[]
        if a.start_hidden:toggle_logs(True);widths.append(float(sample()['nav_size'].split(',')[0]))
        for _ in range(2):
            toggle_logs(False);widths.append(float(sample()['nav_size'].split(',')[0]))
            toggle_logs(True);widths.append(float(sample()['nav_size'].split(',')[0]))
        after=sample();capture(hwnd,out/'after.png')
        changed=any(abs(w-custom)>1 for w in widths)
        result=dict(passed=True,reproduced=changed,before_width=custom,widths=widths,start_hidden=a.start_hidden,save_hidden=a.save_hidden,language=a.language,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
        if a.expect_bug:assert changed,result
        else:
            assert not changed,result
            assert after['page']==before['page'] and after['mask']==before['mask'] and after['right_count']==before['right_count']
            assert after['dock3']==before['dock3'] and after['dock4']==before['dock4'] and after['layout_builds']==before['layout_builds'],after
            assert all(int(after.get('logdock'+str(i),'0'))>0 for i in range(3)),after
            if a.save_hidden:toggle_logs(False);after=sample()
            click(after['view']);wait(lambda s:s.get('save_control','0,0')!='0,0');capture(hwnd,out/'view-menu.png')
            s=sample();assert s['items']=='menu.inspector,menu.logs,menu.title_bar,menu.save_layout_config' and s['save_separator']=='1',s;click(s['save_control'])
            wait(lambda s:s.get('save_count')=='1');saved=tomllib.loads(state.read_text(encoding='utf-8'));assert saved['ui']['dock'] and saved['window']['width']==1500 and saved['ui']['logs']==(not a.save_hidden),saved
            assert cfg.read_bytes()==original
            result.update(explicit_save=True,no_reset_entry=True,separator=True)
    finally:close(proc,hwnd)
    if not a.expect_bug:
        proc,hwnd,log=launch('restart')
        try:
            s=wait(lambda s:s.get('right_count')=='2' and ('logdock2' in s)==(not a.save_hidden))
            assert abs(float(s['nav_size'].split(',')[0])-custom)<=1,s
            if a.save_hidden:
                toggle_logs(True);s=sample();assert abs(float(s['nav_size'].split(',')[0])-custom)<=1 and all(int(s.get('logdock'+str(i),'0'))>0 for i in range(3)),s
            capture(hwnd,out/'restarted.png');result['restart_preserved']=True
        finally:close(proc,hwnd)
    assert cfg.read_bytes()==original;result['workflow_unchanged']=True
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
