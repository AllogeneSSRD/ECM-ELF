"""Native DX11 navigation ordering, right dock tabs and modal pet following.

All UI/business files are isolated; no compute workers or Vulkan are started.
The pet check holds Windows in its real SC_MOVE loop while the frame loop pauses.
"""
import argparse,ctypes as c,hashlib,json,re,subprocess,time,tomllib
from ctypes import wintypes as w
from pathlib import Path
from win32_probe import window,close,capture,user
from renderer_probe import renderer_clean

def main():
    parser=argparse.ArgumentParser()
    for name in ['exe','out']:parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();exe=a.exe.resolve();out=a.out.resolve();out.mkdir()
    cfg=out/'ecm.toml';cfg.write_text('[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n');original=cfg.read_bytes()
    state=out/'ui.toml';state.write_text('[ui]\nlanguage="zh"\nlogs=false\ninspector=false\nscale=1\ntoasts=false\n[pet]\nenabled=true\nanimate=false\n')
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    user.GetWindowLongPtrW.argtypes=[w.HWND,c.c_int];user.GetWindowLongPtrW.restype=c.c_ssize_t
    logs=[]
    def launch(name):
        log=out/(name+'.log');logs.append(log);startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-nvml','--pet','--pet-static'],cwd=out,startupinfo=startup)
        hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,100,350,1500,1000,0x14);user.ShowWindow(hwnd,4)
        return proc,hwnd,log
    def sample(log):
        text=log.read_text(encoding='utf-8');merged={}
        for prefix in ['usability_sample:','navigation_sample:','presentation_sample:']:
            rows=[line for line in text.splitlines() if line.startswith(prefix)]
            if rows:merged.update(dict(re.findall(r'(\w+)=([^\s]+)',rows[-1])))
        return merged
    def wait(proc,log,predicate):
        end=time.monotonic()+12
        while time.monotonic()<end:
            s=sample(log)
            if s and predicate(s):return s
            assert proc.poll() is None;time.sleep(.06)
        raise AssertionError(log.read_text(encoding='utf-8')[-3000:])
    def coords(value):return tuple(map(int,value.split(',')))
    def pointer(hwnd,value,action=0):
        x,y=coords(value);assert x>0 and y>0,value
        user.PostMessageW(hwnd,0x8021,action,(y<<16)|(x&65535))
    def click(hwnd,value):
        for action in [0,1,2]:pointer(hwnd,value,action);time.sleep(.14)
    def drag(hwnd,source,target,hold=.45):
        pointer(hwnd,source);time.sleep(.15);pointer(hwnd,source,1);time.sleep(hold)
        sx,sy=coords(source);tx,ty=coords(target)
        for i in range(1,6):pointer(hwnd,f'{round(sx+(tx-sx)*i/5)},{round(sy+(ty-sy)*i/5)}');time.sleep(.09)
        time.sleep(.2);pointer(hwnd,target,2);time.sleep(.25)
    def rect(hwnd):
        r=w.RECT();assert user.GetWindowRect(hwnd,c.byref(r));return (r.left,r.top,r.right,r.bottom)
    proc,hwnd,log=launch('first');modal_result={}
    try:
        s=wait(proc,log,lambda v:v.get('order')=='0123456' and v.get('title')=='0' and 'chrome2' in v)
        assert not user.GetWindowLongPtrW(hwnd,-16)&0xC00000
        assert float(s['rounding'])==8 and float(s['height'])>=40 and float(s['gap'])>=8
        # A fast drag does not start navigation reordering.
        pointer(hwnd,s['nav4']);time.sleep(.12);pointer(hwnd,s['nav4'],1);time.sleep(.04);pointer(hwnd,s['nav1']);time.sleep(.04);pointer(hwnd,s['nav1'],2);time.sleep(.3)
        assert sample(log)['order']=='0123456'
        s=sample(log);x,y=coords(s['nav1']);drag(hwnd,s['nav4'],f'{x},{y-10}')
        s=wait(proc,log,lambda v:v.get('order')=='0412356' and coords(v['nav0'])[1]<coords(v['nav4'])[1]<coords(v['nav1'])[1])
        pointer(hwnd,s['nav4']);time.sleep(.15);capture(hwnd,out/'navigation-hover.png')
        drag(hwnd,s['nav4'],s['drop']);s=wait(proc,log,lambda v:v.get('mask')=='16' and v.get('right_count')=='1' and int(v.get('dock4','0'))>0)
        drag(hwnd,s['nav3'],s['drop']);s=wait(proc,log,lambda v:v.get('mask')=='24' and v.get('right_count')=='2' and v.get('dock3')==v.get('dock4') and int(v.get('dock3','0'))>0 and v.get('right')=='3')
        assert s['tools']=='0' and s['back']=='0,0'
        assert abs(coords(s['tool3'])[1]-coords(s['tool4'])[1])<2
        click(hwnd,s['tool4']);s=wait(proc,log,lambda v:v.get('right')=='4');capture(hwnd,out/'right-tabs.png')
        click(hwnd,s['close4']);s=wait(proc,log,lambda v:v.get('mask')=='8' and coords(v['nav4'])[0]>0 and v.get('right_count')=='1' and coords(v['nav0'])[1]<coords(v['nav4'])[1])
        assert s['order']=='0412356'
        drag(hwnd,s['nav4'],s['drop']);s=wait(proc,log,lambda v:v.get('mask')=='24' and v.get('right_count')=='2' and v.get('dock3')==v.get('dock4') and coords(v['tool3'])[0]>500 and coords(v['tool4'])[0]>500)
        # Stay in the real native move loop, and compare OS window geometry before
        # another application frame can run. This reproduced the old lag.
        pet=window(proc.pid,'ecm_gui_v2_pet');time.sleep(.3)
        user.PostMessageW(hwnd,0x112,0xF010,0);time.sleep(.35)
        count=log.read_text(encoding='utf-8').count('presentation_sample:');owner_before=rect(hwnd);pet_before=rect(pet);moves=[]
        try:
            for dx,dy in [(180,40),(280,80)]:
                user.SetWindowPos(hwnd,None,owner_before[0]+dx,owner_before[1]+dy,1500,1000,0x14)
                owner=rect(hwnd);position=rect(pet)
                expected=(pet_before[0]+owner[0]-owner_before[0],pet_before[1]+owner[1]-owner_before[1])
                assert position[:2]==expected,(position,expected,owner)
                assert log.read_text(encoding='utf-8').count('presentation_sample:')==count
                moves.append(dict(owner=owner,pet=position))
        finally:
            user.PostMessageW(hwnd,0x100,0x1B,1);user.PostMessageW(hwnd,0x101,0x1B,0xC0000001)
        wait(proc,log,lambda v:log.read_text(encoding='utf-8').count('presentation_sample:')>count)
        modal_result=dict(frame_loop_paused=True,followed_before_next_frame=True,moves=moves)
    except BaseException:
        if proc.poll() is None:capture(hwnd,out/'failure-first.png')
        raise
    finally:
        user.PostMessageW(hwnd,0x100,0x1B,1);time.sleep(.1);close(proc,hwnd)
    saved=tomllib.loads(state.read_text(encoding='utf-8'));assert saved['ui']['nav_order']=='0412356' and saved['ui']['right_pages']==24 and not saved['window']['title_bar']
    proc,hwnd,log=launch('restart')
    try:
        s=wait(proc,log,lambda v:v.get('mask')=='24' and v.get('order')=='0412356' and v.get('right_count')=='2' and int(v.get('dock3','0'))>0 and v.get('dock3')==v.get('dock4'))
        click(hwnd,s['tool3']);s=wait(proc,log,lambda v:v.get('right')=='3');click(hwnd,s['close3']);s=wait(proc,log,lambda v:v.get('mask')=='16' and coords(v['tool4'])[0]>500 and coords(v['tool4'])[1]>20)
        click(hwnd,s['tool4']);s=wait(proc,log,lambda v:v.get('right')=='4');click(hwnd,s['close4']);s=wait(proc,log,lambda v:v.get('mask')=='0' and v.get('right_count')=='0' and coords(v['nav3'])[0]>0 and coords(v['nav4'])[0]>0)
        capture(hwnd,out/'returned-navigation.png')
    finally:close(proc,hwnd)
    assert cfg.read_bytes()==original
    for log in logs:assert renderer_clean(log.read_text(encoding='utf-8'),'dx11')
    result=dict(passed=True,rounded_navigation=True,long_press_order_saved=True,quick_drag_ignored=True,right_native_horizontal_tabs=True,close_returns_to_original_navigation_position=True,restart=True,default_frameless=True,pet_modal_move=modal_result,business_unchanged=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),scope='DX11 and same-monitor Windows move loop; no computation or Vulkan rendering')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
