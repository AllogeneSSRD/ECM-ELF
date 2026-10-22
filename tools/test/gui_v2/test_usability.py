"""Exercise page dragging, live language controls, snapshot charts and sidecar restart."""
import argparse,ctypes as c,hashlib,json,os,re,subprocess,time,tomllib
from pathlib import Path
from win32_probe import window,close,user,capture

def main():
    parser=argparse.ArgumentParser()
    for name in ['exe','out','layers']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir();exe=args.exe.resolve()
    cfg=out/'ecm.toml';cfg.write_text('[shared]\nworktodo="worktodo.txt"\nb2_ratio=0\n[stage1]\ntarget="cpu"\n[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n',encoding='utf-8');before=cfg.read_bytes()
    state=out/'ui.toml';env=os.environ.copy();env['VK_LAYER_PATH']=str(args.layers.resolve())
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p
    assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    logs=[]
    def launch(name,extra=()):
        log=out/(name+'.log');logs.append(log)
        startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--validation','--no-nvml',*extra],cwd=out,env=env,startupinfo=startup)
        hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,60,80,1920,1200,0x14);user.ShowWindow(hwnd,4)
        return proc,hwnd,log
    def snapshots(log):
        return [dict(re.findall(r'(\w+)=([-\d.,]+)',line)) for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('usability_sample:')]
    def wait(proc,log,predicate):
        end=time.monotonic()+12
        while time.monotonic()<end:
            values=snapshots(log)
            if values and predicate(values[-1]):return values[-1]
            assert proc.poll() is None;time.sleep(.05)
        raise AssertionError(log.read_text(encoding='utf-8')[-3000:])
    def point(value):
        x,y=map(int,value.split(','));assert x>=0 and y>=0,value;return (y<<16)|(x&65535)
    def click(hwnd,value):
        packed=point(value);user.PostMessageW(hwnd,0x8021,0,packed);time.sleep(.2)
        user.PostMessageW(hwnd,0x8021,1,packed);time.sleep(.2);user.PostMessageW(hwnd,0x8021,2,packed);time.sleep(.3)
    def drag(hwnd,source,target):
        start=point(source);end=point(target);user.PostMessageW(hwnd,0x8021,0,start);time.sleep(.2)
        user.PostMessageW(hwnd,0x8021,1,start);time.sleep(.2)
        sx,sy=map(int,source.split(','));tx,ty=map(int,target.split(','))
        for i in range(1,9):
            x=round(sx+(tx-sx)*i/8);y=round(sy+(ty-sy)*i/8);user.PostMessageW(hwnd,0x8021,0,(y<<16)|(x&65535));time.sleep(.1)
        time.sleep(.2);user.PostMessageW(hwnd,0x8021,2,end);time.sleep(.4)
    proc,hwnd,log=launch('first')
    try:
        s=wait(proc,log,lambda v:v['page']=='0' and int(v['nav4'].split(',')[0])>0)
        drag(hwnd,s['nav4'],s['drop']);s=wait(proc,log,lambda v:v['mask']=='16')
        drag(hwnd,s['nav3'],s['drop']);s=wait(proc,log,lambda v:v['mask']=='24' and int(v['calculate'].split(',')[0])>0)
        click(hwnd,s['calculate']);s=wait(proc,log,lambda v:v['charts']=='1');capture(hwnd,out/'right-probability-charts.png')
        click(hwnd,s['nav5']);s=wait(proc,log,lambda v:v['page']=='5' and int(v['cn'].split(',')[0])>0)
        click(hwnd,s['cn']);s=wait(proc,log,lambda v:v['zh']=='1');capture(hwnd,out/'settings-zh.png')
        click(hwnd,s['en']);s=wait(proc,log,lambda v:v['zh']=='0')
        click(hwnd,s['cn']);s=wait(proc,log,lambda v:v['zh']=='1');click(hwnd,s['save'])
        saved=tomllib.loads(state.read_text(encoding='utf-8'));assert saved['ui']['right_pages']==24 and saved['ui']['language']=='zh'
        click(hwnd,s['settings2']);s=wait(proc,log,lambda v:v['settings_tab']=='2' and int(v['search'].split(',')[0])>0)
        click(hwnd,s['search'])
        for letter in 'target':user.PostMessageW(hwnd,0x102,ord(letter),0)
        s=wait(proc,log,lambda v:v['matches']=='1');capture(hwnd,out/'settings-search.png')
        for letter in '_no_such_field':user.PostMessageW(hwnd,0x102,ord(letter),0)
        s=wait(proc,log,lambda v:v['matches']=='0')
        drag(hwnd,s['tool4'],s['back']);s=wait(proc,log,lambda v:v['mask']=='8' and v['page']=='4')
        user.SetWindowPos(hwnd,None,60,80,800,650,0x14);time.sleep(.6);capture(hwnd,out/'narrow-right-tools.png')
    except BaseException:
        capture(hwnd,out/'failure-first.png');raise
    finally:close(proc,hwnd)
    assert proc.returncode==0
    proc,hwnd,log=launch('restart',['--theme','light'])
    try:
        s=wait(proc,log,lambda v:v['mask']=='8' and v['zh']=='1' and int(v['calculate'].split(',')[0])>0)
        click(hwnd,s['calculate']);s=wait(proc,log,lambda v:v['charts']=='1');capture(hwnd,out/'restored-light-charts.png')
        drag(hwnd,s['tool3'],s['back']);s=wait(proc,log,lambda v:v['mask']=='0' and v['page']=='3')
    finally:close(proc,hwnd)
    assert proc.returncode==0 and cfg.read_bytes()==before
    proc,hwnd,log=launch('font-rescue',['--simulate-missing-cjk','--language','en'])
    try:
        s=wait(proc,log,lambda v:int(v['nav5'].split(',')[0])>0);click(hwnd,s['nav5'])
        s=wait(proc,log,lambda v:v['page']=='5' and int(v['cn'].split(',')[0])>0);click(hwnd,s['cn']);time.sleep(.8)
        s=wait(proc,log,lambda v:v['zh']=='0');count=log.read_text(encoding='utf-8').count('fonts: ');time.sleep(.8)
        assert log.read_text(encoding='utf-8').count('fonts: ')==count and count<=3
        assert 'Chinese font unavailable' in log.read_text(encoding='utf-8');capture(hwnd,out/'font-rescue.png')
    finally:close(proc,hwnd)
    assert proc.returncode==0
    for log in logs:assert 'validation_summary: errors=0 warnings=0' in log.read_text(encoding='utf-8')
    result=dict(passed=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),drag_to_right_and_back=True,
                clicked_live_language_both_directions=True,clicked_snapshot=True,explicit_save_and_restart=True,
                settings_search_and_empty_state=True,live_font_rescue_without_retry_loop=True,
                actual_snapshot_dark_and_light=True,
                narrow_right_tools=True,business_unchanged=True,limits=['Pointer events use the explicit test seam; physical IME and different monitor DPI remain unqualified.'])
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
