"""Exercise the ELY-style settings shell using actual controls; no computation."""
import argparse,ctypes as c,hashlib,json,re,subprocess,time,tomllib
from pathlib import Path
from win32_probe import window,close,capture,user
from renderer_probe import renderer_clean
from language_probe import select_language

def main():
    p=argparse.ArgumentParser()
    for key in ['exe','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir();exe=a.exe.resolve()
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    cfg=out/'ecm.toml';cfg.write_text('[shared]\nworktodo="worktodo.txt"\nb2_ratio=0\n[stage1]\ntarget="cpu"\n[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n');original=cfg.read_bytes()
    state=out/'ui.toml';state.write_text('[ui]\npage=0\nlanguage="en"\nscale=1\nlogs=false\ninspector=false\n')
    startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    def launch(name):
        log=out/(name+'.log');proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-nvml','--no-pet'],cwd=out,startupinfo=startup)
        hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,40,40,1920,1200,0x14);user.ShowWindow(hwnd,4);return proc,hwnd,log
    proc,hwnd,log=launch('settings')
    def rows(prefix):
        content=log.read_text(encoding='utf-8');end=content.rfind('\nworker_layout_sample:')
        content=content[:end] if end>=0 else ''
        return [dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in content.splitlines() if line.startswith(prefix+':')]
    def wait(prefix,predicate):
        end=time.monotonic()+12
        while time.monotonic()<end:
            values=rows(prefix)
            if values and predicate(values[-1]):return values[-1]
            assert proc.poll() is None;time.sleep(.07)
        raise AssertionError(log.read_text(encoding='utf-8')[-5000:])
    def click(value):
        x,y=map(int,value.split(','));assert x>0 and y>0,value
        for action in [0,1,2]:
            before=log.read_text(encoding='utf-8').count('\nworker_layout_sample:')
            user.PostMessageW(hwnd,0x8021,action,(y<<16)|(x&65535));end=time.monotonic()+8
            while log.read_text(encoding='utf-8').count('\nworker_layout_sample:')<before+2:
                assert proc.poll() is None and time.monotonic()<end;time.sleep(.04)
    def category(i):
        s=wait('settings_sample',lambda v:v['open']=='1' and int(v['category'+str(i)].split(',')[0])>0)
        click(s['category'+str(i)]);return wait('settings_sample',lambda v:v['category']==str(i))
    try:
        nav=wait('usability_sample',lambda v:int(v['nav5'].split(',')[0])>0)
        click(nav['nav5']);s=wait('settings_sample',lambda v:v['open']=='1' and v['category']=='0' and v['stacked']=='0')
        assert s['icons']=='7';assert rows('usability_sample')[-1]['page']=='0'
        assert all(int(s['category'+str(i)].split(',')[0])>0 for i in range(6))
        capture(hwnd,out/'appearance-dark.png');click(s['theme1']);time.sleep(.4);capture(hwnd,out/'appearance-light.png')
        category(1);capture(hwnd,out/'layout.png')
        category(2);gpu=wait('gpu_settings',lambda v:int(v['range3'].split(',')[0])>0);click(gpu['range3']);wait('gpu_settings',lambda v:v['seconds']=='120');capture(hwnd,out/'monitor.png')
        category(3);wait('pet_skin_sample',lambda v:int(v['skin0'].split(',')[0])>0);capture(hwnd,out/'pet.png')
        category(4);capture(hwnd,out/'workflow.png')
        category(5);v=wait('usability_sample',lambda v:v['settings_category']=='5' and int(v['search'].split(',')[0])>0)
        click(v['search'])
        for letter in 'target':user.PostMessageW(hwnd,0x102,ord(letter),0)
        wait('usability_sample',lambda v:v['matches']=='1');capture(hwnd,out/'computation-search.png')
        for letter in '_no_such_field':user.PostMessageW(hwnd,0x102,ord(letter),0)
        wait('usability_sample',lambda v:v['matches']=='0')
        category(0);select_language(hwnd,log,'chineseSimplified.xml',click_handler=click);wait('usability_sample',lambda v:v['zh']=='1')
        select_language(hwnd,log,'english.xml',click_handler=click);wait('usability_sample',lambda v:v['zh']=='0')
        v=rows('usability_sample')[-1];click(v['save']);time.sleep(.3)
        saved=tomllib.loads(state.read_text(encoding='utf-8'));assert saved['ui']['theme']=='light' and saved['ui']['gpu_history_seconds']==120 and saved['ui']['page']==0
        s=rows('settings_sample')[-1];before=tuple(map(float,s['min'].split(',')+s['max'].split(',')))
        click(s['expand']);s=wait('settings_sample',lambda v:float(v['max'].split(',')[0])-float(v['min'].split(',')[0])>before[2]-before[0]+100)
        click(s['expand']);wait('settings_sample',lambda v:abs(float(v['max'].split(',')[0])-float(v['min'].split(',')[0])-(before[2]-before[0]))<5)
        user.SetWindowPos(hwnd,None,40,40,800,650,0x14);s=wait('settings_sample',lambda v:v['stacked']=='1' and float(v['max'].split(',')[0])<=800 and float(v['max'].split(',')[1])<=650)
        assert float(s['content'].split(',')[3])-float(s['content'].split(',')[1])>100
        capture(hwnd,out/'narrow.png');click(s['close']);wait('settings_sample',lambda v:v['open']=='0')
        nav=rows('usability_sample')[-1];click(nav['nav5']);wait('settings_sample',lambda v:v['open']=='1')
        user.PostMessageW(hwnd,0x100,27,1);user.PostMessageW(hwnd,0x101,27,0xC0000001);wait('settings_sample',lambda v:v['open']=='0')
    except BaseException:
        if proc.poll() is None:capture(hwnd,out/'failure.png')
        raise
    finally:close(proc,hwnd)
    assert cfg.read_bytes()==original and renderer_clean(log.read_text(encoding='utf-8'),'dx11')
    # An old sidecar with docked Settings must open the new shell and retain a real central page.
    state.write_text('[ui]\npage=5\nright_pages=32\nright_page=5\nlanguage="zh"\nscale=1\nlogs=false\ninspector=false\n')
    proc,hwnd,log=launch('legacy')
    try:
        s=wait('settings_sample',lambda v:v['open']=='1');v=rows('usability_sample')[-1];assert v['page']=='0' and v['mask']=='0';capture(hwnd,out/'legacy-settings.png')
    finally:close(proc,hwnd)
    saved=tomllib.loads(state.read_text(encoding='utf-8'));assert saved['ui']['page']==0 and saved['ui']['right_pages']==0
    report=dict(passed=True,navigation_icons=7,settings_outside_page_order=True,six_categories=True,theme_cards=True,live_language=True,monitor_range=True,computation_search_and_empty_state=True,ui_save=True,workspace_preserved=True,maximize_restore=True,narrow_category_selector=True,close_and_escape=True,legacy_sidecar_migrated=True,business_unchanged=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),scope='Idle isolated DX11 UI, no computation or Vulkan rendering')
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
