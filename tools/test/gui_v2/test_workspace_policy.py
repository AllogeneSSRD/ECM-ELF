"""Native log-following, native Hide tab bar persistence and curve-time fitting.

Uses isolated, protocol-aware synthetic workers; no GPU sampling or ECM work.
"""
import argparse,ctypes as c,hashlib,json,re,subprocess,time,tomllib
from pathlib import Path
from win32_probe import window,close,capture,user
from renderer_probe import renderer_clean


def main():
    p=argparse.ArgumentParser()
    for key in ['exe','worker','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();exe=a.exe.resolve();worker=a.worker.resolve();out=a.out.resolve();out.mkdir()
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    queue=out/'worktodo.txt';queue.write_text('[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n')
    cfg=out/'ecm.toml';cfg.write_text('[shared]\nworktodo='+json.dumps(str(queue))+'\n[gui]\nNumWorkers=1\nexe='+json.dumps(str(worker))+'\nexit_confirm="stop"\ngraceful_stop_ms=1500\n[workers.1]\nautostart=1\nextra_args="--scenario v2-presentation"\n')
    original=cfg.read_bytes();state=out/'ui.toml';state.write_text('[ui]\npage=0\nlogs=true\nright_pages=16\nright_page=4\ninspector=false\nscale=1\nfont_size=15\nlanguage="en"\ntoasts=false\n')
    def launch(name):
        log=out/(name+'.log');si=subprocess.STARTUPINFO();si.dwFlags=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
        proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-nvml','--no-pet'],cwd=out,startupinfo=si)
        hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,30,30,1500,1200,0x14);user.ShowWindow(hwnd,4)
        return proc,hwnd,log
    proc,hwnd,log=launch('policy')
    def sample():
        text=log.read_text(encoding='utf-8');end=text.rfind('\nview_sample:');text=text[:end] if end>=0 else ''
        prefixes=('view_sample:','workspace_policy:','usability_sample:','navigation_sample:','presentation_sample:','worker_scale:','worker_layout_sample:')
        latest={}
        for line in text.splitlines():
            if line.startswith(prefixes):latest[line.split(':',1)[0]]=line
        result={}
        for line in latest.values():result.update(dict(re.findall(r'(\w+)=([^\s]+)',line)))
        return result
    def wait(predicate,seconds=18):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            row=sample()
            if row and predicate(row):return row
            assert proc.poll() is None;time.sleep(.06)
        raise AssertionError(sample())
    def pointer(pos,action):
        x,y=map(round,pos);before=log.read_text(encoding='utf-8').count('\nview_sample:')
        user.PostMessageW(hwnd,0x8021,action,(y<<16)|(x&65535));end=time.monotonic()+8
        while log.read_text(encoding='utf-8').count('\nview_sample:')<before+2:
            assert proc.poll() is None and time.monotonic()<end;time.sleep(.025)
    def coords(key):return tuple(map(float,sample()[key].split(',')))
    def click(key):
        time.sleep(.12);pos=coords(key)
        for action in [0,1,2]:pointer(pos,action)
        time.sleep(.12)
    def menu(key):
        click('view');wait(lambda r:r.get(key,'0,0')!='0,0');click(key)
    def page(index,visible):
        click('nav'+str(index));return wait(lambda r:r['page']==str(index) and r['logs_visible']==str(int(visible)))
    def move_right(index):
        start,end=coords('nav'+str(index)),coords('drop');pointer(start,0);pointer(start,1);time.sleep(.4)
        for i in range(1,9):pointer((start[0]+(end[0]-start[0])*i/8,start[1]+(end[1]-start[1])*i/8),0)
        pointer(end,2)
    def all_hidden(row):return all(row.get('preference'+str(i))=='1' and row.get('hidden'+str(i))=='1' for i in range(3))
    try:
        r=wait(lambda r:r.get('logs_visible')=='1' and int(r.get('workspace_speed','0'))>=6 and r.get('ymin0')=='0' and int(r.get('node2','0'))>0)
        before_width=float(r['nav_size'].split(',')[0]);builds=r['layout_builds'];base_max=float(r['ymax0'])
        click('auto_control');r=wait(lambda r:r['auto_y']=='1' and float(r['ymin0'])>0)
        assert 0<float(r['ymin0'])<1.75 and 2.25<float(r['ymax0'])<base_max,r
        capture(hwnd,out/'auto-y.png')
        page(1,False);menu('logs_control');wait(lambda r:r['logs_visible']=='1' and r['logs_enabled']=='1')
        page(3,False);page(0,True);page(1,False)
        menu('keep_control');wait(lambda r:r['keep']=='1' and r['logs_visible']=='1')
        page(3,True);menu('keep_control');wait(lambda r:r['keep']=='0' and r['logs_visible']=='0');page(0,True)
        r=sample();assert r['layout_builds']==builds and abs(float(r['nav_size'].split(',')[0])-before_width)<=1,r
        move_right(0);wait(lambda r:r['mask']=='17' and r['page']=='1' and r['right']=='0' and r['logs_visible']=='1')
        click('tool4');wait(lambda r:r['right']=='4' and r['logs_visible']=='0')
        click('tool0');wait(lambda r:r['right']=='0' and r['logs_visible']=='1')
        click('close0');wait(lambda r:r['mask']=='16' and r['logs_visible']=='0');page(0,True)
        for i in range(3):
            click('menu'+str(i));wait(lambda r:r.get('hide_control','0,0')!='0,0');click('hide_control')
            wait(lambda r:r['preference'+str(i)]=='1' and r['hidden'+str(i)]=='1')
        capture(hwnd,out/'hidden-tab-bars.png')
        page(1,False);assert all_hidden(sample());page(0,True);assert all_hidden(sample())
        # A responsive dock rebuild must carry the role preferences to new nodes.
        user.SetWindowPos(hwnd,None,30,30,900,1100,0x14);wait(all_hidden)
        user.SetWindowPos(hwnd,None,30,30,1500,1200,0x14);wait(all_hidden)
        # Multi-tab native docks show their tabs, then restore the single-tab choice.
        move_right(3);wait(lambda r:r['mask']=='24' and r['right_count']=='2' and r['hidden2']=='0' and r['preference2']=='1')
        click('close3');wait(lambda r:r['mask']=='16' and r['right_count']=='1' and all_hidden(r))
        page(1,False);menu('keep_control');wait(lambda r:r['keep']=='1' and r['logs_visible']=='1')
        menu('save_control');wait(lambda r:r['save_count']=='1')
        saved=tomllib.loads(state.read_text(encoding='utf-8'))['ui']
        assert all(saved[key] for key in ['logs','logs_keep_visible','nav_hide_tab_bar','workspace_hide_tab_bar','right_hide_tab_bar','worker_chart_auto_y']),saved
        capture(hwnd,out/'saved.png')
    except BaseException:
        capture(hwnd,out/'failure.png');raise
    finally:close(proc,hwnd)
    assert renderer_clean(log.read_text(encoding='utf-8'),'dx11')
    proc,hwnd,log=launch('restart')
    try:
        wait(lambda r:r.get('page')=='1' and r.get('keep')=='1' and r.get('logs_visible')=='1' and all_hidden(r))
        page(0,True);r=wait(lambda r:int(r.get('workspace_speed','0'))>=6 and r['auto_y']=='1' and float(r['ymin0'])>0)
        capture(hwnd,out/'restart.png')
    except BaseException:
        capture(hwnd,out/'restart-failure.png');raise
    finally:close(proc,hwnd)
    assert cfg.read_bytes()==original and renderer_clean(log.read_text(encoding='utf-8'),'dx11')
    # Legacy sidecars already stored the native layout flags. Adopt those flags
    # when the new role keys are absent, rather than overwrite them with false.
    keys={'nav_hide_tab_bar','workspace_hide_tab_bar','right_hide_tab_bar'}
    state.write_text('\n'.join(line for line in state.read_text(encoding='utf-8').splitlines() if line.partition('=')[0].strip() not in keys)+'\n',encoding='utf-8')
    proc,hwnd,log=launch('legacy')
    try:
        wait(all_hidden);capture(hwnd,out/'legacy-layout.png')
    finally:close(proc,hwnd)
    assert renderer_clean(log.read_text(encoding='utf-8'),'dx11')
    result=dict(passed=True,log_follow_workers=True,workers_visible_in_right_tabs=True,legacy_hidden_layout_adopted=True,manual_other_view=True,pin_across_pages=True,navigation_width_unchanged=True,native_hide_tab_menu=True,three_roles_restart=True,responsive_rebuild=True,multi_tab_restores_single_preference=True,auto_y_current_samples=True,auto_y_restart=True,workflow_unchanged=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),scope='DX11 synthetic worker only; no ECM/GPU task')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
