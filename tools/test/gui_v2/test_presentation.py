"""Exercise real worker tabs, per-worker log settings and native hidden chrome."""
import argparse,ctypes as c,hashlib,json,re,subprocess,time,tomllib
from ctypes import wintypes as w
from pathlib import Path
from win32_probe import window,close,capture,user
from renderer_probe import renderer_clean

def main():
    p=argparse.ArgumentParser()
    for name in ['exe','worker','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--language',choices=['en','zh'],default='zh');p.add_argument('--light',action='store_true')
    a=p.parse_args();out=a.out.resolve();out.mkdir();exe=a.exe.resolve();worker=a.worker.resolve()
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    user.GetWindowLongPtrW.argtypes=[w.HWND,c.c_int];user.GetWindowLongPtrW.restype=c.c_ssize_t
    user.SendMessageW.argtypes=[w.HWND,w.UINT,w.WPARAM,w.LPARAM];user.SendMessageW.restype=c.c_ssize_t
    user.IsZoomed.argtypes=[w.HWND];user.IsIconic.argtypes=[w.HWND]
    queue=out/'worktodo.txt';queue.write_text('[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n[Worker #2]\nECMSTAGE1=1,2,521,-1,100,1\n[Worker #3]\nECMSTAGE2=1,2,137,-1,"ed_m137.save",0,0,10\n')
    cfg=out/'ecm.toml';cfg.write_text('[shared]\nworktodo='+json.dumps(str(queue))+'\n[gui]\nNumWorkers=3\nexit_confirm="stop"\ngraceful_stop_ms=1500\nexe='+json.dumps(str(worker))+'\nstage2_exe='+json.dumps(str(worker))+'\n'+''.join(f'[workers.{i}]\nname="{name}"\nautostart=1\nextra_args="--scenario {scenario}"\n' for i,name,scenario in [(1,'Alpha','v2-presentation'),(2,'Beta','v2-presentation'),(3,'Edwards Stage2','v2-presentation-stage2')]),encoding='utf-8')
    original=cfg.read_bytes();state=out/'ui.toml';state.write_text('[ui]\npage=0\nlogs=true\ninspector=false\ntheme="'+('light' if a.light else 'dark')+'"\n[window]\ntitle_bar=false\n')
    log=out/'ui.log';startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-nvml','--no-pet','--language',a.language,'--ui-scale','1.25'],cwd=out,startupinfo=startup)
    hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,40,40,1920,1200,0x14);user.ShowWindow(hwnd,4)
    def wait(predicate,seconds=12):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            text=log.read_text(encoding='utf-8');rows=[dict(re.findall(r'(\w+)=([-\d.,]+)',line)) for line in text.splitlines() if line.startswith('presentation_sample:')]
            if rows and predicate(rows[-1],text):return rows[-1]
            assert proc.poll() is None,text[-2000:];time.sleep(.05)
        raise AssertionError(log.read_text(encoding='utf-8')[-2500:])
    def click(value):
        x,y=map(int,value.split(','));packed=(y<<16)|(x&65535)
        for action in [0,1,2]:user.PostMessageW(hwnd,0x8021,action,packed);time.sleep(.15)
        time.sleep(.15)
    try:
        s=wait(lambda v,t:v['log_worker']=='1' and 'worker 3: state QueueEmpty' in t)
        assert not user.GetWindowLongPtrW(hwnd,-16)&0xC00000
        point=w.POINT(*map(int,s['drag'].split(',')));assert user.ClientToScreen(hwnd,c.byref(point))
        assert user.SendMessageW(hwnd,0x84,0,((point.y&65535)<<16)|(point.x&65535))==2
        time.sleep(4.2);capture(hwnd,out/'workers-and-alpha.png')
        s=wait(lambda v,t:int(v.get('speed_samples','0'))>=6 and 'table_eta=150 s | log_eta=150 s (2 m 30 s)' in t)
        assert int(s['action1'].split(',')[0])<int(s['worker1'].split(',')[0])
        assert 0<float(s['line_pitch'])-float(s['font_height'])<=3.5
        # A spanning task row is an opaque band after table borders are merged.
        from PIL import Image
        image=Image.open(out/'workers-and-alpha.png');rect=w.RECT();user.GetWindowRect(hwnd,c.byref(rect))
        origin=w.POINT(0,0);assert user.ClientToScreen(hwnd,c.byref(origin))
        x1,y1,x2,y2=map(int,s['summary1'].split(','));ox,oy=origin.x-rect.left,origin.y-rect.top
        samples=[image.getpixel((x+ox,y1+2+oy)) for x in range(x1+12,x2-12)]
        assert len(set(samples))==1,sorted(set(samples))
        click(s['log3']);s=wait(lambda v,t:v['log_worker']=='3' and 'shape=B1=260000000  B2=1300000000000  S=4283 bits  D=1381380  P=126720' in t)
        capture(hwnd,out/'stage2-shape.png')
        click(s['log1']);s=wait(lambda v,t:v['log_worker']=='1')
        click(s['log2']);s=wait(lambda v,t:v['log_worker']=='2');click(s['raw_button']);s=wait(lambda v,t:v['log_worker']=='2' and v['raw']=='1')
        click(s['follow_button']);s=wait(lambda v,t:v['log_worker']=='2' and v['follow']=='0');capture(hwnd,out/'beta-raw.png')
        click(s['log1']);s=wait(lambda v,t:v['log_worker']=='1' and v['raw']=='0' and v['follow']=='1')
        click(s['log2']);s=wait(lambda v,t:v['log_worker']=='2' and v['raw']=='1' and v['follow']=='0')
        click(s['log0']);s=wait(lambda v,t:v['log_worker']=='0');capture(hwnd,out/'application-log.png')
        click(s['worker1']);s=wait(lambda v,t:v['log_worker']=='1')
        restore_x=int(s['chrome1'].split(',')[0]);click(s['chrome1'])
        # Native maximize completes before the next rendered audit sample. Wait
        # for the new button geometry so restore cannot click a stale position.
        s=wait(lambda v,t:bool(user.IsZoomed(hwnd)) and int(v['chrome1'].split(',')[0])>restore_x+100)
        click(s['chrome1']);wait(lambda v,t:not user.IsZoomed(hwnd) and abs(int(v['chrome1'].split(',')[0])-restore_x)<10)
        s=wait(lambda v,t:v['title']=='0');click(s['chrome0'])
        end=time.monotonic()+5
        while not user.IsIconic(hwnd) and time.monotonic()<end:time.sleep(.05)
        assert user.IsIconic(hwnd);user.ShowWindow(hwnd,9);s=wait(lambda v,t:not user.IsIconic(hwnd))
        # Resize while the caption is hidden; native resizing borders remain.
        assert user.GetWindowLongPtrW(hwnd,-16)&0x40000
        user.SetWindowPos(hwnd,None,80,80,1120,850,0x14);time.sleep(.6)
        wait(lambda v,t:float(v.get('log_height','0'))>=2*float(v['font_height']))
        capture(hwnd,out/'narrow-workers.png')
        s=wait(lambda v,t:v['title']=='0');click(s['chrome2']);proc.wait(timeout=12)
    except Exception:
        if proc.poll() is None:capture(hwnd,out/'failure.png')
        raise
    finally:
        if proc.poll() is None:close(proc,hwnd)
    assert proc.returncode==0 and renderer_clean(log.read_text(encoding='utf-8'),'dx11')
    saved=tomllib.loads(state.read_text(encoding='utf-8'));assert not saved['window']['title_bar'] and cfg.read_bytes()==original
    restart_log=out/'restart.log';restart=subprocess.run([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(restart_log),'--frames','8','--no-nvml','--no-pet'],cwd=out,startupinfo=startup,timeout=20)
    assert restart.returncode==0 and 'UI state not loaded' not in restart_log.read_text(encoding='utf-8')
    assert not tomllib.loads(state.read_text(encoding='utf-8'))['window']['title_bar']
    report=dict(passed=True,language=a.language,theme='light' if a.light else 'dark',workers=3,log_tabs=4,action_first=True,summary_covers_grid=True,table_eta_seconds=True,log_eta_seconds_and_reduced=True,stage2_shape_line=True,speed_samples=6,compact_log_rows=True,per_worker_raw_follow=True,table_selects_log=True,hidden_caption=True,caption_hit_test=True,maximize_restore=True,minimize_restore=True,resize=True,custom_close=True,title_preference_restart=True,business_unchanged=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),scope='Protocol-aware synthetic workers, no real ECM calculation')
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
