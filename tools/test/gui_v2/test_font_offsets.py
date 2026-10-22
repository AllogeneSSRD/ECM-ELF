"""Live primary/secondary font offsets and restart; idle isolated DX11 UI."""
import argparse,ctypes as c,hashlib,json,re,subprocess,time,tomllib
from pathlib import Path
from win32_probe import window,close,user,capture
from renderer_probe import renderer_clean

def main():
    p=argparse.ArgumentParser()
    for key in ['exe','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();exe=a.exe.resolve();out=a.out.resolve();out.mkdir();cfg=out/'ecm.toml'
    cfg.write_text('[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n');original=cfg.read_bytes()
    state=out/'ui.toml';state.write_text('[ui]\npage=0\nlanguage="zh"\nscale=1\nlogs=true\ninspector=false\n')
    log=out/'ui.log';startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-nvml','--no-pet'],cwd=out,startupinfo=startup)
    hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,40,40,1920,1400,0x14);user.ShowWindow(hwnd,4)
    def rows(prefix):
        return [dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.read_text(encoding='utf-8').splitlines() if line.startswith(prefix+':')]
    def wait(predicate):
        end=time.monotonic()+12
        while time.monotonic()<end:
            values=rows('typography_sample')
            if values and predicate(values[-1]):return values[-1]
            assert proc.poll() is None;time.sleep(.06)
        raise AssertionError(log.read_text(encoding='utf-8')[-2500:])
    def click(value,offset=0):
        x,y=map(int,value.split(','));x+=offset;assert x>0 and y>0
        for action in [0,1,2]:user.PostMessageW(hwnd,0x8021,action,(y<<16)|(x&65535));time.sleep(.14)
    try:
        s=wait(lambda v:float(v['primary_button'])==16 and float(v['secondary_button'])==15)
        fonts_before=log.read_text(encoding='utf-8').count('fonts: ')
        user.PostMessageW(hwnd,0x8022,7,5)
        s=wait(lambda v:int(v['primary_control'].split(',')[0])>0 and float(v['secondary_note'])==13)
        click(s['primary_control'],25);s=wait(lambda v:3<=int(v['primary_offset'])<=6 and float(v['primary_size'])==15+int(v['primary_offset']))
        click(s['secondary_control'],-70);s=wait(lambda v:-5<=int(v['secondary_offset'])<=-3 and float(v['secondary_size'])==15+int(v['secondary_offset']) and float(v['secondary_note'])==15+int(v['secondary_offset']))
        primary=int(s['primary_offset']);secondary=int(s['secondary_offset']);capture(hwnd,out/'offset-settings.png')
        user.PostMessageW(hwnd,0x8022,7,0)
        s=wait(lambda v:float(v['primary_button'])==15+primary and float(v['secondary_button'])==15)
        presentation=rows('presentation_sample')[-1];assert float(presentation['font_height'])==float(presentation['body_font'])==15 and presentation['log_body']=='1'
        assert log.read_text(encoding='utf-8').count('fonts: ')==fonts_before
        capture(hwnd,out/'compact-workers.png')
    finally:close(proc,hwnd)
    saved=tomllib.loads(state.read_text(encoding='utf-8'));assert saved['ui']['primary_font_offset']==primary and saved['ui']['secondary_font_offset']==secondary
    restart=out/'restart.log';r=subprocess.run([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(restart),'--audit-app','--frames','12','--no-nvml','--no-pet'],cwd=out,startupinfo=startup,timeout=25)
    assert r.returncode==0 and f'primary_offset={primary} secondary_offset={secondary}' in restart.read_text(encoding='utf-8')
    assert cfg.read_bytes()==original and renderer_clean(log.read_text(encoding='utf-8'),'dx11')
    result=dict(passed=True,primary_offset=primary,secondary_offset=secondary,body_log_unchanged=True,primary_secondary_roles_distinct=True,no_font_file_reload=True,restart=True,business_unchanged=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
