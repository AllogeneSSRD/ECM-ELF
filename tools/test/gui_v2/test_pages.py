"""Render all pages in the selected backend, wide/narrow and both languages."""
import argparse,ctypes as c,hashlib,json,os,subprocess,time
from pathlib import Path
from renderer_probe import renderer_args,renderer_clean
from win32_probe import window,capture,close,user
def main():
    p=argparse.ArgumentParser();p.add_argument('--renderer',choices=['dx11','vulkan'],default='dx11');p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--layers',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir();exe=a.exe.resolve();env=os.environ.copy();env['VK_LAYER_PATH']=str(a.layers.resolve())
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    cfg=out/'ecm.toml';cfg.write_text('[gui]\nNumWorkers=2\n[workers.1]\nautostart=0\n[workers.2]\nautostart=0\n');before=cfg.read_bytes()
    row={'status':'F','exponent':127,'worktype':'ECM','b1':100,'stage':1,'factors':['233'],'Edwards':{},'curves':1,'program':{'name':'ECM-ELF','version':'fixture'},'timestamp':'2026-10-21 00:00:00','checksum':{'version':1,'checksum':'1234ABCD'}}
    (out/'results.json.txt').write_text(json.dumps(row)+'\n');screens=[]
    for zh in [False,True]:
        log=out/f'pages-{zh}.log';state=out/f'ui-{zh}.toml'
        startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app',*renderer_args(a.renderer),'--language','zh' if zh else 'en'],cwd=out,env=env,startupinfo=startup)
        hwnd=window(proc.pid,'ecm_gui_v2');user.ShowWindow(hwnd,4)
        try:
            for page in range(7):
                user.PostMessageW(hwnd,0x8022,7,page);end=time.monotonic()+8
                while time.monotonic()<end:
                    if ('settings_sample: open=1' if page==5 else f'page={page} ') in log.read_text(encoding='utf-8'):break
                    assert proc.poll() is None;time.sleep(.08)
                else:raise AssertionError(f'Page {page} failed')
                image=f'page-{page}-{zh}.png';capture(hwnd,out/image);screens.append(image)
            user.SetWindowPos(hwnd,None,40,40,800,600,0x14);user.PostMessageW(hwnd,0x8022,7,5);time.sleep(.4);capture(hwnd,out/f'narrow-{zh}.png')
        finally:close(proc,hwnd)
        assert proc.returncode==0 and renderer_clean(log.read_text(encoding='utf-8'),a.renderer)
    assert cfg.read_bytes()==before
    result=dict(renderer=a.renderer,passed=True,pages=7,languages=2,wide_narrow=True,screens=screens,workflow_unchanged=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),limits=['Page navigation uses explicit audit seam; real IME composition remains unqualified'])
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
