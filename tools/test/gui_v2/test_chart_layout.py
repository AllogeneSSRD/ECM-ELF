"""One bounded chart-window drag; CPU/API evidence cannot qualify desktop refresh.

Defaults to DX11 and one native window. Multi-viewport is explicit; do not rerun
the optional Vulkan mode on the affected desktop while its freeze is unresolved.
"""
import argparse,ctypes as c,hashlib,json,os,re,subprocess,time
from pathlib import Path
from renderer_probe import renderer_args,renderer_clean
from win32_probe import window,close,user,capture
def frame_stats(log,offset):
    values=[float(v) for v in re.findall(r'frame_sample: ms=([\d.]+)',log[offset:])]
    return dict(frame_samples=len(values),maximum_drag_frame_ms=max(values,default=0),cpu_frame_under_1s=bool(values) and max(values)<1000)

def main():
    p=argparse.ArgumentParser();p.add_argument('--renderer',choices=['dx11','vulkan'],default='dx11')
    for key in ['exe','out','layers']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--page',type=int,default=3);p.add_argument('--monitor',action='store_true');p.add_argument('--trace',action='store_true')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--single-viewport',action='store_true');mode.add_argument('--multi-viewport',action='store_true')
    a=p.parse_args();out=a.out.resolve();out.mkdir()
    if a.trace and a.renderer!='vulkan':p.error('--trace requires --renderer vulkan')
    env=os.environ.copy();env['VK_LAYER_PATH']=str(a.layers.resolve());log=out/'layout.log';state=out/'ui.toml'
    state.write_text(f'[ui]\npage={a.page}\ninspector=false\nlogs=false\n',encoding='utf-8')
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    extra=[] if a.monitor else ['--no-nvml']
    if a.trace:extra+=['--trace-render']
    extra+=['--multi-viewport' if a.multi_viewport else '--single-viewport']
    proc=subprocess.Popen([str(a.exe.resolve()),'--ui-state',str(state),'--diagnostics',str(log),*renderer_args(a.renderer),'--audit-app','--audit-performance',*extra],cwd=out,env=env)
    hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,80,80,1400,1000,0x14);user.ShowWindow(hwnd,4)
    def pointer(x,y,button=0):user.PostMessageW(hwnd,0x8021,button,(round(y)<<16)|(round(x)&65535));time.sleep(.13)
    result=dict(renderer=a.renderer,exe_sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest(),page=a.page,no_nvml=not a.monitor,single_viewport=not a.multi_viewport,workers_started=False,desktop_refresh_qualified=False)
    try:
        time.sleep(1.5)
        if a.page==3:
            # The probability toolbar is above parameters in this isolated, known layout.
            pointer(290,215);pointer(290,215,1);pointer(290,215,2);time.sleep(.5)
            assert 'calculated=1' in log.read_text(encoding='utf-8'),'Snapshot calculate control was not reached'
        if a.page==4 and a.monitor:assert 'gpu=1' in log.read_text(encoding='utf-8')
        capture(hwnd,out/'before.png');offset=len(log.read_text(encoding='utf-8'))
        pointer(330,65);pointer(330,65,1)
        for i in range(1,9):pointer(330+550*i/8,65+220*i/8)
        pointer(880,285,2);time.sleep(1)
        capture(hwnd,out/'after.png')
        assert 'docked=0' in log.read_text(encoding='utf-8')[offset:],'The workspace was not actually undocked'
    finally:close(proc,hwnd)
    final_log=log.read_text(encoding='utf-8')
    # Include frames completed between the screenshot and final process exit.
    result.update(frame_stats(final_log,offset))
    result.update(exit=proc.returncode,renderer_ok=renderer_clean(final_log,a.renderer))
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert result['cpu_frame_under_1s'] and result['renderer_ok'] and proc.returncode==0
if __name__=='__main__':main()
