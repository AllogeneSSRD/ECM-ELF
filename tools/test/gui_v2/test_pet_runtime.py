"""Check native pet top/bottom anchors against the owner's actual client rectangle."""
import argparse,ctypes as c,json,os,subprocess,time
from ctypes import wintypes as w
from pathlib import Path
from renderer_probe import renderer_args,renderer_clean
from win32_probe import window,close,user,capture
def main():
    p=argparse.ArgumentParser();p.add_argument('--renderer',choices=['dx11','vulkan'],default='dx11')
    for key in ['exe','out','layers']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir();env=os.environ.copy();env['VK_LAYER_PATH']=str(a.layers.resolve());log=out/'pet.log'
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    user.ClientToScreen.argtypes=[w.HWND,c.POINTER(w.POINT)];user.ClientToScreen.restype=w.BOOL
    proc=subprocess.Popen([str(a.exe.resolve()),'--ui-state',str(out/'ui.toml'),'--diagnostics',str(log),'--pet','--pet-static','--audit-app','--no-nvml',*renderer_args(a.renderer)],cwd=out,env=env)
    hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,200,0,1000,800,0x14);user.ShowWindow(hwnd,4);pet=window(proc.pid,'ecm_gui_v2_pet');time.sleep(.4)
    def rect(target):
        value=w.RECT();assert user.GetWindowRect(target,c.byref(value));return value
    def client():
        value=w.RECT();origin=w.POINT();assert user.GetClientRect(hwnd,c.byref(value)) and user.ClientToScreen(hwnd,c.byref(origin));return origin.y,origin.y+value.bottom
    try:
        user.PostMessageW(hwnd,0x8022,8,5);time.sleep(.4)
        bottom=rect(pet);assert bottom.bottom==client()[1],(bottom.bottom,client());capture(pet,out/'doze-bottom.png')
        user.SetWindowPos(hwnd,None,200,400,1000,800,0x14);time.sleep(.4)
        top=rect(pet);assert top.bottom==client()[0],(top.bottom,client());capture(pet,out/'doze-top.png')
        user.SendMessageW.restype=c.c_ssize_t;assert user.SendMessageW(pet,0x21,hwnd,0x2010001)==3
    finally:close(proc,hwnd)
    assert renderer_clean(log.read_text(encoding='utf-8'),a.renderer)
    result=dict(renderer=a.renderer,passed=True,client_bottom_anchor=True,client_top_anchor=True,sleep_native_capture=True,no_activation=True)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
