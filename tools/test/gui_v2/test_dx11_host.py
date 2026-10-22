"""Bounded default-DX11 startup, real resize/minimize, and loaded-module identity."""
import argparse,ctypes as c,hashlib,json,os,subprocess,time
from ctypes import wintypes as w
from pathlib import Path
from win32_probe import window,close,user,kernel,psapi,capture
from renderer_probe import renderer_clean

def modules(pid):
    kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
    kernel.CloseHandle.argtypes=[w.HANDLE]
    psapi.EnumProcessModulesEx.argtypes=[w.HANDLE,c.POINTER(c.c_void_p),w.DWORD,c.POINTER(w.DWORD),w.DWORD]
    psapi.GetModuleFileNameExW.argtypes=[w.HANDLE,c.c_void_p,w.LPWSTR,w.DWORD]
    handle=kernel.OpenProcess(0x410,False,pid);assert handle
    try:
        values=(c.c_void_p*2048)();needed=w.DWORD();assert psapi.EnumProcessModulesEx(handle,values,c.sizeof(values),c.byref(needed),3)
        assert needed.value<=c.sizeof(values)
        result=[]
        for item in values[:needed.value//c.sizeof(c.c_void_p)]:
            name=c.create_unicode_buffer(32768);assert psapi.GetModuleFileNameExW(handle,item,name,len(name));result.append(Path(name.value).name.lower())
        return sorted(result)
    finally:kernel.CloseHandle(handle)

def main():
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir();log=out/'dx11.log'
    env=os.environ.copy();env.pop('VULKAN_SDK',None);env.pop('VK_LAYER_PATH',None);env.pop('VK_INSTANCE_LAYERS',None)
    proc=subprocess.Popen([str(a.exe.resolve()),'--ui-state',str(out/'ui.toml'),'--diagnostics',str(log),'--no-nvml'],cwd=out,env=env)
    hwnd=window(proc.pid,'ecm_gui_v2');user.ShowWindow(hwnd,4)
    try:
        end=time.monotonic()+12
        while 'frame: 1' not in log.read_text(encoding='utf-8'):
            assert proc.poll() is None and time.monotonic()<end;time.sleep(.05)
        loaded=modules(proc.pid);assert 'd3d11.dll' in loaded and 'dxgi.dll' in loaded
        assert 'vulkan-1.dll' not in loaded and 'nvoglv64.dll' not in loaded
        for x,y,width,height in [(80,80,1400,1000),(100,100,900,700),(80,80,1400,1000)]:
            assert user.SetWindowPos(hwnd,None,x,y,width,height,0x14);time.sleep(.2)
        user.ShowWindow(hwnd,6);time.sleep(.3);user.ShowWindow(hwnd,9);time.sleep(.3);capture(hwnd,out/'restored.png')
    finally:close(proc,hwnd)
    text=log.read_text(encoding='utf-8');assert proc.returncode==0 and renderer_clean(text,'dx11')
    assert text.count('dx11_swapchain: rebuild=')>=2
    report=dict(passed=True,renderer='dx11',default_without_renderer_override=True,vulkan_loader_not_loaded=True,loaded_modules=loaded,resize_minimize_restore=True,exe_sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest(),desktop_refresh_qualified=False)
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
