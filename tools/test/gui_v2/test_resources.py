"""Warm pet resources, then repeat enable/disable and check process ownership."""
import argparse,ctypes as c,json,os,subprocess,time
from pathlib import Path
from win32_probe import window,close,process_sample,user
def main():
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--layers',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir();log=out/'resources.log';g=json.loads(Path(__file__).with_name('performance_gates.json').read_text())
    env=os.environ.copy();env['VK_LAYER_PATH']=str(a.layers.resolve());startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    proc=subprocess.Popen([str(a.exe.resolve()),'--ui-state',str(out/'ui.toml'),'--diagnostics',str(log),'--no-nvml','--validation','--audit-performance'],cwd=out,env=env,startupinfo=startup);hwnd=window(proc.pid,'ecm_gui_v2');user.ShowWindow(hwnd,4)
    user.GetGuiResources.argtypes=[c.c_void_p,c.c_ulong];user.GetGuiResources.restype=c.c_ulong
    def toggle(on):user.PostMessageW(hwnd,0x8022,9,on);time.sleep(.24)
    try:
        time.sleep(1)
        for _ in range(5):toggle(1);toggle(0)
        before=process_sample(proc);gdi_before=user.GetGuiResources(proc._handle,0)
        for _ in range(20):toggle(1);toggle(0)
        after=process_sample(proc);gdi_after=user.GetGuiResources(proc._handle,0)
        assert gdi_after-gdi_before<=g['toggle_gdi_handle_growth_max']
        assert after['private_bytes']-before['private_bytes']<=g['toggle_private_bytes_growth_max']
    finally:close(proc,hwnd)
    text=log.read_text(encoding='utf-8');assert 'validation_summary: errors=0 warnings=0' in text and 'memory_audit_end: bytes=0 allocations=0' in text
    result=dict(passed=True,warm_cycles=5,measured_cycles=20,gdi_before=gdi_before,gdi_after=gdi_after,private_bytes_growth=after['private_bytes']-before['private_bytes'],explicit_vulkan_released=True)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
