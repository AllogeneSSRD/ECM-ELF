"""Click real ImGui controls through the explicit app input seam; native wheel."""
import argparse,ctypes as c,json,os,re,subprocess,time
from pathlib import Path
from win32_probe import window,close,user,capture
def main():
    p=argparse.ArgumentParser()
    for name in ['exe','worker','out','layers']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir();log=out/'interaction.log';cfg=out/'ecm.toml';worker=a.worker.resolve()
    cfg.write_text('[gui]\nNumWorkers=1\ngraceful_stop_ms=1200\nexit_confirm="stop"\nexe='+json.dumps(str(worker))+'\n[workers.1]\nautostart=0\nextra_args="--scenario v2-checkpoint-stop"\n');(out/'worktodo.txt').write_text('[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n')
    row={'status':'NF','exponent':127,'worktype':'ECM','b1':100,'b2':100,'Edwards':{},'curves':1,'program':{'name':'ECM-ELF','version':'fixture'},'timestamp':'2026-10-21 00:00:00','checksum':{'version':1,'checksum':'1234ABCD'}}
    records=[json.dumps(row) for _ in range(299)];row['status']='F';row['stage']=1;row['factors']=['233'];records.append(json.dumps(row));(out/'results.json.txt').write_text('\n'.join(records)+'\n')
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    env=os.environ.copy();env['VK_LAYER_PATH']=str(a.layers.resolve());startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    proc=subprocess.Popen([str(a.exe.resolve()),'--config',str(cfg),'--ui-state',str(out/'ui.toml'),'--diagnostics',str(log),'--audit-app','--validation','--no-nvml'],cwd=out,env=env,startupinfo=startup);hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,80,200,1500,1000,0x14);user.ShowWindow(hwnd,4)
    def snapshots():
        return [dict(re.findall(r'(\w+)=([-\d.,]+)',line)) for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('interaction_sample:')]
    def wait(predicate):
        end=time.monotonic()+10
        while time.monotonic()<end:
            values=snapshots()
            if values and predicate(values[-1]):return values[-1]
            assert proc.poll() is None;time.sleep(.035)
        raise AssertionError(log.read_text(encoding='utf-8')[-2000:])
    def move(value):
        x,y=map(int,value.split(','));assert x>=0 and y>=0;user.PostMessageW(hwnd,0x8021,0,(y<<16)|(x&65535));return (y<<16)|(x&65535)
    def click(value):
        packed=move(value);time.sleep(.15);user.PostMessageW(hwnd,0x8021,1,packed);time.sleep(.15);user.PostMessageW(hwnd,0x8021,2,packed);time.sleep(.2)
    try:
        user.PostMessageW(hwnd,0x8022,7,1);s=wait(lambda v:v['page']=='1' and v['visible']=='300')
        click(s['f']);s=wait(lambda v:v['visible']=='1');click(s['nf']);s=wait(lambda v:v['visible']=='299');capture(hwnd,out/'filter-chips.png')
        move(s['wheel']);time.sleep(.2);offset=len(snapshots());user.PostMessageW(hwnd,0x20a,((-120)&65535)<<16,0);time.sleep(.8)
        scroll=[float(v['scroll']) for v in snapshots()[offset:]];positive=sorted(set(v for v in scroll if v>0));assert len(positive)>=3,scroll
        user.PostMessageW(hwnd,0x8022,7,0);s=wait(lambda v:v['page']=='0' and int(v['start'].split(',')[0])>0);click(s['start'])
        end=time.monotonic()+5
        while 'started pid=' not in log.read_text(encoding='utf-8') and time.monotonic()<end:time.sleep(.04)
        assert 'started pid=' in log.read_text(encoding='utf-8');s=wait(lambda v:int(v['toasts'])>0);capture(hwnd,out/'start-toast.png');click(s['stop'])
        end=time.monotonic()+5
        while 'stop requested (waiting for a checkpoint' not in log.read_text(encoding='utf-8') and time.monotonic()<end:time.sleep(.04)
        assert 'stop requested (waiting for a checkpoint' in log.read_text(encoding='utf-8');capture(hwnd,out/'stop-toast.png')
    finally:close(proc,hwnd)
    assert 'validation_summary: errors=0 warnings=0' in log.read_text(encoding='utf-8')
    result=dict(passed=True,filter_chip_F_NF=True,smooth_native_wheel_positions=positive,clicked_start_stop=True,toast_visible=True,limits=['ImGui pointer injection is explicitly test-only; native WM_MOUSEWHEEL is real host handling; no physical IME/desktop mouse qualification'])
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
