"""Native DX11 GPU clock popover: real read-only queries, never Apply/Reset.

All saves use isolated TOML files. Hardware writes are tested by the fake NVML
DLL in gpu_clock_test.cpp instead of changing a display adapter's clock policy.
"""
import argparse,ctypes as c,hashlib,json,re,subprocess,time,tomllib
from pathlib import Path
from win32_probe import window,close,capture,user
from renderer_probe import renderer_clean

def main():
    p=argparse.ArgumentParser()
    for key in ['exe','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();exe=a.exe.resolve();out=a.out.resolve();out.mkdir();results=[]
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    with (out/'supported-clocks.txt').open('w',encoding='utf-8') as f:
        subprocess.run(['nvidia-smi','-q','-d','SUPPORTED_CLOCKS'],stdout=f,stderr=subprocess.STDOUT,check=True)
    for language,font in [('en',15),('zh',15),('zh',24)]:
        folder=out/f'{language}{font}';folder.mkdir();cfg=folder/'ecm.toml';state=folder/'ui.toml';log=folder/'ui.log'
        cfg.write_text('# clock test\n[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n')
        state.write_text(f'[ui]\npage=4\nlogs=false\ninspector=false\nscale=1\nfont_size={font}\nlanguage="{language}"\n')
        startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-pet'],cwd=folder,startupinfo=startup)
        hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,30,30,1920,1400,0x14);user.ShowWindow(hwnd,4)
        def records():
            text=log.read_text(encoding='utf-8');cut=text.rfind('\ninteraction_sample:')
            text=text[:cut] if cut>=0 else ''
            rows={}
            for line in text.splitlines():
                if line.startswith('gpu_clock:'):
                    row=dict(re.findall(r'(\w+)=([^\s]+)',line));rows[int(row['index'])]=row
            return rows
        def wait(predicate):
            end=time.monotonic()+15
            while time.monotonic()<end:
                rows=records()
                if predicate(rows):return rows
                assert proc.poll() is None;time.sleep(.06)
            raise AssertionError(records())
        def pointer(pos,action):
            x,y=map(round,pos);text=log.read_text(encoding='utf-8');before=text.count('\ninteraction_sample:')
            user.PostMessageW(hwnd,0x8021,action,(y<<16)|(x&65535))
            end=time.monotonic()+8
            while time.monotonic()<end:
                if log.read_text(encoding='utf-8').count('\ninteraction_sample:')>=before+2:return
                assert proc.poll() is None;time.sleep(.035)
            raise AssertionError('No rendered audit frame')
        def xy(value):return tuple(map(float,value.split(',')))
        def ready(r):
            if r['open']!='1' or r['busy']!='0' or r['range']=='0,0':return False
            rect=list(map(float,r['rect'].split(',')))
            return all(rect[0]<=xy(r[key])[0]<=rect[2] and rect[1]<=xy(r[key])[1]<=rect[3] for key in ['apply','save','reset','refresh','min','max'])
        def click(pos):
            for action in [0,1,2]:pointer(pos,action)
        try:
            rows=wait(lambda r:len(r)>=1 and all(xy(v['gear'])[0]>0 for v in r.values()))
            devices={};driver_retries=[]
            for index in rows:
                click(xy(records()[index]['gear']))
                settled=wait(lambda r:r[index]['open']=='1' and r[index]['busy']=='0' and r[index]['completed']!='0')[index]
                if settled['error']!='0':
                    assert settled['error']=='999',settled
                    capture(hwnd,folder/f'gpu{index}-driver-error.png');driver_retries.append(settled)
                    comparison=subprocess.run(['nvidia-smi','-i',settled['uuid'],'--query-supported-clocks=graphics','--format=csv,noheader,nounits'],capture_output=True,text=True)
                    (folder/f'gpu{index}-driver-error-smi.txt').write_text(comparison.stdout+comparison.stderr,encoding='utf-8')
                    completed=int(settled['completed']);click(xy(settled['refresh']))
                    wait(lambda r:int(r[index]['completed'])>completed and r[index]['busy']=='0')
                r=wait(lambda r:ready(r[index]) and r[index]['completed']!='0')[index]
                assert r['error']=='0',r
                lo,hi=map(int,r['range'].split(','));assert 0<lo<=hi<100000,r
                query=subprocess.check_output(['nvidia-smi','-i',r['uuid'],'--query-supported-clocks=graphics','--format=csv,noheader,nounits']).decode()
                (folder/f'gpu{index}-supported.csv').write_text(query,encoding='utf-8')
                clocks=[int(v.strip()) for v in query.splitlines() if v.strip().isdigit()]
                if clocks:assert (lo,hi)==(min(clocks),max(clocks)),(r,clocks)
                devices[index]=dict(uuid=r['uuid'],low=lo,high=hi,smi_compared=bool(clocks))
                capture(hwnd,folder/f'gpu{index}-popover.png')
                click((1860,1320));wait(lambda r:r[index]['open']=='0')
            # Reopen the first adapter and move the min handle. Dismissal keeps
            # edits, reopening preserves them, and Save only writes the draft.
            first=min(devices);click(xy(records()[first]['gear']))
            r=wait(lambda r:ready(r[first]))[first]
            low,high=xy(r['low']),xy(r['high']);pointer(low,0);pointer(low,1)
            pointer(((low[0]+high[0])*.5,low[1]),0);pointer(((low[0]+high[0])*.5,low[1]),2)
            edited=records()[first]['draft'];assert edited!=r['draft'],(r,records()[first])
            click((1860,1320));wait(lambda r:r[first]['open']=='0')
            click(xy(records()[first]['gear']));r=wait(lambda r:ready(r[first]))[first];assert r['draft']==edited
            click(xy(r['save']));time.sleep(.15)
            saved=tomllib.loads(cfg.read_text(encoding='utf-8'));low_mhz,high_mhz=edited.split(',')
            assert f"{r['uuid']}:{low_mhz}-{high_mhz}" in saved['gui']['gpu_clock_ranges']
            assert saved['gui']['NumWorkers']==1 and saved['workers']['1']['autostart']==0
            capture(hwnd,folder/'saved.png')
            click((1860,1320));wait(lambda r:r[first]['open']=='0')
            user.SetWindowPos(hwnd,None,30,30,900,850,0x14);time.sleep(.5)
            click(xy(records()[first]['gear']));r=wait(lambda r:ready(r[first]))[first]
            rect=list(map(float,r['rect'].split(',')));assert 0<=rect[0]<rect[2]<=900 and 0<=rect[1]<rect[3]<=850,rect
            for key in ['apply','save','reset','refresh','min','max']:
                x,y=xy(r[key]);assert rect[0]<=x<=rect[2] and rect[1]<=y<=rect[3],(key,r)
            capture(hwnd,folder/'narrow.png')
            results.append(dict(language=language,font=font,devices=devices,draft=edited,narrow_rect=rect,driver_retries=driver_retries))
        except BaseException:
            capture(hwnd,folder/'failure.png');raise
        finally:close(proc,hwnd)
        assert renderer_clean(log.read_text(encoding='utf-8'),'dx11')
        sources=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('gpu_clock_source:')]
        results[-1]['query_fallback_devices']=sorted({int(row['index']) for row in sources if row['smi']=='1'})
    result=dict(passed=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),hardware_commands_executed=False,cases=results)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
