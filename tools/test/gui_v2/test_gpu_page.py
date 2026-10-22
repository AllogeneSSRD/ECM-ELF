"""DX11 GPU page/settings/hover regression with real read-only driver metadata.

Uses isolated UI/business files. Never starts compute workers or creates a CUDA
context; CUDA device queries only cross-check physical UUIDs and SM counts.
"""
import argparse,csv,ctypes as c,hashlib,json,re,subprocess,time,tomllib,uuid
from pathlib import Path
from win32_probe import window,close,capture,user
from renderer_probe import renderer_clean

def reference():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,compute_cap,uuid','--format=csv,noheader']).decode()
    rows={int(row[0]):dict(name=row[1].strip(),driver=row[2].strip(),capability=row[3].strip(),uuid=row[4].strip()) for row in csv.reader(raw.splitlines())}
    cuda=c.WinDLL('nvcuda.dll');assert cuda.cuInit(0)==0
    count=c.c_int();assert cuda.cuDeviceGetCount(c.byref(count))==0
    sms={}
    for i in range(count.value):
        dev=c.c_int();buf=(c.c_ubyte*16)();n=c.c_int()
        assert cuda.cuDeviceGet(c.byref(dev),i)==0 and cuda.cuDeviceGetUuid(buf,dev)==0
        assert cuda.cuDeviceGetAttribute(c.byref(n),16,dev)==0
        sms['GPU-'+str(uuid.UUID(bytes=bytes(buf)))]=n.value
    nvml=c.CDLL('nvml.dll');assert nvml.nvmlInit_v2()==0
    try:
        version=c.c_int();assert nvml.nvmlSystemGetCudaDriverVersion_v2(c.byref(version))==0
        nvml.nvmlDeviceGetNumGpuCores.argtypes=[c.c_void_p,c.POINTER(c.c_uint)]
        for index,row in rows.items():
            handle=c.c_void_p();cores=c.c_uint()
            assert nvml.nvmlDeviceGetHandleByIndex_v2(index,c.byref(handle))==0
            assert nvml.nvmlDeviceGetNumGpuCores(handle,c.byref(cores))==0
            row.update(sms=sms[row['uuid']],cores=cores.value,cuda=version.value)
    finally:nvml.nvmlShutdown()
    return rows

def main():
    p=argparse.ArgumentParser()
    for key in ['exe','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();exe=a.exe.resolve();out=a.out.resolve();out.mkdir();expected=reference();results=[]
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p
    assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    def records(text,prefix):return [dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in text.splitlines() if line.startswith(prefix+':')]
    for language,seconds in [('zh',15),('en',120)]:
        folder=out/language;folder.mkdir();state=folder/'ui.toml';log=folder/'ui.log';cfg=folder/'ecm.toml'
        cfg.write_text('[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n');before=cfg.read_bytes()
        state.write_text(f'[ui]\npage=4\nlanguage="{language}"\nscale=1\nlogs=false\ninspector=false\n')
        startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-pet'],cwd=folder,startupinfo=startup)
        hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,40,40,1920,1400,0x14);user.ShowWindow(hwnd,4)
        def wait(predicate,timeout=15):
            end=time.monotonic()+timeout
            while time.monotonic()<end:
                text=log.read_text(encoding='utf-8');result=predicate(text)
                if result:return result
                assert proc.poll() is None,text[-2000:];time.sleep(.06)
            raise AssertionError(log.read_text(encoding='utf-8')[-4000:])
        def point(value,click=False):
            x,y=map(int,value.split(','));assert 0<x<1920 and 0<y<1400,value
            for action in ([0,1,2] if click else [0]):user.PostMessageW(hwnd,0x8021,action,(y<<16)|(x&65535));time.sleep(.15)
        def devices(text):
            latest={int(row['index']):row for row in records(text,'gpu_detail')}
            return latest if len(latest)==len(expected) and all(row['ready']=='1' and row['charts']=='3' for row in latest.values()) else None
        try:
            actual=wait(devices)
            for index,want in expected.items():
                got=actual[index];assert got['driver']==want['driver'],(got,want)
                assert f"{got['major']}.{got['minor']}"==want['capability'],(got,want)
                for field in ['sms','cores','cuda']:assert int(got[field])==want[field],(field,got,want)
            first=min(expected);point(actual[first]['header'])
            wait(lambda text:any(row['index']==str(first) and row['hovered']=='1' for row in records(text,'gpu_detail')[-len(expected):]))
            capture(hwnd,folder/'gpu-hover.png')
            text=log.read_text(encoding='utf-8');lines=text.splitlines()
            tooltip=[line.split('tooltip=',1)[1] for line in lines if line.startswith(f'gpu_text: index={first} tooltip=')][-1]
            assert tooltip.startswith(expected[first]['name']+'\t') and tooltip.endswith('NVML / 500 ms'),tooltip
            assert expected[first]['driver'] in tooltip and 'SM ' in tooltip and str(expected[first]['cores']) in tooltip,tooltip
            corners=[line for line in lines if line.startswith(f'gpu_text: index={first} usage=')][-1]
            assert all(word not in corners for word in ['上限','max','limit']),corners
            assert ' / 100%' in corners and ' | ' in corners and ' MHz  |  ' in corners,corners
            basic=[line.split('basic=',1)[1] for line in lines if line.startswith(f'gpu_text: index={first} basic=')][-1]
            assert not re.search(r'\d/|/\d',basic) and basic.count(' | ')==4,basic
            nav=records(text,'usability_sample')[-1];point(nav['nav5'],True)
            wait(lambda text:records(text,'usability_sample')[-1]['page']=='5')
            settings=wait(lambda text: records(text,'gpu_settings')[-1] if int(records(text,'gpu_settings')[-1]['range0'].split(',')[0])>0 else None)
            assert settings['seconds']=='60',settings
            point(settings['range0' if seconds==15 else 'range3'],True)
            wait(lambda text:records(text,'gpu_settings')[-1]['seconds']==str(seconds))
            capture(hwnd,folder/'settings.png')
            point(records(log.read_text(encoding='utf-8'),'usability_sample')[-1]['nav4'],True)
            wait(lambda text:records(text,'usability_sample')[-1]['page']=='4' and devices(text))
            capture(hwnd,folder/'gpu.png')
        except BaseException:
            capture(hwnd,folder/'failure.png');raise
        finally:close(proc,hwnd)
        assert renderer_clean(log.read_text(encoding='utf-8'),'dx11') and cfg.read_bytes()==before
        saved=tomllib.loads(state.read_text(encoding='utf-8'));assert saved['ui']['gpu_history_seconds']==seconds
        restart=folder/'restart.log'
        run=subprocess.run([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(restart),'--audit-app','--frames','12','--no-pet'],cwd=folder,startupinfo=startup,timeout=25)
        assert run.returncode==0 and records(restart.read_text(encoding='utf-8'),'gpu_settings')[-1]['seconds']==str(seconds)
        results.append(dict(language=language,devices=len(actual),three_charts_each=True,metadata_matches_driver=True,tooltip=True,spacing=True,time_range=seconds,persisted=True))
    result=dict(passed=True,cases=results,reference=expected,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),business_unchanged=True,scope='DX11, actual driver metadata only; no compute workers, CUDA contexts or Vulkan rendering')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
