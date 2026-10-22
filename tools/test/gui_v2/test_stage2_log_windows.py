"""Native log windows, measured phase widths and an elastic Stage2 progress bar.

Uses only the ownership-aware synthetic worker; no ECM or Vulkan work.
"""
import argparse,ctypes as c,hashlib,json,re,subprocess,time,tomllib
from pathlib import Path
from win32_probe import window,close,capture,user

def main():
    p=argparse.ArgumentParser()
    for key in ['exe','worker','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--font-size',type=int,default=15);p.add_argument('--width',type=int,default=1600)
    a=p.parse_args();out=a.out.resolve();out.mkdir();exe=a.exe.resolve();worker=a.worker.resolve()
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    queue=out/'worktodo.txt';queue.write_text('[Worker #1]\nECMSTAGE2=1,2,137,-1,"ed_m137.save",0,0,10\n')
    cfg=out/'ecm.toml';cfg.write_text('[shared]\nworktodo='+json.dumps(str(queue))+'\nb2_ratio=5000\n[gui]\nNumWorkers=2\nexit_confirm="stop"\ngraceful_stop_ms=2000\nexe='+json.dumps(str(worker))+'\nstage2_exe='+json.dumps(str(worker))+'\n[workers.1]\nname="Phase fixture"\nautostart=1\nextra_args="--scenario v2-stage2-phases"\n[workers.2]\nname="Idle"\nautostart=0\n');original=cfg.read_bytes()
    state=out/'ui.toml';state.write_text('[ui]\nfont_size='+str(a.font_size)+'\npage=0\nworker=1\nlogs=true\ninspector=false\nscale=1\nlanguage="zh"\n[window]\ntitle_bar=false\n')
    log=out/'ui.log';si=subprocess.STARTUPINFO();si.dwFlags=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
    proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-nvml','--no-pet'],cwd=out,startupinfo=si)
    hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,40,40,a.width,1000,0x14);user.ShowWindow(hwnd,4)
    def sample():
        lines=log.read_text(encoding='utf-8').splitlines();rows=[s for s in lines if s.startswith('presentation_sample:')]
        return dict(re.findall(r'(\w+)=([^\s]+)',rows[-1])) if rows else {}
    def wait(predicate,seconds=12):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            s=sample()
            if s and predicate(s):return s
            assert proc.poll() is None;time.sleep(.08)
        raise AssertionError(log.read_text(encoding='utf-8')[-3500:])
    def click(pos):
        x,y=map(int,pos.split(','));assert x>0 and y>0
        for action in [0,1,2]:user.PostMessageW(hwnd,0x8021,action,(y<<16)|(x&65535));time.sleep(.15)
    try:
        first=wait(lambda s:s['log_worker']=='1' and s.get('stage2_phase')=='5' and '1' in s.get('timing_sources','') and float(s['stage2_fraction'])>0)
        assert first['log_nested']=='0' and int(first['logdock0'])>0 and first['logdock0']==first['logdock1']==first['logdock2'],first
        assert float(first['stage2_width'])>160,first
        assert float(first['table_scroll_y'])<.5 and float(first['table_scroll_x'])<.5 and float(first['table_data_height'])+.5>=float(first['table_required']),first
        later=wait(lambda s:float(s['stage2_fraction'])>float(first['stage2_fraction'])+.002)
        assert float(later['stage2_fraction'])<1
        summaries=[line for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('worker_summary: index=1 ')]
        actual=json.loads(summaries[-1].split(' text=',1)[1]);lines=actual.splitlines()
        assert len(lines)==3 and lines[0].startswith('任务：') and 'M137' in lines[0],lines
        assert 'Edwards' in lines[1] and 'param=' not in lines[1] and 'B1=260000000' in lines[1] and 'B2=5000*B1=1300000000000' in lines[1] and 'sigma=18' in lines[1],lines
        assert lines[2].startswith('阶段：') and 'B1=' not in lines[2] and '曲线' not in lines[2],lines
        capture(hwnd,out/'stage2-measured-segments.png')
        user.SetWindowPos(hwnd,None,40,40,1850,1000,0x14)
        wide=wait(lambda s:float(s['stage2_width'])>float(first['stage2_width'])+120)
        capture(hwnd,out/'stage2-stretched.png')
        click(wide['log0']);manager=wait(lambda s:s['log_worker']=='0');capture(hwnd,out/'manager-tab.png')
        click(manager['log2']);idle=wait(lambda s:s['log_worker']=='2');click(idle['raw_button']);wait(lambda s:s['log_worker']=='2' and s['raw']=='1')
        click(sample()['log1']);wait(lambda s:s['log_worker']=='1' and s['raw']=='0');capture(hwnd,out/'separate-worker-settings.png')
        cache=Path(str(state)+'.stage2-timing.json');doc=json.loads(cache.read_text(encoding='utf-8'));assert doc['schema']==1 and doc['records']
        assert any(v[2]>0 and v[5]>0 for v in doc['records'].values())
    finally:close(proc,hwnd)
    assert proc.returncode==0 and cfg.read_bytes()==original
    saved=tomllib.loads(state.read_text(encoding='utf-8'));assert saved['ui']['stage2_timing_source']=='auto'
    result=dict(passed=True,native_log_windows=3,previous_run_sources=True,realtime_advance=True,elastic_width=[float(first['stage2_width']),float(wide['stage2_width'])],per_worker_raw=True,three_summary_lines=True,ratio_expanded=True,business_unchanged=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
