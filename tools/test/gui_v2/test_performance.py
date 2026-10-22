"""Five independent paired runs; gate file is frozen before collecting samples."""
import argparse,ctypes as c,hashlib,json,os,platform,re,shutil,statistics,subprocess,time
from pathlib import Path
from win32_probe import window,close,process_sample,user,kernel
ROOT=Path(__file__).resolve().parents[3]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def percentile(values,q=.95):
    values=sorted(values);return values[min(len(values)-1,int(q*len(values)))] if values else None
def main():
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--worker',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir();gates_path=Path(__file__).with_name('performance_gates.json');g=json.loads(gates_path.read_text());shutil.copy2(gates_path,out/'frozen-gates.json')
    exe=a.exe.resolve();old=a.old.resolve();worker=a.worker.resolve();shutil.copytree(ROOT/'src/ecm/application/gui/localization',out/'localization')
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    kernel.GetTickCount64.restype=c.c_ulonglong
    env=os.environ.copy();env.pop('VK_LAYER_PATH',None);env.pop('VULKAN_SDK',None)
    startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    raw=[];conditions=['idle','pet-static','pet-animate','floating','minimized','logs-workers','long-results']
    record=dict(gates_sha256=digest(gates_path),exe_sha256=digest(exe),old_sha256=digest(old),worker_sha256=digest(worker),system=platform.platform(),cpu=platform.processor(),tool='GetProcessTimes + GetProcessMemoryInfo + QPC CPU/present submit audit',sample_boundary='2s warmup; 3s elapsed CPU delta; memory end sample; foreground hidden/nonactivating except minimized; same 1280x850 physical size, 144 DPI, English, opaque themes; NVML 500ms',limits=g['limits'],samples=raw)
    record['power_scheme']=subprocess.run(['powercfg','/getactivescheme'],capture_output=True).stdout.decode(errors='replace').strip()
    record['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,power.limit','--format=csv,noheader'],capture_output=True).stdout.decode(errors='replace').strip()
    for condition in conditions:
        for iteration in range(g['runs_per_condition']):
            for newer in [False,True] if iteration%2==0 else [True,False]:
                run=out/f'{condition}-{iteration}-{"v2" if newer else "v1"}';run.mkdir();cfg=run/'ecm.toml';log=run/'app.log';count=4 if condition=='logs-workers' else 1
                cfg.write_text(f'[shared]\nworktodo="worktodo.txt"\n[gui]\nNumWorkers={count}\nrefresh_hz=10\ngpu_poll_ms=500\nexit_confirm="stop"\ngraceful_stop_ms=1200\nexe={json.dumps(str(worker))}\n'+''.join(f'[workers.{i}]\nautostart={1 if condition=="logs-workers" else 0}\nextra_args="--scenario v2-performance --lines 4000"\n' for i in range(1,count+1)))
                (run/'worktodo.txt').write_text(''.join(f'[Worker #{i}]\nECMSTAGE1=1,2,127,-1,100,1\n' for i in range(1,count+1)))
                if condition=='long-results':
                    row={'status':'NF','exponent':127,'worktype':'ECM','b1':100,'b2':100,'Edwards':{},'curves':1,'program':{'name':'ECM-ELF','version':'fixture'},'timestamp':'2026-10-21 00:00:00','checksum':{'version':1,'checksum':'1234ABCD'}}
                    (run/'results.json.txt').write_text((json.dumps(row)+'\n')*10000)
                command=[str(exe if newer else old),'--config',str(cfg),'--language','en' if newer else 'english']
                if newer:
                    command+=['--ui-state',str(run/'ui.toml'),'--diagnostics',str(log),'--renderer','vulkan','--audit-performance']
                    if condition.startswith('pet-'):command+=['--pet']+(['--pet-static'] if condition=='pet-static' else [])
                    if condition=='floating':command+=['--multi-viewport','--detach-probe']
                proc=subprocess.Popen(command,cwd=run,env=env,startupinfo=startup,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);hwnd=window(proc.pid,'ecm_gui_v2' if newer else 'ecm_gui')
                try:
                    user.SetWindowPos(hwnd,None,80,350,*g['physical_window'],0x14);user.ShowWindow(hwnd,6 if condition=='minimized' else 4)
                    if newer and condition=='long-results':user.PostMessageW(hwnd,0x8022,7,1)
                    time.sleep(g['warmup_seconds']);begin=process_sample(proc);start=time.monotonic();responses=[]
                    frame_begin=len(re.findall(r'frame_sample: ms=',log.read_text(encoding='utf-8'))) if newer else 0
                    if newer and condition!='minimized':
                        for sequence in range(1,6):
                            sent=kernel.GetTickCount64();user.PostMessageW(hwnd,0x8022,10,sequence);end=time.monotonic()+2
                            while time.monotonic()<end:
                                match=re.search(rf'response_sample: seq={sequence} ticks=(\d+)',log.read_text(encoding='utf-8'))
                                if match:responses.append(int(match[1])-sent);break
                                time.sleep(.005)
                            else:raise AssertionError('Unresponsive UI')
                    while time.monotonic()-start<g['sample_seconds']:time.sleep(.05)
                    finish=process_sample(proc);duration=time.monotonic()-start
                    sample=dict(condition=condition,iteration=iteration,version='v2' if newer else 'v1',duration_seconds=duration,one_core_cpu_percent=100*(finish['cpu_seconds']-begin['cpu_seconds'])/duration,private_bytes=finish['private_bytes'],working_set_bytes=finish['working_set_bytes'],dpi=user.GetDpiForWindow(hwnd),responses_to_submit_ms=responses)
                    if newer:
                        lines=log.read_text(encoding='utf-8');frames=[float(v) for v in re.findall(r'frame_sample: ms=([\d.]+)',lines)][frame_begin:]
                        sample.update(frame_submit_p95_ms=percentile(frames),explicit_vulkan_bytes=max(map(int,re.findall(r'gpu_bytes=(\d+)',lines)),default=0),frame_samples=len(frames))
                    raw.append(sample);(out/'raw.json').write_text(json.dumps(record,indent=2))
                finally:close(proc,hwnd)
                if newer:assert 'memory_audit_end: bytes=0 allocations=0' in log.read_text(encoding='utf-8')
            print(f'{condition} pair {iteration+1}/5',flush=True)
    summary={};failures=[]
    for condition in conditions:
        summary[condition]={}
        for version in ['v1','v2']:
            values=[v for v in raw if v['condition']==condition and v['version']==version]
            summary[condition][version]={k:statistics.median(v[k] for v in values) for k in ['one_core_cpu_percent','private_bytes','working_set_bytes']}
        one,two=summary[condition]['v1'],summary[condition]['v2']
        if two['one_core_cpu_percent']-one['one_core_cpu_percent']>g['one_core_cpu_increase_percentage_points_max']:failures.append(condition+': CPU delta')
        if two['private_bytes']-one['private_bytes']>g['private_bytes_increase_max'] or two['private_bytes']>g['private_bytes_absolute_max']:failures.append(condition+': private memory')
        if condition=='minimized' and two['one_core_cpu_percent']>g['minimized_one_core_cpu_percent_max']:failures.append('minimized CPU')
    for condition in ['pet-static','pet-animate']:
        idle=summary['idle']['v2'];pet=summary[condition]['v2']
        if pet['one_core_cpu_percent']-idle['one_core_cpu_percent']>g['pet_cpu_increase_percentage_points_max']:failures.append(condition+': pet CPU')
        if pet['private_bytes']-idle['private_bytes']>g['pet_private_bytes_increase_max']:failures.append(condition+': pet memory')
    frames=[v['frame_submit_p95_ms'] for v in raw if v['version']=='v2' and v.get('frame_submit_p95_ms') is not None]
    responses=[ms for v in raw for ms in v['responses_to_submit_ms']]
    if max(frames)>g['v2_frame_cpu_and_submit_p95_ms_max']:failures.append('frame submit P95')
    if percentile(responses)>g['v2_message_to_submit_p95_ms_max']:failures.append('message-to-submit P95')
    record.update(passed=not failures,summary=summary,failures=failures,response_to_submit_p95_ms=percentile(responses),max_condition_frame_p95_ms=max(frames),unqualified=['Worker throughput, total VRAM, physical input latency, controlled power and independent clean machine are not measured by this test'])
    (out/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:record[k] for k in ['passed','summary','failures','response_to_submit_p95_ms','max_condition_frame_p95_ms']}));assert not failures,failures
if __name__=='__main__':main()
