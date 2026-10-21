"""Five paired isolated tiny Stage1 jobs under both frontends; no user queue."""
import argparse,hashlib,json,re,statistics,subprocess,time
from pathlib import Path
from win32_probe import window,close,user
def main():
    p=argparse.ArgumentParser()
    for name in ['exe','old','worker','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir();exe=a.exe.resolve();old=a.old.resolve();worker=a.worker.resolve();samples=[]
    startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    for iteration in range(5):
        for newer in ([False,True] if iteration%2==0 else [True,False]):
            run=out/f'{iteration}-{"v2" if newer else "v1"}';run.mkdir();(run/'saves').mkdir();cfg=run/'ecm.toml';queue=run/'worktodo.txt';log=run/'screen.log'
            queue.write_text('[Worker #1]\nECMSTAGE1=1,2,991,-1,'+json.dumps(str(run/'saves/m0_991_1e4.save'))+',0,0,8\n')
            cfg.write_text('[shared]\nworktodo='+json.dumps(str(queue))+'\n[stage1]\ncurve="mont"\ntarget="gpu"\ngpu_param=0\nsigma=10\nauto_stage2=0\nckpt_seconds=0\ntmp_dir='+json.dumps(str(run/'saves'))+'\nlog_file='+json.dumps(str(log))+'\nfinished='+json.dumps(str(run/'finished.txt'))+'\nstage1_submission_file='+json.dumps(str(run/'results.json.txt'))+'\n[gui]\nNumWorkers=1\nrefresh_hz=10\ngpu_poll_ms=500\nexit_confirm="stop"\nexe='+json.dumps(str(worker))+'\n[workers.1]\nautostart=1\n')
            cmd=[str(exe if newer else old),'--config',str(cfg)]
            if newer:cmd+=['--ui-state',str(run/'ui.toml'),'--diagnostics',str(run/'gui.log')]
            proc=subprocess.Popen(cmd,cwd=run,startupinfo=startup,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);hwnd=window(proc.pid,'ecm_gui_v2' if newer else 'ecm_gui');user.ShowWindow(hwnd,4);start=time.monotonic()
            try:
                end=time.monotonic()+60
                while time.monotonic()<end:
                    text=log.read_text(encoding='utf-8',errors='replace') if log.exists() else ''
                    if 'queue done, 1 task' in text:break
                    assert proc.poll() is None;time.sleep(.05)
                else:raise AssertionError(text[-3000:])
                assert 'ECMSTAGE1=' not in queue.read_text();times=re.findall(r'gputime=([\d.]+) ms',text);assert times,text[-3000:]
                gpu_ms=sum(map(float,times));assert gpu_ms>0
                hits=sorted(set(re.findall(r'factor\[\d+\]=(\d+) curve=(\d+) sigma=(\d+) param=(\d+)',text)));assert hits
                if samples:assert hits==samples[0]['hits'],'Fixed curve/param/sigma changed factors'
                sample=dict(iteration=iteration,version='v2' if newer else 'v1',worker_reported_gpu_ms=gpu_ms,wall_after_hwnd_seconds=time.monotonic()-start,task='M991 B1=10000 B2=0 curves=8 mont/PARAM0 sigma=10',hits=hits,worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest());samples.append(sample);(out/'raw.json').write_text(json.dumps(samples,indent=2))
            finally:close(proc,hwnd)
    first=statistics.median(v['worker_reported_gpu_ms'] for v in samples if v['version']=='v1');second=statistics.median(v['worker_reported_gpu_ms'] for v in samples if v['version']=='v2');ratio=first/second;g=json.loads(Path(__file__).with_name('performance_gates.json').read_text())
    report=dict(passed=ratio>=g['worker_throughput_ratio_min'],samples=samples,median_v1_gpu_ms=first,median_v2_gpu_ms=second,relative_throughput=ratio,limits=['Tiny isolated task; existing external GPU load/power policy not controlled','GPU event interval reported by worker; excludes GUI startup and not a full end-to-end ECM benchmark'])
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));assert report['passed']
if __name__=='__main__':main()
