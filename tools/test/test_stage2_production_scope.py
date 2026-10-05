"""Observe production D scope decisions, then stop each direct internal worker.

This is a selector regression gate, not a completed-curve arithmetic test. Direct
workers create no child process; no save, ini or work queue is changed.
"""
import argparse,hashlib,json,os,re,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--save',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    p.add_argument('--baby-device',type=int,choices=(0,1),default=0)
    p.add_argument('--shift-scale',type=int,choices=(0,1))
    p.add_argument('--ptx-reduce',type=int,choices=(0,1))
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();save=a.save.resolve();sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    save_bytes=save.read_bytes();save_sha=hashlib.sha256(save_bytes).hexdigest()
    line=save_bytes.split(b'\n',1)[0];fingerprint=14695981039346656037
    for byte in line:fingerprint=((fingerprint^byte)*1099511628211)&((1<<64)-1)
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_ARENA_CAP_KB='6451200',NTT_NO_PROGRESS='1',NTT_BABY_DEVICE=str(a.baby_device))
    if a.shift_scale is not None:env['NTT_GL_SHIFT_SCALE']=str(a.shift_scale)
    if a.ptx_reduce is not None:env['NTT_GL_PTX_REDUCE']=str(a.ptx_reduce)
    rows=[]
    for flag in (1,0):
        overrides=(None,'NTT_S4_OLDTAIL','NTT_S4_OFF')
        if a.baby_device:overrides+=('NTT_BABY_DEVICE_ALLOC_FAIL','NTT_BABY_DEVICE_CHECK',
                                   'NTT_BABY_DEVICE_TEST','NTT_BABY_DEVICE_TEST_BAD','NTT_BABY_DEVICE_MAX_MB')
        for override in overrides:
            name=f'{flag}_{override or "default"}';log=out/(name+'.log');result=out/(name+'.jsonl')
            ee=env|{'NTT_GL_SHORT_REDUCE':str(flag)}
            if override:ee[override]='0' if override=='NTT_BABY_DEVICE_MAX_MB' else '1'
            enabled=int(override is None and (not a.baby_device or flag==1) and not a.shift_scale and not a.ptx_reduce)
            version=('resident_baby_v1' if a.baby_device else 'resident_short_v1' if flag else 'resident_shape_v1') if enabled else 'legacy_56_1'
            cmd=[str(exe),'--curve-worker','--save',str(save),'--record-offset','0',
                 '--record-hash',str(fingerprint),'--record-index','1','--b2','100000000000',
                 '--d','330330','--device',str(a.device),'--results',str(result)]
            assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha
            observed=None;started=time.monotonic()
            with log.open('wb') as output:
                child=subprocess.Popen(cmd,env=ee,stdout=output,stderr=subprocess.STDOUT)
                try:
                    while time.monotonic()-started<30:
                        text=log.read_text(encoding='utf-8',errors='replace')
                        observed=re.search(r'd_model: requested=(\d+) enabled=(\d+) version=(\S+).*gl_short=(\d+)',text)
                        if observed or child.poll() is not None:break
                        time.sleep(.05)
                finally:
                    if child.poll() is None:child.terminate()
                    child.wait(timeout=10)
            text=log.read_text(encoding='utf-8',errors='replace')
            checks={'decision':bool(observed) and observed.group(1,2,3)==('1',str(enabled),version),
                    'reducer':bool(observed) and observed[4]==str(flag),
                    'no_completed_curve':'stage2_full_wall:' not in text,
                    'no_result':not result.exists(),
                    'save_unchanged':hashlib.sha256(save.read_bytes()).hexdigest()==save_sha}
            if a.shift_scale is not None:
                checks['scale_control']=f'gl_shift_scale={a.shift_scale}' in text
            if a.ptx_reduce is not None:
                checks['ptx_control']=f'gl_ptx={a.ptx_reduce}' in text
            rows.append(dict(name=name,command=cmd,env={k:v for k,v in ee.items() if k.startswith('NTT_')},
                             checks=checks,controlled_stop=True,exit=child.returncode))
            print(name,checks,flush=True)
    assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha
    data=dict(exe=str(exe),sha256=sha,save_sha256=save_sha,device=a.device,
              scope='Observe selector and stop direct workers; curves not completed',runs=rows,
              passed=sum(sum(r['checks'].values()) for r in rows),
              failed=sum(sum(not v for v in r['checks'].values()) for r in rows))
    (out/'summary.json').write_text(json.dumps(data,indent=2))
    return int(data['failed']!=0)


if __name__=='__main__':raise SystemExit(main())
