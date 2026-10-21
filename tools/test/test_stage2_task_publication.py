"""GPU1: interrupted NF stays private; completing the task seals one batch."""
import argparse
import ctypes
from ctypes import wintypes
import json
import hashlib
from pathlib import Path
import subprocess
import time
from test_edwards_gpu_pipeline import stage1, checksum
from test_result_output_consistency import crc_valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    saved=out/'points.save';n=(1<<347)-1;lines=[]
    for sigma in range(17,21):
        _,x,z=stage1(n,sigma,20);x=x*pow(z,-1,n)%n
        lines.append(f'METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM; SIGMA={sigma}; B1=20; N=0x{n:x}; X=0x{x:x}; STATUS=POINT; CHECKSUM={checksum(n,x,sigma,20,"POINT")};\n')
    saved.write_text(''.join(lines),encoding='utf-8')
    queue=out/'worktodo.txt';task='ECMSTAGE2=1,2,347,-1,points.save,1000000000,0,4\n';queue.write_text(task)
    ini=out/'ecm.toml';ini.write_text('[shared]\nuser=""\ncomputer=""\n[stage1]\nverbose=true\n[stage2]\n'
        f'stage2_save_dir={json.dumps(str(out))}\nstage2_finished={json.dumps(str(out/"finished.txt"))}\n')
    public=out/'results.json.txt';inside=out/'internal.jsonl'
    binary_sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest()
    command=list(map(str,[a.exe.resolve(),'--config',ini,'--device',1,'--once','--worktodo',queue,
        '--results',inside,'--submission',public,'--short-calibration',0,'--tune-runtime',0,
        '--auto-carrier',0,'--d',30030,'--log-level',2,'--log',out/'engine.log']))
    class Security(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('descriptor',wintypes.LPVOID),('inherit',wintypes.BOOL)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateEventW.argtypes=[ctypes.POINTER(Security),wintypes.BOOL,wintypes.BOOL,wintypes.LPCWSTR]
    kernel.CreateEventW.restype=wintypes.HANDLE
    kernel.SetEvent.argtypes=[wintypes.HANDLE];kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    sa=Security(ctypes.sizeof(Security),None,True);event=kernel.CreateEventW(ctypes.byref(sa),True,False,None)
    assert event
    with (out/'stop.log').open('wb') as log:
        proc=subprocess.Popen(command+['--stage2-stop-handle',str(event)],cwd=out,stdout=log,stderr=log,
                              close_fds=False,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                assert proc.poll() is None,'task finished before stop probe'
                if 'Generate and normalize baby points' in (out/'stop.log').read_text(encoding='utf-8',errors='replace'):
                    break
                time.sleep(.005)
            else:raise AssertionError('no baby phase observed')
            assert kernel.SetEvent(event)
            assert proc.wait(timeout=120)==0
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            kernel.CloseHandle(event)
    rows=[json.loads(s) for s in inside.read_text(encoding='utf-8').splitlines()]
    assert 0<len(rows)<4 and queue.read_text()==task
    assert not public.exists(),'unfinished NF task was exported'
    count=len(rows)
    r=subprocess.run(command,cwd=out,capture_output=True,timeout=180)
    (out/'resume.log').write_bytes(r.stdout+r.stderr);assert not r.returncode
    rows=[json.loads(s) for s in inside.read_text(encoding='utf-8').splitlines()]
    submitted=[json.loads(s) for s in public.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==4 and not queue.read_text().strip()
    assert len(submitted)==1 and submitted[0]['status']=='NF' and submitted[0]['curves']==4
    assert all(map(crc_valid,rows+submitted))
    assert hashlib.sha256(a.exe.read_bytes()).hexdigest()==binary_sha256
    report=dict(complete=True,device=1,binary_sha256=binary_sha256,completed_before_stop=count,total=4,
        unfinished_nf_not_exported=True,resume_no_recalculation=True,final_nf_batches=1,crc_valid=True)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
