"""Fail a real periodic snapshot at a safe stop, then resume the preserved task."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from stage1_checkpoint_decode import batch, progressed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,default=1)
    args=parser.parse_args();exe=args.exe.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    class Security(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('descriptor',wintypes.LPVOID),('inherit',wintypes.BOOL)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateEventW.argtypes=[ctypes.POINTER(Security),wintypes.BOOL,wintypes.BOOL,wintypes.LPCWSTR]
    kernel.CreateEventW.restype=wintypes.HANDLE
    kernel.SetEvent.argtypes=[wintypes.HANDLE];kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    report=dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),device=args.device,cases=[],complete=False)
    for model,algorithm in [('mont','ladder'),('mont','resident'),('mont','prac'),('ed','binary'),('ed','naf')]:
        name=model+'_'+algorithm;folder=out/name;folder.mkdir()
        task='ECM2=1,2,347,-1,100000,1000000,8,20260922\n'
        queue=folder/'worktodo.txt';queue.write_text(task)
        ini=folder/'ecm.toml'
        ini.write_text('[shared]\nuser=""\ncomputer=""\nworktodo='+json.dumps(str(queue))+
            '\n[stage1]\ncurve='+json.dumps(model)+'\ntarget="gpu"\ngpu_param=0\ndevice='+str(args.device)+
            '\nauto_stage2=0\nckpt_seconds=0.001\ntmp_dir='+json.dumps(str(folder))+'\nlog_file=""\n',encoding='utf-8')
        env=dict(os.environ,ECM_GPU_STAGE1_ALGO=algorithm if model=='mont' else 'ladder')
        for key in tuple(env):
            if key.startswith('ECM_PRAC_') or key=='ECM_GPU_STAGE1_SAMPLE_SECONDS' or key.startswith('ECM_EDWARDS_'):env.pop(key)
        if model=='ed':env['ECM_EDWARDS_STAGE1_ALGO']=algorithm
        event=kernel.CreateEventW(ctypes.byref(Security(ctypes.sizeof(Security),None,True)),True,False,None)
        assert event
        command=[str(exe),'--config',str(ini)]
        with (folder/'failed.log').open('wb') as log:
            process=subprocess.Popen(command+['--stage1-stop-handle',str(event)],cwd=folder,env=env,
                stdout=log,stderr=log,close_fds=False,creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline=time.monotonic()+120;blocker=None
                while time.monotonic()<deadline:
                    assert process.poll() is None,(name,'exited before injection')
                    candidates=[p for p in folder.glob('*.ckpt') if '.bak' not in p.name]
                    try:
                        if candidates and progressed(batch(candidates[0])[1]):
                            blocker=Path(str(candidates[0])+'.new');blocker.mkdir();break
                    except (PermissionError,FileNotFoundError,FileExistsError):pass
                    time.sleep(.01)
                else:raise AssertionError((name,'no arithmetic snapshot'))
                assert kernel.SetEvent(event)
                assert process.wait(timeout=120)!=0
            finally:
                if process.poll() is None:process.kill();process.wait()
                kernel.CloseHandle(event)
        text=(folder/'failed.log').read_text(encoding='utf-8',errors='replace')
        assert 'checkpoint write failed' in text,(name,text[-2500:])
        assert queue.read_text()==task,(name,'persistence failure consumed queue row')
        assert not list(folder.glob('*.jsonl')),(name,'failed batch published completed ledger')
        batch(candidates[0])
        blocker.rmdir()
        with (folder/'resumed.log').open('wb') as log:
            resumed=subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=log,timeout=240)
        assert resumed.returncode==0 and not queue.read_text().strip(),name
        assert not list(folder.glob('*.ckpt')),(name,'successful task did not seal checkpoint')
        report['cases'].append(name);(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PASS',name,flush=True)
    report['complete']=True;(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
