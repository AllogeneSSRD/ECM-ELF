"""Completed Edwards checkpoints can finish publication with no usable device."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from test_edwards_gpu_pipeline import checkpoint, records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,default=1)
    args=parser.parse_args();exe=args.exe.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    report=dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),cases=[],complete=False)
    for algorithm in ['binary','naf']:
        folder=out/algorithm;folder.mkdir();blocked=folder/'blocked';blocked.write_text('publication failure fixture\n')
        queue=folder/'worktodo.txt';task='ECM2=1,2,127,-1,1000,10000,4,20260922\n';queue.write_text(task)
        ini=folder/'ecm.toml'
        config='[shared]\nuser=""\ncomputer=""\nworktodo='+json.dumps(str(queue))+\
            '\n[stage1]\ncurve="ed"\ntarget="gpu"\nauto_stage2=0\nckpt_seconds=0\ntmp_dir='+\
            json.dumps(str(folder))+'\nstage1_submission_file='+json.dumps(str(blocked/'results.json.txt'))+'\nlog_file=""\n'
        ini.write_text(config,encoding='utf-8')
        env=dict(os.environ,ECM_EDWARDS_STAGE1_ALGO=algorithm)
        env.pop('ECM_EDWARDS_DBL_ALGO',None)
        command=[str(exe),'--config',str(ini)]
        failed=subprocess.run(command+['-d',str(args.device)],cwd=folder,env=env,capture_output=True,timeout=120)
        (folder/'failed.log').write_bytes(failed.stdout+failed.stderr)
        assert failed.returncode!=0 and queue.read_text()==task
        path=folder/'.e_s1_m127_1000.ckpt';saved=checkpoint(path)
        assert saved['done']==saved['total'] and saved['count']==4
        snapshot=path.read_bytes();(folder/'completed_checkpoint.bin').write_bytes(snapshot)
        save=next(folder.glob('e_*.save'));before=records(save);assert len(before)==4
        blocked.rename(folder/'blocked_evidence')
        # Invalid device index would fail cudaSetDevice if the completed path used it.
        env['CUDA_VISIBLE_DEVICES']=''
        resumed=subprocess.run(command+['-d','999'],cwd=folder,env=env,capture_output=True,timeout=120)
        (folder/'resumed.log').write_bytes(resumed.stdout+resumed.stderr)
        text=(resumed.stdout+resumed.stderr).decode('utf-8','replace')
        assert resumed.returncode==0 and not queue.read_text().strip(),text[-3000:]
        assert 'host validation only; no device preparation' in text
        assert not path.exists() and records(save)==before
        assert len((folder/'stage1_results.jsonl').read_text().splitlines())==4
        report['cases'].append(dict(algorithm=algorithm,curves=4,completed_snapshot_sha256=hashlib.sha256(snapshot).hexdigest()))
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',algorithm,flush=True)
    report['complete']=True;(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
