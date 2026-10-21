"""Exercise real task/file collaborators with a private fake computation seam."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();exe=a.exe.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    identity=hashlib.sha256(exe.read_bytes()).hexdigest()
    report=dict(complete=False,binary_sha256=identity,cases=[],gpu_runs=0)
    for mode in ['complete','factor','prepare','throw','publication','stop','stop_publication','bad_begin','execution','execution_stop']:
        proc=subprocess.run([str(exe),mode,str(out/mode)],cwd=out,capture_output=True,text=True,errors='replace',timeout=30)
        text=proc.stdout+proc.stderr;(out/(mode+'.log')).write_text(text,encoding='utf-8')
        assert proc.returncode==0 and f'task_contract: mode={mode}' in text,(mode,text)
        report['cases'].append(mode)
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    assert hashlib.sha256(exe.read_bytes()).hexdigest()==identity
    report['complete']=True
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
