"""Run the real queue completion protocol with only computation substituted."""
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
    report=dict(complete=False,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),cases=[],gpu_runs=0)
    for mode in ['empty','read_error','mark','mark_failure','complete','factor','auto_stage2',
                 'prepare','publication','stop_before','stop','finished_failure','advance_failure','device','persistence']:
        proc=subprocess.run([str(exe),mode,str(out/mode)],cwd=out,capture_output=True,text=True,errors='replace',timeout=30)
        log=proc.stdout+proc.stderr;(out/(mode+'.log')).write_text(log,encoding='utf-8')
        assert proc.returncode==0 and f'queue_contract: mode={mode}' in log,(mode,log)
        report['cases'].append(mode)
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    report['complete']=True
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
