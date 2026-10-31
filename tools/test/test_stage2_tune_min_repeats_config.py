"""Public CLI/TOML minimum-repeat boundaries, using dry-run only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--save',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cases=[('minimum_one','stage2_tune_min_repeats=1\nstage2_tune_runtime_samples=1\n',[],True,''),
        ('default_two','',[],True,''),('zero','stage2_tune_min_repeats=0\n',[],False,'stage2_tune_min_repeats'),
        ('too_large','stage2_tune_min_repeats=1001\n',[],False,'stage2_tune_min_repeats'),
        ('below_minimum','stage2_tune_min_repeats=3\nstage2_tune_runtime_samples=2\n',[],False,'tune-min-repeats exceeds'),
        ('cli_overrides','stage2_tune_min_repeats=3\nstage2_tune_runtime_samples=2\n',['--tune-min-repeats','1'],True,'')]
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_'))}
    report=dict(complete=False,exe_sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest(),cases=[])
    for name,body,extra,valid,message in cases:
        folder=out/name;folder.mkdir();config=folder/'ecm.toml';config.write_text('[stage2]\n'+body,encoding='utf-8')
        command=[str(a.exe.resolve()),'--config',str(config),'--save',str(a.save.resolve()),'--curves','1',
            '--b2','27015978990','--device','1','--dry-run','--short-calibration','0','--tune-runtime','0']+extra
        result=subprocess.run(command,cwd=folder,env=env,capture_output=True,timeout=30)
        text=(result.stdout+result.stderr).decode('utf-8',errors='replace');(folder/'console.log').write_text(text,encoding='utf-8')
        if (result.returncode==0)!=valid or (message and message not in text):raise ValueError(name+': '+text)
        report['cases'].append(dict(name=name,command=command,exit_code=result.returncode))
    report['complete']=True;(out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('minimum repeat configuration: 6 dry-run checks passed')


if __name__=='__main__':main()
