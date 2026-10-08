"""Reject uncalibrated arithmetic profiles and mismatched production/HostOnly builds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--save',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise ValueError('use a fresh output directory')
    out.mkdir(parents=True);exes=[a.baseline.resolve(),a.candidate.resolve()]
    identity=[]
    for mask,exe in enumerate(exes):
        build=read(exe.parent/'build_manifest.json')
        if build['add_sub_mask']!=mask or build['engine'] not in (('development',) if mask==0 else ('development','production')) or sha(exe)!=build['sha256'].lower():raise ValueError('build settings')
        objects={name:sha(exe.parent/'_objects'/(name+'.obj')) for name in build['objects']}
        if objects!={k:v.lower() for k,v in build['objects'].items()}:raise ValueError('object identity')
        identity.append(dict(binary_sha256=sha(exe),manifest_sha256=sha(exe.parent/'build_manifest.json'),objects=objects))
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')};env['CUDA_LAUNCH_BLOCKING']='0'
    env['PSModulePath']=str(Path(os.environ['WINDIR'])/'System32/WindowsPowerShell/v1.0/Modules')
    report=dict(complete=False,identity=identity,tool_sha256=sha(__file__),save_sha256=sha(a.save),runs=[])
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    def verify():
        if sha(__file__)!=report['tool_sha256'] or sha(a.save)!=report['save_sha256']:raise ValueError('tool/save changed')
        for exe,want in zip(exes,identity):
            if sha(exe)!=want['binary_sha256'] or sha(exe.parent/'build_manifest.json')!=want['manifest_sha256']:raise ValueError('binary/manifest changed')
            for name,digest in want['objects'].items():
                if sha(exe.parent/'_objects'/(name+'.obj'))!=digest:raise ValueError('object changed')
    def run(name,command,expected,token):
        verify();proc=subprocess.run(command,env=env,capture_output=True,timeout=120)
        dest=out/(name+'.log');dest.write_bytes(proc.stdout+proc.stderr)
        if proc.returncode!=expected or token not in dest.read_text(encoding='utf-8',errors='replace'):raise ValueError(name+' wrong rejection/result')
        report['runs'].append(dict(name=name,command=command,exit=proc.returncode,token=token,log_sha256=sha(dest)))
        (out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n');verify()
    builder=ROOT/'tools/build/build_stage2_local.ps1'
    for mask in [0,2,3,4]:
        dest=out/f'invalid_production_m{mask}'
        command=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(builder),'-Engine','production','-AddSubMask',str(mask),'-Build',str(dest)]
        run(f'production_m{mask}',command,1,'Production requires' if mask<4 else 'AddSubMask')
        if dest.exists():raise ValueError('rejected build created output')
    run('hostonly_mismatch',['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(builder),'-Engine','development',
        '-GlBackend','ptx','-SplitCompile','6','-AddSubMask','1','-HostOnly','-Build',str(exes[0].parent)],1,'HostOnly requires identical CUDA')
    ini=out/'manual.ini';ini.write_text('device=1\n')
    for mask,exe in enumerate(exes):
        common=[str(exe),'--ini',str(ini),'--save',str(a.save.resolve()),'--device','1','--log-level','debug']
        run(f'auto_m{mask}',common+['--auto-b2','--plan-only','--cost-profile',str(out/'absent.cprof')],2,
            'no calibrated cost profile for selected NTT add/sub' if mask else 'cannot lock cost profile for reading')
        run(f'dplan_m{mask}',common+['--plan-only','--b2','2011326186870','--d','1381380','--arena-mb','6300'],0,
            f'ntt_addsub_arithmetic: mask={mask}')
        text=(out/f'dplan_m{mask}.log').read_text();plans=[json.loads(s) for s in text.splitlines() if s.startswith('{')]
        if len(plans)!=1 or plans[0]['curves_executed'] or plans[0]['calibrated'] or plans[0]['model']!='legacy_56_1':raise ValueError('uncalibrated D planning policy')
    report['note']='Both D plans already fall back because cache payload v2 is uncalibrated; the explicit arithmetic guard also prevents future reuse of old rates. Auto guard distinguishes the compiled candidate before profile I/O.'
    report['complete']=True;(out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(complete=True,passed=len(report['runs']))))


if __name__=='__main__':main()
