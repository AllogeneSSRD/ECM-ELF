"""Six useful fixture calls exercise bounded 64/32 learning and fastest adoption.

Repeated copies are timing fixtures, never independent ECM coverage. No live
queue, save or tune file is modified. Read-only plan adoption is checked last.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from bench_stage2_point_routes import sha,validate_clock
from bench_stage2_version import load,write,stamp,check_gpu_processes
from gpu_telemetry import TelemetryRecorder
from bench_stage2_n_scaling import fields
from result_format import normalize_result
from stage2_tune_tables import summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-run',type=Path,help='Audit a completed raw run after a collector failure; no learn rerun')
    a=p.parse_args();study=load(a.study/'plan.json');defaults=load(Path(study['defaults'])/'plan.json')
    source=next(r for r in defaults['runs'] if r['case']=='m8011_n7995_s26' and r['engine']=='gpu-carrier')
    case=next(c for c in defaults['cases'] if c['id']==source['case'])
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe=a.exe.resolve();identity=sha(exe)
    saved=out/'points.save';point=Path(case['save']).read_text(encoding='utf-8-sig')
    saved.write_text(point*6,encoding='utf-8')
    config=out/'ecm.toml';shutil.copy2(source['command'][source['command'].index('--config')+1],config)
    profile=out/'tune.toml'
    command=[str(exe)]+[x.replace(source['prefix'],str(out/'learn')) for x in source['command'][1:]]
    def set_arg(cmd,key,value):
        if key in cmd:cmd[cmd.index(key)+1]=str(value)
        else:cmd.extend([key,str(value)])
    for k,v in [('--save',saved),('--config',config),('--d',source['D']),('--curves',6),
        ('--tune-profile',profile),('--tune-runtime',1),('--giant-chain','auto'),
        ('--groot-layout',1),('--giant-tail-balance',1),('--b2-round-chain',0)]:set_arg(command,k,v)
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')}
    env['NTT_NO_PROGRESS']='1'
    report=dict(complete=False,started_utc=stamp(),exe_sha256=identity,case=case,independent_curves=False)
    def invoke(name,cmd):
        write(out/(name+'.command.json'),cmd);telemetry=TelemetryRecorder(1,out/(name+'.telemetry.jsonl'))
        try:
            check_gpu_processes(telemetry.monitor,out,out/(name+'.before.json'),allowed=study['allowed_processes'])
            with (out/(name+'.console.log')).open('wb') as log:
                proc=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
            check_gpu_processes(telemetry.monitor,out,out/(name+'.after.json'),allowed=study['allowed_processes'])
        finally:telemetry.close()
        if proc.returncode:raise ValueError('Worker failed; logs retained')
        events=[json.loads(s) for s in (out/(name+'.console.log')).read_text(encoding='utf-8').splitlines() if s.startswith('{')]
        return [s for s in events if s.get('type')=='tune_selection']
    original=a.reuse_run.resolve() if a.reuse_run else out
    if a.reuse_run:
        previous=load(original/'learn.command.json')
        if Path(previous[0]).resolve()!=exe or (original/'points.save').read_bytes()!=saved.read_bytes():
            raise ValueError('Cannot reuse a different binary/input run')
        shutil.copy2(original/'tune.toml',profile)
        report['raw_source_run']=str(original)
    else:invoke('learn',command)
    # The parent deliberately filters child stdout. The durable receipt carries
    # the actual per-curve selection even when no JSON reaches the console.
    receipts=[json.loads(s) for s in (original/'learn.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    selections=[r['stage2-engine']['tune_plan'] for r in receipts]
    if len(selections)!=6 or [s['giant_chain_block'] for s in selections[:4]]!=[64,64,32,32]:
        raise ValueError('Bounded chain pair sequence differs: '+str(selections))
    if not all(s['runtime_probe'] for s in selections[:4]) or any(s['runtime_probe'] for s in selections[4:]):
        raise ValueError('Measured pair not adopted after bounded probes')
    for s in selections[4:]:
        values=[c['mean_seconds'] for c in s['runtime_candidates'] if c['samples']>=2]
        if s['mean_seconds']!=min(values):raise ValueError('Not the fastest admitted mean')
    rows=summaries(tomllib.loads(profile.read_text(encoding='utf-8')))
    if {r['giant_chain_block'] for r in rows}!={32,64} or len(rows)!=2 or any(r['repeats']<2 for r in rows):
        raise ValueError('Chain identities pooled or observations absent')
    if len(receipts)!=6 or any(r['status']!='NF' or r['source-build']!=identity or normalize_result(r)['bad_factors'] for r in receipts):
        raise ValueError('Incomplete clean NF receipts')
    text=(original/'learn.engine.log').read_text(encoding='utf-8')
    clocks=[fields(s,'stage2_full_wall') for s in text.splitlines() if s.startswith('stage2_full_wall:')]
    checks=[fields(s,'s4_multiply_stats') for s in text.splitlines() if s.startswith('s4_multiply_stats:')]
    if len(clocks)!=6 or len(checks)!=6:raise ValueError('Missing per-curve arithmetic evidence')
    for c,s in zip(clocks,checks):
        validate_clock(c)
        if min(int(s[k]) for k in ('gmp_checked','gmp_selftest_cases'))<=0 or any(int(s[k]) for k in ('gmp_check_bad','gmp_selftest_bad')):
            raise ValueError('Missing/failed mandatory checks')
    plan=command.copy();set_arg(plan,'--curves',1);set_arg(plan,'--tune-runtime',0);plan.append('--plan-only')
    before=sha(profile);adopt=invoke('adopt',plan)
    if len(adopt)!=1 or adopt[0]['runtime_probe'] or sha(profile)!=before:raise ValueError('Read-only adoption changed profile')
    if adopt[0]['mean_seconds']!=min(c['mean_seconds'] for c in adopt[0]['runtime_candidates'] if c['samples']>=2):
        raise ValueError('Read-only adoption lost fastest chain')
    if sha(exe)!=identity:raise ValueError('Binary changed')
    report.update(complete=True,finished_utc=stamp(),selections=selections,rows=rows,adopted=adopt[0])
    write(out/'summary.json',report);print(json.dumps(dict(complete=True,chain_sequence=[s['giant_chain_block'] for s in selections])))


if __name__=='__main__':main()
