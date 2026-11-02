"""Useful checked curves qualify S4 auto, repeats, region reuse and read-only use."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from bench_stage2_point_matrix import command,environment
from bench_stage2_point_routes import sha
from bench_stage2_runtime_matrix import completed_records
from bench_stage2_version import check_gpu_processes,load,write
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result
from stage2_tune_tables import summaries


def identity(r):return tuple(int(r[k]) for k in ('reduce_coop','reduce_threads','reduce_kernel'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('study','exe','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();source=load(a.study/'plan.json');case=next(c for c in source['cases'] if c['bits']>8192)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'bin').mkdir();(out/'bin/.gitignore').write_text('*\n')
    for f in ('ecm_cuda_stage2.exe','gmp-10.dll','build_manifest.json'):shutil.copy2(a.exe.parent/f,out/'bin'/f)
    shutil.copy2(__file__,out/'test_snapshot.py');binary=sha(out/'bin/ecm_cuda_stage2.exe')
    report=dict(complete=False,exe_sha256=binary,calls=[])
    def run(name,profile,curves,minimum=2,backend='auto',learn=True,d=30030,kernel=0):
        folder=out/name;folder.mkdir();cell=dict(B2=300750450,D=d,P=2880 if d!=6006 else 720,carrier=0,layout=1)
        cmd=command(out,folder,case,cell,profile,'32',False)
        point=(folder/'points.save').read_text();(folder/'points.save').write_text(point*curves)
        for flag,v in (('--curves',curves),('--tune-min-repeats',minimum),('--tune-runtime',int(learn))):cmd[cmd.index(flag)+1]=str(v)
        cmd+=['--baby-backend','coop32','--giant-backend','coop32','--seed-backend','coop32',
              '--reduce-backend',backend,'--reduce-kernel',str(kernel)]
        write(folder/'command.json',cmd);monitor=TelemetryRecorder(1,folder/'telemetry.jsonl',interval=.2)
        try:
            check_gpu_processes(monitor.monitor,out,folder/'before.json',allowed=source['allowed_processes'])
            with (folder/'console.log').open('wb') as log:r=subprocess.run(cmd,cwd=folder,env=environment(),stdout=log,stderr=subprocess.STDOUT,timeout=600)
            check_gpu_processes(monitor.monitor,out,folder/'after.json',allowed=source['allowed_processes'])
        finally:monitor.close()
        if r.returncode:raise ValueError('Native S4 auto call failed: '+str(folder))
        records=completed_records(folder/'result.jsonl')
        if len(records)!=curves:raise ValueError('Incomplete curves')
        for row in records:
            n=normalize_result(row)
            if row['status']!='NF' or row['source-build']!=binary or n['N_hex']!=case['N_hex'] or n['B2']!=cell['B2'] or \
               n['sigma']!=case['sigma'] or n['B1']!=case['b1'] or row['d']!=d or n['carrier_exponent'] or n['bad_factors'] or n['factors']:
                raise ValueError('Input/geometry/arithmetic changed')
        modes=[identity(r['stage2-engine']) for r in records]
        report['calls'].append(dict(name=name,modes=modes,folder=str(folder)));write(out/'summary.json',report)
        print('reduce_auto:',name,modes,flush=True);return modes
    profile=out/'paired.toml';modes=run('minimum2',profile,8)
    if modes[:6]!=[(0,0,0)]*2+[(32,128,1)]*2+[(32,128,2)]*2:raise ValueError('S4 repeats/order not respected')
    rows=[r for r in summaries(tomllib.loads(profile.read_text(encoding='utf-8-sig'))) if r['source']=='measured']
    best=min(rows,key=lambda r:r['mean_seconds']+r.get('point_check_seconds',0));winner=identity(best)
    if modes[6:]!=[winner]*2:raise ValueError('Fastest repeated S4 not retained')
    before=sha(profile)
    if run('readonly',profile,1,learn=False)!=[winner] or sha(profile)!=before:raise ValueError('Read-only selection failed')
    # Same transform interval: measure only the already qualified winner at a new D.
    if run('nearby_D',profile,2,d=27300)!=[winner]*2:raise ValueError('Paired S4 region not reused')
    # Different NTT cliff: establish a fresh scalar comparison.
    if run('cliff_D',profile,1,d=6006)!=[(0,0,0)]:raise ValueError('Unmeasured NTT cliff crossed')
    lone=out/'forced.toml'
    if run('forced',lone,2,backend='coop32',kernel=2)!=[(32,128,2)]*2:
        raise ValueError('Forced S4 variant lost in internal child')
    before=sha(lone)
    if run('unpaired_readonly',lone,1,learn=False)!=[(0,0,0)] or sha(lone)!=before:raise ValueError('Forced-only S4 qualified auto')
    report['complete']=True;write(out/'summary.json',report)


if __name__=='__main__':main()
