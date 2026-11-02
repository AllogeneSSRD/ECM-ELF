"""Fixed-geometry scalar/cooperative seed A/B, using verified matrix saves.

No learning, hardware changes or live queues. Diagnostic leaf/GMP checks are
explicitly separate from the timing population. All calls and first samples stay.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from bench_stage2_point_matrix import command, environment
from bench_stage2_point_routes import sha, validate_clock
from bench_stage2_version import load, write, stamp, check_gpu_processes
from bench_stage2_n_scaling import fields
from bench_stage2_runtime_matrix import completed_records
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result
from stage2_d_catalog import phi


def collect(out, folder, case, cell, plan, mode, checks):
    cmd=command(out,folder,case,cell,folder/'unused_tune.toml',str(cell['chain']),False)
    cmd[cmd.index('--device')+1]=str(plan['protocol']['device'])
    seed_mode=mode if plan.get('component','seed')=='seed' else 'coop32'
    cmd+=['--baby-backend','coop32','--giant-backend','coop32','--seed-backend',seed_mode]
    if plan.get('component')=='reduce':
        cmd+=['--reduce-backend',mode]
        if plan.get('reduce_kernel'):cmd+=['--reduce-kernel',str(plan['reduce_kernel'])]
    env=environment()
    if checks:
        env.update(NTT_GIANT_SEED_CHECK='1',NTT_DEVICE_GLEAF_CHECK='1',
                   NTT_S4_CARRY_TRACE='1',NTT_TARGET_LEAF_HASH='1')
    write(folder/'command.json',cmd)
    monitor=TelemetryRecorder(plan['protocol']['device'],folder/'telemetry.jsonl',interval=.2)
    start=time.perf_counter()
    try:
        check_gpu_processes(monitor.monitor,out,folder/'before.json',allowed=plan['allowed_processes'])
        with (folder/'console.log').open('wb') as log:
            result=subprocess.run(cmd,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        write(folder/'exit.json',dict(returncode=result.returncode))
        check_gpu_processes(monitor.monitor,out,folder/'after.json',allowed=plan['allowed_processes'])
        if result.returncode:raise ValueError('Failed curve: '+str(folder))
    finally:
        monitor.close()
    raw=completed_records(folder/'result.jsonl')
    if len(raw)!=1:raise ValueError('Expected one complete curve')
    r=raw[0];value=normalize_result(r)
    expected=dict(N_hex=case['N_hex'],sigma=case['sigma'],B1=case['b1'],B2=cell['B2'],
                  carrier_exponent=cell['carrier'],curve=case.get('curve','ed'))
    if any(value[k]!=v for k,v in expected.items()) or r['source-build']!=plan['exe_sha256'] or \
       r['status']!='NF' or value['factors'] or value['bad_factors']:
        raise ValueError('Input, arithmetic result or binary identity differs')
    if r['d']!=cell['D'] or r['poly-size']!=cell['P']:raise ValueError('Geometry changed')
    text=(folder/'engine.log').read_text(encoding='utf-8')+'\n'+(folder/'debug.log').read_text(encoding='utf-8')
    clock=fields(text,'stage2_full_wall');validate_clock(clock,checks)
    labels=('s4_multiply_stats','real_batched_split','real_batched_wall','real_baby',
            'real_giant_seed_pair','real_giant_base','real_giant_seed','device_gleaf',
            'groot_layout','target_descent_values','real_batched_rootfold','s4_coop_reduce_mode')
    stats={key:fields(text,key,False) for key in labels}
    s4=stats['s4_multiply_stats']
    if min(int(s4[k]) for k in ('gmp_checked','gmp_selftest_cases'))<=0 or \
       any(int(s4[k]) for k in ('gmp_check_bad','gmp_selftest_bad')):
        raise ValueError('Mandatory GMP check missing or failed')
    actual=32 if seed_mode=='coop32' and (cell['carrier'] or case['bits'])>8192 else 0
    pair=stats['real_giant_seed_pair']
    if int(pair['seed_backend'])!=actual or int(pair['base_backend'])!=actual or int(pair['base_nonunits']):
        raise ValueError('Unexpected seed/base arithmetic or nonunit fallback')
    if plan.get('component')=='reduce':
        actual_reduce=32 if mode=='coop32' and not cell['carrier'] and case['bits']>8192 else 0
        reduce=stats['s4_coop_reduce_mode']
        modern='reduce_coop' in r['stage2-engine']
        requested=actual_reduce if modern else (32 if mode=='coop32' else 0)
        if int(reduce['enabled'])!=actual_reduce or int(reduce['requested'])!=requested:
            raise ValueError('Unexpected S4 reduction policy')
        if modern:
            expected_kernel=plan.get('reduce_kernel') or 1
            if tuple(r['stage2-engine'][k] for k in ('reduce_coop','reduce_threads','reduce_kernel'))!=(actual_reduce,128 if actual_reduce else 0,expected_kernel if actual_reduce else 0):
                raise ValueError('Unexpected actual S4 variant')
    if int(stats['groot_layout']['balanced'])!=cell['layout']:raise ValueError('Layout changed')
    if checks:
        leaf=stats['device_gleaf']
        if int(leaf['checked_groups'])!=int(leaf['groups']) or int(leaf['checked_leaf_words'])!=int(leaf['device_leaf_words']) or \
           int(stats['real_giant_seed']['checked_words'])<=0 or not stats['target_descent_values']:
            raise ValueError('Diagnostic GMP coverage incomplete')
    counts=[s for s in text.splitlines() if s.startswith(('ntt_length_counts:','ntt_convolution_counts:'))]
    if not counts:raise ValueError('Missing NTT workload counts')
    return dict(cell=cell['id'],mode=mode,folder=str(folder),seconds=float(clock['total']),clock=clock,
                process_seconds=time.perf_counter()-start,stats=stats,ntt_counts=counts,device_identity=monitor.identity)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component',choices=('seed','reduce'),default='seed')
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--background-snapshot',type=Path)
    p.add_argument('--reduce-kernel',type=int,choices=(1,2))
    p.add_argument('--arithmetic',choices=('both','ordinary','carrier'),default='both')
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',action='append',default=[])
    p.add_argument('--repeats',type=int,choices=(1,2,3),default=1)
    p.add_argument('--checks',action='store_true')
    p.add_argument('--d',type=int)
    p.add_argument('--b2',type=int)
    a=p.parse_args();source=load(a.study/'plan.json')
    if not source['complete']:raise ValueError('Incomplete source matrix')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'bin';binary.mkdir();(binary/'.gitignore').write_text('*\n')
    manifest=load(a.exe.parent/'build_manifest.json')
    if manifest['sha256'].lower()!=sha(a.exe):raise ValueError('Binary does not match build manifest')
    for unit in manifest['compile_units'].values():
        if sha(unit['entry']['output'])!=unit['object_sha256'].lower():raise ValueError('Changed compile object')
    for f in (a.exe.resolve(),a.exe.resolve().parent/'gmp-10.dll'):
        shutil.copy2(f,binary/f.name)
    shutil.copy2(a.exe.parent/'build_manifest.json',out/'build_manifest.json')
    shutil.copy2(__file__,out/'collector_snapshot.py')
    cases=[dict(c) for c in source['cases'] if (not a.case and c['bits']>8192) or c['id'] in a.case]
    if not cases or set(a.case)-{c['id'] for c in cases}:raise ValueError('Unknown/empty cases')
    (out/'inputs').mkdir()
    for c in cases:
        save=out/'inputs'/Path(c['save']).name;shutil.copy2(c['save'],save)
        if sha(save)!=c['save_sha256']:raise ValueError('Changed Stage1 input')
        c['save']=str(save)
    cells=[dict(c) for c in source['cells'] if c['case'] in {x['id'] for x in cases} and
           (a.arithmetic=='both' or bool(c['carrier'])==(a.arithmetic=='carrier'))]
    for c in cells:
        if a.d:c.update(D=a.d,P=phi(a.d)//2)
        if a.b2:c['B2']=a.b2
    plan=dict(complete=False,component=a.component,reduce_kernel=a.reduce_kernel,started_utc=stamp(),source_study=str(a.study.resolve()),
              exe_sha256=sha(binary/a.exe.name),dll_sha256=sha(binary/'gmp-10.dll'),
              source_matrix_sha256=sha(a.study/'plan.json'),allowed_processes=source['allowed_processes'],
              cases=cases,cells=cells,runs=[],failures=[],diagnostic=a.checks,
              protocol=dict(device=a.device,repeats=a.repeats,warmups=0,learning=False,arithmetic=a.arithmetic,
                            batch_mb=256,arena_mb=6300,fold_mb=640))
    if a.background_snapshot:
        bg=load(a.background_snapshot)
        if not bg.get('approved_background_by_user') or bg['device']!=a.device:raise ValueError('Wrong background authorization')
        probe=TelemetryRecorder(a.device,out/'identity_telemetry.jsonl',interval=.2)
        try:
            if probe.identity['uuid']!=bg['uuid']:raise ValueError('Background UUID mismatch')
        finally:probe.close()
        plan['allowed_processes']=plan['allowed_processes']+bg['allowed_processes']
        plan['approved_background_snapshot']=bg;shutil.copy2(a.background_snapshot,out/'background_snapshot.json')
    write(out/'summary.json',plan)
    try:
        for ci,cell in enumerate(cells):
            case=next(c for c in cases if c['id']==cell['case']);baseline=None
            for repeat in range(a.repeats):
                order=('scalar','coop32') if (ci+repeat)%2==0 else ('coop32','scalar')
                for mode in order:
                    folder=out/'calls'/cell['id']/f'{repeat}_{mode}';folder.mkdir(parents=True)
                    row=collect(out,folder,case,cell,plan,mode,a.checks)
                    signature={k:row['stats']['s4_multiply_stats'][k] for k in ('launches','poly_muls','coeffs_reduced')}
                    signature['ntt']=row['ntt_counts']
                    if a.checks:
                        leaf=row['stats']['target_descent_values'];signature['leaf']={k:leaf[k] for k in ('words','hash')}
                    if baseline is not None and signature!=baseline:raise ValueError('A/B work or normalized leaves differ')
                    baseline=signature;row['repeat']=repeat;plan['runs'].append(row)
                    write(out/'summary.json',plan)
                    print(a.component+'_curve:',cell['id'],repeat,mode,row['seconds'],flush=True)
        if sha(a.exe)!=plan['exe_sha256']:raise ValueError('Source binary changed during experiment')
        plan.update(complete=True,finished_utc=stamp())
    except Exception as e:
        plan['failures'].append(str(e));raise
    finally:
        write(out/'summary.json',plan)


if __name__=='__main__':main()
