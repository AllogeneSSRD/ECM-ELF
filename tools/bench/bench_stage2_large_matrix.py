"""Frozen large-B2 Stage2 geometry baseline and same-binary S4 comparisons.

Reuse exact saved Edwards inputs and the large actual Prime95 B2 values. No
live queue, hardware changes, calibration, warmup or hidden learning runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

from bench_stage2_version import load, write, stamp, check_gpu_processes
from bench_stage2_point_matrix import environment, records
from bench_stage2_runtime_matrix import completed_records
from bench_stage2_n_scaling import fields
from bench_edwards_gpu_stage1 import validate
from analyze_stage2_point_matrix import partition
from gpu_telemetry import TelemetryRecorder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(exe, folder, repo):
    exe=exe.resolve();build=exe.parent;folder.mkdir(parents=True);(folder/'.gitignore').write_text('*\n')
    manifest_path=build/'build_manifest.json'
    if not manifest_path.exists():manifest_path=build/'manifest.json'
    m=load(manifest_path);expected=m.get('sha256',m.get('exe_sha256'))
    assert sha(exe)==expected.lower(),exe
    objects={}
    for unit in m.get('compile_units',{}).values():objects[unit['entry']['output']]=unit['object_sha256']
    for unit in m.get('reused',{}).values():objects[unit['object']]=unit['sha256']
    for path in build.glob('*.obj'):objects[str(path)]=sha(path)
    for name,digest in objects.items():assert sha(name)==digest.lower(),name
    source_map={}
    with zipfile.ZipFile(folder/'sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name,digest in m['source_hashes'].items():
            path=Path(name) if Path(name).is_absolute() else repo/name
            assert sha(path)==digest.lower(),name
            label=path.relative_to(repo).as_posix() if path.is_relative_to(repo) else 'external/'+hashlib.sha256(str(path).encode()).hexdigest()[:12]+'/'+path.name
            archive.write(path,label);source_map[name]=label
    shutil.copy2(exe,folder/'ecm_cuda_stage2.exe');shutil.copy2(build/'gmp-10.dll',folder/'gmp-10.dll')
    shutil.copy2(manifest_path,folder/manifest_path.name)
    identity=dict(exe_sha256=sha(exe),dll_sha256=sha(folder/'gmp-10.dll'),objects=objects,
                  manifest_sha256=sha(manifest_path),sources_sha256=sha(folder/'sources.zip'),source_map=source_map)
    write(folder/'frozen.json',identity);return identity


def prepare(a,out):
    repo=Path(__file__).resolve().parents[2]
    source=load(a.study/'plan.json');reference=load(a.reference/'plan.json')
    assert source['complete'] and len(source['cells'])==26 and not source['failures']
    out.mkdir(parents=True,exist_ok=False);identity=freeze(a.exe,out/'bin',repo)
    shutil.copy2(__file__,out/'collector.py');shutil.copy2(a.study/'plan.json',out/'source_plan.json')
    cases=[];(out/'inputs').mkdir()
    for original in source['cases']:
        case=dict(original);save=out/'inputs'/Path(case['save']).name
        assert sha(case['save'])==case['save_sha256'];shutil.copy2(case['save'],save);case['save']=str(save)
        n=int(case['N_hex'],16);assert n.bit_length()==case['bits'] and ((1<<case['exponent'])-1)%n==0
        validate(save,n,case['b1'],1,case['sigma'],'ed');cases.append(case)
    cells=[]
    for original in source['cells']:
        case=next(c for c in cases if c['id']==original['case'])
        matches=[r for r in reference['prime95_rows'] if r['N_hex'].lower()==case['N_hex'].lower() and
                 r['b1']==case['b1'] and r['b2_requested']==260000000000]
        assert len(matches)==1,case['id']
        cells.append(dict(id=original['id'],case=case['id'],B2=matches[0]['b2'],nominal_B2=260000000000,
                          carrier=original['carrier'],chain=original['chain'],D=0,P=0,layout=-1))
    if a.baseline:
        old=load(a.baseline/'plan.json');assert old['complete'] and not old['failures']
        for cell in cells:
            previous=next(c for c in old['cells'] if c['id']==cell['id'])
            assert (cell['B2'],cell['carrier'],cell['chain'])==(previous['B2'],previous['carrier'],previous['chain'])
            for key in ('D','P','layout'):cell[key]=previous[key]
        shutil.copy2(a.baseline/'plan.json',out/'baseline_plan.json')
    plan=dict(complete=False,started_utc=stamp(),identity=identity,exe_sha256=identity['exe_sha256'],
              cases=cases,cells=cells,runs=[],failures=[],allowed_processes=source['allowed_processes'],
              treatments=[int(x) for x in a.kernels.split(',')],phase='paired' if a.baseline else 'baseline',
              protocol=dict(device=1,warmups=0,calibration=False,learning=False,batch_mb=256,arena_mb=6300,
                            fold_mb=640,geometry='Baseline chooses D/layout once; paired run freezes it; inherited finite chain',
                            backends='baby/giant/seed coop32 request; S4 coop32; unsupported routes retain native fallback',
                            timing='engine init+main; curve wall separate; mandatory checks retained',
                            repetitions=1,recheck='Repeat both arms if >3% and >0.05s regression; keep initial samples'))
    write(out/'plan.json',plan);return plan


def collect(plan,out,cell,kernel,label,checks=False):
    case=next(c for c in plan['cases'] if c['id']==cell['case'])
    folder=out/'calls'/cell['id']/label;folder.mkdir(parents=True,exist_ok=False)
    config=folder/'ecm.toml';config.write_text('[stage2]\nstage2_short_calibration=false\n')
    command=[str(out/'bin/ecm_cuda_stage2.exe'),'--config',str(config),'--save',case['save'],'--curves','1',
             '--device','1','--b2',str(cell['B2']),'--d',str(cell['D']),'--carrier-exponent',str(cell['carrier']),
             '--auto-carrier','0','--groot-layout','auto' if cell['layout']<0 else str(cell['layout']),
             '--giant-chain',str(cell['chain']),'--giant-tail-balance','0','--b2-round-chain','0',
             '--baby-backend','coop32','--giant-backend','coop32','--seed-backend','coop32',
             '--reduce-backend','coop32','--reduce-threads','128','--reduce-kernel',str(kernel),
             '--tune-runtime','0','--tune-profile',str(folder/'unused.toml'),'--short-calibration','0',
             '--short-ntt-batch-tune','0','--arena-mb','6300','--owner-budget-mb','640','--batch-mb','256',
             '--relocation','0','--factor-only','--ntt-counts','1','--log-level','phases',
             '--results',str(folder/'result.jsonl'),'--submission',str(folder/'submission.txt'),
             '--log',str(folder/'engine.log'),'--debug-log-file',str(folder/'debug.log')]
    env=environment()
    if checks:env.update(NTT_TARGET_LEAF_HASH='1',NTT_GIANT_SEED_CHECK='1',NTT_DEVICE_GLEAF_CHECK='1',NTT_S4_CARRY_TRACE='1')
    write(folder/'command.json',command);monitor=TelemetryRecorder(1,folder/'telemetry.jsonl',interval=.2)
    try:
        check_gpu_processes(monitor.monitor,out,folder/'before.json',allowed=plan['allowed_processes'])
        begin=time.perf_counter()
        with (folder/'console.log').open('wb') as log:
            result=subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        process=time.perf_counter()-begin;write(folder/'exit.json',dict(returncode=result.returncode))
        check_gpu_processes(monitor.monitor,out,folder/'after.json',allowed=plan['allowed_processes'])
    finally:monitor.close()
    assert result.returncode==0,folder
    raw=completed_records(folder/'result.jsonl');assert len(raw)==1,folder
    if not cell['D']:
        text=(folder/'debug.log').read_text(encoding='utf-8')
        layout=fields(text,'groot_layout');cell.update(D=raw[0]['d'],P=raw[0]['poly-size'],layout=int(layout['balanced']))
    row=records(plan,folder,case,cell)[0]
    debug=(folder/'engine.log').read_text(encoding='utf-8')+'\n'+(folder/'debug.log').read_text(encoding='utf-8')
    console=(folder/'console.log').read_text(encoding='utf-8')
    diagnostic_labels=('real_batched_rootfold','real_batched_folddevice','ntt_workspace_stats','s4_coop_reduce_mode',
                       'scaled_frontier_device','real_giant_seed_pair','device_gleaf','rawupload','real_batched_prepare')
    diagnostics={key:fields(debug,key,False) for key in diagnostic_labels}
    expected=32 if not cell['carrier'] and case['bits']>8192 else 0
    engine=raw[0]['stage2-engine']
    assert (engine['reduce_coop'],engine['reduce_kernel'])==(expected,kernel if expected else 0),folder
    ntt=[s for s in debug.splitlines() if s.startswith('ntt_length_counts:')];assert len(ntt)==1,folder
    wall=fields(console,'curve_done');row['curve_wall']=float(wall['wall'])
    row['phases']=partition(row)
    row.update(cell=cell['id'],kernel=kernel,label=label,folder=str(folder),diagnostic=checks,
               process_seconds=process,diagnostics=diagnostics,ntt=ntt[0],
               root={k:diagnostics['real_batched_rootfold'][k] for k in ('digest_words','digest_sum','digest_xor')},
               actual_backends={k:engine[k] for k in ('baby_coop','giant_coop','seed_coop','reduce_coop','reduce_threads','reduce_kernel')})
    row['file_sha256']={n:sha(folder/n) for n in ('command.json','result.jsonl','engine.log','debug.log','console.log','exit.json')}
    return row


def workload(row):
    """Arithmetic and scheduling evidence; excludes noisy clocks and policies."""
    return dict(ntt=row['ntt'],root=row['root'],
                s4={k:row['s4'][k] for k in ('launches','poly_muls','coeffs_reduced','gmp_checked','gmp_selftest_cases')},
                points=[{k:p[k] for k in ('npts','route','block','words')} for p in row['policies']])


def append_call(plan,out,cell,kernel,repeat):
    row=collect(plan,out,cell,kernel,f'k{kernel}_r{repeat}')
    siblings=[r for r in plan['runs'] if r['cell']==cell['id']]
    if siblings:assert workload(row)==workload(siblings[0]),'Paired work/output differs: '+cell['id']
    if (out/'baseline_plan.json').exists():
        previous=load(out/'baseline_plan.json')
        old=next(r for r in previous['runs'] if r['cell']==cell['id'])
        assert workload(row)==workload(old),'Baseline work/output differs: '+cell['id']
    plan['runs'].append(row);write(out/'plan.json',plan)
    print('large_matrix',cell['id'],'kernel',kernel,'repeat',repeat,'D/P',cell['D'],cell['P'],'engine',row['seconds'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path);p.add_argument('--baseline',type=Path)
    p.add_argument('--study',type=Path,default=Path('data/experiments/stage2_giant_auto_20261101/matrix01'))
    p.add_argument('--reference',type=Path,default=Path('data/experiments/edwards_stage2_reference_20261017/matrix_01'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--kernels',default='2')
    p.add_argument('--resume',action='store_true');p.add_argument('--max-runs',type=int,default=0)
    p.add_argument('--wide-repeats',type=int,choices=(1,3),default=1,
                   help='Independent repeats for affected ordinary >8192-bit cells; always keep first calls')
    a=p.parse_args();out=a.output.resolve()
    if not a.resume and not a.exe:p.error('a fresh study needs --exe')
    plan=load(out/'plan.json') if a.resume else prepare(a,out)
    assert not plan['failures'] and sha(out/'collector.py')==sha(__file__),'source changed or failed study: use a new directory'
    assert sha(out/'bin/ecm_cuda_stage2.exe')==plan['exe_sha256']
    new=0
    try:
        for index,cell in enumerate(plan['cells']):
            order=plan['treatments'] if index%2==0 else list(reversed(plan['treatments']))
            for kernel in order:
                if any(r['cell']==cell['id'] and r['kernel']==kernel for r in plan['runs']):continue
                append_call(plan,out,cell,kernel,0);new+=1
                if a.max_runs and new>=a.max_runs:return
        # Keep qualification separate from the first pass. Reverse both the
        # cell and treatment order, without dropping any cold/slow first sample.
        rechecks=[]
        for cell in plan['cells']:
            case=next(c for c in plan['cases'] if c['id']==cell['case'])
            rows=[r for r in plan['runs'] if r['cell']==cell['id'] and r['label'].endswith('_r0')]
            affected=not cell['carrier'] and case['bits']>8192
            ref=next((r for r in rows if r['kernel']==plan['treatments'][0]),None)
            regression=bool(ref) and any(r['seconds']>ref['seconds']*1.03 and r['seconds']>ref['seconds']+.05 for r in rows)
            if (affected and a.wide_repeats==3) or regression:rechecks.append(cell)
        plan['recheck_cells']=[c['id'] for c in rechecks]
        for repeat in (1,2):
            for cell in (list(reversed(rechecks)) if repeat==1 else rechecks):
                for kernel in (list(reversed(plan['treatments'])) if repeat==1 else plan['treatments']):
                    if any(r['cell']==cell['id'] and r['kernel']==kernel and r['label']==f'k{kernel}_r{repeat}' for r in plan['runs']):continue
                    append_call(plan,out,cell,kernel,repeat);new+=1
                    if a.max_runs and new>=a.max_runs:return
        plan.update(complete=True,finished_utc=stamp())
    except Exception as error:
        plan['failures'].append(str(error));raise
    finally:write(out/'plan.json',plan)


if __name__=='__main__':main()
