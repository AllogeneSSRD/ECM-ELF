"""Learn point backends at frozen small-B2 geometries, then run held-out curves.

GPU1 only, no clock writes, no real work queue. Training repeats are useful cost
observations on repeated saved points; they add no independent ECM coverage.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from bench_stage2_version import load,write,stamp,check_gpu_processes
from bench_stage2_point_matrix import records,environment
from bench_stage2_point_routes import sha
from bench_stage2_runtime_matrix import StopEvent,completed_records
from bench_stage2_n_scaling import fields
from gpu_telemetry import TelemetryRecorder


def prepare(a,out):
    source=load(a.study/'plan.json')
    if not source['complete'] or len(source['cells'])!=26:raise ValueError('Need completed 13-width small-B2 source')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('bin','inputs','profiles','calls'):(out/name).mkdir()
    (out/'bin/.gitignore').write_text('*\n')
    for name in ('ecm_cuda_stage2.exe','gmp-10.dll','build_manifest.json'):shutil.copy2(a.exe.resolve().parent/name,out/'bin'/name)
    manifest=load(out/'bin/build_manifest.json')
    if manifest['sha256'].lower()!=sha(out/'bin/ecm_cuda_stage2.exe'):raise ValueError('Binary manifest mismatch')
    shutil.copy2(a.study/'plan.json',out/'source_plan.json');shutil.copy2(__file__,out/'collector_snapshot.py')
    reference_hash=None
    if a.reference_layout:
        shutil.copy2(a.reference_layout,out/'reference_layout.json');reference_hash=sha(out/'reference_layout.json')
    cases=[]
    for case in source['cases']:
        case=dict(case);save=out/'inputs'/Path(case['save']).name
        if sha(Path(case['save']))!=case['save_sha256']:raise ValueError('Original save changed')
        shutil.copy2(case['save'],save);case['save']=str(save);cases.append(case)
    cells=[]
    for cell in source['cells']:
        cell={k:cell[k] for k in ('id','case','B2','D','P','layout','carrier')}
        cell['chain']=next(d['chain'] for d in source['decisions'] if d['cell']==cell['id'])
        cell['reference_seconds']=next(d['row']['seconds'] for d in source['evaluations'] if d['cell']==cell['id'] and not d['balance'])
        cells.append(cell)
    allowed=[]
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    for pid in a.allow_gpu_pid:
        names={s.split(',',2)[2].strip() for s in processes.splitlines() if s.split(',')[1].strip()==str(pid)}
        if len(names)!=1 or not Path(next(iter(names))).is_absolute():raise ValueError('Cannot bind allowed process')
        allowed.append(dict(pid=pid,process_name=next(iter(names))))
    reference_version=completed_records(Path(source['evaluations'][0]['folder'])/'result.jsonl')[0]['program']['version']
    plan=dict(complete=False,program_version=a.version,reference_version=reference_version,started_utc=stamp(),exe_sha256=sha(out/'bin/ecm_cuda_stage2.exe'),
        dll_sha256=sha(out/'bin/gmp-10.dll'),source_sha256=sha(out/'source_plan.json'),
        source_study=str(a.study.resolve()),reference_layout_sha256=reference_hash,cases=cases,cells=cells,allowed_processes=allowed,
        training=[],decisions=[],evaluations=[],failures=[],protocol=dict(device=1,min_repeats=2,
        batch_mb=256,arena_mb=6300,fold_mb=640,warmups=0,holdout_repeats=1,
        geometry='Source D/layout/chain and actual B2 fixed; ordinary/carrier separate',
        training='Fresh per-cell profile; stop once required backend comparisons have two observations; retain one in-flight extra',
        evaluation='Freeze profile; independently run scalar/scalar and automatic winner, alternating order',
        retest='If automatic exceeds current scalar or historical source by both 5% and 20ms, add two independent pairs; retain first',
        hardware='No clock/power writes; log before/after process gates and per-call NVML'))
    write(out/'plan.json',plan);return plan


def command(out,folder,case,cell,profile,learning=False,control=False):
    point=next(s for s in Path(case['save']).read_text(encoding='utf-8-sig').splitlines() if s.startswith('METHOD='))
    saved=folder/'points.save';saved.write_text((point+'\n')*(16 if learning else 1),encoding='utf-8')
    config=folder/'ecm.toml';config.write_text('[stage2]\nstage2_short_calibration=false\n')
    return [str(out/'bin/ecm_cuda_stage2.exe'),'--config',str(config),'--save',str(saved),
        '--curves',str(16 if learning else 1),'--device','1','--b2',str(cell['B2']),'--d',str(cell['D']),
        '--carrier-exponent',str(cell['carrier']),'--auto-carrier','0','--groot-layout',str(cell['layout']),
        '--giant-chain',str(cell['chain']),'--giant-tail-balance','0','--b2-round-chain','0',
        '--baby-backend',('scalar' if control else 'auto'),'--giant-backend',('scalar' if control else 'auto'),
        '--tune-min-repeats','2','--tune-runtime',str(int(learning)),'--tune-profile',str(profile),
        '--short-calibration','0','--short-ntt-batch-tune','0','--arena-mb','6300','--owner-budget-mb','640',
        '--batch-mb','256','--relocation','0','--factor-only','--ntt-counts','1','--log-level','phases',
        '--results',str(folder/'result.jsonl'),'--submission',str(folder/'submission.txt'),
        '--log',str(folder/'engine.log'),'--debug-log-file',str(folder/'debug.log')]


def learned(raw,wide):
    count=Counter((r['stage2-engine']['baby_coop'],r['stage2-engine']['giant_coop']) for r in raw)
    if count[0,0]<2:return False
    if not wide:return True
    return count[32,0]>=2 and any(v>=2 for (b,g),v in count.items() if g==32) and any(v>=2 for (b,g),v in count.items() if g==16)


def capture(plan,out,folder,case,cell):
    rows=records(plan,folder,case,cell);raw=completed_records(folder/'result.jsonl')
    trace=(folder/'engine.log').read_text(encoding='utf-8')+'\n'+(folder/'debug.log').read_text(encoding='utf-8')
    digests=[fields(s,'real_batched_rootfold') for s in trace.splitlines() if s.startswith('real_batched_rootfold:')]
    counts=[s for s in trace.splitlines() if s.startswith('ntt_length_counts:')]
    walls=[fields(s,'curve_done') for s in (folder/'console.log').read_text(encoding='utf-8').splitlines() if s.startswith('curve_done:')]
    if len(digests)!=len(raw) or len(counts)!=len(raw) or len(walls)!=len(raw):raise ValueError('Missing work/digest/wall evidence')
    for row,result,digest,ntt,wall in zip(rows,raw,digests,counts,walls):
        engine=result['stage2-engine']
        if result['program']['version']!=plan['program_version']:raise ValueError('Version label differs from actual binary')
        row.update(baby=engine['baby_coop'],giant=engine['giant_coop'],curve_wall=float(wall['wall']),
            root=[digest[k] for k in ('digest_words','digest_sum','digest_xor')],ntt=ntt)
        if row['chain']!=cell['chain'] or row['balance']:raise ValueError('Fixed point geometry changed')
    return rows


def run(plan,out,cell,label,profile,learning=False,control=False):
    case=next(c for c in plan['cases'] if c['id']==cell['case']);folder=out/'calls'/cell['id']/label
    folder.mkdir(parents=True,exist_ok=False);cmd=command(out,folder,case,cell,profile,learning,control)
    wide=8192<(cell['carrier'] or case['bits'])<=16384
    stop=StopEvent() if learning else None;monitor=TelemetryRecorder(1,folder/'telemetry.jsonl',interval=.2)
    requested=False;begin=time.perf_counter();proc=None
    try:
        check_gpu_processes(monitor.monitor,out,folder/'before.json',allowed=plan['allowed_processes'])
        if stop:cmd+=['--stage2-stop-handle',str(stop.handle)]
        write(folder/'command.json',cmd)
        extra=dict(startupinfo=stop.startup(),close_fds=True) if stop else {}
        with (folder/'console.log').open('wb') as log:
            proc=subprocess.Popen(cmd,cwd=folder,env=environment(),stdout=log,stderr=subprocess.STDOUT,**extra)
            while proc.poll() is None:
                try:pending=completed_records(folder/'result.jsonl') if learning and (folder/'result.jsonl').exists() else []
                except json.JSONDecodeError:pending=[] # A result append may be in flight.
                if learning and not requested and learned(pending,wide):
                    stop.request();requested=True;write(folder/'stop.json',dict(reason='required_backend_comparisons_complete'))
                if time.perf_counter()-begin>1800:
                    if stop:stop.request()
                    proc.wait(timeout=1800);raise TimeoutError(str(folder))
                time.sleep(.01 if learning else .2)
        check_gpu_processes(monitor.monitor,out,folder/'after.json',allowed=plan['allowed_processes'])
        if proc.returncode:raise ValueError('Worker failed: '+str(folder))
    finally:
        if proc is not None and proc.poll() is None:
            if stop:stop.request()
            else:proc.terminate()
            try:proc.wait(timeout=900)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        monitor.close()
        if stop:stop.close()
    rows=capture(plan,out,folder,case,cell)
    if learning and not learned(completed_records(folder/'result.jsonl'),wide):raise ValueError('Backend comparisons incomplete')
    hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='points.save'}
    return dict(cell=cell['id'],label=label,folder=str(folder),rows=rows,process_seconds=time.perf_counter()-begin,file_sha256=hashes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path);p.add_argument('--exe',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--version',default='0.57.9')
    p.add_argument('--reference-layout',type=Path,help='Frozen layout/Prime95 summary, used only as historical reference')
    p.add_argument('--resume',action='store_true');p.add_argument('--max-cells',type=int,default=26)
    p.add_argument('--allow-gpu-pid',type=int,action='append',default=[])
    a=p.parse_args();out=a.output.resolve()
    if not a.resume and (not a.study or not a.exe):p.error('Need --study and --exe')
    plan=load(out/'plan.json') if a.resume else prepare(a,out)
    if plan['complete'] or plan['failures']:raise ValueError('Completed or failed evidence must be preserved')
    if (out/'collector_snapshot.py').read_bytes()!=Path(__file__).read_bytes():raise ValueError('Collector changed')
    if sha(out/'bin/ecm_cuda_stage2.exe')!=plan['exe_sha256'] or sha(out/'bin/gmp-10.dll')!=plan['dll_sha256'] or sha(out/'source_plan.json')!=plan['source_sha256']:raise ValueError('Frozen source/binary changed')
    try:
        for index,cell in enumerate(plan['cells'][:a.max_cells]):
            profile=out/'profiles'/(cell['id']+'.toml');frozen=profile.with_suffix('.frozen.toml')
            if not any(t['cell']==cell['id'] for t in plan['training']):
                entry=run(plan,out,cell,'training',profile,learning=True);plan['training'].append(entry)
                shutil.copy2(profile,frozen)
                write(out/'plan.json',plan);print('trained',cell['id'],[(r['baby'],r['giant']) for r in entry['rows']],flush=True)
            digest=sha(frozen)
            if not any(t['cell']==cell['id'] for t in plan['decisions']):
                folder=out/'calls'/cell['id']/'decision';folder.mkdir()
                case=next(c for c in plan['cases'] if c['id']==cell['case'])
                cmd=command(out,folder,case,cell,frozen)+['--plan-only'];write(folder/'command.json',cmd)
                r=subprocess.run(cmd,cwd=folder,env=environment(),capture_output=True,text=True,timeout=90)
                (folder/'console.log').write_text(r.stdout+r.stderr)
                if r.returncode:raise ValueError('Frozen planning failed')
                events=[json.loads(s) for s in r.stdout.splitlines() if s.startswith('{')]
                choice=next(e for e in events if e.get('type')=='tune_selection')
                valid=[c for c in choice['runtime_candidates'] if c['samples']>=2]
                score=lambda c:c['mean_seconds']+c.get('point_check_seconds',0)
                selected=next(c for c in valid if (c['baby_coop'],c['giant_coop'])==(choice['baby_coop'],choice['giant_coop']))
                if choice['runtime_probe'] or score(selected)>min(map(score,valid))+1e-8:raise ValueError('Not the fastest qualified measured route')
                plan['decisions'].append(dict(cell=cell['id'],selection=choice,profile_sha256=digest));write(out/'plan.json',plan)
            for label in (('scalar','auto') if index%2==0 else ('auto','scalar')):
                if any(e['cell']==cell['id'] and e['label']==label for e in plan['evaluations']):continue
                plan['evaluations'].append(run(plan,out,cell,label,frozen,control=label=='scalar'));write(out/'plan.json',plan)
            current=[e for e in plan['evaluations'] if e['cell']==cell['id']]
            auto=next(e['rows'][0]['seconds'] for e in current if e['label']=='auto')
            scalar=next(e['rows'][0]['seconds'] for e in current if e['label']=='scalar')
            if any(auto>r*1.05 and auto-r>.02 for r in (scalar,cell['reference_seconds'])):
                for repeat in (1,2):
                    for mode in ('auto','scalar'):
                        label=mode+'_retest'+str(repeat)
                        if not any(e['cell']==cell['id'] and e['label']==label for e in plan['evaluations']):
                            plan['evaluations'].append(run(plan,out,cell,label,frozen,control=mode=='scalar'));write(out/'plan.json',plan)
            if sha(frozen)!=digest:raise ValueError('Held-out run wrote frozen tune')
            print('validated',index+1,26,cell['id'],'scalar',scalar,'auto',auto,flush=True)
        plan['complete']=len({e['cell'] for e in plan['evaluations']})==26
        if plan['complete']:plan['finished_utc']=stamp()
    except Exception as error:
        plan['failures'].append(str(error));raise
    finally:write(out/'plan.json',plan)


if __name__=='__main__':main()
