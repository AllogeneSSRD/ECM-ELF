"""Measure normal runtime D/carrier learning on the full exact-N Stage2 matrix.

Use continuous worktodo queues so queue-aware search and preparation reuse are
active. Report the final profile's best admitted repeated mean, with its actual
sample count. Replayed Stage1 points do not add independent ECM coverage.
"""
import argparse
import ctypes
from ctypes import wintypes
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import shutil
import statistics as st
import subprocess
import sys
import time

from bench_stage2_version import ROOT, load, write, stamp, validate, check_gpu_processes
from bench_stage2_n_scaling import Monitor, fields, number
from result_format import normalize_result
sys.path.insert(0,str(ROOT/'tools/test'))
from check_stage2_version import audit as audit_baseline
from stage2_tune_tables import summaries
import tomllib


def log_rows(text,label):
    return [fields(line,label) for line in text.splitlines() if line.startswith(label+':')]


def observation_key(sample):
    return tuple(sample.get(k,0) for k in ('target_bits','b1','b2','d','carrier_exponent',
        'fold_resident','frontier_resident','giant_chunk_points','giant_chain_min','giant_force_ladder'))


def queue_rows(folder,case,cell,version,count,planned_count=None,allow_unfinished_tail=False):
    raw=[json.loads(line) for line in (folder/'result.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    text=(folder/'engine.log').read_text(encoding='utf-8')
    labels=['stage2_full_wall','s4_multiply_stats','real_shape','real_batched_split','ntt_workspace_stats']
    logs={k:log_rows(text,k) for k in labels}
    parent=re.findall(r'curve_done:.*?\bwall=([\d.]+)',(folder/'console.log').read_text(encoding='utf-8'))
    valid_logs=all(count<=len(v)<=count+int(allow_unfinished_tail) for v in logs.values())
    if len(raw)!=count or len(parent)!=count or not valid_logs:
        raise ValueError('Incomplete queue results, timings or arithmetic records')
    result=[]
    for i,record in enumerate(raw):
        row=normalize_result(record);phase=logs['stage2_full_wall'][i];shape=logs['real_shape'][i];check=logs['s4_multiply_stats'][i]
        identity=dict(N_hex=case['N_hex'],B1=case['b1'],B2=cell['B2'],sigma=case['sigma'],curve='ed')
        if any(row[k]!=v for k,v in identity.items()) or row['program']['version']!=version:
            raise ValueError('Input identity or version differs')
        if record['status']!='NF' or row['status']!='stage2_completed' or row['factors'] or row['bad_factors']:
            raise ValueError('Need complete, mathematically checked NF records')
        choice=row['tune_plan'];d,p=number(shape['D']),number(shape['P']);carrier=row['carrier_exponent']
        if row['requested_D']!=0 or carrier not in (0,case['exponent']) or (d,p,carrier)!=(choice['D'],choice['P'],choice['carrier_exponent']):
            raise ValueError('Executed geometry differs from automatic choice')
        if choice['model']!='runtime_shape_mean_v2' or (not choice['runtime_probe'] and choice['reason']!='runtime_measured_fastest'):
            raise ValueError('Expected the current runtime selector')
        if number(shape['S_bits'])!=(carrier or case['bits']) or number(shape['giant_points'])!=cell['B2']//d+2:
            raise ValueError('Arithmetic width or covered interval differs')
        if number(shape['num_poly_g'])!=(cell['B2']//d+2+p-1)//p:
            raise ValueError('G polynomial count differs')
        if phase['clean']!='1' or abs(float(phase['total'])-float(phase['init'])-float(phase['main']))>.003:
            raise ValueError('Unclosed full-engine timing')
        if min(int(check[k]) for k in ('gmp_checked','gmp_selftest_cases'))<=0 or any(int(check[k]) for k in ('gmp_check_bad','gmp_selftest_bad')):
            raise ValueError('Missing or failed arithmetic checks')
        result.append(dict(id=f'{cell["id"]}_{i+1:02d}',cell=cell['id'],case=case['id'],B2=cell['B2'],nominal_B2=cell['nominal_B2'],
            index=i+1,remaining=(planned_count or count)-i,folder=str(folder),D=d,P=p,carrier_exponent=carrier,selection=choice,
            engine_seconds=float(phase['total']),parent_seconds=float(parent[i]),outer_seconds=row['seconds'],
            phase_seconds=phase,shape=shape,s4=check,stats={k:logs[k][i] for k in labels[3:]},
            observation=record.get('tune_observation',{}).get('sample')))
    return result


def completed_records(path):
    if not path.exists():return []
    lines=path.read_bytes().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b'\n') and line.strip()]


def stopping_observation(records,threshold=3):
    """The selected full shape has three observations and remains fastest.

    Probe pairs alone never stop a queue. Compare the newly updated mean with
    the other admitted repeated candidates, since its third observation can
    change the ranking. This is an operational stop rule, not search exhaustion.
    """
    groups=defaultdict(list)
    for index,record in enumerate(records,1):
        sample=record.get('tune_observation',{}).get('sample')
        if not sample:continue
        key=observation_key(sample);groups[key].append(sample['mean_seconds'])
        choice=record['stage2-engine']['tune_plan']
        if choice['runtime_probe'] or choice['reason']!='runtime_measured_fastest' or len(groups[key])<threshold:continue
        mean=st.mean(groups[key]);others=[r['mean_seconds'] for r in choice['runtime_candidates']
            if r['samples']>=2 and (r['D'],r['carrier_exponent'])!=(sample['d'],sample['carrier_exponent'])]
        if others and mean>min(others):continue
        return dict(index=index,repeats=len(groups[key]),key=list(key),mean_seconds=mean,
            D=sample['d'],carrier_exponent=sample['carrier_exponent'])
    return None


class StopEvent:
    """Inheritable, unnamed Windows event; native parent finishes its curve."""
    def __init__(self):
        class Security(ctypes.Structure):
            _fields_=[('length',wintypes.DWORD),('descriptor',ctypes.c_void_p),('inherit',wintypes.BOOL)]
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        self.api.CreateEventW.argtypes=[ctypes.POINTER(Security),wintypes.BOOL,wintypes.BOOL,wintypes.LPCWSTR]
        self.api.CreateEventW.restype=wintypes.HANDLE
        self.api.SetEvent.argtypes=[wintypes.HANDLE];self.api.SetEvent.restype=wintypes.BOOL
        self.api.CloseHandle.argtypes=[wintypes.HANDLE]
        attributes=Security(ctypes.sizeof(Security),None,True)
        self.handle=self.api.CreateEventW(ctypes.byref(attributes),True,False,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
    def startup(self):
        info=subprocess.STARTUPINFO();info.lpAttributeList={'handle_list':[self.handle]};return info
    def request(self):
        if not self.api.SetEvent(self.handle):raise ctypes.WinError(ctypes.get_last_error())
    def close(self):
        if self.handle:self.api.CloseHandle(self.handle);self.handle=None


def task(case,cell,save,count):
    return f'ECMSTAGE2=1,2,{case["exponent"]},-1,"{save}",{cell["B2"]},0,{count},"{",".join(map(str,case["known"]))}"\n'


def prepare(args):
    baseline=args.baseline.resolve();audit_baseline(baseline)
    imported=None
    if args.import_study:
        imported=load(args.import_study/'plan.json')
        manual=args.import_study/'authorized_manual_stop.json'
        if manual.exists():
            stopped=load(manual)
            if imported['complete'] or imported['failures'] or imported.get('active',{}).get('cell')!=stopped['cell']:
                raise ValueError('Import does not match the authorized immediate stop')
        else:
            boundary=load(args.import_study/'authorized_protocol_stop.json')
            if imported.get('active') or imported['complete'] or len(imported['failures'])!=1:
                raise ValueError('Import needs the completed authorized boundary stop')
            if imported['queues'][-1]['cell']!=boundary['finish_active_queue'] or boundary['boundary_gate'] not in imported['failures'][0]['error']:
                raise ValueError('Source failure is not the recorded authorized boundary gate')
        if imported['program_version']!=args.version or Path(imported['baseline']).resolve()!=baseline:
            raise ValueError('Import version or exact baseline differs')
    old=load(baseline/'plan.json');out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'bin').mkdir();(out/'inputs').mkdir()
    for name in ('ecm_cuda_stage2.exe','gmp-10.dll'):shutil.copy2(args.exe.resolve().parent/name,out/'bin'/name)
    shutil.copy2(__file__,out/'collector_snapshot.py')
    cases=[]
    for source in old['cases']:
        case=dict(source);save=out/'inputs'/Path(case['save']).name;shutil.copy2(case['save'],save);case['save']=str(save)
        validate(save,int(case['N_hex'],16),case['b1'],1,case['sigma'],'ed');cases.append(case)
    cells=[]
    for ref in old['prime95_rows']:
        if ref['curve_type']!='Edwards' or not ref['benchmark_eligible']:continue
        case=next(c for c in cases if c['N_hex']==ref['N_hex'] and c['sigma']==int(ref['s']) and c['b1']==ref['b1'])
        cells.append(dict(id=f'{case["id"]}_b{ref["b2"]}',case=case['id'],B2=ref['b2'],nominal_B2=ref['b2_requested']))
    assert len(cells)==26
    # One small queue checks the full path; investigate the widest/highest cell next.
    first=cells[0];cells=[first]+sorted(cells[1:],key=lambda c:(next(x['bits'] for x in cases if x['id']==c['case']),c['B2']),reverse=True)
    plan=dict(schema=3,program_version=args.version,source_exe=str(args.exe.resolve()),baseline=str(baseline),
        started_utc=stamp(),device=1,curves_per_condition=args.curves,sample_limit=10,cases=cases,cells=cells,
        stop_repeats=3,
        queues=[],runs=[],decisions=[],failures=[],complete=False,protocol=dict(
            primary='Best admitted repeated runtime-profile mean of stage2_full_wall.total=init+main',
            sample_count='Actual retained observations per chosen full shape, at least 2 and at most 10',
            queue='Continuous worktodo queue per condition, cap 48; stop after the selected fastest full shape has three observations',
            stopping='Native inherited event finishes the active curve. Up to one already-started extra curve is retained. This heuristic does not prove exhaustive search.',
            order='Smallest low-B2 first, then decreasing actual cofactor width and B2',
            replay='Same independently verified point repeated; no new independent ECM coverage',
            warmups=0,arena_mb=6300,fold_mb=640,batch_mb=256,
            secondary='All engine times; curve_done.wall queue timing; separate process wall with 20ms exit polling (retained old queues: 250ms)',
            settings='GPU1 only; no clock/power/priority changes; short calibration, relocation and NTT counts off'))
    manifest=load(args.exe.resolve().parent/'build_manifest.json')
    context={k:manifest[k] for k in ('architecture','engine','gl_backend','gl_fixed_mode','outer_unroll_u','add_sub_mask','split_compile','toolkit') if k in manifest}
    context.update(program_version=args.version,source_exe=str(args.exe.resolve()))
    if imported:
        for name in ('ecm_cuda_stage2.exe','gmp-10.dll'):
            if (args.import_study/'bin'/name).read_bytes()!=(out/'bin'/name).read_bytes():raise ValueError('Imported build differs')
        for old_queue in imported['queues']:
            queue=dict(old_queue,planned_count=imported['curves_per_condition'],completed_count=imported['curves_per_condition'],
                finish_reason='retained_prechange_full_queue',origin=str(args.import_study.resolve()))
            plan['queues'].append(queue)
        plan['runs']=imported['runs']
        plan['imported_study']=str(args.import_study.resolve())
        if manual.exists():
            cell=next(c for c in cells if c['id']==stopped['cell']);case=next(c for c in cases if c['id']==cell['case'])
            folder=Path(stopped['folder']);count=stopped['completed_records']
            rows=queue_rows(folder,case,cell,args.version,count,imported['curves_per_condition'],True)
            plan['runs'].extend(rows)
            plan['queues'].append(dict(cell=cell['id'],folder=str(folder),command=load(folder/'command.json'),
                process_seconds=None,engine_sum=sum(r['engine_seconds'] for r in rows),parent_sum=sum(r['parent_seconds'] for r in rows),
                finished_utc=stopped['stopped_utc'],planned_count=imported['curves_per_condition'],completed_count=count,
                finish_reason='retained_user_stopped_queue',origin=str(args.import_study.resolve()),
                telemetry_unavailable='Collector was stopped by explicit user request before the in-memory series was flushed'))
        plan['imported_calls']=len(plan['runs'])
        shutil.copy2(args.import_study/'plan.json',out/'import_source_plan.json')
        for name in ('stage2_ecm_tune.toml','initial_import_profile.toml'):
            shutil.copy2(args.import_study/'stage2_ecm_tune.toml',out/name)
    write(out/'context.json',context);write(out/'plan.json',plan);return plan


def run(args):
    out=args.output.resolve();plan=load(out/'plan.json') if args.resume else prepare(args)
    if plan['schema']!=3 or plan['complete'] or plan['failures']:raise ValueError('Wrong schema, completed study or recorded failure needs review')
    if Path(__file__).read_bytes()!=(out/'collector_snapshot.py').read_bytes():raise ValueError('Collector changed; preserve the study')
    monitor=Monitor(1);plan['gpu']={k:v for k,v in monitor.identity.items() if k!='uuid'}
    if '4060 Laptop' not in plan['gpu']['name']:raise ValueError('Unexpected GPU1')
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')};env['NTT_NO_PROGRESS']='1'
    cases={c['id']:c for c in plan['cases']};profile=out/'stage2_ecm_tune.toml';accepted=0
    try:
        for cell in plan['cells']:
            if any(q['cell']==cell['id'] for q in plan['queues']):continue
            case=cases[cell['case']];folder=out/'queues'/cell['id'];folder.mkdir(parents=True,exist_ok=False)
            point=next(s for s in Path(case['save']).read_text(encoding='utf-8-sig').splitlines() if s.startswith('METHOD='))
            save=folder/'replay.save';save.write_text((point+'\n')*plan['curves_per_condition'],encoding='utf-8')
            queue=folder/'worktodo.txt';line=task(case,cell,save,plan['curves_per_condition']);queue.write_text(line,encoding='utf-8');(folder/'worktodo.initial.txt').write_text(line,encoding='utf-8')
            ini=folder/'ecm.toml';ini.write_text('user=\ncomputer=\nstage2_finished=finished.txt\n',encoding='utf-8')
            command=[str(out/'bin/ecm_cuda_stage2.exe'),'--ini',str(ini),'--worktodo',str(queue),'--once','--device','1',
                '--d','0','--auto-carrier','1','--tune-runtime','1','--tune-runtime-samples','10','--runtime-prepare-cache','1',
                '--short-calibration','0','--tune-profile',str(profile),'--arena-mb','6300','--owner-budget-mb','640','--batch-mb','256',
                '--relocation','0','--ntt-counts','0','--factor-only','--log-level','phases',
                '--log',str(folder/'engine.log'),'--results',str(folder/'result.jsonl'),'--submission',str(folder/'submission.txt')]
            stop=StopEvent();command+=['--stage2-stop-handle',str(stop.handle)]
            write(folder/'command.json',command);check_gpu_processes(monitor,out,folder/'before.json')
            plan['active']=dict(cell=cell['id'],folder=str(folder),started_utc=stamp());write(out/'plan.json',plan)
            print('RUN QUEUE',len(plan['queues'])+1,'/26',cell['id'],plan['curves_per_condition'],'curves',flush=True)
            started=time.perf_counter();telemetry=[];observed=0;trigger=None;next_sample=0
            with (folder/'console.log').open('wb') as log:
                process=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,
                    startupinfo=stop.startup(),close_fds=True)
                while process.poll() is None:
                    if time.perf_counter()>=next_sample:
                        telemetry.append(monitor.sample());next_sample=time.perf_counter()+.25
                    result_file=folder/'result.jsonl'
                    if result_file.exists():
                        records=completed_records(result_file);completed=len(records)
                        if completed>observed:
                            observed=completed;print('PROGRESS',cell['id'],observed,'/',plan['curves_per_condition'],flush=True)
                            if trigger is None:
                                trigger=stopping_observation(records,plan['stop_repeats'])
                                if trigger:
                                    trigger['observed_completed']=completed
                                    stop.request();write(folder/'stop_request.json',trigger)
                                    print('STOP REQUEST',cell['id'],'selected repeats',trigger['repeats'],'at curve',trigger['index'],flush=True)
                    if time.perf_counter()-started>args.timeout:
                        process.kill();process.wait();raise TimeoutError(cell['id'])
                    time.sleep(.02)
            stop.close()
            elapsed=time.perf_counter()-started;write(folder/'telemetry.json',telemetry)
            check_gpu_processes(monitor,out,folder/'after.json')
            if process.returncode:raise ValueError('Queue worker failed: '+str(folder))
            completed=len(completed_records(folder/'result.jsonl'))
            rows=queue_rows(folder,case,cell,plan['program_version'],completed,plan['curves_per_condition'])
            if not 1<=completed<=plan['curves_per_condition']:raise ValueError('Unexpected executed count')
            console=(folder/'console.log').read_text(encoding='utf-8')
            if completed<plan['curves_per_condition']:
                if not trigger or f'stage2_stopped: curves_this_run={completed} ' not in console:
                    raise ValueError('Early queue exit lacks the native graceful-stop confirmation')
                if queue.read_text(encoding='utf-8-sig')!=line:raise ValueError('Early stop changed the original task')
            elif queue.read_text(encoding='utf-8-sig').strip():raise ValueError('Full task did not finish')
            shutil.copy2(profile,folder/'profile_after.toml')
            record=dict(cell=cell['id'],folder=str(folder),command=command,process_seconds=elapsed,
                engine_sum=sum(r['engine_seconds'] for r in rows),parent_sum=sum(r['parent_seconds'] for r in rows),finished_utc=stamp(),
                planned_count=plan['curves_per_condition'],completed_count=completed,
                finish_reason='selected_repeats_3' if trigger else 'queue_limit',stop_trigger=trigger)
            plan['queues'].append(record);plan['runs'].extend(rows);plan.pop('active',None);write(out/'plan.json',plan);accepted+=1
            print('DONE QUEUE',cell['id'],'engine',round(record['engine_sum'],3),'wall',round(record['parent_sum'],3),flush=True)
            if args.max_queues and accepted>=args.max_queues:return
        frozen=out/'stage2_ecm_tune_frozen.toml';shutil.copy2(profile,frozen)
        document=tomllib.loads(frozen.read_text(encoding='utf-8-sig'));samples=summaries(document)
        for cell in plan['cells']:
            if any(s['cell']==cell['id'] for s in plan['decisions']):continue
            case=cases[cell['case']];folder=out/'decisions'/cell['id'];folder.mkdir(parents=True,exist_ok=False)
            queue=folder/'worktodo.txt';queue.write_text(task(case,cell,Path(case['save']),1),encoding='utf-8')
            ini=folder/'ecm.toml';ini.write_text('user=\ncomputer=\nstage2_finished=finished.txt\n',encoding='utf-8')
            command=[str(out/'bin/ecm_cuda_stage2.exe'),'--ini',str(ini),'--worktodo',str(queue),'--once','--plan-only',
                '--device','1','--d','0','--auto-carrier','1','--tune-runtime','0','--short-calibration','0',
                '--tune-profile',str(frozen),'--arena-mb','6300','--owner-budget-mb','640','--batch-mb','256',
                '--relocation','0','--ntt-counts','0','--factor-only']
            write(folder/'command.json',command);before=frozen.read_bytes()
            check_gpu_processes(monitor,out,folder/'before.json')
            native=subprocess.run(command,cwd=folder,env=env,capture_output=True,text=True,errors='replace',timeout=180)
            (folder/'console.log').write_text(native.stdout+native.stderr,encoding='utf-8')
            if native.returncode or frozen.read_bytes()!=before:raise ValueError('Read-only best-decision query failed')
            events=[json.loads(s) for s in native.stdout.splitlines() if s.startswith('{')]
            choice=next(s for s in events if s.get('type')=='tune_selection')
            if choice['runtime_probe'] or choice['runtime_samples']<2 or choice['reason']!='runtime_measured_fastest':raise ValueError('No repeated best decision')
            eligible=[s for s in samples if s['source']=='measured' and s.get('benchmark_kind')=='runtime_save' and
                (s['target_bits'],s['b1'],s['b2'],s['d'],s['carrier_exponent'])==(case['bits'],case['b1'],cell['B2'],choice['D'],choice['carrier_exponent']) and
                s['repeats']==choice['runtime_samples'] and abs(s['mean_seconds']-choice['mean_seconds'])<1e-8]
            if len(eligible)!=1:raise ValueError('Native best decision does not identify one exact profile row')
            sample=eligible[0];key=observation_key(sample)
            observations=[r for r in plan['runs'] if r['observation'] and observation_key(r['observation'])==key]
            if len(observations)!=sample['repeats'] or abs(st.mean(r['engine_seconds'] for r in observations)-sample['mean_seconds'])>2e-6:
                raise ValueError('Best mean differs from retained raw observations')
            decision=dict(cell=cell['id'],folder=str(folder),selection=choice,sample=sample,run_ids=[r['id'] for r in observations])
            plan['decisions'].append(decision);write(out/'plan.json',plan)
        if profile.read_bytes()!=frozen.read_bytes():raise ValueError('Active profile changed after freezing')
        plan.update(complete=True,finished_utc=stamp());write(out/'plan.json',plan)
        print('COMPLETE',len(plan['runs']),'calls;',len(plan['decisions']),'best repeated profile decisions;',frozen,flush=True)
    except Exception as error:
        if 'stop' in locals() and stop.handle:
            stop.request();stop.close()
        plan['failures'].append(dict(utc=stamp(),error=str(error)));write(out/'plan.json',plan);raise
    finally:monitor.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path);p.add_argument('--exe',type=Path);p.add_argument('--version',default='0.39.3')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--curves',type=int,default=48);p.add_argument('--max-queues',type=int,default=0)
    p.add_argument('--import-study',type=Path,help='Retain completed queues after the documented authorized protocol boundary')
    p.add_argument('--timeout',type=float,default=86400)
    a=p.parse_args()
    if not a.resume and not(a.baseline and a.exe):p.error('New study needs --baseline and --exe')
    if not 8<=a.curves<=1000:p.error('--curves must be 8..1000')
    run(a)
