"""Verify tune-driven Auto B2 in plan-only, real workers and private INI queues.

Provided Stage1 costs are decision-test inputs, not measured Stage1 benchmarks.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tomllib

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--save',type=Path,required=True)
    parser.add_argument('--wide-profile',type=Path,required=True)
    parser.add_argument('--wide-save',type=Path,required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[p.resolve() for p in [args.exe,args.profile,args.save,args.wide_profile,args.wide_save]]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    identities={str(p):sha(p) for p in sources}
    profile=tomllib.loads(args.profile.read_text(encoding='utf-8'))
    assert all(s['target_bits']==521 and s['b1']==20 for s in profile['ecm'].values())
    ini=out/'bench.ini';ini.write_text('verbose=false\nstage2_debug_log=false\n',encoding='utf-8')
    policy=profile['policy']
    common=[str(args.exe.resolve()),'--ini',str(ini),'--device',str(args.device),
            '--batch-mb',str(policy['batch_mb']),'--arena-mb',str(policy['arena_mb']),
            '--owner-budget-mb',str(policy['fold_mb']),'--log-level','quiet']
    plans={};curves={};rejected=[]
    def run(name,extra,success=True,reason=''):
        try:
            proc=subprocess.run(common+list(map(str,extra)),cwd=ROOT,capture_output=True,text=True,
                                errors='replace',timeout=300)
        except subprocess.TimeoutExpired as error:
            decode=lambda value:value.decode('utf-8',errors='replace') if isinstance(value,bytes) else value or ''
            (out/(name+'.console.log')).write_text(decode(error.stdout)+decode(error.stderr),encoding='utf-8')
            (out/'timeout.json').write_text(json.dumps(dict(case=name,timeout_seconds=300,
                command=list(map(str,error.cmd))),indent=2)+'\n',encoding='utf-8')
            raise
        (out/(name+'.console.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        assert (proc.returncode==0)==success,(name,proc.stdout,proc.stderr)
        assert not reason or reason in proc.stdout+proc.stderr,(name,reason,proc.stderr)
        if not success:rejected.append(name)
        return [json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{')]
    def choice(name,extra=(),t1=3,success=True,reason=''):
        rows=run(name,['--save',args.save.resolve(),'--auto-b2','--tune-profile',args.profile.resolve(),
                       '--stage1-seconds-per-curve',t1,'--plan-only',*extra],success,reason)
        if not success:return
        auto=next(x for x in rows if x.get('type')=='stage2_auto_plan')
        plan=next(x for x in rows if x.get('type')=='stage2_plan')
        assert auto['source']=='full_ecm_tune' and auto['schema']==3
        assert all(auto[k]==plan[k] for k in ['B2','D','carrier_exponent','P','I','G'])
        assert auto['T1']==t1 and auto['T1_source']=='provided_per_curve'
        assert math.isclose(auto['T2'],auto['ratio_adjust']*auto['engine_seconds'],rel_tol=1e-12)
        k=.11343+.88657*(math.log10(auto['B2']/20)/2)**(1.96617-.06781*math.log10(20))
        assert math.isclose(auto['K'],k,rel_tol=1e-12)
        assert math.isclose(auto['score'],k/(t1+auto['ratio_adjust']*auto['guarded_engine_seconds']),rel_tol=1e-12)
        assert plan['curve_workspace_memory']['initial_free_snapshot_fits']
        assert not any(x.get('type')=='tune_selection' for x in rows)
        plans[name]=auto;return auto
    default=choice('default')
    low=choice('low_stage1',t1=.01);high=choice('high_stage1',t1=100)
    assert low['B2']<default['B2']<high['B2']
    adjusted=choice('adjusted',['--stage2-ratio-adjust','2'])
    assert adjusted['B2']<default['B2'] and adjusted['ratio_adjust']==2
    fixed=choice('fixed_unseen',['--auto-min-b2','5200000000','--auto-max-b2','5200000000'])
    assert fixed['B2']==5200000000 and fixed['model']=='linear_giant_points_v1' and fixed['range_limited']
    locked=choice('fixed_d',['--d','30030'])
    assert locked['D']==30030 and locked['model']=='measured_exact_scope_v1'
    ordinary=choice('fixed_ordinary',['--carrier-exponent','0'])
    assert ordinary['carrier_exponent']==0
    priority=choice('tune_priority',['--cost-profile',out/'absent.cprof'])
    assert priority['B2']==default['B2']
    amortized=choice('provided_t1_batch',['--stage1-batch','16'])
    assert amortized['stage1_batch']==16 and amortized['B2']==default['B2']
    assert math.isclose(amortized['score'],default['score'],rel_tol=1e-12)
    for name,extra,reason in [
        ('outside_low',['--auto-min-b2','1'],'lower bound outside'),
        ('outside_high',['--auto-max-b2','52000000000'],'upper bound outside'),
        ('reverse',['--auto-min-b2','20000000000','--auto-max-b2','10000000000'],'upper bound outside'),
        ('policy',['--batch-mb','64'],'policy mismatch'),
        ('debug',['--log-level','debug'],'debug_log'),
        ('carrier_illegal',['--carrier-exponent','6001'],'does not divide'),
    ]:choice(name,extra,success=False,reason=reason)
    run('missing_t1',['--save',args.save.resolve(),'--auto-b2','--tune-profile',args.profile.resolve(),'--plan-only'],
        False,'requires --stage1-seconds-per-curve')
    run('missing_profiles',['--save',args.save.resolve(),'--auto-b2','--plan-only'],False,'Auto B2 requires')
    run('legacy_unchanged',['--save',args.save.resolve(),'--auto-b2','--cost-profile',out/'absent.cprof','--plan-only'],
        False,'no calibrated cost profile for selected NTT add/sub')
    untagged=out/'no_model.toml'
    untagged.write_text(args.profile.read_text(encoding='utf-8').replace('prediction_model = "linear_giant_points_v1"\n',''),encoding='utf-8')
    choice('no_model_inside',['--tune-profile',untagged,'--auto-min-b2','5200000000','--auto-max-b2','5200000000'],
           success=False,reason='no measured or qualified')
    derived=out/'synthetic_arena_refusal.toml'
    derived.write_text(args.profile.read_text(encoding='utf-8').replace('arena_mb = 6300','arena_mb = 1')
                       .replace('arena_cap_kb = 6451200','arena_cap_kb = 1024'),encoding='utf-8')
    # Admission is independent of the cost-search breadth. Restrict this
    # synthetic refusal to one exact scope; exhaustively replaying every
    # predicted B2 with an impossible arena duplicates expensive CPU plans.
    choice('memory_refusal',['--tune-profile',derived,'--arena-mb','1','--d','60060',
                            '--auto-min-b2','2600000000','--auto-max-b2','2600000000'],
           success=False,reason='fits current joint memory')
    ini_fixed=out/'fixed_d.ini'
    ini_fixed.write_text(ini.read_text()+'stage2_d=30030\n',encoding='utf-8')
    assert choice('ini_d',['--ini',ini_fixed])['D']==30030
    def curve(name,options):
        result,log=out/(name+'.jsonl'),out/(name+'.log')
        run(name,[*options,'--curves','1','--results',result,'--log',log])
        row=json.loads(result.read_text(encoding='utf-8'))
        assert row['auto_b2'] and row['requested_B2']==row['requested_D']==0
        assert 'tune_plan' not in row
        auto=row['auto_plan']
        assert auto['B2']==row['B2'] and auto['carrier_exponent']==row['carrier_exponent']
        text=log.read_text(encoding='utf-8')
        assert 'gmp_selftest_bad=0' in text and 'gmp_check_bad=0' in text
        assert row['hits']==row['bad_factors']==0
        assert re.search(r'real_batched_folddevice: requested=1 enabled=1 fallback=none\b',text)
        assert re.search(r'scaled_frontier_device: requested=1 enabled=1\b',text)
        wall=next(x for x in text.splitlines() if x.startswith('stage2_full_wall:'))
        assert 'clean=1' in wall
        actual=float(re.search(r'\btotal=([0-9.]+)',wall).group(1))
        error=abs(actual-auto['engine_seconds'])/actual
        assert error<=.08,(name,error,auto,actual)
        curves[name]=dict(actual_seconds=actual,predicted=auto['engine_seconds'],relative_error=error,auto_plan=auto)
        return row
    row=curve('actual_unseen',['--save',args.save.resolve(),'--auto-b2','--tune-profile',args.profile.resolve(),
                              '--stage1-seconds-per-curve','3','--auto-min-b2','5200000000','--auto-max-b2','5200000000'])
    assert row['B2']==fixed['B2'] and row['auto_plan']['D']==fixed['D']
    wide_options=['--save',args.wide_save.resolve(),'--auto-b2','--tune-profile',args.wide_profile.resolve(),
                  '--stage1-seconds-per-curve','30']
    rows=run('wide_plan',[*wide_options,'--plan-only'])
    wide=next(r for r in rows if r.get('type')=='stage2_auto_plan')
    assert wide['carrier_exponent']==6011 and wide['D']==120120
    plans['wide']=wide
    rows=run('wide_lock_carrier',[*wide_options,'--carrier-exponent','6011','--plan-only'])
    carrier=next(r for r in rows if r.get('type')=='stage2_auto_plan')
    assert carrier['carrier_exponent']==6011 and carrier['B2']==wide['B2'] and carrier['D']==wide['D']
    wide_row=curve('wide_auto',wide_options)
    assert wide_row['auto_plan']['D']==wide['D'] and wide_row['B2']==wide['B2'] and wide_row['carrier_exponent']==6011
    rows=run('wide_lock_ordinary',[*wide_options,'--carrier-exponent','0','--plan-only'])
    assert next(r for r in rows if r.get('type')=='stage2_auto_plan')['carrier_exponent']==0
    # Private two-record queue; all paths are inside this fresh evidence folder.
    qdir=out/'queue';qdir.mkdir();save=qdir/'m521.save'
    save.write_bytes(args.save.read_bytes()*2)
    queue=qdir/'worktodo.txt';task='ECMSTAGE2=tune-auto,1,2,521,-1,"m521.save",0,0,2,""\n'
    queue.write_text(task,encoding='utf-8');qini=qdir/'ecm.ini'
    qini.write_text('[gpu]\ndevice='+str(args.device)+'\n[queue]\nstage2_worktodo=worktodo.txt\nstage2_finished=finished.txt\n'
                    'tmp_dir='+str(qdir)+'\n[stage2]\nstage2_auto_b2=1\nstage2_tune_profile='+os.path.relpath(args.profile.resolve(),qdir)+
                    '\nstage1_seconds_per_curve=3\nstage2_batch_mb=256\nstage2_arena_mb=6300\nstage2_fold_mb=640\n'
                    'stage2_results_file=results.jsonl\nstage2_log_file=stage2.log\n',encoding='utf-8')
    rows=run('queue_plan',['--ini',qini,'--worktodo',queue,'--plan-only'])
    assert sum(r.get('type')=='stage2_auto_plan' for r in rows)==2
    assert queue.read_text()==task and not (qdir/'finished.txt').exists()
    run('queue_execute',['--ini',qini,'--worktodo',queue,'--once'])
    result_path=qdir/'results.jsonl'
    results=[json.loads(x) for x in result_path.read_text(encoding='utf-8').splitlines()]
    assert len(results)==2 and all(r['auto_b2'] and r['requested_B2']==0 and r['bad_factors']==r['hits']==0 for r in results)
    assert all('tune_plan' not in r and r['auto_plan']['source']=='full_ecm_tune' for r in results)
    assert 'ECMSTAGE2' not in queue.read_text() and task in (qdir/'finished.txt').read_text()
    before=result_path.read_bytes();run('queue_completed_rerun',['--ini',qini,'--worktodo',queue,'--once'])
    assert result_path.read_bytes()==before
    assert identities=={str(p):sha(p) for p in sources}
    report=dict(passed=True,binary_sha256=sha(args.exe.resolve()),plans=plans,curves=curves,rejected=rejected,
                queue_curves=2,queue_finished_rerun_preserved=True,provided_stage1_costs_are_test_inputs=True,
                legacy_qualification_unchanged=True,joint_selection_not_reselected=True,
                total_scope='engine stage2_full_wall.total; Stage1/process/planning/publication excluded')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,plans=len(plans),rejected=len(rejected),actual_curves=len(curves)+2,
                         maximum_time_error=max(c['relative_error'] for c in curves.values()))))


if __name__=='__main__':main()
