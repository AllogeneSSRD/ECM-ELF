"""Validate tune Auto B2 benefit ranking against remeasured finite candidates.

T1 is an independently completed Stage1 profile median. Each candidate receives
one warmup and interleaved full Stage2 repeats. This checks a stated finite set,
not global ECM success probability or unseen B2 optimality. Raw process overhead
is recorded separately and does not silently change the engine-cost contract.
"""
import argparse,hashlib,importlib.util,json,math,os,re,statistics,subprocess,time,tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
TIME_ERROR_LIMIT=.08
RANK_LOSS_LIMIT=.05

def benefit(b1,b2):
    if b1<2 or b2<=b1:raise ValueError('invalid ECM bounds')
    return .11343+.88657*(math.log10(b2/b1)/2)**(1.96617-.06781*math.log10(b1))

def profit(b1,b2,t1,t2,ratio):
    if not all(math.isfinite(x) and x>0 for x in [t1,t2,ratio]):raise ValueError('invalid benefit costs')
    return benefit(b1,b2)/(t1+ratio*t2)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['exe','profile','stage1-profile','save','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',type=int,required=True);p.add_argument('--stage1-batch',type=int,required=True)
    p.add_argument('--stage1-exponent',choices=['lcm','choose12'],default='lcm')
    p.add_argument('--ratio',type=float,default=1);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--min-candidates',type=int,default=2);p.add_argument('--timeout',type=float,default=1800)
    a=p.parse_args()
    if a.device<0 or not 1<=a.stage1_batch<=1048576 or not 2<=a.repeats<=1000 or a.min_candidates<2 or not math.isfinite(a.ratio) or a.ratio<=0 or not math.isfinite(a.timeout) or a.timeout<=0:
        p.error('invalid device/batch/repeats/ratio/timeout')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    paths=[a.exe.resolve(),a.profile.resolve(),a.stage1_profile.resolve(),a.save.resolve(),Path(__file__).resolve(),
           ROOT/'tools/bench/validate_stage2_tune_selection.py']
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();identities={str(x):sha(x) for x in paths}
    s=importlib.util.spec_from_file_location('fixed',paths[-1]);fixed=importlib.util.module_from_spec(s);s.loader.exec_module(fixed)
    profile=tomllib.loads(a.profile.read_text(encoding='utf-8-sig'))
    stage1=tomllib.loads(a.stage1_profile.read_text(encoding='utf-8-sig'))
    assert a.profile.stat().st_size<=64*1048576 and a.stage1_profile.stat().st_size<=16*1048576
    with a.save.open(encoding='utf-8-sig') as stream:line=next(x for x in stream if x.strip())
    fields=dict(re.findall(r'(\w+)\s*=\s*([^;]+)',line));n=int(fields['N'].strip(),0);b1=int(fields['B1'])
    bits=n.bit_length();kind='mersenne' if (n+1)&n==0 else 'generic'
    matches=[x for x in stage1['stage1'].values() if x['target_bits']==bits and x['b1']==b1 and
        x['batch']==a.stage1_batch and x['modulus_kind']==kind and x['exponent']==a.stage1_exponent]
    assert len(matches)==1,'no unique measured Stage1 scope'
    t1=statistics.median(matches[0]['seconds']);assert t1>0 and t1==matches[0]['median_seconds']
    groups={}
    for row in profile['ecm'].values():
        if row['target_bits']!=bits or row['b1']!=b1 or not row['fold_resident'] or not row['frontier_resident']:continue
        carrier=row['carrier_exponent']
        if carrier and ((1<<carrier)-1)%n:continue
        if not carrier and row['modulus_kind']!=kind:continue
        groups.setdefault(tuple(row[k] for k in fixed.SCOPE),[]).append(row)
    assert groups,'no covered Stage2 scope'
    ini=out/'ecm.ini';ini.write_text('verbose=false\nstage2_debug_log=false\n',encoding='utf-8')
    policy=profile['policy'];common=[str(a.exe.resolve()),'--ini',str(ini),'--save',str(a.save.resolve()),
        '--device',str(a.device),'--tune-profile',str(a.profile.resolve()),'--batch-mb',str(policy['batch_mb']),
        '--arena-mb',str(policy['arena_mb']),'--owner-budget-mb',str(policy['fold_mb']),'--log-level','quiet']
    auto=['--auto-b2','--stage1-tune-profile',str(a.stage1_profile.resolve()),'--stage1-batch',str(a.stage1_batch),
        '--stage1-exponent',a.stage1_exponent,'--stage2-ratio-adjust',str(a.ratio)]
    report=dict(complete=False,source_identities=identities,b1=b1,target_bits=bits,t1_seconds=t1,ratio=a.ratio,
        stage1_batch=a.stage1_batch,exponent=a.stage1_exponent,repeats=a.repeats,warmups=1,
        time_error_limit=TIME_ERROR_LIMIT,rank_loss_limit=RANK_LOSS_LIMIT,candidates=[],memory_rejected=[],
        total_scope='score=K/(measured T1+ratio*remeasured Stage2 engine total); finite tested candidates')
    def publish():(out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def run(name,extra):
        start=time.perf_counter();proc=subprocess.run(common+extra,capture_output=True,text=True,errors='replace',timeout=a.timeout)
        (out/(name+'.console.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        assert proc.returncode==0,(name,proc.stderr)
        return [json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{')],time.perf_counter()-start
    publish()
    try:
        rows,_=run('auto_plan',auto+['--plan-only']);choice=next(x for x in rows if x.get('type')=='stage2_auto_plan')
        assert choice['T1_source']=='measured_stage1_profile' and choice['T1']==t1
        assert math.isclose(choice['K'],benefit(b1,choice['B2']),rel_tol=1e-12)
        assert math.isclose(choice['score'],profit(b1,choice['B2'],t1,choice['guarded_engine_seconds'],a.ratio),rel_tol=1e-12)
        report['auto_plan']=choice;selected=(choice['B2'],choice['D'],choice['carrier_exponent'])
        candidates={selected}
        for key,group in groups.items():
            for row in group:candidates.add((row['b2'],row['d'],row['carrier_exponent']))
        for index,(b2,d,carrier) in enumerate(sorted(candidates)):
            extra=['--b2',str(b2),'--d',str(d),'--carrier-exponent',str(carrier)]
            rows,_=run(f'candidate_{index}_plan',extra+['--plan-only'])
            pick=next(x for x in rows if x.get('type')=='tune_selection');plan=next(x for x in rows if x.get('type')=='stage2_plan')
            if not pick['selected']:
                report['memory_rejected'].append(dict(b2=b2,d=d,carrier=carrier,reason=pick['reason']));continue
            assert plan['curve_workspace_memory']['initial_free_snapshot_fits']
            group=next(values for key,values in groups.items() if key[-1]==d and key[2]==carrier)
            prediction=fixed.predict(group,b2,profile['profile'].get('prediction_model')=='linear_giant_points_v1')
            assert prediction is not None,'native accepted an independently ineligible candidate'
            native=pick.get('estimated_seconds',pick.get('median_seconds'));assert math.isclose(native,prediction['seconds'],rel_tol=1e-10)
            assert math.isclose(pick['rank_seconds'],prediction['rank'],rel_tol=1e-10)
            report['candidates'].append(dict(index=index,b2=b2,d=d,carrier=carrier,selected=(b2,d,carrier)==selected,
                prediction=prediction,seconds=[],process_seconds=[],warmup_seconds=None))
        assert len(report['candidates'])>=a.min_candidates
        assert any(x['selected'] for x in report['candidates']);publish()
        # Measure the automatic choice through the real unforced production route.
        for repeat in range(a.repeats+1):
            ordered=report['candidates'][::1 if repeat%2==0 else -1]
            for candidate in ordered:
                stem=f'candidate_{candidate["index"]}_repeat{repeat}';receipt=out/(stem+'.jsonl');log=out/(stem+'.log')
                extra=auto if candidate['selected'] else ['--b2',str(candidate['b2']),'--d',str(candidate['d']),
                    '--carrier-exponent',str(candidate['carrier'])]
                _,process=run(stem,extra+['--curves','1','--results',str(receipt),'--log',str(log)])
                row=json.loads(receipt.read_text(encoding='utf-8'));assert row['status']=='stage2_completed' and row['hits']==row['bad_factors']==0
                assert row['B2']==candidate['b2'] and row['carrier_exponent']==candidate['carrier']
                if candidate['selected']:
                    assert row['auto_plan']['D']==candidate['d'] and row['auto_plan']['T1']==t1 and 'tune_plan' not in row
                else:assert row['tune_plan']['D']==candidate['d']
                text=log.read_text(encoding='utf-8')
                assert 'gmp_selftest_bad=0' in text and 'gmp_check_bad=0' in text
                assert re.search(r'real_batched_folddevice: requested=1 enabled=1 fallback=none\b',text)
                assert re.search(r'scaled_frontier_device: requested=1 enabled=1\b',text)
                wall=next(x for x in text.splitlines() if x.startswith('stage2_full_wall:'));assert 'clean=1' in wall
                total=float(re.search(r'\btotal=([0-9.]+)',wall)[1])
                if repeat:candidate['seconds'].append(total);candidate['process_seconds'].append(process)
                else:candidate['warmup_seconds']=total
                publish();print(json.dumps(dict(candidate=candidate['index'],b2=candidate['b2'],d=candidate['d'],repeat=repeat,seconds=total)),flush=True)
        for candidate in report['candidates']:
            total=statistics.median(candidate['seconds']);candidate['actual_median_seconds']=total
            candidate['relative_error']=abs(total-candidate['prediction']['seconds'])/total
            candidate['actual_score']=profit(b1,candidate['b2'],t1,total,a.ratio)
            candidate['process_median_seconds']=statistics.median(candidate['process_seconds'])
        best=max(report['candidates'],key=lambda c:c['actual_score']);actual=next(x for x in report['candidates'] if x['selected'])
        report['rank_loss']=1-actual['actual_score']/best['actual_score'];report['actual_best_index']=best['index'];publish()
        assert all(x['relative_error']<=TIME_ERROR_LIMIT for x in report['candidates']),'time prediction error exceeds fixed limit'
        assert report['rank_loss']<=RANK_LOSS_LIMIT,'Auto B2 benefit loss exceeds fixed finite-set limit'
        assert all(sha(Path(path))==h for path,h in identities.items()),'source data changed'
        report['complete']=True;publish()
    except Exception as error:report['failure']=str(error);publish();raise
    print(json.dumps(dict(passed=True,candidates=len(report['candidates']),rank_loss=report['rank_loss'],
        actual_curves=len(report['candidates'])*(a.repeats+1))))

if __name__=='__main__':main()
