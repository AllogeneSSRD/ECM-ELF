"""P3 native planning, owner fallback, complete leaf and check-drain regressions.

Consume existing independently verified Stage1 fixtures. Never use user queues
or alter device settings. Timings are functional output, not performance data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ecm_report'))
from result_format import normalize_result
if hasattr(sys,'set_int_max_str_digits'): sys.set_int_max_str_digits(0)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(value):
    if isinstance(value,dict):
        return {k:stable(v) for k,v in value.items() if k not in {
            'program','source-build','timestamp','checksum','result_checksum','seconds','elapsed','time',
            'free_bytes','arena_cap_bytes'} and not k.endswith('_seconds')}
    if isinstance(value,list): return [stable(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--small-save',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    a.fixtures=a.fixtures.resolve();a.small_save=a.small_save.resolve()
    collector=out/'collector.py';collector.write_bytes(Path(__file__).read_bytes())
    exes={'old':a.baseline.resolve(),'new':a.exe.resolve()}
    inputs={str(v):sha(v) for v in [*exes.values(),a.fixtures,a.small_save]}
    fixtures=json.loads(a.fixtures.read_text())['fixtures']
    for case in fixtures: assert sha(case['save'])==case['save_sha256']
    config=out/'fixture.toml';config.write_text('[stage2]\nstage2_short_calibration=false\nstage2_tune_runtime=false\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    clean.update(NTT_D_MODEL='0',NTT_TARGET_LEAF_HASH='1')
    result=dict(complete=False,inputs=inputs,collector_sha256=sha(collector),device=a.device,pairs=[],plans=[],rejections=[],performance_samples=0)
    def persist(): (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    def invoke(name,side,save,flags=(),environment=None,plan=False,poison=False,level='debug'):
        folder=out/(name+'_'+side);folder.mkdir();record=folder/'result.jsonl';log=folder/'engine.log';debug=folder/'debug.log'
        cmd=list(map(str,[exes[side],'--config',config,'--save',save,'--b2',13230,'--d',210,
            '--device',a.device,'--curves',1,'--carrier-exponent',0,'--short-calibration',0,'--tune-runtime',0,
            '--factor-only','--results',record,'--log',log,'--debug-log-file',debug,'--log-level',level,*flags]))
        if plan: cmd.append('--plan-only')
        env=clean|(environment or {})
        (folder/'invocation.json').write_text(json.dumps(dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith('NTT_')}),indent=2)+'\n')
        proc=subprocess.run(cmd,cwd=folder,env=env,capture_output=True,timeout=300)
        (folder/'stdout.log').write_bytes(proc.stdout+proc.stderr)
        text=(proc.stdout+proc.stderr).decode('utf-8','replace')
        for f in (log,debug):
            if f.exists(): text+='\n'+f.read_text(encoding='utf-8',errors='replace')
        records=[normalize_result(json.loads(s)) for s in record.read_text().splitlines()] if record.exists() else []
        if poison:
            assert proc.returncode==2 and not records,(name,side,'corruption accepted')
            assert re.search(r'FATAL|mismatch|failed|error',text,re.I),(name,side,'missing rejection reason')
            return None,text
        assert proc.returncode==0,(name,side,proc.returncode)
        if plan:
            rows=[json.loads(s) for s in text.splitlines() if s.startswith('{') and '"type":"stage2_plan"' in s]
            assert len(rows)==1 and rows[0]['curves_executed']==0
            assert 'mont_selftest:' not in text and not records
            return stable(rows[0]),text
        assert len(records)==1,(name,side,'result count')
        for token in ('mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0','gmp_selftest_bad=0','gmp_check_bad=0','pending=0'):
            assert token in text,(name,side,token)
        assert not re.search(r'(?:mismatches|gmp_check_bad|gmp_selftest_bad)=[1-9]',text)
        leaves=re.findall(r'target_descent_values: ([^\r\n]+)',text);assert leaves,(name,side,'complete target leaves')
        n=int(records[0]['N_hex'],16)
        assert not records[0]['bad_factors']
        assert all(1<int(f)<n and n%int(f)==0 for f in records[0]['factors'])
        def fields(prefix):
            rows=re.findall(r'^'+prefix+r': (.*)$',text,re.M);assert rows,(name,side,prefix)
            return dict(re.findall(r'(\w+)=(\S+)',rows[-1]))
        ledger,wall=fields('real_batched_wall'),fields('stage2_full_wall')
        assert abs(float(ledger['sum'])-float(wall['main']))<.003,(name,side,'timing ledger closure')
        coverage=fields('s4_multiply_stats')
        coverage={k:coverage[k] for k in ('launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks')}
        return dict(leaves=leaves[-1],fields=stable(records),coverage=coverage),text
    cases=[(c['name'],c['save'],[],{},c) for c in fixtures]
    for name,flags,env,prefix,reason in [
        ('budget_fallback',['--owner-budget-mb','0'],{},'real_batched_folddevice','budget'),
        ('baby_alloc',[],{'NTT_BABY_DEVICE_ALLOC_FAIL':'1'},None,None),
        ('fold_alloc',[],{'NTT_FOLD_DEVICE_ALLOC_FAIL':'1'},'real_batched_folddevice','allocation_fixture'),
        ('frontier_alloc',[],{'NTT_SCALED_FRONTIER_ALLOC_FAIL':'1'},'scaled_frontier_device','allocation_fixture'),
        ('async_ring_one',[],{'NTT_S4_ORACLE_RING':'1'},None,None),
        ('phase_trim',[],{'NTT_PHASE_TRIM_RAW':'1','NTT_OWNER_TRIM_FUSE':'1'},None,None),
        ('unfused_pack',['--pack-zero','0'],{},None,None),
    ]:
        cases.append((name,a.small_save,flags,env,dict(prefix=prefix,reason=reason)))
    fixture_flags={key:'1' for key in ('NTT_XADD6_TEST','NTT_S4_MERSENNE_TEST',
        'NTT_S4_FLAT_TEST','NTT_S4_FINAL_READBACK_TEST','NTT_S4_OUTPUT_WINDOW_TEST','NTT_SCALED_TEST',
        'NTT_S3_PRODUCTS_TEST','NTT_FOLD_FLAT_TEST','NTT_FOLD_DEVICE_TEST','NTT_GFINV_SEG_TEST',
        'NTT_GFINV_BATCH_TEST','NTT_GROOT_DEVICE_TEST','NTT_S4_GROOT_TEST','NTT_ARENA_WORKSPACE_TEST')}
    cases.append(('diagnostic_fixtures',a.small_save,[],fixture_flags,{}))
    generic=next(c for c in fixtures if c['name']=='generic8193')
    cases.append(('baby_diagnostic_fixture',generic['save'],[],{'NTT_BABY_DEVICE_TEST':'1'},generic))
    for name,save,flags,env,case in cases:
        pair=[]
        for side in exes:
            row,text=invoke(name,side,save,flags,env);pair.append(row)
            if case.get('expected_factor'): assert any(int(f)%case['expected_factor']==0 for f in row['fields'][0]['factors'])
            if case.get('unit'):
                hashes=re.findall(r'^descent_values:.*?\bhash=(\S+)',text,re.M)
                assert hashes[-1]==case['expected_leaf_hash']['65'],(name,side,'independent GMP monic leaf oracle')
            if case.get('prefix'):
                fields=dict(re.findall(r'(\w+)=(\S+)',re.findall(r'^'+case['prefix']+r': (.*)$',text,re.M)[-1]))
                assert fields['enabled']=='0' and fields['fallback']==case['reason'],(name,side,fields)
        assert pair[0]==pair[1],(name,'observable fields changed')
        result['pairs'].append(dict(case=name,target_leaves_equal=True,result_fields_equal=True));persist();print(name,'PASS',flush=True)
    for name,flags in [('fixed',[]),('auto',['--d','0']),('carrier',['--carrier-exponent','254']),('relocation',['--relocation','1'])]:
        pair=[invoke('plan_'+name,s,a.small_save,flags,plan=True)[0] for s in exes]
        assert pair[0]==pair[1],('plan_'+name,'geometry/request/memory fields changed')
        result['plans'].append(dict(case=name,fields_equal=True,curves_executed=0));persist()
    for level in ('quiet','curve','phases','batches','debug'):
        for key,check in [('NTT_S4_ORACLE_TEST_BAD',None),('NTT_GSCALE_DEVICE_TEST_BAD','NTT_GSCALE_DEVICE_CHECK'),
                          ('NTT_SCALED_ROOT_TEST_BAD','NTT_SCALED_ROOT_CHECK'),('NTT_SCALED_FRONTIER_TEST_BAD','NTT_SCALED_FRONTIER_CHECK')]:
            env={key:'1'}|(dict([(check,'1')]) if check else {})
            for side in exes: invoke('reject_'+key+'_'+level,side,a.small_save,environment=env,poison=True,level=level)
            result['rejections'].append(dict(check=key,level=level,both_rejected=True));persist()
    # The existing synthetic baby fixture changes N while a Mersenne point
    # constant from the real curve is active. Preserve its baseline rejection;
    # the generic-modulus invocation above verifies the fixture's working scope.
    for side in exes:
        _,text=invoke('baseline_baby_mersenne_limit',side,a.small_save,
            environment={'NTT_BABY_DEVICE_TEST':'1'},poison=True)
        assert 'baby device affine GMP mismatch point=1' in text
    result['baseline_diagnostic_limit']='baby fixture under active Mersenne point constant: same rejection in both builds'
    assert inputs=={str(v):sha(v) for v in [*exes.values(),a.fixtures,a.small_save]}
    result['complete']=True;persist()
    print(json.dumps(dict(complete=True,pairs=len(result['pairs']),plans=len(result['plans']),rejections=len(result['rejections']))))


if __name__=='__main__':main()
