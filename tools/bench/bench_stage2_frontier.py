"""Same-binary resident descent gates and fixed-D ABBA/BAAB full-curve timings."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parents[2]
HELPER=ROOT/'tools/bench/bench_stage2_production.py'
spec=importlib.util.spec_from_file_location('production_bench',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read,sha,fields=helper.read,helper.sha,helper.fields


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--save',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',type=Path)
    p.add_argument('--mode',choices=('gate','controls','timing','timing-wide'),required=True)
    a=p.parse_args();out=a.output.resolve();exe=a.exe.resolve()
    if out.exists():raise ValueError('use a fresh output directory')
    out.mkdir(parents=True);identity=helper.freeze(exe);build=read(exe.parent/'build_manifest.json')
    if (build['engine'],build['gl_fixed_mode'],build['outer_unroll_u'],build['add_sub_mask'])!=('development',3,0,1):raise ValueError('wrong compiled engine/arithmetic')
    reference=read(a.reference);fixtures=read(a.fixtures)
    if not reference['complete'] or not fixtures['complete']:raise ValueError('unfinished reference')
    expected=next(r for r in reference['runs'] if r['category']=='timing' and r['mask']==1)
    save=a.save.resolve()
    if sha(save)!=reference['cases'][0]['save_sha256']:raise ValueError('large saved input differs')
    big=dict(name='m4423_large',save=str(save),save_sha256=sha(save),D=1381380,B2=2011326186870,points=1456028,factor=0)
    small=[]
    for c in fixtures['fixtures']:
        for points in ((2,65,66) if c['unit'] else (66,)):
            small.append(dict(name=f'{c["name"]}_i{points}',save=c['save'],save_sha256=c['save_sha256'],D=210,
                B2=c['B1']+1 if points==2 else 210*(points-2),points=points,factor=c['expected_factor'],
                expected_leaf=c['expected_leaf_hash'].get(str(points)),N_hex=c['N_hex'],B1=c['B1'],resident_expected=points>24))
    full=next(c for c in fixtures['fixtures'] if c['name']=='generic16384')
    tail=dict(name='generic16384_chunk_tail',save=full['save'],save_sha256=full['save_sha256'],D=30030,
        B2=30030*(66305-2),points=66305,factor=0,N_hex=full['N_hex'],B1=full['B1'])
    if a.mode=='gate':cases=[big]+small+[tail]
    elif a.mode=='controls':cases=[next(c for c in small if c['name']=='generic16384_i66'),big]
    elif a.mode=='timing':cases=[big]
    else:cases=[dict(name=c['name'],save=c['save'],save_sha256=c['save_sha256'],D=30030,B2=30030*(32768-2),points=32768,factor=0,N_hex=c['N_hex'],B1=c['B1']) for c in fixtures['fixtures'] if c['unit']]
    if a.mode.startswith('timing'):
        if not a.gate:raise ValueError('--gate required')
        gate=read(a.gate)
        if not gate['complete'] or gate['mode']!='gate' or gate['identity']!=identity:raise ValueError('gate identity differs')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_D_MODEL='0',NTT_GIANT_SEED_PAIR='1',NTT_GIANT_BASE_CPU='0',NTT_GIANT_CHAIN_BLOCK='64',NTT_GIANT_CHAIN_MIN='32768',
        NTT_NO_PROGRESS='1',CUDA_LAUNCH_BLOCKING='0',NTT_OUTER_NARROW='0',NTT_FOLD_OWNER_REUSE='3',NTT_GSCALE_DEVICE='1',NTT_SCALED_ROOT_DEVICE='1')
    ini=out/'manual.ini';ini.write_text('device=1\n')
    data=dict(complete=False,identity=identity,mode=a.mode,tool_sha256=sha(__file__),helper_sha256=sha(HELPER),reference_sha256=sha(a.reference),fixtures_sha256=sha(a.fixtures),cases=cases,runs=[])
    if a.gate:data['gate_sha256']=sha(a.gate)
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    def persist():(out/'measurements.json').write_text(json.dumps(data,indent=2)+'\n')
    def verify():
        if sha(__file__)!=data['tool_sha256'] or sha(HELPER)!=data['helper_sha256'] or helper.freeze(exe)!=identity:raise ValueError('frozen experiment changed')
        for c in cases:
            if sha(c['save'])!=c['save_sha256']:raise ValueError('saved input changed')
    def run(c,mode,category,index,extra=None,want_enabled=None,want_fallback=None,exit_code=0,token=None):
        verify();name=f'{c["name"]}_{category}_{index}_f{mode}'
        use=env.copy();use['NTT_SCALED_FRONTIER_DEVICE']=str(mode)
        if category=='gate' and c['D']==210:use['NTT_SCALED_CHECK']='1'
        if category=='gate' and c['name'].endswith('chunk_tail'):use.update(NTT_GFINV_SEG_CHECK='1',NTT_GIANT_SEED_CHECK='1',NTT_GIANT_CHAIN_CHECK='1')
        if extra:use.update(extra)
        log=out/(name+'.log');result=out/(name+'.jsonl');driver=out/(name+'_driver.log')
        command=[str(exe),'--ini',str(ini),'--save',c['save'],'--device','1','--b2',str(c['B2']),'--d',str(c['D']),'--arena-mb','6300','--factor-only','--log',str(log),'--results',str(result)]
        proc=subprocess.run(command,env=use,capture_output=True,timeout=900);driver.write_bytes(proc.stdout+proc.stderr)
        text=log.read_text() if log.exists() else '';combined=text+driver.read_text(errors='replace')
        if proc.returncode!=exit_code:raise ValueError(name+' unexpected exit '+str(proc.returncode))
        row=dict(name=name,case=c['name'],frontier=mode,category=category,command=command,environment={k:v for k,v in use.items() if k.startswith('NTT_') or k=='CUDA_LAUNCH_BLOCKING'},exit=proc.returncode,driver_sha256=sha(driver))
        if token:
            if token not in combined:raise ValueError('missing expected token '+token)
        if exit_code:
            row['token']=token
        else:
            records=[json.loads(s) for s in result.read_text().splitlines()]
            if len(records)!=1 or 'stage2_complete: curves=1' not in driver.read_text():raise ValueError('incomplete native result')
            record=records[0];n=int(record['N_hex'],16)
            if (record['B2'],record['requested_D'],record['device'],record['bad_factors'])!=(c['B2'],c['D'],1,0):raise ValueError('wrong native input')
            if c.get('N_hex') and (record['N_hex'].lower()!=c['N_hex'] or record['B1']!=c['B1'] or record['sigma']!=26):raise ValueError('wrong wide saved point')
            if any(not 1<int(f)<n or n%int(f) for f in record['factors']):raise ValueError('improper factor')
            if c['factor'] and not any(int(f)%c['factor']==0 for f in record['factors']):raise ValueError('known factor missing')
            for t in ['ntt_addsub_arithmetic: mask=1','ntt_outer_offsets: narrow_mask=0','stage1_skipped=1','mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0','gmp_selftest_bad=0','gmp_check_bad=0','pending=0']:
                if t not in text:raise ValueError('missing check '+t)
            for key,prefix in [('wall','stage2_full_wall'),('leaf','descent_values'),('coverage','s4_multiply_stats'),('ledger','real_batched_wall'),('root','scaled_root_device'),('scaled','scaled_descent'),('frontier_stats','scaled_frontier_device')]:row[key]=fields(text,prefix)
            f=row['frontier_stats'];enabled=mode if want_enabled is None else want_enabled
            if int(f['requested'])!=mode or int(f['enabled'])!=enabled:raise ValueError('frontier admission differs')
            if want_fallback and f['fallback']!=want_fallback:raise ValueError('frontier fallback differs')
            if enabled:
                p=int(row['root']['coefficients']);w=(n.bit_length()+63)//64
                if int(f['metadata_bytes'])!=24*p or int(f['leaf_d2h_bytes'])!=8*p*w:raise ValueError('frontier component accounting')
                if category in ('timing','warmup') and (f['check_d2h_bytes']!='0' or row['root']['d2h_bytes']!='0'):raise ValueError('extra diagnostic/root copies in timing')
            if abs(float(row['ledger']['sum'])-float(row['wall']['main']))>.003:raise ValueError('main ledger does not close')
            if c.get('expected_leaf') and row['leaf']['hash']!=c['expected_leaf']:raise ValueError('independent monic leaf differs')
            if c['name']=='m4423_large':
                if row['leaf']!=expected['leaf']:raise ValueError('original complete large leaf differs')
                for key in ['N_hex','B1','B2','sigma','requested_D','device','factors','bad_factors']:
                    if record[key]!=expected['result'][key]:raise ValueError('original input/result differs')
                for key in ['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']:
                    if row['coverage'][key]!=expected['coverage'][key]:raise ValueError('large default coverage differs')
            if use.get('NTT_SCALED_CHECK')=='1' or use.get('NTT_SCALED_FRONTIER_CHECK')=='1':
                if row['scaled']['checked_states']!=row['scaled']['states'] or row['scaled']['checked_words']!=row['scaled']['words']:raise ValueError('incomplete independent GMP nodes')
            if c['name'].endswith('chunk_tail') and category=='gate':
                if 'giant_chain_check:' not in text or 'mismatches=0' not in text:raise ValueError('missing full chain check')
            row.update(result=record,log_sha256=sha(log),result_sha256=sha(result))
        data['runs'].append(row);persist();verify();print(name,row.get('wall',{}).get('total',exit_code),flush=True)
    persist()
    for c in cases:
        if a.mode=='controls':
            if c['name']=='m4423_large':
                checks=[('budget476',dict(NTT_SCALED_FRONTIER_MAX_MB='476'),0,'budget',0,None),
                    ('budget477',dict(NTT_SCALED_FRONTIER_MAX_MB='477'),1,None,0,None)]
            else:
                checks=[('fixtures',dict(NTT_SCALED_TEST='1',NTT_SCALED_FRONTIER_TEST='1',NTT_SCALED_FRONTIER_CHECK='1'),1,None,0,'scaled_frontier_fixture: cases=150 bad=0'),
                ('pageable',dict(NTT_S4_ASYNC='0',NTT_SCALED_FRONTIER_CHECK='1'),1,None,0,None),
                ('full_window',dict(NTT_S4_OUTPUT_WINDOW='0',NTT_SCALED_FRONTIER_CHECK='1'),1,None,0,None),
                ('budget',dict(NTT_SCALED_FRONTIER_MAX_MB='0'),0,'budget',0,None),
                ('allocation',dict(NTT_SCALED_FRONTIER_ALLOC_FAIL='1'),0,'allocation_fixture',0,None),
                ('owner_off',dict(NTT_FOLD_DEVICE_MAX_MB='0'),0,'root_unavailable',0,None),
                ('root_off',dict(NTT_SCALED_ROOT_DEVICE='0'),0,'root_unavailable',0,None),
                ('poison',dict(NTT_SCALED_FRONTIER_CHECK='1',NTT_SCALED_FRONTIER_TEST_BAD='1'),1,None,2,'scaled frontier GMP node mismatch'),
                ('unchecked_poison',dict(NTT_SCALED_FRONTIER_TEST_BAD='1'),0,None,2,'scaled frontier poison requires GMP check'),
                ('invalid_flag',dict(NTT_SCALED_FRONTIER_DEVICE='2'),0,None,2,'NTT_SCALED_FRONTIER_DEVICE must be 0 or 1')]
                checks.extend((f'reuse{mask}',dict(NTT_FOLD_OWNER_REUSE=str(mask),NTT_SCALED_FRONTIER_CHECK='1'),1,None,0,None) for mask in [0,1,2])
            for name,extra,enabled,fallback,code,token in checks:run(c,1,'control_'+name,0,extra,enabled,fallback,code,token)
        elif a.mode=='gate':
            for mode in [0,1]:
                enabled=mode if c.get('resident_expected',True) else 0
                run(c,mode,'gate',mode,want_enabled=enabled,want_fallback='root_unavailable' if mode and not enabled else None)
        else:
            for mode in [0,1]:run(c,mode,'warmup',mode)
            for index,mode in enumerate([0,1,1,0,1,0,0,1]):run(c,mode,'timing',index)
        valid=[r for r in data['runs'] if r['case']==c['name'] and r['exit']==0]
        if len({r['leaf']['hash'] for r in valid})!=1 or len({tuple(r['result']['factors']) for r in valid})!=1:raise ValueError('complete output differs')
        if a.mode!='controls':
            for key in ['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']:
                if len({r['coverage'][key] for r in valid})!=1:raise ValueError('required coverage differs')
    if a.mode.startswith('timing'):
        data['summary']={}
        for c in cases:
            selected=[r for r in data['runs'] if r['case']==c['name'] and r['category']=='timing']
            means={str(m):statistics.mean(float(r['wall']['total']) for r in selected if r['frontier']==m) for m in [0,1]};groups=[]
            for start in [0,4]:
                group=selected[start:start+4];v={m:statistics.mean(float(r['wall']['total']) for r in group if r['frontier']==m) for m in [0,1]};groups.append(100*(1-v[1]/v[0]))
            data['summary'][c['name']]=dict(full_mean_seconds=means,reduction_percent=100*(1-means['1']/means['0']),groups_reduction_percent=groups)
    verify();data['complete']=True;persist();print(json.dumps(data.get('summary',dict(passed=len(data['runs'])))))


if __name__=='__main__':main()
