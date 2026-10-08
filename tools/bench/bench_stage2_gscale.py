"""Same-binary resident Gamma/root preparation gates and ABBA+BAAB timing.

Gates compare the entire corrected H with GMP and known monic leaf fingerprints,
including owner fallback, G1, nonunits and real device corruption rejection.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from bench_stage2_production import freeze, fields, read, sha

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=('gamma','scaled-root'),default='gamma')
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--save',type=Path)
    p.add_argument('--mode',choices=('gate','gate-large','gate-wide','gate-layout','primitive','timing','timing-wide'),required=True)
    p.add_argument('--gmp-ecm',type=Path,default=Path('D:/code/GIMPS/gmp-ecm/ecm-7.0.5-znver3/ecm.exe'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();exe=a.exe.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('use a fresh output directory')
    if a.target=='scaled-root' and a.mode=='primitive':p.error('primitive is Gamma-only')
    identity=freeze(exe);tool_sha=sha(__file__);fixture_sha=sha(a.fixtures)
    helper_path=ROOT/'tools/bench/bench_stage2_production.py';helper_sha=sha(helper_path)
    prepared=read(a.fixtures)
    if not prepared['complete']:raise ValueError('independent fixture generation incomplete')
    if read(exe.parent/'build_manifest.json')['engine']!='development':raise ValueError('same-binary switch requires development')
    cases=[dict(c,D=210,B2=210*63,points=65) for c in prepared['fixtures']]
    primitive_identity=None
    if a.mode=='primitive':
        ref_path=ROOT/'tools/stat/suyama_mont_ref.py'
        spec=importlib.util.spec_from_file_location('gscale_ref',ref_path)
        reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        primitive_identity=dict(reference_sha256=sha(ref_path),gmp_sha256=sha(a.gmp_ecm))
        cases=[]
        for bits,offset in ((61,1),(509,3),(521,1),(1279,1),(2203,1),(8191,1)):
            for selected_offset in (range(3,100,2) if bits==509 else (offset,)):
                n=(1<<bits)-selected_offset;expr=f'2^{bits}-{selected_offset}'
                try:q=reference.stage1(26,20,n)
                except ValueError:continue
                if q['gcd']==1 and q['x'] and math.gcd(q['x'],n)==1:break
            else:raise ValueError('no normalizable primitive input')
            if q['gcd']!=1 or not q['x'] or math.gcd(q['x'],n)!=1:raise ValueError('primitive save not normalizable')
            saved=out/f'bits{bits}.save';gmp_save=out/f'bits{bits}_gmp.save'
            proc=subprocess.run([str(a.gmp_ecm),'-param','0','-sigma','26','-c','1','-save',str(gmp_save),'20','20'],input=(expr+'\n').encode(),capture_output=True,timeout=120)
            (out/f'bits{bits}_gmp.log').write_bytes(proc.stdout+proc.stderr)
            import re
            match=re.search(r'\bX=(0x[0-9a-fA-F]+)',gmp_save.read_text())
            if proc.returncode or not match or int(match[1],16)!=q['x']:raise ValueError('primitive CPU/GMP Stage1 mismatch')
            saved.write_text(f'METHOD=ECM; PARAM=0; SIGMA=26; B1=20; N={expr}; X=0x{q["x"]:x}; CHECKSUM={20*26*n*q["x"]%4294967291};\n')
            cases.append(dict(name=f'bits{bits}',N_hex=f'{n:x}',B1=20,save=str(saved),save_sha256=sha(saved),D=210,B2=13230,points=65,unit=False,expected_factor=0,gmp_save_sha256=sha(gmp_save)))
    if a.mode in ('timing','gate-large'):
        if not a.save:p.error('--save required')
        cases=[dict(name='m4423_large',save=str(a.save.resolve()),save_sha256=sha(a.save),D=1381380,
                    B2=2011326186870,points=1456028,unit=True,expected_factor=0)]
    elif a.mode in ('timing-wide','gate-wide'):
        cases=[dict(c,D=30030,B2=30030*32766,points=32768) for c in prepared['fixtures'] if c['unit']]
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_D_MODEL='0',NTT_GIANT_SEED_PAIR='1',NTT_GIANT_BASE_CPU='0',
               NTT_GIANT_CHAIN_BLOCK='64',NTT_GIANT_CHAIN_MIN='32768',NTT_FOLD_OWNER_REUSE='3',
               NTT_FOLD_DEVICE_MAX_MB='640',NTT_NO_PROGRESS='1',CUDA_LAUNCH_BLOCKING='0')
    ini=out/'manual.ini';ini.write_text('device=1\n')
    data=dict(identity=identity,tool_sha256=tool_sha,helper_sha256=helper_sha,fixture_sha256=fixture_sha,primitive_identity=primitive_identity,mode=a.mode,target=a.target,cases=cases,runs=[],complete=False)
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    (out/'helper.py').write_bytes(helper_path.read_bytes())
    def verify():
        if sha(__file__)!=tool_sha or sha(helper_path)!=helper_sha or sha(a.fixtures)!=fixture_sha or freeze(exe)!=identity:raise ValueError('identity changed')
        for name,want in identity['sources'].items():
            if sha(exe.parent/'sources'/name)!=want:raise ValueError('compiled snapshot changed: '+name)
        for c in cases:
            if sha(c['save'])!=c['save_sha256']:raise ValueError('save changed')
            if a.mode=='primitive' and sha(Path(c['save']).with_name(c['name']+'_gmp.save'))!=c['gmp_save_sha256']:raise ValueError('GMP primitive save changed')
        if primitive_identity and (sha(ref_path)!=primitive_identity['reference_sha256'] or sha(a.gmp_ecm)!=primitive_identity['gmp_sha256']):raise ValueError('primitive reference changed')
    def persist():(out/'measurements.json').write_text(json.dumps(data,indent=2)+'\n')
    def run(c,name,mode,category='gate',extra=None,reject=None):
        verify();log=out/(name+'.log');result=out/(name+'.jsonl')
        use=env|{'NTT_GSCALE_DEVICE':str(mode)}|(extra or {})
        if a.target=='scaled-root':
            use=env|{'NTT_GSCALE_DEVICE':'1','NTT_SCALED_ROOT_DEVICE':str(mode)}|(extra or {})
        cmd=[str(exe),'--ini',str(ini),'--save',c['save'],'--b2',str(c['B2']),'--d',str(c['D']),
             '--device','1','--arena-mb','6300','--results',str(result),'--log',str(log),'--factor-only']
        proc=subprocess.run(cmd,env=use,capture_output=True,timeout=900)
        driver=out/(name+'_driver.log');driver.write_bytes(proc.stdout+proc.stderr)
        verify();text=log.read_text(encoding='utf-8') if log.exists() else ''
        if reject:
            if proc.returncode==0 or reject not in text+proc.stdout.decode('utf-8','replace')+proc.stderr.decode('utf-8','replace') or result.exists():
                raise ValueError('corruption/configuration not rejected: '+name)
            row=dict(name=name,category='rejection',exit=proc.returncode,expected=reject)
        else:
            if proc.returncode:raise ValueError(name+' failed, inspect driver')
            records=[json.loads(s) for s in result.read_text(encoding='utf-8').splitlines()]
            if len(records)!=1:raise ValueError('record count')
            r=records[0];n=int(r['N_hex'],16)
            if r['bad_factors'] or r['B2']!=c['B2'] or r['requested_D']!=c['D'] or r['device']!=1:raise ValueError('input mismatch')
            if any(not 1<int(f)<n or n%int(f) for f in r['factors']):raise ValueError('invalid factor')
            if c.get('N_hex') and (r['N_hex'].lower()!=c['N_hex'] or r['B1']!=c['B1'] or r['sigma']!=26):raise ValueError('saved input mismatch')
            if c.get('expected_factor') and not any(int(f)%c['expected_factor']==0 for f in r['factors']):raise ValueError('nonunit factor missing')
            for token in ('mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0','gmp_selftest_bad=0','gmp_check_bad=0','pending=0'):
                if token not in text:raise ValueError('mandatory check missing: '+token)
            leaf=fields(text,'descent_values');scale=fields(text,'real_gscale_device')
            if c.get('unit') and c.get('expected_leaf_hash') and c['points'] in (2,65,66) and leaf['hash']!=c['expected_leaf_hash'][str(c['points'])]:raise ValueError('independent monic leaf mismatch')
            root=fields(text,'scaled_root_device') if a.target=='scaled-root' else {}
            if category in ('timing','warmup'):
                if root:
                    if root['enabled']!=str(mode) or root['checked_words']!='0' or root['check_d2h_bytes']!='0':raise ValueError('formal root path or coverage wrong')
                    if mode and (root['h2d_bytes']!='24' or int(root['d2h_bytes'])!=8*int(root['coefficients'])*((n.bit_length()+63)//64)):raise ValueError('root transfers wrong')
                if scale['enabled']!=str(mode if a.target=='gamma' else 1) or scale['check_d2h_bytes']!='0' or scale['checked_words']!='0':raise ValueError('formal device path or diagnostic coverage wrong')
                if (mode or a.target=='scaled-root') and int(scale['h2d_bytes'])!=8*((n.bit_length()+63)//64):raise ValueError('scalar upload bytes wrong')
            if mode and use.get('NTT_GSCALE_DEVICE_CHECK')=='1' and scale['enabled']=='1':
                if int(scale['checked_words'])!=int(scale['coefficients'])*((n.bit_length()+63)//64):raise ValueError('corrected H check incomplete')
            if use.get('NTT_GSCALE_DEVICE_TEST')=='1':
                fixture=fields(text,'gscale_device_fixture')
                if int(fixture['cases'])!=12 or int(fixture['words'])!=528*((n.bit_length()+63)//64) or fixture['bad']!='0':raise ValueError('primitive coverage missing')
            if root and root['enabled']=='1' and use.get('NTT_SCALED_ROOT_CHECK')=='1':
                if int(root['checked_words'])!=int(root['coefficients'])*((n.bit_length()+63)//64):raise ValueError('root check incomplete')
            if use.get('NTT_SCALED_CHECK')=='1':
                scaled=fields(text,'scaled_descent')
                if scaled['checked_words']!=scaled['words'] or scaled['checked_states']!=scaled['states']:raise ValueError('GMP scaled states incomplete')
            row=dict(name=name,case=c['name'],category=category,mode=mode,result=r,leaf=leaf,scale=scale,root=root,
                     wall=fields(text,'stage2_full_wall'),phases=fields(text,'real_batched_split'),coverage=fields(text,'s4_multiply_stats'),
                     owner=fields(text,'real_batched_folddevice'),result_sha256=sha(result),log_sha256=sha(log))
        row.update(command=cmd,environment={k:v for k,v in use.items() if k.startswith('NTT_') or k=='CUDA_LAUNCH_BLOCKING'},driver_sha256=sha(driver))
        data['runs'].append(row);persist();print(name,row.get('wall',{}).get('total','passed'),flush=True)
    if a.mode=='gate-layout':cases[:]=[c for c in cases if c['name']=='generic16384']
    persist()
    checked={'NTT_GSCALE_DEVICE_CHECK':'1'} if a.target=='gamma' else {'NTT_SCALED_ROOT_CHECK':'1'}
    small_checked=checked if a.target=='gamma' else checked|{'NTT_SCALED_CHECK':'1'}
    if a.mode=='gate-layout':
        if a.target!='scaled-root':p.error('gate-layout requires scaled-root')
        c=cases[0];run(c,c['name']+'_cpu',0)
        for reuse in (0,1,2,3):run(c,c['name']+'_reuse'+str(reuse),1,extra=small_checked|{'NTT_FOLD_OWNER_REUSE':str(reuse)})
    elif a.mode in ('gate-large','gate-wide'):
        for c in cases:
            run(c,c['name']+'_cpu',0)
            run(c,c['name']+'_device',1,extra=checked)
    elif a.mode=='primitive':
        for c in cases:
            for point in (0,1):run(c,c['name']+'_point'+str(point),1,extra={'NTT_POINT_MERSENNE':str(point),'NTT_GSCALE_DEVICE_TEST':'1','NTT_GSCALE_DEVICE_CHECK':'1'})
    elif a.mode=='gate':
        for c in cases:
            run(c,c['name']+'_cpu',0)
            run(c,c['name']+'_device',1,extra=small_checked|({'NTT_GSCALE_DEVICE_TEST':'1'} if a.target=='gamma' else {}))
        c=next(c for c in cases if c['name']=='generic16384')
        for count in (2,66):
            short=dict(c,name='short'+str(count),B2=c['B1']+1 if count==2 else 210*(count-2),points=count)
            for m in (0,1):run(short,short['name']+'_'+str(m),m,extra=small_checked)
        for name,extra in [('budget',{'NTT_FOLD_DEVICE_MAX_MB':'0'}),('allocation',{'NTT_FOLD_DEVICE_ALLOC_FAIL':'1'})]:
            run(dict(c,name=name),name+'_cpu',0,extra=extra)
            run(dict(c,name=name),name+'_device',1,extra=extra|small_checked)
        if a.target=='gamma':
            run(c,'device_corruption',1,extra={'NTT_GSCALE_DEVICE_CHECK':'1','NTT_GSCALE_DEVICE_TEST_BAD':'1'},reject='Gamma device GMP mismatch')
            run(c,'poison_without_check',1,extra={'NTT_GSCALE_DEVICE_TEST_BAD':'1'},reject='Gamma poison requires full GMP check')
            run(c,'invalid_flag',2,reject='NTT_GSCALE_DEVICE must be 0 or 1')
        else:
            run(c,'device_corruption',1,extra=checked|{'NTT_SCALED_ROOT_TEST_BAD':'1'},reject='scaled root device mismatch')
            run(c,'poison_without_check',1,extra={'NTT_SCALED_ROOT_TEST_BAD':'1'},reject='scaled root poison requires full root check')
            run(c,'invalid_flag',2,reject='NTT_SCALED_ROOT_DEVICE must be 0 or 1')
    else:
        for c in cases:
            for i,m in enumerate((0,1)):run(c,c['name']+'_warm_'+str(i),m,'warmup')
            for i,m in enumerate((0,1,1,0,1,0,0,1)):run(c,c['name']+'_timing_'+str(i),m,'timing')
    groups={}
    for row in data['runs']:
        if row['category']=='rejection':continue
        groups.setdefault(row['case'],[]).append(row)
    for name,rows in groups.items():
        if len({r['leaf']['hash'] for r in rows})!=1 or len({tuple(sorted(r['result']['factors'])) for r in rows})!=1:raise ValueError('full leaf/factor mismatch: '+name)
        if a.target=='scaled-root' and a.mode in ('gate','gate-large','gate-wide','gate-layout'):
            baseline=next(r for r in rows if r['mode']==0)
            for r in rows:
                extra=int(r['root'].get('enabled')=='1' and r['environment'].get('NTT_SCALED_ROOT_CHECK')=='1')
                for k,delta in (('launches',extra),('poly_muls',extra),('coeffs_reduced',extra*int(r['root']['coefficients'])),('gmp_selftest_cases',0)):
                    if int(r['coverage'][k])!=int(baseline['coverage'][k])+delta:raise ValueError('diagnostic root coverage differs: '+k)
        for k in ('launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks'):
            if a.target=='scaled-root' and a.mode in ('gate','gate-large','gate-wide','gate-layout'):continue # exact extra multiplication accounted above
            if len({r['coverage'][k] for r in rows})!=1:raise ValueError('NTT/S4 coverage differs: '+k)
    if a.mode not in ('gate','gate-large','gate-wide','gate-layout','primitive'):
        data['summary']={}
        for name,rows in groups.items():
            means={str(m):statistics.mean(float(r['wall']['total']) for r in rows if r['mode']==m and r['category']=='timing') for m in (0,1)}
            timed=[r for r in rows if r['category']=='timing']
            if [r['mode'] for r in timed]!=[0,1,1,0,1,0,0,1]:raise ValueError('formal sequence changed')
            group_reductions=[]
            for start in (0,4):
                sub=timed[start:start+4]
                gm={m:statistics.mean(float(r['wall']['total']) for r in sub if r['mode']==m) for m in (0,1)}
                group_reductions.append(100*(1-gm[1]/gm[0]))
            data['summary'][name]=dict(full_mean_seconds=means,reduction_percent=100*(1-means['1']/means['0']),groups_reduction_percent=group_reductions)
    verify();data.update(complete=True,passed=len(data['runs']),failed=0);persist()
    print(json.dumps(data.get('summary',{'passed':data['passed']})),flush=True)

if __name__=='__main__':main()
