"""Frozen native subtraction qualification and fixed-D full-curve cross-build timing."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HELPER=ROOT/'tools/bench/bench_stage2_production.py'
spec=importlib.util.spec_from_file_location('production_bench',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read,sha,fields=helper.read,helper.sha,helper.fields


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--save',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('gate','timing','timing-wide'),required=True)
    p.add_argument('--gate',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise ValueError('use a fresh output directory')
    out.mkdir(parents=True);exes=[a.baseline.resolve(),a.candidate.resolve()]
    identity=[helper.freeze(exe) for exe in exes]
    for mask,exe in enumerate(exes):
        build=read(exe.parent/'build_manifest.json')
        if (build['engine'],build['gl_fixed_mode'],build['outer_unroll_u'],build.get('add_sub_mask'))!=('development',3,0,mask):raise ValueError('compiled native settings differ')
    ref=read(a.reference);prepared=read(a.fixtures)
    if not ref['complete'] or not prepared['complete']:raise ValueError('incomplete reference')
    expected=next(r for r in ref['runs'] if r['category']=='timing' and r['key']=='candidate')
    saved=a.save.resolve()
    if sha(saved)!=ref['cases'][0]['save_sha256']:raise ValueError('different large saved input')
    big=dict(name='m4423_large',save=str(saved),save_sha256=sha(saved),D=1381380,B2=2011326186870,points=1456028,unit=True,factor=0)
    if a.mode=='gate':
        cases=[big]
        for c in prepared['fixtures']:
            for points in ((2,65,66) if c['unit'] else (66,)):
                cases.append(dict(name=f'{c["name"]}_i{points}',save=c['save'],save_sha256=c['save_sha256'],D=210,
                    B2=210*(points-2),points=points,unit=c['unit'],factor=c['expected_factor'],
                    expected_leaf=c['expected_leaf_hash'].get(str(points)),fixture_name=c['name'],N_hex=c['N_hex'],B1=c['B1']))
                # I=2 needs a positive B2 above B1; floor(B2/D)=0 still gives two points.
                if points==2:cases[-1]['B2']=c['B1']+1
        full=next(c for c in prepared['fixtures'] if c['name']=='generic16384')
        cases.append(dict(name='generic16384_chunk_tail',save=full['save'],save_sha256=full['save_sha256'],D=30030,
            B2=30030*(66305-2),points=66305,unit=True,factor=0,fixture_name=full['name'],N_hex=full['N_hex'],B1=full['B1']))
    elif a.mode=='timing':cases=[big]
    else:
        cases=[dict(name=c['name'],save=c['save'],save_sha256=c['save_sha256'],D=30030,B2=30030*(32768-2),
            points=32768,unit=True,factor=0,N_hex=c['N_hex'],B1=c['B1']) for c in prepared['fixtures'] if c['unit']]
        if [c['name'] for c in cases]!=['generic8193','m16381','generic16384']:raise ValueError('wide matrix changed')
    if a.mode!='gate':
        if not a.gate:raise ValueError('--gate required')
        gate=read(a.gate)
        if not gate['complete'] or gate['mode']!='gate' or gate['identity']!=identity:raise ValueError('native gate identity')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_D_MODEL='0',NTT_GIANT_SEED_PAIR='1',NTT_GIANT_BASE_CPU='0',NTT_GIANT_CHAIN_BLOCK='64',
               NTT_GIANT_CHAIN_MIN='32768',NTT_NO_PROGRESS='1',CUDA_LAUNCH_BLOCKING='0',NTT_OUTER_NARROW='0',
               NTT_FOLD_OWNER_REUSE='3',NTT_GSCALE_DEVICE='1',NTT_SCALED_ROOT_DEVICE='1')
    ini=out/'manual.ini';ini.write_text('device=1\n')
    data=dict(complete=False,identity=identity,mode=a.mode,tool_sha256=sha(__file__),helper_sha256=sha(HELPER),
        reference_sha256=sha(a.reference),fixtures_sha256=sha(a.fixtures),cases=cases,runs=[],environment=env_filter(env))
    if a.gate:data['gate_sha256']=sha(a.gate)
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    def persist():(out/'measurements.json').write_text(json.dumps(data,indent=2)+'\n')
    def verify():
        if sha(__file__)!=data['tool_sha256'] or sha(HELPER)!=data['helper_sha256']:raise ValueError('collector/helper changed')
        for exe,want in zip(exes,identity):
            if sha(exe)!=want['binary_sha256'] or sha(exe.parent/'build_manifest.json')!=want['build_sha256']:raise ValueError('compiled build changed')
            for name,digest in want['sources'].items():
                if sha(exe.parent/'sources'/name)!=digest:raise ValueError('compiled source snapshot changed')
        for c in cases:
            if sha(c['save'])!=c['save_sha256']:raise ValueError('save changed')
    def run(c,mask,category,index):
        verify();name=f'{c["name"]}_{category}_{index}_m{mask}';log=out/(name+'.log');result=out/(name+'.jsonl')
        command=[str(exes[mask]),'--ini',str(ini),'--save',c['save'],'--device','1','--b2',str(c['B2']),'--d',str(c['D']),
            '--arena-mb','6300','--factor-only','--log',str(log),'--results',str(result)]
        use=env.copy()
        if category=='gate' and c['D']==210:use.update(NTT_SCALED_TEST='1',NTT_SCALED_CHECK='1')
        if category=='gate' and c['name'].endswith('chunk_tail'):use.update(NTT_GFINV_SEG_CHECK='1',NTT_GIANT_SEED_CHECK='1',NTT_GIANT_CHAIN_CHECK='1')
        proc=subprocess.run(command,env=use,capture_output=True,timeout=900)
        driver=out/(name+'_driver.log');driver.write_bytes(proc.stdout+proc.stderr)
        if proc.returncode:raise ValueError(name+' nonzero exit; retain raw evidence')
        text=log.read_text();records=[json.loads(s) for s in result.read_text().splitlines()]
        if len(records)!=1:raise ValueError('result count')
        record=records[0];leaf=fields(text,'descent_values');coverage=fields(text,'s4_multiply_stats');wall=fields(text,'stage2_full_wall')
        if (record['B2'],record['requested_D'],record['device'],record['bad_factors'])!=(c['B2'],c['D'],1,0):raise ValueError('actual input mismatch')
        if c.get('N_hex') and (record['N_hex'].lower()!=c['N_hex'] or record['B1']!=c['B1'] or record['sigma']!=26):raise ValueError('wide input mismatch')
        n=int(record['N_hex'],16)
        if any(not 1<int(f)<n or n%int(f) for f in record['factors']):raise ValueError('improper factor')
        if c['factor'] and not any(int(f)%c['factor']==0 for f in record['factors']):raise ValueError('known nonunit factor missing')
        if c.get('expected_leaf') and leaf['hash']!=c['expected_leaf']:raise ValueError('independent complete monic evaluation mismatch')
        for token in ['stage1_skipped=1','mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0',
            'gmp_selftest_bad=0','gmp_check_bad=0','pending=0',f'ntt_addsub_arithmetic: mask={mask}','ntt_outer_offsets: narrow_mask=0']:
            if token not in text:raise ValueError('required check missing: '+token)
        dmodel=fields(text,'d_model')
        if dmodel['enabled']!='0' or dmodel['requested']!='0':raise ValueError('fixed-D cost policy changed')
        if c['name']=='m4423_large':
            if leaf!=expected['leaf']:raise ValueError('complete large leaf differs from production reference')
            for key in ['N_hex','B1','B2','sigma','requested_D','device','factors','bad_factors']:
                if record[key]!=expected['result'][key]:raise ValueError('large reference input/result differs')
            for key in ['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']:
                if coverage[key]!=expected['coverage'][key]:raise ValueError('large arithmetic coverage changed')
        ledger=fields(text,'real_batched_wall');root=fields(text,'scaled_root_device')
        if abs(float(ledger['sum'])-float(wall['main']))>.003:raise ValueError('precise main ledger does not close')
        if root['requested']!='1' or root['checked_words']!='0' or root['check_d2h_bytes']!='0':raise ValueError('root/check policy differs')
        scaled=fields(text,'scaled_descent')
        if category=='gate' and c['D']==210:
            if 'scaled_fixture: cases=150' not in text or 'bad=0' not in fields_line(text,'scaled_fixture'):raise ValueError('GMP scaled fixtures missing')
            if scaled['checked_states']!=scaled['states'] or scaled['checked_words']!=scaled['words']:raise ValueError('complete independent GMP nodes missing')
        chain={}
        if category=='gate' and c['name'].endswith('chunk_tail'):
            w=(n.bit_length()+63)//64;p=2880;chunk=p*((max(p,(256<<20)//(16*w))+p-1)//p)
            pieces=[min(chunk,c['points']-lo) for lo in range(0,c['points'],chunk)];points=sum(v for v in pieces if v>=32768)
            r=[dict(re.findall(r'(\w+)=(\S+)',s)) for s in re.findall(r'^giant_chain_check: (.*)$',text,re.M)]
            if sum(int(t['points']) for t in r)!=points or any(int(t['mismatches']) for t in r):raise ValueError('full chain affine check missing')
            chain=dict(checked_chain_points=points,ladder_tail_points=c['points']-points)
        row=dict(name=name,case=c['name'],mask=mask,category=category,command=command,exit=proc.returncode,environment=env_filter(use),
            wall=wall,leaf=leaf,result=record,coverage=coverage,ledger=ledger,root=root,scaled=scaled,chain=chain,
            phases=fields(text,'real_batched_split'),log_sha256=sha(log),result_sha256=sha(result),driver_sha256=sha(driver))
        data['runs'].append(row);persist();verify();print(name,wall['total'],flush=True)
    persist()
    for c in cases:
        if a.mode=='gate':
            for index,mask in enumerate([0,1]):run(c,mask,'gate',index)
        else:
            for mask in [0,1]:run(c,mask,'warmup',mask)
            for index,mask in enumerate([0,1,1,0,1,0,0,1]):run(c,mask,'timing',index)
        selected=[r for r in data['runs'] if r['case']==c['name']]
        if len({r['leaf']['hash'] for r in selected})!=1 or len({tuple(r['result']['factors']) for r in selected})!=1:raise ValueError('cross-build output mismatch')
        for key in ['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']:
            if len({r['coverage'][key] for r in selected})!=1:raise ValueError('cross-build default coverage mismatch')
    if a.mode!='gate':
        data['summary']={}
        for c in cases:
            selected=[r for r in data['runs'] if r['case']==c['name'] and r['category']=='timing']
            means={str(mask):statistics.mean(float(r['wall']['total']) for r in selected if r['mask']==mask) for mask in [0,1]};groups=[]
            for start in [0,4]:
                group=selected[start:start+4];m={mask:statistics.mean(float(r['wall']['total']) for r in group if r['mask']==mask) for mask in [0,1]};groups.append(100*(1-m[1]/m[0]))
            data['summary'][c['name']]=dict(full_mean_seconds=means,reduction_percent=100*(1-means['1']/means['0']),groups_reduction_percent=groups)
    verify();data['complete']=True;persist();print(json.dumps(data.get('summary',dict(passed=len(data['runs'])))))


def env_filter(env):return {k:v for k,v in env.items() if k.startswith('NTT_') or k=='CUDA_LAUNCH_BLOCKING'}
def fields_line(text,prefix):return re.findall(r'^'+re.escape(prefix)+r': (.*)$',text,re.M)[-1]


if __name__=='__main__':main()
