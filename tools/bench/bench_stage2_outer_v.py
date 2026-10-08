"""Fixed-input same-binary Stage2 matrix for the two cooperative offset widths."""
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
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True,help='Completed original production timing matrix')
    p.add_argument('--save',type=Path,required=True)
    p.add_argument('--mode',choices=('gate','timing'),required=True)
    p.add_argument('--fixtures',type=Path)
    p.add_argument('--gate',type=Path)
    a=p.parse_args();exe=a.exe.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('use a fresh output directory')
    out.mkdir(parents=True)
    identity=helper.freeze(exe);build=read(exe.parent/'build_manifest.json')
    if build['engine']!='development' or build['gl_fixed_mode']!=3 or build['outer_unroll_u']!=0:
        raise ValueError('require development PTX3/original unroll')
    ref=read(a.reference)
    if not ref['complete']:raise ValueError('incomplete formal reference')
    expected=next(r for r in ref['runs'] if r['category']=='timing' and r['key']=='candidate')
    save=a.save.resolve();save_sha=sha(save)
    if save_sha!=ref['cases'][0]['save_sha256']:raise ValueError('different large saved input')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_D_MODEL='0',NTT_GIANT_SEED_PAIR='1',NTT_GIANT_BASE_CPU='0',NTT_GIANT_CHAIN_BLOCK='64',
               NTT_GIANT_CHAIN_MIN='32768',NTT_NO_PROGRESS='1',CUDA_LAUNCH_BLOCKING='0',
               NTT_FOLD_OWNER_REUSE='3',NTT_GSCALE_DEVICE='1',NTT_SCALED_ROOT_DEVICE='1')
    ini=out/'manual.ini';ini.write_text('device=1\n')
    cases=[dict(name='m4423_large',save=str(save),save_sha256=save_sha,D=1381380,B2=2011326186870,unit=True,factor=0)]
    if a.mode=='gate':
        if not a.fixtures:raise ValueError('--fixtures required for gate')
        fixtures=read(a.fixtures)
        if not fixtures['complete']:raise ValueError('incomplete independently validated saves')
        cases += [dict(name=c['name'],save=c['save'],save_sha256=c['save_sha256'],D=210,B2=210*64,
                       unit=c['unit'],factor=c['expected_factor']) for c in fixtures['fixtures']]
    else:
        if not a.gate:raise ValueError('--gate required for timing')
        gate=read(a.gate)
        if not gate['complete'] or gate['identity']!=identity:raise ValueError('native gate differs')
    data=dict(complete=False,identity=identity,reference_sha256=sha(a.reference),helper_sha256=sha(HELPER),
              tool_sha256=sha(__file__),mode=a.mode,cases=cases,runs=[],environment={k:v for k,v in env.items() if k.startswith('NTT_') or k=='CUDA_LAUNCH_BLOCKING'})
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    def persist():(out/'measurements.json').write_text(json.dumps(data,indent=2)+'\n')
    def verify():
        if sha(exe)!=identity['binary_sha256'] or sha(__file__)!=data['tool_sha256'] or sha(HELPER)!=data['helper_sha256']:
            raise ValueError('binary/tool changed')
        for name,want in identity['sources'].items():
            if sha(exe.parent/'sources'/name)!=want:raise ValueError('source snapshot changed')
        for c in cases:
            if sha(c['save'])!=c['save_sha256']:raise ValueError('save changed')
    def run(c,mask,category,index):
        verify();name=f'{c["name"]}_{category}_{index}_v{mask}';log=out/(name+'.log');results=out/(name+'.jsonl')
        command=[str(exe),'--ini',str(ini),'--save',c['save'],'--device','1','--b2',str(c['B2']),'--d',str(c['D']),
                 '--arena-mb','6300','--factor-only','--log',str(log),'--results',str(results)]
        use=env|{'NTT_OUTER_NARROW':str(mask)}
        if category=='gate' and c['name']!='m4423_large':use.update(NTT_SCALED_TEST='1',NTT_SCALED_CHECK='1')
        r=subprocess.run(command,env=use,capture_output=True,timeout=900)
        driver=out/(name+'_driver.log');driver.write_bytes(r.stdout+r.stderr)
        if r.returncode:raise ValueError(name+' nonzero exit; retain logs')
        text=log.read_text();records=[json.loads(s) for s in results.read_text().splitlines()]
        if len(records)!=1:raise ValueError('result count')
        result=records[0];leaf=fields(text,'descent_values');coverage=fields(text,'s4_multiply_stats');wall=fields(text,'stage2_full_wall')
        if (result['B2'],result['requested_D'],result['device'],result['bad_factors'])!=(c['B2'],c['D'],1,0):raise ValueError('result input mismatch')
        n=int(result['N_hex'],16)
        if any(not 1<int(f)<n or n%int(f) for f in result['factors']):raise ValueError('invalid factor')
        if c['factor'] and not any(int(f)%c['factor']==0 for f in result['factors']):raise ValueError('known nonunit factor missing')
        for token in ('mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0','gmp_selftest_bad=0','gmp_check_bad=0','pending=0',f'ntt_outer_offsets: narrow_mask={mask}'):
            if token not in text:raise ValueError('required check missing: '+token)
        if c['name']=='m4423_large':
            if leaf!=expected['leaf']:raise ValueError('large complete leaf differs from production reference')
            for k in ('N_hex','B1','B2','sigma','requested_D','device','factors','bad_factors'):
                if result[k]!=expected['result'][k]:raise ValueError('different large input/result')
            for k in ('launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks'):
                if coverage[k]!=expected['coverage'][k]:raise ValueError('coverage differs: '+k)
        ledger=fields(text,'real_batched_wall')
        if abs(float(ledger['sum'])-float(wall['main']))>.003:raise ValueError('precise main ledger does not close')
        root=fields(text,'scaled_root_device')
        if root['requested']!='1' or root['checked_words']!='0':raise ValueError('root policy/check overhead changed')
        row=dict(name=name,case=c['name'],mask=mask,category=category,command=command,environment={k:v for k,v in use.items() if k.startswith('NTT_') or k=='CUDA_LAUNCH_BLOCKING'},
                 wall=wall,leaf=leaf,result=result,coverage=coverage,ledger=ledger,root=root,
                 phases=fields(text,'real_batched_split'),log_sha256=sha(log),result_sha256=sha(results),driver_sha256=sha(driver))
        data['runs'].append(row);persist();verify();print(name,wall['total'],flush=True)
    persist()
    for c in cases:
        if a.mode=='gate':
            for index,mask in enumerate((0,3)):run(c,mask,'gate',index)
        else:
            for index,mask in enumerate((0,3)):run(c,mask,'warmup',index)
            for index,mask in enumerate((0,3,3,0,3,0,0,3)):run(c,mask,'timing',index)
        rows=[r for r in data['runs'] if r['case']==c['name']]
        if len({r['leaf']['hash'] for r in rows})!=1 or len({tuple(r['result']['factors']) for r in rows})!=1:raise ValueError('mode output mismatch')
    if a.mode=='timing':
        rows=[r for r in data['runs'] if r['category']=='timing']
        means={str(mask):statistics.mean(float(r['wall']['total']) for r in rows if r['mask']==mask) for mask in (0,3)}
        groups=[]
        for lo in (0,4):
            sub=rows[lo:lo+4];m={mask:statistics.mean(float(r['wall']['total']) for r in sub if r['mask']==mask) for mask in (0,3)}
            groups.append(100*(1-m[3]/m[0]))
        data['summary']=dict(full_mean_seconds=means,reduction_percent=100*(1-means['3']/means['0']),groups_reduction_percent=groups)
    data['complete']=True;persist();print(json.dumps(data.get('summary',dict(passed=len(data['runs'])))))


if __name__=='__main__':main()
