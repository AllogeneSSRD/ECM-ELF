"""Frozen cross-build Stage2 comparison for independent production CUDA.

Gate mode checks real default chain geometry and complete output fingerprints.
Timing mode uses a predeclared warmup, ABBA+BAAB matrix with every sample retained.
Both sides use paired GPU seeds, C64, threshold32768 and the same required checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fields(text,prefix):
    rows=re.findall(r'^'+re.escape(prefix)+r': (.*)$',text,re.M)
    if not rows:raise ValueError('missing '+prefix)
    return dict(re.findall(r'(\w+)=(\S+)',rows[-1]))
def freeze(exe):
    b=read(exe.parent/'build_manifest.json');f=exe.parent/'frozen_sources_manifest.json'
    sources={n:h.lower() for n,h in b['source_hashes'].items()}
    if sha(exe)!=b['sha256'].lower():raise ValueError('binary/build mismatch')
    if not f.exists():
        for name,want in sources.items():
            if sha(ROOT/name)!=want:raise ValueError('compiled source changed: '+name)
            dest=exe.parent/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
        f.write_text(json.dumps(dict(binary_sha256=sha(exe),sources=sources),indent=2)+'\n')
    frozen=read(f)
    if frozen['binary_sha256']!=sha(exe) or frozen['sources']!=sources:raise ValueError('snapshot/build mismatch')
    return dict(binary_sha256=sha(exe),build_sha256=sha(exe.parent/'build_manifest.json'),snapshot_sha256=sha(f),sources=sources)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--mode',choices=('gate','timing','timing-wide'),required=True)
    p.add_argument('--fixtures',type=Path)
    p.add_argument('--save',type=Path)
    p.add_argument('--case',choices=('generic8193','m16381','generic16384'),
                   help='Restrict timing-wide to one predeclared valid input for diagnosis; default tests all three')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-gates',action='store_true',help='Preserve verified gate rows and recover a completed raw invocation after a collector-only rejection')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()) and not a.resume_gates:raise ValueError('use a fresh output directory')
    if a.resume_gates and a.mode!='gate':raise ValueError('timing matrices cannot use gate recovery')
    if a.case and a.mode!='timing-wide':raise ValueError('--case requires timing-wide')
    exes={k:v.resolve() for k,v in [('baseline',a.baseline),('candidate',a.candidate)]}
    identity={k:freeze(v) for k,v in exes.items()}
    if read(exes['candidate'].parent/'build_manifest.json').get('engine')!='production':raise ValueError('candidate is not production')
    tool_sha=sha(__file__)
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_D_MODEL='0',NTT_GIANT_SEED_PAIR='1',NTT_GIANT_BASE_CPU='0',
               NTT_GIANT_CHAIN_BLOCK='64',NTT_GIANT_CHAIN_MIN='32768',NTT_NO_PROGRESS='1',CUDA_LAUNCH_BLOCKING='0')
    ini=out/'manual.ini';ini.write_text('device=1\n')
    if a.mode=='gate':
        if not a.fixtures:p.error('--fixtures required for gate mode')
        prepared=read(a.fixtures)
        if not prepared['complete']:raise ValueError('unfinished independent fixtures')
        cases=[dict(name=c['name'],save=c['save'],save_sha256=c['save_sha256'],D=30030,
                    B2=30030*(32768-2),points=32768,unit=c['unit'],factor=c['expected_factor']) for c in prepared['fixtures']]
        full=next(c for c in prepared['fixtures'] if c['name']=='generic16384')
        cases.append(dict(name='generic16384_chunk_tail',save=full['save'],save_sha256=full['save_sha256'],D=30030,
                          B2=30030*(66305-2),points=66305,unit=True,factor=0))
        sequence=['baseline','candidate'];categories=['gate']
    elif a.mode=='timing-wide':
        if not a.fixtures:p.error('--fixtures required for wide timing')
        prepared=read(a.fixtures)
        if not prepared['complete']:raise ValueError('unfinished independent fixtures')
        cases=[dict(name=c['name'],save=c['save'],save_sha256=c['save_sha256'],D=30030,
                    B2=30030*(32768-2),points=32768,unit=True,factor=0,
                    N_hex=c['N_hex'],B1=c['B1'],sigma=26) for c in prepared['fixtures']
               if c['name'] in ('generic8193','m16381','generic16384')]
        if len(cases)!=3:raise ValueError('wide timing needs all three valid unit fixtures')
        if a.case:cases=[c for c in cases if c['name']==a.case]
        sequence=['baseline','candidate','candidate','baseline','candidate','baseline','baseline','candidate']
        categories=['warmup','timing']
    else:
        if not a.save:p.error('--save required for timing mode')
        cases=[dict(name='m4423_large',save=str(a.save.resolve()),save_sha256=sha(a.save),D=1381380,
                    B2=2011326186870,points=1456028,unit=True,factor=0)]
        sequence=['baseline','candidate','candidate','baseline','candidate','baseline','baseline','candidate']
        categories=['warmup','timing']
    data=dict(identity=identity,tool_sha256=tool_sha,mode=a.mode,cases=cases,sequence=sequence,runs=[],complete=False)
    if a.resume_gates:
        initial=read(out/'measurements.json')
        if initial['complete'] or initial['identity']!=identity or initial['cases']!=cases or initial['sequence']!=sequence:
            raise ValueError('recovery must use exactly the original gate plan and compiled sources')
        if sha(out/'collector_initial.py')!=initial['tool_sha256']:raise ValueError('retain exact original collector')
        rejected=read(out/'initial_rejection.json')
        for name,h in rejected['files'].items():
            if sha(out/name)!=h:raise ValueError('initial rejected evidence changed: '+name)
        for row in initial['runs']:
            if sha(row['log'])!=row['log_sha256'] or sha(Path(row['log']).with_suffix('.jsonl'))!=row['result_sha256']:
                raise ValueError('previous verified row changed')
        data=initial
        data.setdefault('continuations',[]).append(dict(tool_sha256=tool_sha,
            reason='Derive affine coverage from the actual default chain/ladder chunks; preserve all previously verified arithmetic and raw outputs.'))
    (out/('collector_continuation.py' if a.resume_gates else 'collector.py')).write_bytes(Path(__file__).read_bytes())
    def persist():(out/'measurements.json').write_text(json.dumps(data,indent=2)+'\n')
    def verify():
        if sha(__file__)!=tool_sha:raise ValueError('collector changed')
        for key,exe in exes.items():
            ident=identity[key]
            if sha(exe)!=ident['binary_sha256'] or sha(exe.parent/'build_manifest.json')!=ident['build_sha256'] or sha(exe.parent/'frozen_sources_manifest.json')!=ident['snapshot_sha256']:
                raise ValueError('binary/receipt changed')
            for name,h in ident['sources'].items():
                if sha(exe.parent/'sources'/name)!=h:raise ValueError('frozen source changed')
        for c in cases:
            if sha(c['save'])!=c['save_sha256']:raise ValueError('save changed')
    def run(c,key,category,index):
        verify();name=f'{c["name"]}_{category}_{index}_{key}';log=out/(name+'.log');result=out/(name+'.jsonl')
        if any(row['name']==name for row in data['runs']):return
        cmd=[str(exes[key]),'--ini',str(ini),'--save',c['save'],'--b2',str(c['B2']),'--d',str(c['D']),
             '--device','1','--arena-mb','6300','--results',str(result),'--log',str(log),'--factor-only']
        if read(exes[key].parent/'build_manifest.json').get('engine')=='production':
            cmd+=['--log-level','debug']
        use=env.copy()
        if category=='gate':use.update(NTT_GFINV_SEG_CHECK='1',NTT_GIANT_SEED_CHECK='1',NTT_GIANT_CHAIN_CHECK='1' if c['unit'] else '0')
        recovered=a.resume_gates and log.is_file() and result.is_file()
        if recovered:
            if 'stage2_complete: curves=1' not in (out/(name+'_driver.log')).read_text(encoding='utf-8'):
                raise ValueError('unrecorded invocation did not complete successfully')
        else:
            proc=subprocess.run(cmd,env=use,capture_output=True,timeout=900)
            (out/(name+'_driver.log')).write_bytes(proc.stdout+proc.stderr)
            if proc.returncode:raise ValueError(name+' failed; inspect raw logs')
        verify()
        r=read_lines(result);text=log.read_text(encoding='utf-8')
        for token in ('mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0',
                      'gmp_selftest_bad=0','gmp_check_bad=0','pending=0'):
            if token not in text:raise ValueError('required check missing: '+token)
        if r['B2']!=c['B2'] or r['requested_D']!=c['D'] or r['device']!=1 or r['bad_factors']:raise ValueError('result/input mismatch')
        if c.get('N_hex') and (r['N_hex'].lower()!=c['N_hex'] or r['B1']!=c['B1'] or r['sigma']!=c['sigma']):
            raise ValueError('wide saved arithmetic input mismatch')
        n=int(r['N_hex'],16)
        if any(not 1<int(f)<n or n%int(f) for f in r['factors']):raise ValueError('bad factor')
        if c['factor'] and not any(int(f)%c['factor']==0 for f in r['factors']):raise ValueError('known nonunit factor missing')
        pair=fields(text,'real_giant_seed_pair');base=fields(text,'real_giant_base')
        if pair['base_builds']!='1' or base['gpu_builds']!='1' or base['cpu_builds']!='0':raise ValueError('default paired base not executed')
        if c['name']=='base_nonunit16384' and (pair['base_nonunits']!='1' or pair['chunks']!='0'):raise ValueError('nonunit base fallback missing')
        if c['unit'] and int(pair['chunks'])<1:raise ValueError('paired kernel did not execute')
        if category=='gate' and c['unit']:
            affine=[dict(re.findall(r'(\w+)=(\S+)',s)) for s in re.findall(r'^giant_chain_check: (.*)$',text,re.M)]
            shape=fields(text,'real_shape');p=int(shape['P'].split('=')[-1]);w=(n.bit_length()+63)//64
            chunk=p*((max(p,(256<<20)//(16*w))+p-1)//p)
            parts=[min(chunk,c['points']-lo) for lo in range(0,c['points'],chunk)]
            chain_points=sum(v for v in parts if v>=32768)
            if sum(int(v['points']) for v in affine)!=chain_points or any(int(v['mismatches']) for v in affine):raise ValueError('chain affine check incomplete')
        coverage=fields(text,'s4_multiply_stats')
        row=dict(name=name,case=c['name'],key=key,category=category,command=cmd,environment={k:v for k,v in use.items() if k.startswith('NTT_') or k=='CUDA_LAUNCH_BLOCKING'},
                 result=r,leaf=fields(text,'descent_values'),pair=pair,base=base,coverage=coverage,
                 wall=fields(text,'stage2_full_wall'),phases=fields(text,'real_batched_split'),log=str(log),log_sha256=sha(log),result_sha256=sha(result),recovered_raw=recovered)
        if category=='gate' and c['unit']:row['affine_chain_points']=chain_points;row['ladder_tail_points']=c['points']-chain_points
        data['runs'].append(row);persist();print(name,row['wall']['total'],'s',flush=True)
    def read_lines(path):
        rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        if len(rows)!=1:raise ValueError('wrong record count')
        return rows[0]
    persist()
    for c in cases:
        for category in categories:
            for index,key in enumerate(['baseline','candidate'] if category=='warmup' else sequence):run(c,key,category,index)
        rows=[r for r in data['runs'] if r['case']==c['name']]
        if len({r['leaf']['hash'] for r in rows})!=1 or len({tuple(sorted(r['result']['factors'])) for r in rows})!=1:
            raise ValueError('cross-build complete leaf/factor mismatch')
        for stat in ('launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks'):
            if len({r['coverage'][stat] for r in rows})!=1:raise ValueError('arithmetic coverage changed: '+stat)
    if a.mode!='gate':
        summaries={}
        for c in cases:
            means={key:statistics.mean(float(r['wall']['total']) for r in data['runs'] if r['case']==c['name'] and r['key']==key and r['category']=='timing') for key in exes}
            summaries[c['name']]=dict(full_mean_seconds=means,reduction_percent=100*(1-means['candidate']/means['baseline']))
        data['summary']=summaries if a.mode=='timing-wide' else summaries[cases[0]['name']]
    verify();data.update(complete=True,passed=len(data['runs']),failed=0);persist()
    print(json.dumps(data.get('summary',dict(passed=data['passed']))),flush=True)

if __name__=='__main__':main()
