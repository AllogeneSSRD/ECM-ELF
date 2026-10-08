"""Independent production source, saved arithmetic, verbosity and queue gates.

Reuse independently prepared CPU/GMP-ECM Stage1 fixtures, including complete
monic evaluation fingerprints and real nonunit factors. These are correctness
invocations, not performance samples. Production uses its actual chain threshold.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fields(text,prefix):
    lines=re.findall(r'^'+re.escape(prefix)+r': (.*)$',text,re.M)
    if not lines:raise ValueError('missing '+prefix)
    return dict(re.findall(r'(\w+)=(\S+)',lines[-1]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--gscale-check',action='store_true',help='Check every resident corrected H word and Gamma corruption at all log levels')
    a=p.parse_args();exe=a.exe.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('use a fresh gate directory')
    mf=exe.parent/'build_manifest.json';build=json.loads(mf.read_text(encoding='utf-8-sig'))
    if build.get('engine')!='production' or build['gl_fixed_mode']!=3 or build['outer_unroll_u']!=0:
        raise ValueError('wrong production build')
    if any(name.startswith('tools/bench/') for name in build['source_hashes']):
        raise ValueError('production depends on experiment sources')
    prepared=json.loads(a.fixtures.read_text(encoding='utf-8'))
    if not prepared['complete'] or prepared['identity']['reference_sha256']!=sha(ROOT/'tools/stat/suyama_mont_ref.py'):
        raise ValueError('fixture reference identity changed')
    identity=dict(binary_sha256=sha(exe),manifest_sha256=sha(mf),tool_sha256=sha(__file__),fixture_sha256=sha(a.fixtures))
    def verify():
        if identity!=dict(binary_sha256=sha(exe),manifest_sha256=sha(mf),tool_sha256=sha(__file__),fixture_sha256=sha(a.fixtures)):
            raise ValueError('gate identity changed')
        if identity['binary_sha256'].upper()!=build['sha256']:raise ValueError('build binary mismatch')
        for name,want in build['source_hashes'].items():
            if sha(ROOT/name).upper()!=want:raise ValueError('compiled source changed: '+name)
    clean={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    clean['CUDA_LAUNCH_BLOCKING']='0'
    if a.gscale_check:clean['NTT_GSCALE_DEVICE_CHECK']='1'
    ini=out/'manual.ini';ini.write_text('[gpu]\ndevice='+str(a.device)+'\n',encoding='utf-8')
    report=dict(identity=identity,runs=[],protocols=[],rejections=[],complete=False,formal_performance_samples=0)
    def persist():
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def call(name,args,env=None,code=0):
        verify();cmd=[str(exe),'--ini',str(ini),*args]
        proc=subprocess.run(cmd,env=clean|(env or {}),capture_output=True,timeout=300)
        (out/(name+'_driver.log')).write_bytes(proc.stdout+proc.stderr)
        verify()
        if proc.returncode!=code:raise ValueError(f'{name}: exit {proc.returncode}, expected {code}')
        return dict(name=name,command=cmd,environment=env or {},exit=proc.returncode,
                    driver_log_sha256=sha(out/(name+'_driver.log'))),proc.stdout.decode('utf-8','replace')+proc.stderr.decode('utf-8','replace')
    def run(case,name,count=65,level='debug',extra=None):
        if sha(case['save'])!=case['save_sha256']:raise ValueError('save changed')
        result=out/(name+'.jsonl');log=out/(name+'.log')
        b2=210*(count-2) if count>2 else case['B1']+1
        env={'NTT_D_MODEL':'0','NTT_GFINV_SEG_CHECK':'1','NTT_GIANT_SEED_CHECK':'1',
             'NTT_FOLD_DEVICE_MAX_MB':'64','NTT_GIANT_CHAIN_CHECK':'1' if case['unit'] else '0'}
        row,stdout=call(name,['--save',case['save'],'--b2',str(b2),'--d','210','--device',str(a.device),
            '--results',str(result),'--log',str(log),'--log-level',level,'--factor-only'],env|(extra or {}))
        records=[json.loads(line) for line in result.read_text(encoding='utf-8').splitlines()]
        if len(records)!=1:raise ValueError('result count')
        r=records[0];n=int(case['N_hex'],16);text=log.read_text(encoding='utf-8')
        if r['N_hex'].lower()!=case['N_hex'] or r['B1']!=case['B1'] or r['B2']!=b2 or r['sigma']!=26 or r['bad_factors']:
            raise ValueError('wrong arithmetic input/result')
        if any(not 1<int(f)<n or n%int(f) for f in r['factors']):raise ValueError('invalid factor')
        if case.get('expected_factor') and not any(int(f)%case['expected_factor']==0 for f in r['factors']):
            raise ValueError('known nonunit factor missing')
        if level=='debug':
            for token in ('stage1_skipped=1','mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0',
                          'gmp_selftest_bad=0','gmp_check_bad=0','pending=0'):
                if token not in text:raise ValueError('mandatory check absent: '+token)
            leaf=fields(text,'descent_values')
            if case['unit'] and leaf['hash']!=case['expected_leaf_hash'][str(count)]:
                raise ValueError('independent complete monic fingerprint mismatch')
            row['leaf']=leaf
            if a.gscale_check:
                scale=fields(text,'real_gscale_device')
                if scale['requested']!='1':raise ValueError('production GPU Gamma policy absent')
                if scale['enabled']=='1' and int(scale['checked_words'])!=int(scale['coefficients'])*((n.bit_length()+63)//64):raise ValueError('full corrected H not checked')
                row['gscale']=scale
        # Checks above run at every verbosity; corruption gates below prove their verdicts survive filtering.
        if level=='quiet' and (text.strip() or stdout.strip()):raise ValueError('quiet leaked progress')
        if level!='debug' and re.search(r'^(tree_level|descent_progress|mont_selftest|s4_div_check):',text,re.M):
            raise ValueError('internal logs leaked')
        if level in ('phases','batches','debug') and 'real_batched_split:' not in text:
            raise ValueError('major phase timing missing')
        if level in ('batches','debug') and count>24 and 'batched_progress:' not in text:
            raise ValueError('batch progress missing')
        if level in ('quiet','curve','phases') and 'batched_progress:' in text:raise ValueError('wrong batch verbosity')
        row.update(result=r,log_sha256=sha(log),result_sha256=sha(result),level=level,count=count)
        report['runs'].append(row);persist();print(name,'passed',flush=True)
        return r
    persist()
    for case in prepared['fixtures']:run(case,case['name'])
    full=next(c for c in prepared['fixtures'] if c['name']=='generic16384')
    for count in (2,66):run(full,'wide_tail_'+str(count),count)
    for kind,extra in [('owner_budget',{'NTT_FOLD_DEVICE_MAX_MB':'0'}),
                       ('owner_alloc',{'NTT_FOLD_DEVICE_ALLOC_FAIL':'1'}),
                       ('baby_alloc',{'NTT_BABY_DEVICE_ALLOC_FAIL':'1'})]:run(full,kind,extra=extra)
    for level in ('quiet','curve','phases','batches'):run(full,'verbosity_'+level,level=level)
    for level in ('quiet','curve','phases','batches','debug'):
        result=out/('bad_'+level+'.jsonl');log=out/('bad_'+level+'.log')
        row,_=call('bad_'+level,['--save',full['save'],'--b2','13230','--d','210','--device',str(a.device),
            '--results',str(result),'--log',str(log),'--log-level',level],{'NTT_S4_ORACLE_TEST_BAD':'1'},code=2)
        text=log.read_text(encoding='utf-8')
        if result.exists() or 'FATAL' not in text or 'GMP' not in text:raise ValueError('filtered corruption accepted')
        row['log_sha256']=sha(log);report['rejections'].append(row);persist()
    if a.gscale_check:
        for level in ('quiet','curve','phases','batches','debug'):
            result=out/('bad_gamma_'+level+'.jsonl');log=out/('bad_gamma_'+level+'.log')
            row,_=call('bad_gamma_'+level,['--save',full['save'],'--b2','13230','--d','210','--device',str(a.device),
                '--results',str(result),'--log',str(log),'--log-level',level],{'NTT_GSCALE_DEVICE_TEST_BAD':'1'},code=2)
            text=log.read_text(encoding='utf-8')
            if result.exists() or 'Gamma device GMP mismatch' not in text:raise ValueError('filtered Gamma corruption accepted')
            row['log_sha256']=sha(log);report['rejections'].append(row);persist()
    # Real wide queue: optional B2/skip/count selects only record 2; Worker logging overrides global.
    saves=out/'three.save';line=Path(full['save']).read_text();saves.write_text(line*3)
    queue=out/'worktodo.txt';task='ECMSTAGE2=1,2,16384,-15,"three.save",13230,1,1'
    queue.write_text(task+'\n');finished=out/'finished.txt';qini=out/'queue.ini'
    qini.write_text(f'worktodo={queue}\nfinished={finished}\ntmp_dir={out}\ndevice={a.device}\nstage2_log_level=debug\n[Worker #1]\nstage2_log_level=quiet\n')
    result=out/'queue.jsonl';log=out/'queue.log'
    row,stdout=call('queue',['--ini',str(qini),'--once','--factor-only','--results',str(result),'--log',str(log)])
    qr=json.loads(result.read_text().strip())
    if qr['record']!=2 or qr['B2']!=13230 or queue.read_text().strip() or finished.read_text().strip()!=task:
        raise ValueError('wide queue selection/transaction failed')
    if stdout.strip() or log.read_text().strip():raise ValueError('worker INI quiet failed')
    report['protocols'].append(row);persist()
    row,text=call('plan',['--save',full['save'],'--b2','13230','--d','210','--device',str(a.device),'--plan-only','--log-level','quiet'])
    plans=[json.loads(s) for s in text.splitlines() if s.startswith('{')]
    if len(plans)!=1 or plans[0]['bits']!=16384 or plans[0]['words']!=256 or plans[0]['curves_executed']!=0:raise ValueError('plan contract')
    report['protocols'].append(row)
    # Obsolete algorithm overrides fail before any queue transaction or result publication.
    obsolete=[('NTT_XADD6','0'),('NTT_S5_ON','1'),('NTT_S4_OLDTAIL','1'),
                      ('NTT_GIANT_SEED_PAIR','0'),('NTT_GIANT_BASE_CPU','1'),('NTT_FUSE_T','11'),
                      ('NTT_FOLD_OWNER_REUSE','0')]
    if a.gscale_check:obsolete += [('NTT_GSCALE_DEVICE','0'),('NTT_GSCALE_DEVICE_TEST','1')]
    for key,value in obsolete:
        queue.write_text(task+'\n');before=queue.read_bytes();result=out/(key+'.jsonl')
        row,text=call(key,['--ini',str(qini),'--once','--results',str(result)],{key:value},code=2)
        if queue.read_bytes()!=before or result.exists() or 'requires '+key not in text:raise ValueError('unsafe configuration transaction')
        report['rejections'].append(row);persist()
    row,text=call('bad_level',['--log-level','verbose'],code=2)
    if 'log level must' not in text:raise ValueError('invalid verbosity accepted')
    report['rejections'].append(row)
    verify();report.update(complete=True,passed=len(report['runs'])+len(report['protocols'])+len(report['rejections']),failed=0)
    persist();print(json.dumps(dict(passed=report['passed'],failed=0)),flush=True)

if __name__=='__main__':main()
