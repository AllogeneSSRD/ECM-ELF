"""Larger 16k correctness/capacity probes; wall times are diagnostic, not A/B claims."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
HELPER=ROOT/'tools/bench/bench_stage2_production.py'
spec=importlib.util.spec_from_file_location('bench',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read,sha,fields=helper.read,helper.sha,helper.fields


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise ValueError('use fresh output')
    out.mkdir(parents=True);exes={k:v.resolve() for k,v in [('baseline',a.baseline),('candidate',a.candidate)]}
    identity={k:helper.freeze(v) for k,v in exes.items()};fixture=read(a.fixtures)
    if not fixture['complete']:raise ValueError('unfinished saved input proof')
    if read(exes['candidate'].parent/'build_manifest.json')['add_sub_mask']!=1:raise ValueError('wrong candidate arithmetic')
    cases=[]
    for name in ['m16381','generic16384']:
        c=next(c for c in fixture['fixtures'] if c['name']==name)
        for owner in ([640] if name=='m16381' else [640,0]):
            cases.append(dict(name=f'{name}_p28800_owner{owner}',save=c['save'],save_sha256=c['save_sha256'],N_hex=c['N_hex'],B1=c['B1'],D=300300,P=28800,I=32768,B2=300300*(32768-2),owner_mb=owner))
    report=dict(complete=False,identity=identity,cases=cases,tool_sha256=sha(__file__),helper_sha256=sha(HELPER),fixtures_sha256=sha(a.fixtures),plans=[],runs=[],formal_performance_samples=0)
    (out/'collector.py').write_bytes(Path(__file__).read_bytes());ini=out/'manual.ini';ini.write_text('device=1\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')};env.update(NTT_D_MODEL='0',CUDA_LAUNCH_BLOCKING='0',NTT_NO_PROGRESS='1')
    def verify():
        if sha(__file__)!=report['tool_sha256'] or sha(HELPER)!=report['helper_sha256'] or sha(a.fixtures)!=report['fixtures_sha256']:raise ValueError('collector/input changed')
        for k,exe in exes.items():
            if helper.freeze(exe)!=identity[k]:raise ValueError('compiled identity changed')
        for c in cases:
            if sha(c['save'])!=c['save_sha256']:raise ValueError('saved input changed')
    def persist():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    persist()
    # The larger D is a planning boundary only: no promise its NTT workspace fits.
    for name in ['m16381','generic16384']:
        c=next(c for c in cases if c['name'].startswith(name))
        for d,pdegree in [(300300,28800),(600600,57600)]:
            verify();label=f'{name}_plan_d{d}'
            cmd=[str(exes['candidate']),'--ini',str(ini),'--save',c['save'],'--device','1','--b2',str(d*(32768-2)),'--d',str(d),'--arena-mb','6300','--plan-only','--log-level','debug']
            proc=subprocess.run(cmd,env=env,capture_output=True,timeout=120);log=out/(label+'.log');log.write_bytes(proc.stdout+proc.stderr)
            if proc.returncode:raise ValueError('capacity plan failed')
            plans=[json.loads(s) for s in log.read_text().splitlines() if s.startswith('{')]
            if len(plans)!=1 or plans[0]['curves_executed']!=0 or plans[0]['P']!=pdegree:raise ValueError('planning boundary differs')
            w=(int(c['N_hex'],16).bit_length()+63)//64;combined=8*w*(7*pdegree+7)+48+24*pdegree
            if (combined>640*2**20)!=(d==600600):raise ValueError('independent combined budget boundary')
            report['plans'].append(dict(name=label,command=cmd,result=plans[0],combined_owner_metadata_bytes=combined,executed_curve=False,log_sha256=sha(log)));persist();print(label,'passed',flush=True)
    for c in cases:
        for key,exe in exes.items():
            verify();name=c['name']+'_'+key;log=out/(name+'.log');result=out/(name+'.jsonl');driver=out/(name+'_driver.log')
            use=env|{'NTT_FOLD_DEVICE_MAX_MB':str(c['owner_mb'])}
            cmd=[str(exe),'--ini',str(ini),'--save',c['save'],'--device','1','--b2',str(c['B2']),'--d',str(c['D']),'--arena-mb','6300','--factor-only','--log-level','debug','--log',str(log),'--results',str(result)]
            proc=subprocess.run(cmd,env=use,capture_output=True,timeout=1800);driver.write_bytes(proc.stdout+proc.stderr)
            if proc.returncode:raise ValueError(name+' failed '+str(proc.returncode))
            records=[json.loads(s) for s in result.read_text().splitlines()];text=log.read_text()
            if len(records)!=1 or 'stage2_complete: curves=1' not in driver.read_text():raise ValueError('unfinished capacity curve')
            r=records[0];n=int(r['N_hex'],16)
            if (r['N_hex'].lower(),r['B1'],r['B2'],r['requested_D'],r['device'],r['bad_factors'])!=(c['N_hex'],c['B1'],c['B2'],c['D'],1,0):raise ValueError('capacity input differs')
            if any(not 1<int(f)<n or n%int(f) for f in r['factors']):raise ValueError('improper factor')
            for token in ['mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0','gmp_selftest_bad=0','gmp_check_bad=0','pending=0']:
                if token not in text:raise ValueError('mandatory capacity check '+token)
            shape=fields(text,'real_shape')
            if int(shape['P'].split('=')[-1])!=c['P'] or int(shape['giant_points'])!=c['I']:raise ValueError('actual capacity geometry differs')
            fold=fields(text,'real_batched_folddevice')
            if int(fold['enabled'])!=int(c['owner_mb']!=0):raise ValueError('actual owner residency differs')
            row=dict(name=name,case=c['name'],key=key,command=cmd,environment={k:v for k,v in use.items() if k.startswith('NTT_') or k=='CUDA_LAUNCH_BLOCKING'},result=r,shape=shape,fold=fold)
            for k,prefix in [('leaf','descent_values'),('coverage','s4_multiply_stats'),('wall','stage2_full_wall'),('ledger','real_batched_wall'),('workspace','ntt_workspace_stats')]:row[k]=fields(text,prefix)
            if abs(float(row['ledger']['sum'])-float(row['wall']['main']))>.003:raise ValueError('capacity main ledger unclosed')
            if key=='candidate':
                row['frontier']=fields(text,'scaled_frontier_device')
                if int(row['frontier']['enabled'])!=int(c['owner_mb']!=0) or row['frontier']['check_d2h_bytes']!='0':raise ValueError('actual capacity frontier scope differs')
            row.update(log_sha256=sha(log),result_sha256=sha(result),driver_sha256=sha(driver));report['runs'].append(row);persist();verify();print(name,row['wall']['total'],'diagnostic seconds',flush=True)
        pair=[r for r in report['runs'] if r['case']==c['name']]
        if pair[0]['leaf']!=pair[1]['leaf'] or pair[0]['result']['factors']!=pair[1]['result']['factors']:raise ValueError('complete large-capacity result differs')
        for k in ['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']:
            if pair[0]['coverage'][k]!=pair[1]['coverage'][k]:raise ValueError('large capacity required coverage differs '+k)
    report['complete']=True;persist();print(json.dumps(dict(complete=True,plans=len(report['plans']),curves=len(report['runs']))))


if __name__=='__main__':main()
