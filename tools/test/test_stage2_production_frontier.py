"""Bind production resident descent to frozen development and larger 16k inputs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
HELPER=ROOT/'tools/bench/bench_stage2_production.py'
spec=importlib.util.spec_from_file_location('bench',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read,sha,fields=helper.read,helper.sha,helper.fields


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();exe=a.exe.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('use fresh output')
    out.mkdir(parents=True);identity=helper.freeze(exe);build=read(exe.parent/'build_manifest.json')
    if build['engine']!='production' or build['add_sub_mask']!=1:raise ValueError('wrong production policy')
    reference=read(a.reference)
    if not reference['complete'] or reference['mode']!='gate':raise ValueError('unfinished development gate')
    expected=[r for r in reference['runs'] if r['frontier']==1]
    if len(expected)!=13 or len({r['case'] for r in expected})!=13:raise ValueError('full development cases required')
    report=dict(complete=False,identity=identity,reference_sha256=sha(a.reference),tool_sha256=sha(__file__),helper_sha256=sha(HELPER),runs=[],formal_performance_samples=0)
    (out/'collector.py').write_bytes(Path(__file__).read_bytes());ini=out/'manual.ini';ini.write_text('device=1\n')
    def persist():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    def verify():
        if sha(__file__)!=report['tool_sha256'] or sha(HELPER)!=report['helper_sha256'] or sha(a.reference)!=report['reference_sha256'] or helper.freeze(exe)!=identity:raise ValueError('frozen comparison changed')
        for name,digest in identity['sources'].items():
            if sha(ROOT/name)!=digest:raise ValueError('compiled production source changed '+name)
    persist()
    for old in expected:
        verify();case=next(c for c in reference['cases'] if c['name']==old['case']);name=case['name']
        if sha(case['save'])!=case['save_sha256']:raise ValueError('saved input changed')
        log=out/(name+'.log');result=out/(name+'.jsonl');driver=out/(name+'_driver.log')
        env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')};env.update(old['environment'])
        cmd=[str(exe),'--ini',str(ini),'--save',case['save'],'--device','1','--b2',str(case['B2']),'--d',str(case['D']),'--arena-mb','6300','--log-level','debug','--factor-only','--log',str(log),'--results',str(result)]
        proc=subprocess.run(cmd,env=env,capture_output=True,timeout=900);driver.write_bytes(proc.stdout+proc.stderr)
        if proc.returncode:raise ValueError(name+' failed '+str(proc.returncode))
        records=[json.loads(s) for s in result.read_text().splitlines()];text=log.read_text()
        if len(records)!=1 or 'stage2_complete: curves=1' not in driver.read_text():raise ValueError('incomplete production curve')
        r=records[0];n=int(r['N_hex'],16)
        for key in ['N_hex','B1','B2','sigma','requested_D','device','factors','bad_factors']:
            if r[key]!=old['result'][key]:raise ValueError('development full result differs '+key)
        if any(not 1<int(f)<n or n%int(f) for f in r['factors']):raise ValueError('improper factor')
        for token in ['ntt_addsub_arithmetic: mask=1','stage1_skipped=1','mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0','gmp_selftest_bad=0','gmp_check_bad=0','pending=0']:
            if token not in text:raise ValueError('mandatory check absent '+token)
        row=dict(name=name,case=case,command=cmd,environment={k:v for k,v in env.items() if k.startswith('NTT_') or k=='CUDA_LAUNCH_BLOCKING'},result=r)
        for key,prefix in [('leaf','descent_values'),('coverage','s4_multiply_stats'),('scaled','scaled_descent'),('frontier','scaled_frontier_device'),('root','scaled_root_device'),('wall','stage2_full_wall'),('ledger','real_batched_wall')]:row[key]=fields(text,prefix)
        if row['leaf']!=old['leaf']:raise ValueError('development complete leaf differs')
        for key in ['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']:
            if row['coverage'][key]!=old['coverage'][key]:raise ValueError('required coverage differs '+key)
        for key in ['enabled','metadata_bytes','f_h2d_bytes','metadata_h2d_bytes','leaf_d2h_bytes','check_d2h_bytes','fallback']:
            if row['frontier'][key]!=old['frontier_stats'][key]:raise ValueError('development resident interface differs '+key)
        if case['D']==210 and (row['scaled']['checked_states']!=row['scaled']['states'] or row['scaled']['checked_words']!=row['scaled']['words']):raise ValueError('all GMP nodes missing')
        if abs(float(row['ledger']['sum'])-float(row['wall']['main']))>.003:raise ValueError('main ledger unclosed')
        if name.endswith('chunk_tail'):
            chain=[dict(re.findall(r'(\w+)=(\S+)',s)) for s in re.findall(r'^giant_chain_check: (.*)$',text,re.M)]
            if sum(int(x['points']) for x in chain)!=66240 or any(x['mismatches']!='0' for x in chain):raise ValueError('full chain and 65-point tail gate')
        row.update(log_sha256=sha(log),result_sha256=sha(result),driver_sha256=sha(driver));report['runs'].append(row);persist();verify();print(name,'passed',flush=True)
    report['complete']=True;persist();print(json.dumps(dict(complete=True,passed=len(report['runs']))))


if __name__=='__main__':main()
