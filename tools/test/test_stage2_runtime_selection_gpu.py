"""Exercise carrier-first production trials and bounded NF sampling on GPU1.

Inputs are previously verified Stage1 saves. All work/output lives in a fresh
ignored directory; no production queue, result file or profile is edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"ecm_report"))
from result_format import normalize_result
import re
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/test'))
from stage2_tune_tables import summaries
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('exe','save','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--exponent',type=int,default=1009)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--sample-limit',type=int,default=10)
    p.add_argument('--factor-save',type=Path)
    p.add_argument('--mont-save',type=Path,help='Verified Montgomery point for the separate Auto B2 planning contract')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    assert 2<=a.sample_limit<=1000
    count=max(10,a.sample_limit+5)
    exe=a.exe.resolve();saved=a.save.read_text(encoding='utf-8-sig').strip()
    n=int(re.search(r'\bN=0x([\da-fA-F]+);',saved)[1],16)
    whole=(1<<a.exponent)-1
    assert whole%n==0 and whole!=n
    known=whole//n
    (out/'curves.save').write_text((saved+'\n')*count,encoding='utf-8')
    queue=out/'worktodo.txt';profile=out/'stage2_ecm_tune.toml'
    queue.write_text(f'ECMSTAGE2=1,2,{a.exponent},-1,curves.save,100000000,0,{count},"{known}"\n',encoding='utf-8')
    ini=out/'ecm.ini'
    ini.write_text('user=\ncomputer=\nverbose=false\nstage2_debug_log=false\n'+
        ('stage2_tune_runtime_samples='+str(a.sample_limit)+'\n' if a.sample_limit!=10 else '')+
        'stage2_finished=finished.txt\n'
        'stage2_save_dir='+str(out)+'\n',encoding='utf-8')
    common=[str(exe),'--ini',str(ini),'--device',str(a.device),'--batch-mb','64',
        '--arena-mb','6300','--owner-budget-mb','640','--short-calibration','0','--factor-only',
        '--submission',str(out/'results.json.txt'),'--tune-profile',str(profile)]
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_') and k!='CUDA_VISIBLE_DEVICES'}
    identities={str(exe):sha(exe),str(a.save.resolve()):sha(a.save)}
    def call(name,args):
        command=common+args;(out/(name+'.command.json')).write_text(json.dumps(command,indent=2),encoding='utf-8')
        r=subprocess.run(command,cwd=out,env=env,capture_output=True,text=True,errors='replace',timeout=600)
        (out/(name+'.log')).write_text(r.stdout+r.stderr,encoding='utf-8')
        assert not r.returncode,(name,r.stdout,r.stderr)
        return r.stdout
    results=out/'internal.jsonl'
    call('production',['--worktodo',str(queue),'--once','--d','30030','--results',str(results),'--log',str(out/'engine.log')])
    rows=[normalize_result(json.loads(s)) for s in results.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==count and all(x['status']=='stage2_completed' and not x['factors'] and not x['bad_factors'] for x in rows)
    assert [r['carrier_exponent'] for r in rows[:4]]==[a.exponent,a.exponent,0,0]
    assert all(r['B2']==100000000 and r['d']==30030 and int(r['N_hex'],16)==n for r in rows)
    sample_rows=summaries(tomllib.loads(profile.read_text(encoding='utf-8')))
    runtime=[s for s in sample_rows if s.get('benchmark_kind')=='runtime_save']
    counts={c:sum(s['repeats'] for s in runtime if s['carrier_exponent']==c) for c in (0,a.exponent)}
    assert sorted(counts.values())==[2,a.sample_limit],counts
    assert len(runtime)==2 and sum(s['repeats'] for s in runtime)==a.sample_limit+2
    assert all(s['verified']==1 and s['stddev_seconds']>0 for s in runtime)
    version=re.search(r'inline constexpr char version\[\] = "([^"]+)"', (ROOT/'src/core/ecm_program_version.h').read_text())[1]
    assert all(s.get('program_version')==version and s.get('updated_at') for s in runtime)
    assert all('program_version' not in c for c in tomllib.loads(profile.read_text(encoding='utf-8'))['condition'].values())
    assert all(not r['tune_plan']['runtime_probe'] for r in rows[4:])
    assert len({r['carrier_exponent'] for r in rows[4:]})==1,'stable conditions repeatedly reverted path'
    assert queue.read_text(encoding='utf-8').strip()==''
    before=sha(profile)
    # Explicit arithmetic locks and disabling runtime tuning retain selection contracts.
    for carrier in (0,a.exponent):
        text=call('locked_'+str(carrier),['--save',str(out/'curves.save'),'--curves','1','--b2','100000000',
            '--d','30030','--carrier-exponent',str(carrier),'--plan-only'])
        plans=[json.loads(s) for s in text.splitlines() if s.startswith('{')]
        selection=next(s for s in plans if s.get('type')=='tune_selection')
        assert selection['carrier_exponent']==carrier and selection['D']==30030
    call('disabled',['--save',str(out/'curves.save'),'--curves','1','--b2','100000000','--d','30030',
        '--tune-runtime','0','--carrier-exponent','0','--results',str(out/'disabled.jsonl'),'--log',str(out/'disabled.engine.log')])
    assert sha(profile)==before
    disabled=normalize_result(json.loads((out/'disabled.jsonl').read_text(encoding='utf-8')))
    assert 'tune_observation' not in disabled
    assert disabled['tune_plan']['model']=='runtime_shape_mean_v2' and not disabled['tune_plan']['runtime_probe']
    if a.mont_save:
        text=call('auto_b2_plan',['--save',str(a.mont_save.resolve()),'--curves','1','--auto-b2',
            '--auto-min-b2','100000000','--auto-max-b2','100000000','--stage1-seconds-per-curve','1',
            '--d','30030','--carrier-exponent','0','--plan-only'])
        auto_rows=[json.loads(s) for s in text.splitlines() if s.startswith('{')]
        assert any(s.get('type')=='tune_selection' and s.get('reason')=='runtime_measured_hysteresis' for s in auto_rows)
        assert sha(profile)==before
    if a.factor_save:
        call('factor',['--save',str(a.factor_save.resolve()),'--curves','1','--b2','100000000','--d','30030',
            '--carrier-exponent','0','--results',str(out/'factor.jsonl'),'--log',str(out/'factor.engine.log')])
        factor=normalize_result(json.loads((out/'factor.jsonl').read_text(encoding='utf-8')))
        assert any(int(f)%1000367==0 for f in factor['factors']) and 'tune_observation' not in factor
        assert sha(profile)==before
    assert all(sha(Path(f))==h for f,h in identities.items())
    report=dict(complete=True,device=a.device,identities=identities,production_curves=count,
        sample_limit=a.sample_limit,default_limit_tested=a.sample_limit==10,
        carrier_first=True,per_path_sample_counts=counts,stable_final_carrier=rows[-1]['carrier_exponent'],
        profile_unchanged_when_disabled=True,read_only_statistics=True,auto_b2_plan=bool(a.mont_save),
        explicit_locks=True,factor_excluded=bool(a.factor_save),
        timing_scope='functional integration only; no performance claim')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
