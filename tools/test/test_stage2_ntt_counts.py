"""Check optional per-curve NTT tables on real Edwards/Montgomery saves, offline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ecm_report'))
from result_format import normalize_result


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tables(text):
    result=[]
    for block in text.split('ntt_count_table:')[1:]:
        rows=[tuple(map(int,m)) for m in re.findall(r'^ntt_count_row:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$',block,re.M)]
        assert rows and [r[0] for r in rows]==sorted({r[0] for r in rows})
        for length,k,c,f,i,total in rows:
            assert length==1<<k and c>0 and (f,i,total)==(2*c,c,3*c)
        histogram=re.search(r'^ntt_length_counts: log2_length:num_of_ntt (.+)$',block,re.M)
        assert histogram and dict((int(k),int(v)) for k,v in re.findall(r'(\d+):(\d+)',histogram[1]))=={r[1]:r[5] for r in rows}
        work=re.search(r'^ntt_work: (.+)$',block,re.M)
        assert work
        fields=dict(re.findall(r'(\w+)=(\S+)',work[1]))
        assert fields['scope']=='curve_polynomial'
        assert int(fields['convolutions'])==sum(r[2] for r in rows)
        assert int(fields['transforms'])==sum(r[5] for r in rows)
        assert abs(float(fields['elements'])-sum(r[0]*r[5] for r in rows))<=max(1,sum(r[0]*r[5] for r in rows)*1e-6)
        result.append(rows)
    return result


def invoke(exe,save,out,name,text='',extra=(),b2=40000,d=210,curves=1,device=1,old=False,good=True):
    prefix=out/name;config=prefix.with_suffix('.toml')
    config.write_text('[shared]\nuser=""\ncomputer=""\n[stage2]\nstage2_short_calibration=false\nstage2_tune_runtime=false\n'+text,encoding='utf-8')
    log=prefix.with_suffix('.engine.log');debug=prefix.with_suffix('.debug.log');result=prefix.with_suffix('.jsonl')
    cmd=list(map(str,[exe,'--config',config,'--save',save,'--b2',b2,'--d',d,'--curves',curves,'--device',device,
        '--short-calibration',0,'--tune-runtime',0,'--factor-only','--results',result,
        '--submission',prefix.with_suffix('.json.txt'),'--log',log,'--debug-log-file',debug,'--log-level','curve',*extra]))
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_') and k!='CUDA_VISIBLE_DEVICES'}
    start=time.perf_counter();proc=subprocess.run(cmd,capture_output=True,env=env,timeout=900,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));wall=time.perf_counter()-start
    stdout=(proc.stdout+proc.stderr).decode('utf-8',errors='replace')
    prefix.with_suffix('.stdout.log').write_text(stdout,encoding='utf-8')
    if not good:
        assert proc.returncode and not result.exists();return dict(name=name,returncode=proc.returncode)
    assert not proc.returncode,name+' failed; inspect ignored evidence'
    normal=log.read_text(encoding='utf-8');detail=debug.read_text(encoding='utf-8')
    receipts=[normalize_result(json.loads(s)) for s in result.read_text().splitlines()]
    assert len(receipts)==curves and all(r['status']=='stage2_completed' and not r['bad_factors'] for r in receipts)
    checks=[dict(re.findall(r'(\w+)=(\S+)',s)) for s in re.findall(r'^s4_multiply_stats: (.+)$',normal+'\n'+detail,re.M)]
    assert checks and all(int(c['gmp_checked'])>0 and int(c['gmp_selftest_cases'])>0 and c['gmp_check_bad']=='0' and c['gmp_selftest_bad']=='0' for c in checks)
    ledger=[dict(re.findall(r'(\w+)=(\S+)',s)) for s in re.findall(r'^stage2_full_wall: (.+)$',normal,re.M)]
    assert len(ledger)==curves and all(c['clean']=='1' for c in ledger)
    leaf=re.findall(r'^(?:target_)?descent_values:.*?hash=(\d+)',detail,re.M)
    assert leaf
    return dict(name=name,command=cmd,process_seconds=wall,engine_seconds=[float(c['total']) for c in ledger],
        tables=[] if old else tables(normal),leaf_hashes=leaf,factors=[r['factors'] for r in receipts],
        legacy_histograms=re.findall(r'^ntt_length_counts: log2_length:num_of_ntt (.+)$',normal,re.M),
        stdout_tables=len(re.findall(r'^ntt_count_table:',stdout,re.M)),result_sha256=sha(result))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--old-exe',type=Path)
    p.add_argument('--ed-save',type=Path,required=True);p.add_argument('--mont-save',type=Path,required=True)
    p.add_argument('--device',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe=a.exe.resolve();ed=a.ed_save.resolve();mont=a.mont_save.resolve()
    report=dict(complete=False,exe_sha256=sha(exe),script_sha256=sha(__file__),device=a.device,runs=[])
    tests=[('default','',[],False),('toml_on','stage2_ntt_counts=true\n',[],True),
        ('worker_off','stage2_ntt_counts=true\n[workers.2.stage2]\nstage2_ntt_counts=false\n',['--worker',2],False),
        ('worker_on','stage2_ntt_counts=false\n[workers.2.stage2]\nstage2_ntt_counts=true\n',['--worker',2],True),
        ('cli_off','stage2_ntt_counts=true\n',['--ntt-counts',0],False),
        ('cli_on','stage2_ntt_counts=false\n',['--ntt-counts',1],True),
        ('quiet_on','stage2_ntt_counts=true\n',['--log-level','quiet'],True)]
    for name,text,extra,on in tests:
        row=invoke(exe,ed,out,name,text,extra,device=a.device)
        assert bool(row['tables'])==on and bool(row['stdout_tables'])==on
        if report['runs']:
            assert (row['leaf_hashes'],row['factors'])==(report['runs'][0]['leaf_hashes'],report['runs'][0]['factors'])
        report['runs'].append(row);print(name,'PASS',flush=True)
    for name,save,curves in [('montgomery',mont,1),('two_curves',ed,2)]:
        row=invoke(exe,save,out,name,'stage2_ntt_counts=true\n',curves=curves,device=a.device)
        assert len(row['tables'])==curves
        if curves==2:assert row['tables'][0]==row['tables'][1] # Geometry repeats, counters reset per child.
        report['runs'].append(row)
    for name,text,extra in [('bad_toml','stage2_ntt_counts=2\n',[]),('bad_cli','',['--ntt-counts',2])]:
        report['runs'].append(invoke(exe,ed,out,name,text,extra,good=False,device=a.device))
    if a.old_exe:
        for name,save in [('old_ed',ed),('old_mont',mont)]:
            old=invoke(a.old_exe.resolve(),save,out,name,old=True,device=a.device)
            new=next(r for r in report['runs'] if r['name']==('toml_on' if name=='old_ed' else 'montgomery'))
            assert old['legacy_histograms']==new['legacy_histograms']
            assert (old['leaf_hashes'],old['factors'])==(new['leaf_hashes'],new['factors'])
            report['runs'].append(old)
    report['complete']=True
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS',len(report['runs']),'NTT counter gates')


if __name__=='__main__':main()
