"""Fixed geometry native Stage2 comparisons for the optional point fold."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--save',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--baseline',type=Path,help='Frozen production binary; compare against candidate point mode1')
    p.add_argument('--baseline-sources',type=Path,help='Its immutable raw source closure')
    p.add_argument('--baseline-point-mersenne',type=int,choices=(0,1),default=0,
                   help='Actual point mode of a frozen baseline (1 for a prior point-fold experiment)')
    p.add_argument('--runs',type=int,choices=(4,8),help='4 ABBA or 8 ABBA+BAAB; default 4 with a baseline, otherwise 8')
    a=p.parse_args();root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();save=a.save.resolve();save_sha=hashlib.sha256(save.read_bytes()).hexdigest()
    manifest=json.loads((exe.parent/'build_manifest.json').read_text(encoding='utf-8-sig'))
    sources={s.split('=',1)[0]:s.split('=',1)[1].lower() for s in manifest['sources'] if '=' in s and (root/s.split('=',1)[0]).is_file()}
    sha=hashlib.sha256(exe.read_bytes()).hexdigest();assert sha==manifest['sha256'].lower()
    baseline=a.baseline.resolve() if a.baseline else None
    old_sha=hashlib.sha256(baseline.read_bytes()).hexdigest() if baseline else None
    if baseline:
        assert a.baseline_sources
        old_manifest=json.loads((baseline.parent/'build_manifest.json').read_text(encoding='utf-8-sig'))
        assert old_manifest['sha256'].lower()==old_sha
        old_sources={s.split('=',1)[0]:s.split('=',1)[1].lower() for s in old_manifest['sources'] if '=' in s and (a.baseline_sources/s.split('=',1)[0]).is_file()}
        assert len(old_sources)>=18
    def verify():
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha
        assert hashlib.sha256(save.read_bytes()).hexdigest()==save_sha
        for name,h in sources.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
        if baseline:
            assert hashlib.sha256(baseline.read_bytes()).hexdigest()==old_sha
            for name,h in old_sources.items():assert hashlib.sha256((a.baseline_sources/name).read_bytes()).hexdigest()==h
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_D_MODEL='0',NTT_ARENA_CAP_KB='6451200',NTT_STAGE1_Q_DUMP='1')
    saved_x=re.search(rb'\bX=(?:0x)?([0-9a-fA-F]+)',save.read_bytes().splitlines()[0])[1].decode().lower().lstrip('0')
    runs=[]
    count=a.runs if a.runs is not None else 4 if baseline else 8
    order=(0,1,1,0,1,0,0,1)[:count]
    for mode in order:
        verify();name=f'{len(runs)+1}_{mode}';selected=baseline if baseline and mode==0 else exe
        command=[str(selected),'--save',str(save),'--b2','2011326186870','--d','1381380','--device',str(a.device),'--results',str(out/(name+'.jsonl')),'--log',str(out/(name+'_engine.log'))]
        actual_mode=a.baseline_point_mersenne if baseline and mode==0 else mode
        with (out/(name+'_driver.log')).open('wb') as log:
            r=subprocess.run(command,env=env|{'NTT_POINT_MERSENNE':str(actual_mode)},stdout=log,stderr=subprocess.STDOUT,timeout=600)
        verify();assert r.returncode==0,(name,r.returncode)
        text=(out/(name+'_engine.log')).read_text(encoding='utf-8',errors='replace')
        for token in ('stage1_skipped=1','gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1','point_arithmetic: xadd6=1','fixed=3','hash=4244971527793015097','signature=c85031f6149bae11'):
            assert token in text,(name,token)
        if not (baseline and mode==0):assert f'point_mersenne_mode: requested={mode} enabled={mode} bits=4423 nw=70' in text
        elif actual_mode:assert f'point_mersenne_mode: requested={actual_mode} enabled={actual_mode} bits=4423 nw=70' in text
        q=re.search(r'real_setup_Q_full: hex=([0-9a-f]+)',text)[1];assert q==saved_x
        result=json.loads((out/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()[-1]);assert result['bad_factors']==0 and result['factors']==[]
        wall=dict(re.findall(r'(init|main|total)=([0-9.]+)',re.search(r'stage2_full_wall: (.*)',text)[1]))
        s4=dict(re.findall(r'(\w+)=([^ ]+)',re.search(r's4_multiply_stats: (.*)',text)[1]))
        coverage={k:s4[k] for k in ('launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks')}
        if runs:assert coverage==runs[0]['coverage']
        runs.append(dict(name=name,mode=mode,actual_point_mode=actual_mode,command=command,wall={k:float(v) for k,v in wall.items()},coverage=coverage))
        print(name,wall,flush=True)
        (out/'measurements.json').write_text(json.dumps(dict(runs=runs),indent=2))
    means={str(mode):statistics.mean(r['wall']['total'] for r in runs if r['mode']==mode) for mode in (0,1)}
    summary=dict(exe=str(exe),sha256=sha,sources=sources,baseline=str(baseline) if baseline else None,baseline_sha256=old_sha,
                 save_sha256=save_sha,Q_sha256=hashlib.sha256(saved_x.encode()).hexdigest(),env=env,device=a.device,runs=runs,means=means,
                 gain_percent=100*(1-means['1']/means['0']),passed=len(runs),failed=0,scope='Fixed save/Q/D/checks, serial whole-curve runs; no CI. Explicit point modes and D/model off.')
    summary['baseline_point_mersenne']=a.baseline_point_mersenne if baseline else None
    (out/'measurements.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('runs','sources','env')}))


if __name__=='__main__':main()
