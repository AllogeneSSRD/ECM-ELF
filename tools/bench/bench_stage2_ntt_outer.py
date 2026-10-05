"""Fixed-save whole Stage2 A/B for the compiler-default and outer ILP schedules."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--baseline-sources',type=Path,required=True)
    p.add_argument('--save',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--runs',type=int,choices=(4,8),default=8)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert not any(out.iterdir()),'Use a fresh output directory'
    exes={0:a.baseline.resolve(),4:a.exe.resolve()};closures={0:a.baseline_sources.resolve(),4:repo}
    manifests={u:json.loads((exe.parent/'build_manifest.json').read_text(encoding='utf-8-sig')) for u,exe in exes.items()}
    sources={}
    for u,m in manifests.items():
        assert m['gl_fixed_mode']==3 and m['architecture']=='sm_89'
        assert m.get('outer_unroll_u',0)==u
        sources[u]={r[1]:r[2].lower() for line in m['sources']
            if (r:=re.fullmatch(r'([^=]+\.(?:cu|cuh|cpp|h|ps1))=([A-Fa-f0-9]{64})',line))}
        assert len(sources[u])==19
    # Permit only the scheduling option, reporting, D guard and build signature.
    changes={name for name in sources[0] if sources[0][name]!=sources[4][name]}
    assert set(sources[0])==set(sources[4])
    assert changes=={'tools/bench/ntt_coop_outer.cuh','tools/bench/stage2_tree_gpu.cu','tools/build/build_ecm_cuda_stage2.ps1'},changes
    saved=a.save.resolve();saved_sha=digest(saved)
    saved_x=re.search(rb'\bX=(?:0x)?([0-9a-fA-F]+)',saved.read_bytes().splitlines()[0])[1].decode().lower().lstrip('0')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_D_MODEL='0',NTT_ARENA_CAP_KB='6451200',
               NTT_STAGE1_Q_DUMP='1',NTT_POINT_MERSENNE='1')
    rows=[];driver_sha=digest(Path(__file__))
    def verify():
        assert digest(saved)==saved_sha and digest(Path(__file__))==driver_sha
        for u,exe in exes.items():
            assert digest(exe)==manifests[u]['sha256'].lower()
            for name,want in sources[u].items():assert digest(closures[u]/name)==want,name
    for u in (0,4,4,0,4,0,0,4)[:a.runs]:
        verify();name=f'{len(rows)+1}_u{u}'
        cmd=[str(exes[u]),'--save',str(saved),'--b2','2011326186870','--d','1381380',
             '--device',str(a.device),'--results',str(out/(name+'.jsonl')),'--log',str(out/(name+'_engine.log'))]
        with (out/(name+'_driver.log')).open('wb') as log:
            run=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        verify();assert run.returncode==0,(name,run.returncode)
        text=(out/(name+'_engine.log')).read_text(encoding='utf-8',errors='replace')
        for token in ('stage1_skipped=1','gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1',
                      'point_arithmetic: xadd6=1','fixed=3','hash=4244971527793015097',
                      'signature=c85031f6149bae11','point_mersenne_mode: requested=1 enabled=1 bits=4423 nw=70',
                      'd_model: requested=0 enabled=0 version=legacy_56_1'):
            assert token in text,(name,token)
        if u:assert f'ntt_outer_schedule: unroll_u={u}' in text
        elif 'ntt_outer_schedule:' in text:assert 'ntt_outer_schedule: unroll_u=0' in text
        else:assert b'NTT_OUTER_UNROLL_U' not in (closures[0]/'tools/bench/ntt_coop_outer.cuh').read_bytes()
        assert re.search(r'real_setup_Q_full: hex=([0-9a-f]+)',text)[1]==saved_x
        result=json.loads((out/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()[-1])
        assert result['bad_factors']==0 and result['factors']==[]
        wall={k:float(v) for k,v in re.findall(r'(init|main|total)=([0-9.]+)',re.search(r'stage2_full_wall: (.*)',text)[1])}
        s4=dict(re.findall(r'(\w+)=([^ ]+)',re.search(r's4_multiply_stats: (.*)',text)[1]))
        coverage={k:s4[k] for k in ('launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks')}
        if rows:assert coverage==rows[0]['coverage']
        rows.append(dict(name=name,unroll_u=u,command=cmd,wall=wall,coverage=coverage))
        print(name,wall,flush=True)
        (out/'measurements.json').write_text(json.dumps(dict(runs=rows),indent=2),encoding='utf-8')
    means={str(u):{phase:statistics.mean(r['wall'][phase] for r in rows if r['unroll_u']==u)
           for phase in ('init','main','total')} for u in (0,4)}
    result=dict(device=a.device,manifests=manifests,sources=sources,driver_sha256=driver_sha,
        save_sha256=saved_sha,Q_sha256=hashlib.sha256(saved_x.encode()).hexdigest(),env=env,runs=rows,means=means,
        gain_percent={phase:100*(1-means['4'][phase]/means['0'][phase]) for phase in ('init','main','total')},
        passed=len(rows),failed=0,scope='Fixed saved Q/B2/D/checks and point1; serial ABBA+BAAB, no CI; D model off')
    (out/'measurements.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(means=means,gain_percent=result['gain_percent'])),flush=True)


if __name__=='__main__':main()
