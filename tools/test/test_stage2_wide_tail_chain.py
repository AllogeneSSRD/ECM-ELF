"""Cross-build wide-tail checks with independent Stage1 and ladder references.

These are diagnostic correctness runs, never formal timing samples. Prepared
fixtures come from test_stage2_wide_native.py and retain GMP Stage1 evidence.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from bench_stage2_production import fields,freeze,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('use a fresh evidence directory')
    fixtures=json.loads(a.fixtures.read_text());assert fixtures['complete']
    exes=dict(baseline=a.baseline.resolve(),candidate=a.candidate.resolve())
    report=dict(complete=False,identities={k:freeze(v) for k,v in exes.items()},
                fixtures_sha256=sha(a.fixtures),collector_sha256=sha(__file__),runs=[])
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_GFINV_SEG_CHECK='1',CUDA_LAUNCH_BLOCKING='0')
    ini=out/'bench.ini';ini.write_text('user=\ncomputer=\n')
    cases=(('generic8193',8191),('generic8193',8192),('generic16384',8192),
           ('factor16384',8192),('base_nonunit16384',8192))
    for name,count in cases:
        case=next(c for c in fixtures['fixtures'] if c['name']==name)
        assert sha(case['save'])==case['save_sha256'] and case['stage1_cpu_gmp_equal']
        pair={}
        for arm,exe in exes.items():
            tag=f'{name}_i{count}_{arm}';folder=out/tag;folder.mkdir()
            command=[exe,'--ini',ini,'--save',case['save'],'--b2',30030*(count-2),'--d',30030,
                     '--device',a.device,'--curves',1,'--short-calibration',0,'--tune-runtime',0,
                     '--carrier-exponent',0,'--arena-mb',6300,'--owner-budget-mb',640,'--batch-mb',256,
                     '--factor-only','--log-level','debug','--results',folder/'internal.jsonl',
                     '--submission',folder/'results.json.txt','--log',folder/'engine.log',
                     '--debug-log-file',folder/'debug.log']
            use=env|dict(NTT_GIANT_CHAIN_CHECK='1' if case['unit'] else '0')
            proc=subprocess.run(list(map(str,command)),cwd=out,env=use,capture_output=True,timeout=1200)
            (folder/'driver.log').write_bytes(proc.stdout+proc.stderr)
            assert proc.returncode==0,tag
            text=(folder/'engine.log').read_text()+'\n'+(folder/'debug.log').read_text()
            result=json.loads((folder/'internal.jsonl').read_text())
            w=(case['bits']+63)//64;minimum=8192 if arm=='candidate' and w>128 else 32768
            expected_chain=count if count>=minimum else 0
            checks=[fields(line,'giant_chain_check') for line in text.splitlines() if line.startswith('giant_chain_check:')]
            if case['unit']:
                assert sum(int(c['points']) for c in checks)==expected_chain
                assert all(int(c['mismatches'])==0 for c in checks)
            if arm=='candidate':
                policy=fields(text,'giant_chain_policy')
                assert int(policy['effective_min'])==minimum and int(policy['npts'])==count
                nonunit=not case['unit']
                assert policy['route']==('chain' if expected_chain and not nonunit else 'ladder')
                if nonunit:
                    fallback=fields(text,'wide_tail_ladder_fallback')
                    assert fallback['reason']==('nonunit_base' if case['base_gcd_hex']!='1' else 'nonunit_segment')
                    assert int(fallback['points'])==count
                    assert int(fallback['index_capacity'])==8192
            assert int(fields(text,'real_shape')['giant_points'])==count
            coverage=fields(text,'s4_multiply_stats')
            assert int(coverage['gmp_selftest_cases'])>0 and int(coverage['gmp_checked'])>0
            assert int(coverage['gmp_selftest_bad'])==int(coverage['gmp_check_bad'])==result['bad_factors']==0
            n=int(case['N_hex'],16)
            assert all(1<int(f)<n and n%int(f)==0 for f in result['factors'])
            if case['expected_factor']:assert any(int(f)%case['expected_factor']==0 for f in result['factors'])
            signature=fields(text,'descent_values'),sorted(result['factors'])
            projective=fields(text,'real_batched_projective')
            assert projective['leaves']==projective['gamma_points']
            if not case['unit']:
                # Once Z is a nonunit, affine_x_gmp_checked records its GCD
                # and uses the projective X word as a placeholder. That word
                # depends on the ladder/chain representative; it is not a
                # canonical polynomial evaluation in the composite ring.
                assert result['factors'] and int(projective['affine_fallback_points'])>0
            pair[arm]=signature
            report['runs'].append(dict(case=name,points=count,arm=arm,expected_chain_points=expected_chain,
                leaf=signature[0],factors=signature[1],coverage=coverage,projective=projective,
                log=str(folder/'engine.log'),log_sha256=sha(folder/'engine.log')))
            (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            print('PASS',tag,flush=True)
        assert pair['baseline'][1]==pair['candidate'][1],name+' factor mismatch'
        assert pair['baseline'][0]==pair['candidate'][0],name+' complete canonical/fallback leaf mismatch'
    report['complete']=True;(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',len(report['runs']),'wide-tail checks; no performance samples')


if __name__=='__main__':main()
