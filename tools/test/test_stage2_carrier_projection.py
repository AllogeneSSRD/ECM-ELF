"""Check wide-tail carrier leaves in the target ring, against ordinary runs.

Extra ladder, segment, group and scale checks make these diagnostic runs;
their times are never performance samples. The input study remains untouched.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from bench_stage2_production import fields,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();study=json.loads((a.study/'plan.json').read_text())
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('use a fresh evidence directory')
    shutil.copy2(__file__,out/'collector_snapshot.py')
    report=dict(complete=False,study_sha256=sha(a.study/'plan.json'),test_sha256=sha(__file__),
                binaries={k:v['sha256'] for k,v in study['binaries'].items()},runs=[],performance_samples=0)
    def persist():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_TARGET_LEAF_HASH='1',NTT_GFINV_SEG_CHECK='1',
               NTT_GSCALE_DEVICE_CHECK='1',NTT_DEVICE_GLEAF_CHECK='1',NTT_GIANT_CHAIN_CHECK='1')
    ini=out/'bench.ini';ini.write_text('user=\ncomputer=\n')
    persist()
    cells=[c for c in study['cells'] if c['mode']=='carrier']
    if not cells:
        cells=[dict(c,id=f'm{c["exponent"]}_low_carrier',mode='carrier',carrier=c['exponent'])
               for c in study['cells'] if c['bound']=='low' and study['cases'][str(c['exponent'])]['bits']>8192]
    if not cells:raise ValueError('wide low-B2 ordinary references are required')
    for cell in cells:
        case=study['cases'][str(cell['exponent'])]
        for model in ('ed','mont'):
            ordinary=next(c for c in study['cells'] if not c['carrier'] and c['exponent']==cell['exponent'] and c['b2']==cell['b2'] and c['d']==cell['d'])
            reference=[r for r in study['runs'] if r['cell']==ordinary['id'] and r['model']==model]
            if not reference:raise ValueError('ordinary completed reference is missing')
            if len({json.dumps(r['stats']['descent_values'],sort_keys=True) for r in reference})!=1:
                raise ValueError('ordinary reference leaves differ')
            for row in reference:
                for path,digest in row['files'].items():
                    if sha(path)!=digest:raise ValueError('ordinary evidence changed')
            expected=reference[0]['stats']['descent_values']
            save=case['saves'][model]
            if sha(save)!=case['save_hashes'][model]:raise ValueError('Stage1 save changed')
            for build,binary in study['binaries'].items():
                if sha(binary['path'])!=binary['sha256']:raise ValueError('binary changed')
                tag=f'{cell["id"]}_{model}_{build}';folder=out/tag;folder.mkdir()
                command=[binary['path'],'--ini',ini,'--save',save,'--b2',cell['b2'],'--d',cell['d'],
                    '--device',study['device'],'--curves',1,'--carrier-exponent',cell['carrier'],
                    '--short-calibration',0,'--tune-runtime',0,'--arena-mb',6300,'--owner-budget-mb',640,
                    '--batch-mb',256,'--factor-only','--log-level','debug','--results',folder/'internal.jsonl',
                    '--submission',folder/'results.json.txt','--log',folder/'engine.log',
                    '--debug-log-file',folder/'debug.log']
                command=list(map(str,command))
                recorded={k:v for k,v in env.items() if k.startswith('NTT_')}
                (folder/'command.json').write_text(json.dumps(dict(command=command,environment=recorded),indent=2)+'\n')
                print('RUN',tag,flush=True)
                proc=subprocess.run(command,cwd=out,env=env,capture_output=True,timeout=1800)
                (folder/'driver.log').write_bytes(proc.stdout+proc.stderr)
                if proc.returncode:raise ValueError('native failure: '+tag)
                result=json.loads((folder/'internal.jsonl').read_text())
                expected_identity=dict(N_hex=case['N_hex'],B1=case['b1'],B2=cell['b2'],sigma=case['sigma'],
                                       curve=model,d=cell['d'],carrier_exponent=cell['carrier'])
                if any(result[key]!=value for key,value in expected_identity.items()):raise ValueError('native work identity changed')
                if result['factors'] or result['bad_factors'] or result['status']!='stage2_completed':
                    raise ValueError('not complete NF')
                text=(folder/'engine.log').read_text()+'\n'+(folder/'debug.log').read_text()
                target=fields(text,'target_descent_values');raw=fields(text,'descent_values')
                if any(target[k]!=expected[k] for k in ('leaves','words','hash')):
                    raise ValueError('complete target-ring leaves differ: '+tag)
                if int(target['target_bits'])!=case['bits'] or int(target['nonzero'])!=int(target['leaves']):
                    raise ValueError('target projection coverage mismatch')
                checks=[fields(line,'giant_chain_check') for line in text.splitlines() if line.startswith('giant_chain_check:')]
                checked=sum(int(c['points']) for c in checks)
                if not checked or any(int(c['mismatches']) for c in checks):raise ValueError('point ladder mismatch')
                if build=='candidate' and checked!=cell['b2']//cell['d']+2:
                    raise ValueError('new carrier tails were not all independently checked')
                s4=fields(text,'s4_multiply_stats');groups=fields(text,'device_gleaf')
                seed=fields(text,'real_giant_seed');scale=fields(text,'real_gscale_device')
                if int(s4['gmp_selftest_bad']) or int(s4['gmp_check_bad']) or not int(s4['gmp_checked']):
                    raise ValueError('GMP failure/missing checks')
                if not all(int(x)>0 for x in (groups['checked_groups'],groups['checked_leaf_words'],seed['segment_checks'],scale['checked_words'])):
                    raise ValueError('missing group/segment/leaf/scale oracle')
                report['runs'].append(dict(cell=cell['id'],model=model,build=build,
                    N_hex=case['N_hex'],b1=case['b1'],b2=cell['b2'],d=cell['d'],sigma=case['sigma'],
                    carrier=cell['carrier'],target=target,raw=raw,chain_points_checked=checked,
                    s4=s4,groups=groups,seed=seed,scale=scale,
                    files={str(f):sha(f) for f in folder.iterdir() if f.is_file()}))
                persist();print('PASS',tag,flush=True)
    report['complete']=True;persist();print('COMPLETE',len(report['runs']),'carrier target-ring checks')


if __name__=='__main__':main()
