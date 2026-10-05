"""CPU frontend fixtures: multiplicities, GP failure and INI/CLI precedence.

These factor-revealing X values test the existing saved-X shortcut; they are
not claimed to be valid completed Stage1 points or GPU Stage2 factor trials.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gp',type=Path,required=True)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise RuntimeError('Use a fresh directory')
    n=(1<<21)-1;x=7**2*127;save=root/'fixture.save'
    save.write_text(f'METHOD=ECM; PARAM=0; SIGMA=6; B1=2; N=(2^21-1); X=0x{x:x}; CHECKSUM={2*6*n*x%4294967291};\n')
    ini=root/'ecm.ini';ini.write_text(f'[stage2]\nstage2_factorize_hits=1\nstage2_gp={a.gp.resolve()}\nstage2_factor_timeout=15\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')};rows=[]
    for name,extra in [('multiplicity',[]),('missing_gp',['--gp',str(root/'missing_gp.exe')])]:
        result=root/(name+'.jsonl');log=root/(name+'.log')
        cmd=[str(a.exe.resolve()),'--ini',str(ini),'--save',str(save),'--b2','3','--device','1',
             '--results',str(result),'--log',str(log),*extra]
        r=subprocess.run(cmd,capture_output=True,cwd=root,env=env,timeout=30)
        (root/(name+'_driver.log')).write_bytes(r.stdout+r.stderr);assert r.returncode==0
        row=json.loads(result.read_text());assert row['status']=='factor_in_saved_X' and row['factors']==[str(x)]
        if name=='multiplicity':
            assert row['factorization_complete']
            parts={int(q['factor']):q['multiplicity'] for q in row['factor_analysis'][0]['prime_powers']}
            assert parts=={7:2,127:1}
        else:
            assert not row['factorization_complete'] and row['prime_factors']==[]
            assert row['factor_analysis'][0]['reason']=='gp_launch_failed'
        rows.append(row)
    (root/'summary.json').write_text(json.dumps(dict(passed=2,failed=0,scope='CPU frontend synthetic saved-X fixtures',rows=rows),indent=2))
    print('factor frontend: 2 passed, 0 failed')


if __name__=='__main__':main()
