"""CPU-only provenance rejection gates for a fixed PTX D fit."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--anchors',type=Path,required=True)
    p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    assert not any(a.output.iterdir()),'Use a fresh output directory'
    repo=Path(__file__).resolve().parents[2]
    original=json.loads(a.measurements.read_text(encoding='utf-8'))
    anchor=json.loads(a.anchors.read_text(encoding='utf-8'))
    expected=json.loads(a.fit.read_text(encoding='utf-8'))
    assert original['feature_profile']==expected['feature_profile']==5
    rows=[]

    def run(name,m,an,accepted=False):
        mf=a.output/(name+'_measurements.json');af=a.output/(name+'_anchors.json')
        mf.write_text(json.dumps(m),encoding='utf-8');af.write_text(json.dumps(an),encoding='utf-8')
        fit=a.output/(name+'_fit.json')
        result=subprocess.run([sys.executable,str(repo/'tools/bench/fit_stage2_d.py'),
            '--measurements',str(mf),'--anchor-measurements',str(af),'--output',str(fit)],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        (a.output/(name+'.log')).write_bytes(result.stdout)
        if accepted:
            assert result.returncode==0,(name,result.stdout[-2000:])
            assert json.loads(fit.read_text(encoding='utf-8'))['rates']==expected['rates']
        else:
            assert result.returncode!=0 and not fit.exists(),(name,result.stdout[-2000:])
            assert b'ValueError:' in result.stdout,(name,result.stdout[-2000:])
        rows.append(dict(name=name,accepted=accepted,exit=result.returncode))

    run('matching',original,anchor,True)
    for key,value in [('sha256','0'*64),('device',99),('Q_line','wrong'),('sources',{}),('gl_fixed_mode',-1)]:
        an=copy.deepcopy(anchor);an[key]=value;run('anchor_'+key,original,an)
    for key,value in [('NTT_GL_PTX_REDUCE','0'),('NTT_BABY_DEVICE','0')]:
        an=copy.deepcopy(anchor);an['env'][key]=value;run('anchor_'+key,original,an)
    an=copy.deepcopy(anchor);an['ntt_weights']['weights']['24']*=1.01
    run('anchor_mixed_weights',original,an)
    an=copy.deepcopy(anchor);an['runs'][0]['phases']['affine']+=1
    run('anchor_altered_phases',original,an)
    an=copy.deepcopy(anchor);an['runs'][0]['features']['D']+=2
    run('anchor_altered_features',original,an)
    for name,key,value in [('runtime_control','gl_fixed_mode',-1),('historical_profile','feature_profile',4)]:
        m=copy.deepcopy(original);m[key]=value;run(name,m,anchor)
    m=copy.deepcopy(original);m['sources']['tools/bench/ntt_poly_probe.cu']='0'*64
    run('different_ntt_sources',m,anchor)
    m=copy.deepcopy(original);m['ntt_weights']['weights']['24']*=1.01
    run('altered_weights',m,anchor)
    m=copy.deepcopy(original);m['ntt_weights']['measurements_sha256']='0'*64
    run('altered_weight_evidence',m,anchor)
    m=copy.deepcopy(original);del m['ntt_weights']['weights']['20']
    run('incomplete_weights',m,anchor)
    m=copy.deepcopy(original);m['runs'][0]['features']['D']+=2
    run('altered_curve_features',m,anchor)
    for key in ('NTT_GL_PTX_REDUCE','NTT_BABY_DEVICE'):
        m=copy.deepcopy(original);m['env'][key]='0';run('measurement_'+key,m,anchor)
    # A declared fixed3 backend cannot validate a raw log from runtime PTX.
    for target in ('measurement','anchor'):
        m=copy.deepcopy(original);an=copy.deepcopy(anchor)
        data=m if target=='measurement' else an
        source=Path(data['runs'][0]['log']);text=source.read_text(encoding='utf-8')
        assert 'ptx=1 fixed=3' in text
        raw=a.output/(target+'_wrong_backend_raw.log')
        raw.write_text(text.replace('ptx=1 fixed=3','ptx=1 fixed=-1'),encoding='utf-8')
        data['runs'][0]['log']=str(raw);run(target+'_wrong_raw_backend',m,an)
    (a.output/'summary.json').write_text(json.dumps(dict(passed=len(rows),failed=0,runs=rows,
        fit_sha256=hashlib.sha256(a.fit.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print('TOTAL',len(rows),'input gates passed / 0 failed')


if __name__=='__main__':main()
