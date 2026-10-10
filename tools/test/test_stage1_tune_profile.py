"""Check native completed-Stage1 cost scopes, malformed inputs and effort grids."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('measure',ROOT/'tools/bench/tune_stage1_cost.py')
    tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    device=dict(uuid_hex='0123456789abcdef0123456789abcdef',sm_major=8,sm_minor=9,cuda_runtime=13030,cuda_driver=13030)
    sample=dict(target_bits=521,modulus_kind='mersenne',b1=20,batch=8,exponent='lcm',repeats=3,
                checked_curves=8,hits=0,bad=0,seconds=[.2,.3,.4],median_seconds=.3,mad_seconds=.1)
    text=tool.profile_text(device,1,3,[sample]);accepted=rejected=0
    def run(name,text=text,bits=521,b1=20,batch=8,exponent='lcm',kind=None,ok=True):
        nonlocal accepted,rejected
        path=out/(name+'.toml');path.write_text(text,encoding='utf-8')
        proc=subprocess.run([str(a.fixture.resolve()),'--stage1-cost',str(path),str(bits),str(b1),str(batch),exponent]+([] if kind is None else [kind]),
                            capture_output=True,text=True,errors='replace',timeout=30)
        (out/(name+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        assert (proc.returncode==0)==ok,(name,proc.stdout,proc.stderr)
        if ok:accepted+=1;assert abs(float(proc.stdout)-.3)<1e-12
        else:rejected+=1
    run('valid');run('bom','\ufeff'+text)
    choose=copy.deepcopy(sample);choose['exponent']='choose12'
    run('choose12',tool.profile_text(device,1,3,[choose]),exponent='choose12')
    generic=copy.deepcopy(sample);generic['modulus_kind']='generic'
    generic_text=tool.profile_text(device,1,3,[generic])
    run('generic',generic_text,kind='generic')
    run('generic_as_mersenne',generic_text,ok=False)
    run('mersenne_as_generic',kind='generic',ok=False)
    for name,kwargs in [('width',dict(bits=522)),('b1',dict(b1=21)),('batch',dict(batch=1)),('exponent',dict(exponent='choose12'))]:
        run(name,ok=False,**kwargs)
    for name,old,new in [
        ('device','sm_minor = 9','sm_minor = 6'),('runtime','cuda_runtime = 13030','cuda_runtime = 13040'),
        ('incomplete','complete = 1','complete = 0'),('failed','failed = 0','failed = 1'),
        ('unchecked','checked_curves = 8','checked_curves = 7'),('hit','hits = 0','hits = 1'),
        ('bad','bad = 0','bad = 1'),('median','median_seconds = 0.3','median_seconds = 0.4'),
        ('mad','mad_seconds = 0.1','mad_seconds = 0.2'),('param','param = 0','param = 3'),
        ('tpi','requested_tpi = 0','requested_tpi = 16'),('cache','exp_cache = "off"','exp_cache = "cache"'),
        ('backend','backend = "cgbn_montgomery"','backend = "cpu"'),('algorithm','algorithm = "ladder"','algorithm = "prac"'),
        ('repeats','repeats = 3','repeats = 2'),('unit','process_seconds_per_curve','projected_seconds_per_curve'),
        ('batch_zero','batch = 8','batch = 0'),('negative','[0.2, 0.3, 0.4]','[-0.2, 0.3, 0.4]'),
        ('nan','[0.2, 0.3, 0.4]','[0.2, nan, 0.4]'),('format','format = 1','format = 99'),
    ]:run(name,text.replace(old,new),ok=False)
    run('duplicate_key',text.replace('b1 = 20','b1 = 20\nb1 = 20'),ok=False)
    run('duplicate_scope',tool.profile_text(device,1,3,[sample,sample]),ok=False)
    previous=None
    for level in range(1,11):
        grid=tool.effort(level)
        if previous:
            for key in ['exponents','b1','batches']:assert set(previous[key])<=set(grid[key])
            assert previous['repeats']<grid['repeats']
        previous=grid
    assert grid['b1'][-1]==260000000 and grid['repeats']==21
    report=dict(accepted=accepted,rejected=rejected,effort_levels=10,exact_scope_only=True,
                completed_not_projected=True,exponent_and_batch_scoped=True)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
