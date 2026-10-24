"""Qualify the local PARI C-API path through 192 bits, without GP on PATH.

GP is invoked separately only as a functional oracle. Native timings are serial
full helper-process wall times, including startup, proofs and point validation.
"""
import argparse,hashlib,json,os,statistics,subprocess,time,winreg
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def product(rows):
    value=1
    for row in rows:value*=int(row['prime'])**int(row['exponent'])
    return value
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['helper','gp','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3);a=p.parse_args();assert a.repeats>=3
    helper=a.helper.resolve();gp=a.gp.resolve();out=a.out.resolve();out.mkdir()
    cases=[dict(name='user',factor='3436193553215096358758503',sigma='1955774944',profile=0)]
    for name,prime in [('p127',(1<<127)-1),('p160',(1<<160)-(1<<31)-1),('p192',(1<<192)-(1<<64)-1)]:
        for profile in [0,1,3,4,5]:cases.append(dict(name=f'{name}-{profile}',factor=str(prime),sigma='26',profile=profile))
    (out/'inputs.json').write_text(json.dumps(cases,indent=2))
    env=os.environ.copy();env['PATH']=str(Path(os.environ['SystemRoot'])/'System32')
    def run(case,backend,stem):
        request=out/(stem+'-in.json');response=out/(stem+'-out.json')
        request.write_text(json.dumps(dict(factor=case['factor'],sigma=case['sigma'],profile=case['profile'],backend=backend,gp=str(gp) if backend==1 else str(out/'absent-gp.exe'))))
        start=time.perf_counter();proc=subprocess.run([str(helper),str(request),str(response)],env=env,capture_output=True,timeout=40 if backend==1 else 8)
        wall=(time.perf_counter()-start)*1000
        (out/(stem+'-stderr.txt')).write_bytes(proc.stderr);value=json.loads(response.read_text())
        assert proc.returncode in ([0] if backend==1 else [0,124]),(stem,proc.returncode,value)
        if backend==2:
            assert value['backend']=='pari' and value['native_engine']=='pari',(stem,value)
            assert wall<7000,(stem,wall) # watchdog/OS scheduling allowance, not completion qualification
        return value,wall
    rows=[];references={}
    for case in cases:
        reference,_=run(case,1,case['name']+'-gp');assert reference['status']=='complete',(case,reference)
        references[case['name']]=reference
        result,wall=run(case,2,case['name']+'-native')
        assert result['status'] in ['complete','partial','resource_limit'],(case,result)
        if result.get('group_computed'):assert result['group_order']==reference['group_order']
        if result['status']=='complete':
            for key in ['group_order','point_order','group_factors','point_factors']:assert result[key]==reference[key],(case,key,result,reference)
            assert result['all_proven'] and result['modulus_primality']=='proven' and result['point_minimality']
            assert product(result['group_factors'])==int(result['group_order']) and product(result['point_factors'])==int(result['point_order'])
        rows.append(dict(**case,bits=int(case['factor']).bit_length(),status=result['status'],wall_ms=wall))
    repeats={}
    for name in ['user','p192-0']:
        case=next(c for c in cases if c['name']==name);samples=[]
        for index in range(a.repeats):
            result,wall=run(case,2,f'{name}-repeat-{index}')
            assert result['status']=='complete' and result['all_proven'] and wall<5000,(name,result,wall)
            for key in ['group_order','point_order','group_factors','point_factors']:assert result[key]==references[name][key]
            samples.append(wall)
        repeats[name]=dict(samples_ms=samples,median_ms=statistics.median(samples),max_ms=max(samples))
    assert any(r['bits']==192 and r['status']=='complete' for r in rows)
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    result=dict(passed=True,cases=rows,repeat_timings=repeats,cpu=cpu,
        boundary='Native helper process startup to exit, serial runs; count, factorization, primality proofs and independent GMP point checks included. Shared desktop; frequency uncontrolled and power not measured. GP timings are not compared.',
        native_gp_path_missing=True,native_path_system_only=True,
        helper_sha256=sha(helper),pari_sha256=sha(helper.parent/'libpari.dll'),gp_sha256=sha(gp))
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
