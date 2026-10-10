"""Check exact giant route work against direct scalar bit counts and NumPy fits."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import tomllib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from stage2_tune_route_cost import MODEL, predict_route, route_work


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--prediction-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    rng=random.Random(6011)
    cases=[(199802,60060,184320,32768,False), (199802,60060,184320,32768,True)]
    cases += [(rng.randint(1,20000),rng.randint(2,200000000),rng.randint(1,50000),
               rng.choice([0,32768,8192]),rng.choice([False,True])) for _ in range(180)]
    for points,d,chunk,minimum,force in cases:
        expected=route_work(points,d,chunk,minimum,force)
        direct=0;chain=ladder=chain_chunks=ladder_chunks=0
        for start in range(0,points,chunk):
            end=min(points,start+chunk)
            if force or end-start<minimum:
                ladder+=end-start;ladder_chunks+=1
                direct+=sum((i*d).bit_length()-1 for i in range(start+1,end+1))
            else:chain+=end-start;chain_chunks+=1
        assert list(expected.values())==[chain,ladder,chain_chunks,ladder_chunks,direct]
        values=list(map(int,subprocess.check_output([str(a.fixture.resolve()),'--giant-work',
            str(points),str(d),str(chunk),str(minimum),str(int(force))],text=True).split()))
        assert values==list(expected.values()),(points,d,chunk,expected,values)
    assert subprocess.run([str(a.fixture.resolve()),'--giant-work','0','6','1','0','0'],capture_output=True).returncode!=0
    assert subprocess.run([str(a.fixture.resolve()),'--giant-work','18446744073709551615','2','1','0','1'],capture_output=True).returncode!=0
    native_log=a.prediction_dir/'mixed_routes.log'
    data=tomllib.loads((a.prediction_dir/'mixed_routes.toml').read_text(encoding='utf-8'))
    b2=(172800+15000-2)*180180
    py=predict_route(list(data['ecm'].values()),b2,MODEL)
    native=json.loads(native_log.read_text(encoding='utf-8'))
    assert py and native['eligible'] and abs(py['seconds']-native['seconds'])<1e-9
    for chunk,minimum,force,expected in [(172800,32768,0,1),(345600,32768,0,0),
                                      (172800,32769,0,0),(172800,32768,1,0)]:
        actual=int(subprocess.check_output([str(a.fixture.resolve()),'--work-policy',
            str(a.prediction_dir/'mixed_routes.toml'),str(chunk),str(minimum),str(force)],text=True))
        assert actual==expected,(chunk,minimum,force,actual)
    report=dict(route_cases=len(cases),invalid_cases=2,independent_scalar_bit_counts=True,
                independent_numpy_regression=True,real_failed_shape=cases[0],runtime_policy_cases=4)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
