"""Compare the GUI helper's five curve identities with actual PARI/GP.

Only isolated request/response files are written. No workers, GPU or dataset edits.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for key in ['helper','gp','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir();helper=a.helper.resolve();gp=a.gp.resolve();cases=[]
    for prime in ['1009','1013','10007']:
        for profile in [0,1,3,4,5]:
            pair=[]
            for backend in [1,2]:
                stem=f'{prime}-{profile}-{backend}';req=out/(stem+'-in.json');res=out/(stem+'-out.json')
                req.write_text(json.dumps(dict(factor=prime,sigma='26',profile=profile,backend=backend,gp=str(gp))),encoding='utf-8')
                run=subprocess.run([str(helper),str(req),str(res)],capture_output=True,timeout=45)
                assert run.returncode==0,(stem,run.stderr);r=json.loads(res.read_text());pair.append(r)
                assert r['status']=='complete' and r['group_computed'] and r['point_minimality'],(stem,r)
                assert r['backend']==('gp' if backend==1 else 'native'),r
                for field,value in [('group_factors','group_order'),('point_factors','point_order')]:
                    product=1
                    for f in r[field]:product*=int(f['prime'])**int(f['exponent'])
                    assert product==int(r[value]),r
            for key in ['group_order','point_order','group_factors','point_factors']:
                assert pair[0][key]==pair[1][key],(prime,profile,key,pair)
            cases.append(dict(prime=prime,profile=profile,group=pair[0]['group_order'],point=pair[0]['point_order']))
    result=dict(passed=True,cases=cases,helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),gp_sha256=hashlib.sha256(gp.read_bytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
