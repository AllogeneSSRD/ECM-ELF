"""Reject mixed provenance and CPU/fallback baby data in a profile4 fit. CPU only."""
import argparse,copy,hashlib,json,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurements',type=Path,required=True);p.add_argument('--anchors',type=Path,required=True)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh directory')
    repo=Path(__file__).resolve().parents[2];m=json.loads(a.measurements.read_text());anchor=json.loads(a.anchors.read_text())
    rows=[]
    def run(name,data,code):
        file=out/(name+'.json');file.write_text(json.dumps(data));result=out/(name+'_fit.json')
        cmd=[sys.executable,str(repo/'tools/bench/fit_stage2_d.py'),'--measurements',str(a.measurements.resolve()),
             '--anchor-measurements',str(file),'--output',str(result)]
        r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20)
        (out/(name+'.log')).write_bytes(r.stdout)
        assert (r.returncode==0)==(code==0),(name,r.stdout[-1500:])
        if code==0:assert json.loads(result.read_text())['rates']==json.loads(a.fit.read_text())['rates']
        else:assert not result.exists() and b'ValueError:' in r.stdout,name
        rows.append(dict(name=name,exit=r.returncode))
    run('matching',anchor,0)
    for key,value in (('sha256','0'*64),('device',99),('Q_line','wrong'),('sources',{})):
        data=copy.deepcopy(anchor);data[key]=value;run('reject_'+key,data,1)
    data=copy.deepcopy(anchor);data['env']['NTT_BABY_DEVICE']='0';run('reject_cpu_controls',data,1)
    data=copy.deepcopy(anchor);data['runs'][0]['phases']['affine']+=1;run('reject_altered_phases',data,1)
    # Same declared controls, but raw log from the CPU branch of the frozen A/B.
    data=copy.deepcopy(anchor);first=Path(data['runs'][0]['log'])
    cpu=first.parent/'1_0.log';assert cpu.exists()
    data['runs'][0]['log']=str(cpu);run('reject_cpu_raw_log',data,1)
    (out/'summary.json').write_text(json.dumps(dict(passed=len(rows),failed=0,fit_sha256=hashlib.sha256(a.fit.read_bytes()).hexdigest(),runs=rows),indent=2))
    print('TOTAL',len(rows),'input gates passed / 0 failed')


if __name__=='__main__':main()
