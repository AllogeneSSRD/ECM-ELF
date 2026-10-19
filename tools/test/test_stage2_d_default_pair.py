"""Check paired-matrix scheduling/resume with a fake collector, without CUDA."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
import bench_stage2_d_default_pair as pair


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    out=ap.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    write=pair.collector.write;load=pair.collector.load
    fake=out/'fixture';fake.mkdir();(fake/'input.txt').write_text('compiled input',encoding='utf-8')
    old_exe=fake/'old.exe';new_exe=fake/'new.exe';old_exe.write_bytes(b'old');new_exe.write_bytes(b'new')
    write(fake/'build_manifest.json',dict(sha256=sha(new_exe),source_hashes={'input.txt':sha(fake/'input.txt')}))
    write(fake/'original.jsonl',{'source-build':sha(old_exe)})
    cases=[dict(id=f'm{i}',N_hex=str(i),sigma=26,b1=50) for i in range(13)]
    refs=[dict(curve_type='Edwards',benchmark_eligible=True,N_hex=str(i),s=26,b1=50,b2=b)
          for i in range(13) for b in (1000,10000)]
    baseline=fake/'historical';baseline.mkdir()
    old=dict(complete=True,failures=[],repeats=3,cases=cases,reference=str(fake/'reference'),
             program_version='0.38.4',exe=str(old_exe),prime95_rows=refs,runs=[dict(prefix=str(fake/'original'))])
    write(baseline/'plan.json',old)

    def ids(repeats):
        return [f'm{i}_b{b}_{mode}_r{r}' for r in range(1,repeats+1) for i in range(13)
                for b in (1000,10000) for mode in (('gpu','gpu-carrier') if r%2 else ('gpu-carrier','gpu'))]
    def prepare(args,study):
        study.mkdir();result=dict(complete=False,runs=[],repeats=args.repeats,exe=str(args.exe))
        write(study/'plan.json',result);return result
    calls=[];interrupt_after=[None]
    def run(args):
        p=args.output/'plan.json';r=load(p)
        if r['complete']:raise AssertionError('Completed collector was restarted')
        missing=[i for i in ids(r['repeats']) if i not in {v['id'] for v in r['runs']}]
        if missing:r['runs'].append(dict(id=missing[0]));calls.append((args.output.name,missing[0]))
        else:r['complete']=True
        write(p,r)
        if interrupt_after[0] is not None and len(calls)==interrupt_after[0]:raise KeyboardInterrupt
    def invoke(output,repeats=1,resume=False):
        argv=['pair','--baseline',str(baseline),'--candidate',str(new_exe),'--candidate-version','0.39.2',
              '--output',str(output),'--repeats',str(repeats)]+(['--resume'] if resume else [])
        with patch.object(pair,'ROOT',fake),patch.object(pair.collector,'prepare',prepare),\
             patch.object(pair.collector,'run',run),patch.object(pair.time,'sleep'),\
             patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):
            pair.main()
    checks=0
    for repeats in (1,2):
        study=out/f'repeats{repeats}';calls.clear();invoke(study,repeats)
        result=load(study/'pair.json');assert result['complete'] and len(calls)==104*repeats;checks+=1
        assert calls==[(s['arm'],s['id']) for s in result['schedule']];checks+=1
        assert all(calls[i][0]!=calls[i+1][0] and calls[i][1]==calls[i+1][1] for i in range(0,len(calls),2));checks+=1
        assert all(load(study/arm/'plan.json')['complete'] for arm in ('baseline','candidate'));checks+=1
    study=out/'resume';calls.clear();interrupt_after[0]=7
    try:invoke(study)
    except KeyboardInterrupt:pass
    else:raise AssertionError('Interruption was not exercised')
    assert not load(study/'pair.json')['complete'] and len(calls)==7;checks+=1
    interrupt_after[0]=None;invoke(study,resume=True)
    assert len(calls)==104 and len(set(calls))==104 and load(study/'pair.json')['complete'];checks+=1
    result=dict(complete=True,checks=checks,CUDA_calls=0,scope='Scheduling, reverse order, completion and resume; fake collector only')
    write(out/'report.json',result);print(json.dumps(result))


if __name__=='__main__':main()
