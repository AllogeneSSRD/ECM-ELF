"""Compare every supported 4608 PRAC policy against the independent CPU field backend."""
import argparse
import hashlib
import json
from pathlib import Path

from test_cuda_prac import rows,run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,default=1)
    args=parser.parse_args()
    exe=args.exe.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    identity=hashlib.sha256(exe.read_bytes()).hexdigest()
    settings=('(2^4423-1)',1000,2,26,'lcm')
    reference=out/'cpu';run(exe,reference,*settings,'cpu',args.device)
    expected=rows(reference);assert len(expected)==2
    policies=[('baseline',cap,16,'none') for cap in (0,168,255)]
    policies += [(variant,cap,16,'none') for variant in ('compact','outline-add','single-add','single-compact')
                 for cap in (168,255)]
    policies += [('single-compact',128,16,'none')]
    policies += [('single-compact',cap,32,'none') for cap in (168,255)]
    policies += [('single-compact',168,16,policy) for policy in ('runtime','np0','m4423')]
    policies += [('baseline',0,32,'none'),('baseline',255,32,'none')]
    report=dict(complete=False,binary_sha256=identity,device=args.device,cases=[])
    for variant,cap,tpi,constants in policies:
        name=f'{variant}_{cap}_tpi{tpi}_{constants}'
        folder=out/name
        text=run(exe,folder,*settings,'prac',args.device,tpi=tpi,registers=cap,
                 variant=variant,constants=constants)
        assert rows(folder)==expected,name
        assert f'CGBN<{tpi},4608>' in text,name
        policy='natural (255)' if cap==255 else '168 (4608 candidate)' if cap==168 else 'per-tier'
        if cap!=128:assert 'PRAC register policy='+policy in text,name
        report['cases'].append(dict(variant=variant,registers=cap,tpi=tpi,constants=constants,curves=2))
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PASS',name,flush=True)
    assert hashlib.sha256(exe.read_bytes()).hexdigest()==identity
    report['complete']=True
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
