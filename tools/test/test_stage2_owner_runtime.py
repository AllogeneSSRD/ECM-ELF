"""Compare owner plans to reported layouts in completed curve evidence."""
import argparse
import json
from pathlib import Path
import re
import subprocess

def numbers(line):
    return {k:int(v) for k,v in re.findall(r'(\w+)=(\d+)(?:\s|$)',line)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--curves',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    evidence=json.loads(a.curves.read_text(encoding='utf-8'))
    if not evidence['complete']:raise ValueError('complete curve evidence is required')
    rows=[]
    for curve in evidence['cases']:
        original=curve['command'];command=[original[0]]
        for i,arg in enumerate(original):
            if arg in ('--ini','--save','--device','--b2','--d','--arena-mb','--batch-mb',
                       '--owner-budget-mb','--carrier-exponent'):
                command.extend([arg,original[i+1]])
        command.extend(['--curves','1','--plan-only'])
        proc=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        (out/(curve['name']+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
        plan=[json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{')][-1]
        m=plan['resident_workspace_memory'];fold=numbers(curve['fold']);frontier=numbers(curve['frontier'])
        assert m['fold_bytes']==fold['layout_bytes']
        if fold['enabled'] and frontier['enabled']:
            assert m['finished'] and m['valid']
            assert m['frontier_bytes']==frontier['metadata_bytes']
            assert m['owner_peak_bytes']==frontier['owner_and_metadata_bytes']
        elif not fold['enabled'] and 'fallback=budget' in curve['fold']:
            assert m['valid'] and not m['finished'] and m['reason']=='fold_budget_refusal'
            assert not m['owner_peak_bytes']
        else:raise ValueError('unsupported runtime owner/fallback evidence')
        rows.append(dict(name=curve['name'],plan=plan,fold=fold,frontier=frontier))
    (out/'results.json').write_text(json.dumps(dict(complete=True,cases=rows),indent=2)+'\n',encoding='utf-8')
    print('PASS: planned fold/frontier extents match runtime layout reports; budget fallback matches')

if __name__=='__main__':main()
