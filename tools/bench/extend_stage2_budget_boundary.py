"""Append a prespecified same-D 1024/640/640/1024 MiB owner boundary control.

Call after the initial matrix finishes, then resume the unchanged runtime driver.
The original matrix and this extension are preserved separately for provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',type=Path,required=True)
    a=ap.parse_args()
    p=a.study/'plan.json';m=a.study/'measurements.json'
    data=json.loads(m.read_text())
    assert data['passed']==48 and len(data['runs'])==len(data['cases'])==48
    assert not (a.study/'boundary_extension.json').exists()
    base=next(r for r in data['cases'] if r['bits']==4423 and r['B2']==8000000000000 and r['config']=='large_resident')
    assert base['plan']['D']==1531530 and base['plan']['P']==138240
    assert 640*(1<<20)<base['plan']['owner_bytes']<1024*(1<<20)
    for name,source in (('matrix_plan.json',p),('matrix_measurements.json',m)):
        assert not (a.study/name).exists()
        (a.study/name).write_bytes(source.read_bytes())
    added=[]
    for rep,budget in enumerate((1024,640,640,1024),1):
        row=dict(base,group='owner640_boundary',rep=rep,fold_mb=budget,
                 config='large_resident' if budget==1024 else 'large_owner640')
        row['name']=f'{len(data["cases"])+1:02d}_{row["group"]}_m4423_b8000000000000_{row["config"]}_r{rep}'
        data['cases'].append(row);added.append(row)
    extension=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   matrix_measurements_sha256=hashlib.sha256(m.read_bytes()).hexdigest(),
                   matrix_plan_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                   cases=added,scope='Same frozen production/Q/D/P/big/arena/checks; only owner 1024 vs 640 MiB; ABBA.')
    (a.study/'boundary_extension.json').write_text(json.dumps(extension,indent=2))
    p.write_text(json.dumps(data,indent=2))


if __name__=='__main__':main()
