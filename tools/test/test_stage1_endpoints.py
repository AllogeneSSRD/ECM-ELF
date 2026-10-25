"""Compare deterministic Stage1 endpoint evidence across two usability runs.

Ignores PROGRAM/WHO/TIME presentation metadata; compares the arithmetic identity,
normalized value and point/factor status of every saved record, including resume.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

FIELDS=('METHOD','PARAM','SIGMA','B1','N','X','STATUS','CHECKSUM','SCALAR','VERSION')

def records(path):
    rows=[]
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.lstrip().startswith(('#',';')):continue
        fields=dict(re.findall(r'(\w+)\s*=\s*([^;]+)',line))
        rows.append(tuple(fields.get(key,'') for key in FIELDS))
    return sorted(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    before,after=args.before.resolve(),args.after.resolve()
    evidence=[];empty=[]
    for current in sorted(after.rglob('*.save')):
        if '.bak' in current.name:continue
        name=current.relative_to(after);previous=before/name
        if not current.stat().st_size:
            empty.append(name.as_posix());continue
        assert previous.is_file(),('missing baseline endpoint',str(name))
        expected,actual=records(previous),records(current)
        assert expected==actual,('Stage1 endpoint differs',str(name))
        evidence.append(dict(file=name.as_posix(),records=len(actual),before_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),after_sha256=hashlib.sha256(current.read_bytes()).hexdigest()))
    assert len(evidence)>=20,('incomplete backend/resume coverage',len(evidence))
    result=dict(complete=True,before=str(before),after=str(after),saved_files=len(evidence),records=sum(v['records'] for v in evidence),empty_preflight_files=empty,files=evidence)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))

if __name__=='__main__':main()
