#!/usr/bin/env python3
"""Compare full projective window CSV bytes across policies on identical inputs."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(data):
    if not data.get('passed'):
        raise ValueError('Window gate must finish successfully before this audit')
    groups = {}
    for row in data['results']:
        if not row['passed'] or not row['diagnostic_dump']:
            raise ValueError('Each audited result needs a passed diagnostic dump')
        key = tuple(row[k] for k in ('bits','B1','curves','sigma','exponent','first','count','work'))
        key += (row['geometry']['container_bits'],row['geometry']['tpi'])
        csv = Path(row['log']).parent/'window_q.csv'
        groups.setdefault(key,[]).append(dict(variant=row['variant'],registers=row['registers'],
            chunk=row['chunk'],path=str(csv),sha256=hashlib.sha256(csv.read_bytes()).hexdigest()))
    compared = 0; comparisons = []
    for key,rows in groups.items():
        expected = rows[0]
        for actual in rows[1:]:
            if actual['sha256'] != expected['sha256']:
                raise ValueError(f'Projective bytes differ: {expected["path"]}, {actual["path"]}')
            compared += 1
            comparisons.append(dict(input_key=key,reference=expected,actual=actual))
    return dict(binary_sha256=data['binary_sha256'],input_groups=len(groups),
        bitwise_comparisons=compared,scope='Same seed, scalar subproduct, container and TPI; compares complete X/Z CSV bytes',
        comparisons=comparisons,passed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=audit(json.loads(a.input.read_text(encoding='utf-8')))
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['bitwise_comparisons'],'bitwise comparisons passed')


if __name__=='__main__': main()
