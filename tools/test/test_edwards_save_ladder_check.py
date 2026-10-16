"""Verify independent ladder agreement and rejection beyond file checksum checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checker',type=Path,required=True)
    p.add_argument('--saves',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False,parents=True)
    report=dict(checker_sha256=hashlib.sha256(a.checker.read_bytes()).hexdigest(),checks=[],complete=False)
    for i,path in enumerate(a.saves):
        result=subprocess.run([str(a.checker.resolve()),str(path.resolve())],capture_output=True,timeout=1800)
        (out/f'valid_{i}.log').write_bytes(result.stdout+result.stderr)
        assert result.returncode==0
        report['checks'].append(dict(save_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),valid=True))
    tokens=dict(t.strip().split('=',1) for t in a.saves[0].read_text().splitlines()[0].split(';') if t.strip())
    n=int(tokens['N'],0);x=(int(tokens['X'],0)+1)%n;sigma=int(tokens['SIGMA']);b1=int(tokens['B1'])
    h=14695981039346656037
    for byte in f'ECM-EDWARDS-AM|1|48LCM|{n:x}|{x:x}|{sigma}|{b1}|POINT'.encode():h=((h^byte)*1099511628211)&((1<<64)-1)
    tokens['X']=hex(x);tokens['CHECKSUM']=str(h)
    corrupt=out/'wrong_point_valid_checksum.save'
    corrupt.write_text('; '.join(k+'='+v for k,v in tokens.items())+';\n')
    result=subprocess.run([str(a.checker.resolve()),str(corrupt)],capture_output=True,timeout=1800)
    (out/'wrong_point.log').write_bytes(result.stdout+result.stderr)
    assert result.returncode!=0 and b'independent ladder differs' in result.stderr
    report.update(wrong_point_with_valid_checksum_rejected=True,complete=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS independent saved-point ladder and valid-checksum wrong-point rejection')


if __name__=='__main__':main()
