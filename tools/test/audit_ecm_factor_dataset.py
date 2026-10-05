"""Independent divisibility and selected point-order audit for the core dataset."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/ecm_dataset'))
from dataset import bounds, digest, gp_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,default=ROOT/'tools/ecm_dataset/ecm_stage2_dataset.sqlite');p.add_argument('--gp',type=Path,nargs='?',const=None)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();db=sqlite3.connect(a.db);db.row_factory=sqlite3.Row
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert not list(db.execute('PRAGMA foreign_key_check'))
    spec=importlib.util.spec_from_file_location('ref',ROOT/'tools/stat/suyama_mont_ref.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    prime_values=set();orders=0;scalar_checks=0
    for r in db.execute('SELECT * FROM factors'):
        assert len(r['value'])==r['digital'] and pow(2,r['exponent'],int(r['value']))==1
        if r['sigma'] is None: continue
        factor=int(r['value']);group=int(r['group_order']);point=int(r['point_order'])
        gf=[(int(q),int(e)) for q,e in json.loads(r['group_factorization'])]
        pf=[(int(q),int(e)) for q,e in json.loads(r['point_factorization'])]
        assert math.prod(q**e for q,e in gf)==group and math.prod(q**e for q,e in pf)==point
        assert group%point==0 and abs(group-factor-1)<=math.isqrt(4*factor)
        _,pairs=bounds(pf,1)
        assert (int(r['b1']),int(r['b2'])) in pairs
        _,a24,x0,z0=ref.suyama_curve(int(r['sigma']),factor)
        # This is x-only Python arithmetic, independent of GP's ellorder and map.
        _,z=ref.ladder(point,x0,z0,a24,factor);assert z==0;scalar_checks+=1
        for q,_ in pf:
            _,z=ref.ladder(point//q,x0,z0,a24,factor);assert z!=0;scalar_checks+=1
        prime_values.add(factor);prime_values.update(q for q,_ in gf+pf);orders+=1
    gp=gp_path(a.gp)
    script=''.join(f'print("PRIME|{q}|",isprime({q}));\n' for q in sorted(prime_values))+'quit();\n'
    result=subprocess.run([str(gp),'-q','-f'],input=script.encode(),capture_output=True,timeout=60)
    lines=[line for line in result.stdout.decode().replace('\r','').splitlines() if line.startswith('PRIME|')]
    assert result.returncode==0 and len(lines)==len(prime_values) and all(line.endswith('|1') for line in lines)
    output=dict(passed=True,factor_rows=db.execute('SELECT count(*) FROM factors').fetchone()[0],
        complete_orders=orders,independent_point_checks=scalar_checks,proven_primes=len(prime_values),
        database_sha256=digest(a.db),gp_sha256=digest(gp),audit_sha256=digest(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2));db.close()


if __name__=='__main__':main()
