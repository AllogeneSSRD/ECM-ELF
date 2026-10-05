"""Check budget argument rejection and a prepared custom-plan geometry (no GPU curves)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrix',type=Path,required=True)
    ap.add_argument('--custom',type=Path,required=True)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[2]
    base=[sys.executable,str(repo/'tools/bench/bench_stage2_budget_scaling.py'),
          '--exe','unused','--sources','unused','--shape-probe','unused','--output','unused']
    invalid=[['--small-big-mb','0'],['--big-mb','384','--small-big-mb','768'],
             ['--small-owner-mb','-1'],['--owner-mb','0'],['--arena-mb','0']]
    for flags in invalid:
        r=subprocess.run(base+flags,capture_output=True,cwd=repo)
        assert r.returncode==2 and b'Need 0<small-big' in r.stderr,flags
    original=json.loads(a.matrix.read_text());custom=json.loads(a.custom.read_text())
    assert len(custom['cases'])==48 and not custom['runs']
    assert custom['configs']['small_big']['big_mb']==512
    assert custom['configs']['large_resident']['fold_mb']==640
    assert custom['configs']['large_owner128']['fold_mb']==0
    orig={c['name']:c for c in original['cases']}
    for c in custom['cases']:
        g=c['plan'];assert g['big_bytes']<=c['big_mb']*(1<<20)
        assert g['arena_est_bytes']<=c['arena_mb']*(1<<20)
        if c['config']!='small_big':assert g==orig[c['name']]['plan'],c['name']
    print('Budget options: 5 invalid CLI cases + 48 custom plans pass; no performance curves launched.')


if __name__=='__main__':main()
