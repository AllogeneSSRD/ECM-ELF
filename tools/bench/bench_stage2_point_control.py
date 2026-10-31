"""Same-build fixed-D tail-trigger control, separate from the optimized matrix."""
import argparse
from pathlib import Path
import shutil

from bench_stage2_point_matrix import command,invoke,records
from bench_stage2_point_routes import sha
from bench_stage2_version import load,write,stamp
from stage2_d_catalog import phi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',default='m8011_n7995_s26');p.add_argument('--d',type=int,default=180180)
    a=p.parse_args();root=a.root.resolve();plan=load(root/'plan.json');out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if not plan['complete']:raise ValueError('Complete the main matrix before the control')
    case=next(c for c in plan['cases'] if c['id']==a.case)
    cell=dict(next(c for c in plan['cells'] if c['case']==a.case and c['carrier']))
    cell.update(D=a.d,P=phi(a.d)//2)
    report=dict(complete=False,source=str(root),cell=cell,exe_sha256=plan['exe_sha256'],started_utc=stamp(),runs=[])
    before=sha(root/'frozen_tune.toml');shutil.copy2(__file__,out/'collector_snapshot.py')
    for chain in (64,32):
        for balance in ((False,True) if chain==64 else (True,False)):
            folder=out/(str(chain)+('_on' if balance else '_off'));folder.mkdir()
            cmd=command(root,folder,case,cell,root/'frozen_tune.toml',chain,balance)
            invoke(plan,root,folder,cmd);rows=records(plan,folder,case,cell)
            if len(rows)!=1 or rows[0]['chain']!=chain or rows[0]['balance']!=balance:raise ValueError('Control geometry changed')
            report['runs'].append(dict(chain=chain,balance=balance,folder=str(folder),row=rows[0]))
            write(out/'summary.json',report);print(chain,balance,rows[0]['seconds'],flush=True)
    if sha(root/'frozen_tune.toml')!=before:raise ValueError('Frozen profile changed')
    report.update(complete=True,finished_utc=stamp());write(out/'summary.json',report)


if __name__=='__main__':main()
