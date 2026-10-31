"""Balanced repeats for large absolute/relative off/on differences; keep originals."""
import argparse
from pathlib import Path
import shutil

from bench_stage2_point_matrix import command,invoke,records
from bench_stage2_point_routes import sha
from bench_stage2_version import load,write,stamp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    plan=load(root/'plan.json')
    if not plan['complete'] or plan['failures']:raise ValueError('Complete the held-out matrix first')
    report=dict(complete=False,source=str(root),source_plan_sha256=sha(root/'plan.json'),started_utc=stamp(),
                rule='abs(on/off-1)>0.03 and abs(on-off)>0.05s; both signs, all original calls retained',cells=[],runs=[])
    for cell in plan['cells']:
        off=next(e['row']['seconds'] for e in plan['evaluations'] if e['cell']==cell['id'] and not e['balance'])
        on=next(e['row']['seconds'] for e in plan['evaluations'] if e['cell']==cell['id'] and e['balance'])
        if abs(on/off-1)>.03 and abs(on-off)>.05:report['cells'].append(cell['id'])
    write(out/'summary.json',report);shutil.copy2(__file__,out/'collector_snapshot.py')
    before=sha(root/'frozen_tune.toml')
    for name in report['cells']:
        cell=next(c for c in plan['cells'] if c['id']==name);case=next(c for c in plan['cases'] if c['id']==cell['case'])
        chain=next(d['chain'] for d in plan['decisions'] if d['cell']==name)
        for index,balance in enumerate((False,True,True,False),1):
            folder=out/name/(str(index)+('_on' if balance else '_off'));folder.mkdir(parents=True,exist_ok=False)
            cmd=command(root,folder,case,cell,root/'frozen_tune.toml',chain,balance)
            elapsed=invoke(plan,root,folder,cmd);row=records(plan,folder,case,cell)
            if len(row)!=1 or row[0]['chain']!=chain or row[0]['balance']!=balance:raise ValueError('Recheck route changed')
            report['runs'].append(dict(cell=name,balance=balance,folder=str(folder),process_seconds=elapsed,row=row[0]))
            write(out/'summary.json',report);print(name,index,row[0]['seconds'],flush=True)
    if sha(root/'frozen_tune.toml')!=before or sha(root/'plan.json')!=report['source_plan_sha256']:raise ValueError('Frozen evidence changed')
    report.update(complete=True,finished_utc=stamp());write(out/'summary.json',report)


if __name__=='__main__':main()
