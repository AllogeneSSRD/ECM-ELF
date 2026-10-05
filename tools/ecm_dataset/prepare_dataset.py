"""Prepare a stratified, exact order corpus before any timed GPU production run."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from dataset import ROOT, GP_SOURCE, analyze, connect, digest, gp_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,default=ROOT/'tools/ecm_dataset/ecm_stage2_dataset.sqlite')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gp',type=Path,nargs='?',const=None,help='GP path/name; omitted or bare --gp searches PATH')
    parser.add_argument('--exponents',type=int,nargs='+',default=[223,431,1367,2657,4933,6977,8171])
    parser.add_argument('--sigmas',type=int,nargs='+',default=[*range(6,18),26])
    parser.add_argument('--max-factor-digits',type=int,default=17)
    parser.add_argument('--timeout',type=float,default=90)
    args=parser.parse_args();db=connect(args.db);gp=gp_path(args.gp)
    if any(not 1<=n<=8192 for n in args.exponents) or any(not 6<=s<1<<64 for s in args.sigmas):
        parser.error('Need production-supported exponents and PARAM0 sigmas')
    args.output.mkdir(parents=True,exist_ok=True)
    summary=[]
    for n in args.exponents:
        rows=list(db.execute('SELECT value,digital FROM factors WHERE exponent=? AND digital BETWEEN 5 AND ? ORDER BY digital,value',
                             (n,args.max_factor_digits)))
        chosen={}
        for digits in (6,12,17):
            if rows:
                row=min(rows,key=lambda r:(abs(r['digital']-digits),int(r['value'])))
                chosen[row['value']]=row
        jobs=[(n,int(f),s) for f in chosen for s in args.sigmas]
        statement=GP_SOURCE.read_text(encoding='utf-8')+'\n'
        for e,f,s in jobs:
            statement+=f'print("BEGIN|{e}|{f}|{s}");ecm_wall=getwalltime();iferr(ecm_param0_order({f},{s}),ecm_error,print("ERROR|",errname(ecm_error)));print("SECONDS|",(getwalltime()-ecm_wall)/1000.0);print("END");\n'
        statement+='quit();\n'
        script=args.output/f'm{n}.gp';script.write_text(statement,encoding='utf-8')
        run=subprocess.run([str(gp),'-q','-f',str(script.resolve())],capture_output=True,timeout=args.timeout)
        text=run.stdout.decode('utf-8',errors='replace').replace('\r','')
        (args.output/f'm{n}.stdout').write_text(text,encoding='utf-8')
        (args.output/f'm{n}.stderr').write_bytes(run.stderr)
        parsed=re.findall(r'^BEGIN\|(\d+)\|(\d+)\|(\d+)\n(.*?)^END$',text,re.M|re.S)
        if run.returncode or len(parsed)!=len(jobs):raise RuntimeError(f'Incomplete GP batch M{n}')
        results=[]
        for e,f,s,evidence in parsed:
            seconds=re.search(r'^SECONDS\|([^\n]+)',evidence,re.M)
            result=analyze(db,int(e),int(f),int(s),gp,retry=True,prepared=(evidence,float(seconds[1])))
            results.append({k:result.get(k) for k in ('value','sigma','status','updated','stage1_min_b1','bounds')})
        summary.append(dict(exponent=n,jobs=len(jobs),complete=sum(r['status']=='complete' for r in results),results=results,
                            script_sha256=digest(script),stdout_sha256=digest(args.output/f'm{n}.stdout')))
        (args.output/'preparation.json').write_text(json.dumps(dict(gp_sha256=digest(gp),order_script_sha256=digest(GP_SOURCE),batches=summary),indent=2),encoding='utf-8')
        print(json.dumps({k:summary[-1][k] for k in ('exponent','jobs','complete')}),flush=True)
    db.close()


if __name__=='__main__':main()
