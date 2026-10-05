"""Build/query the local factor corpus; ingest native Stage2 JSONL results."""
import argparse
import csv
import json
from pathlib import Path
from dataset import ROOT, DEFAULT_DB, analyze, connect, gp_path, import_catalog, ingest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest='command', required=True)
    imp = sub.add_parser('init'); imp.add_argument('--html',type=Path,default=ROOT/'.refactor/Mersenne_exponent_factor_1-9999.html')
    ana = sub.add_parser('analyze'); ana.add_argument('--exponent',type=int,required=True);ana.add_argument('--factor',type=int,required=True)
    ana.add_argument('--sigma',type=int,nargs='+',required=True);ana.add_argument('--retry',action='store_true')
    ing = sub.add_parser('ingest');ing.add_argument('--results',type=Path,required=True);ing.add_argument('--exponent',type=int)
    for p in (ana,ing):
        p.add_argument('--gp',type=Path,nargs='?',const=None,help='GP path/name; omitted or bare --gp searches PATH');p.add_argument('--timeout',type=float,default=30);p.add_argument('--torsion',type=int,choices=(1,12),default=1)
    sub.add_parser('summary')
    exp = sub.add_parser('export');exp.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();db = connect(args.db)
    try:
        if args.command == 'init': print(json.dumps(import_catalog(db,args.html),indent=2))
        elif args.command == 'analyze':
            gp = gp_path(args.gp)
            for sigma in args.sigma:
                print(json.dumps(analyze(db,args.exponent,args.factor,sigma,gp,args.torsion,args.timeout,args.retry)),flush=True)
            best=db.execute('SELECT * FROM factors WHERE exponent=? AND value=?',(args.exponent,str(args.factor))).fetchone()
            print(json.dumps({'best':dict(best) if best else None}))
        elif args.command == 'ingest':print(json.dumps(ingest(db,args.results,gp_path(args.gp),args.torsion,args.timeout,args.exponent)))
        elif args.command == 'summary':
            result = {t:db.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in
                      ('mersennes','factors')}
            result['default_best'] = db.execute('SELECT count(*) FROM factors WHERE sigma IS NOT NULL').fetchone()[0]
            print(json.dumps(result,indent=2))
        elif args.command == 'export':
            args.output.parent.mkdir(parents=True,exist_ok=True)
            rows=db.execute('SELECT * FROM factors ORDER BY exponent,length(value),value')
            with args.output.open('w',encoding='utf-8',newline='') as f:
                writer=csv.writer(f);writer.writerow([x[0] for x in rows.description]);writer.writerows(rows)
    finally:db.close()


if __name__ == '__main__':main()
