"""Exact positive/target-negative Stage2 cases using frozen production binaries.

Preparation and PARI analysis occur outside measured production execution.
GPU0 is reserved for the user's other production task; this runner uses GPU1.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from dataset import ROOT, connect, digest, gp_path


def reference():
    spec=importlib.util.spec_from_file_location('suyama_reference',ROOT/'tools/stat/suyama_mont_ref.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare(db, output, exponents, allow_cofactors=False):
    ref=reference();cases=[];rejected=[]
    for n in exponents:
        choices=[dict(r) for r in db.execute('''SELECT value,sigma,b1,b2 FROM factors
                WHERE exponent=? AND sigma IS NOT NULL''',(n,))
                 if 2<=int(r['b1'])<=5000 and 500<=int(r['b2'])<=2000000 and int(r['b2'])-int(r['b1'])>=50]
        choices.sort(key=lambda r:(-len(r['value']),int(r['b1']),int(r['b2']),int(r['sigma'])))
        for row in choices:
            b1,b2,sigma=int(row['b1']),int(row['b2']),int(row['sigma']);modulus=(1<<n)-1;removed=[]
            try:
                if allow_cofactors:
                    for _ in range(32):
                        _,a24,x0,z0=ref.suyama_curve(sigma,modulus)
                        _,z=ref.ladder(ref.lcm_1_to(b1),x0,z0,a24,modulus)
                        g=math.gcd(z,modulus)
                        if g==1:break
                        if not 1<g<modulus or g%int(row['value'])==0:raise ValueError('Target did not survive Stage1')
                        removed.append(str(g));modulus//=g
                    else:raise ValueError('Too many Stage1 reductions')
                point=ref.stage1(sigma,b1,modulus)
                if point['gcd']!=1:raise ValueError('Stage1 already has a factor')
            except ValueError as e:
                rejected.append(dict(exponent=n,factor=row['value'],sigma=row['sigma'],reason=str(e)));continue
            d=210 if b2-b1>1000 else 30 if b2-b1>150 else 6
            low=b2-4*d
            if low<=b1:raise RuntimeError('Invalid negative control geometry')
            cases.append(dict(exponent=n,N_hex=f'{modulus:x}',sigma=sigma,B1=b1,B2=b2,negative_B2=low,D=d,
                              expected_factor=row['value'],X_hex=f'{point["x"]:x}',
                              checksum=b1*sigma*modulus*point['x']%4294967291,removed_gcds=removed))
            break
        else:raise RuntimeError(f'No valid full-Mersenne Stage1 survivor for M{n}')
    data=dict(schema=2,cases=cases,rejected_preparation_candidates=rejected,
              tools={p:digest(ROOT/p) for p in ('tools/stat/suyama_mont_ref.py','tools/ecm_dataset/run_production_dataset.py')})
    output.mkdir(parents=True,exist_ok=True);(output/'plan.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps({'prepared':len(cases),'exponents':[c['exponent'] for c in cases],
                      'factor_digits':[len(c['expected_factor']) for c in cases],'B1':[c['B1'] for c in cases],
                      'B2':[c['B2'] for c in cases],'rejected_candidates':len(rejected)},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,default=ROOT/'tools/ecm_dataset/ecm_stage2_dataset.sqlite')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-cofactors',action='store_true',help='Verify and remove native Stage1 factors before the target Stage2')
    p.add_argument('--exponents',type=int,nargs='+',default=[223,431,1367,2657,4933,6977,8171])
    p.add_argument('--stage1',type=Path,default=ROOT/'build_cuda_cmake/ecm_cuda.exe')
    p.add_argument('--stage2',type=Path,default=ROOT/'build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe')
    p.add_argument('--factorize-hits',action='store_true');p.add_argument('--gp',type=Path,nargs='?',const=None,
        help='GP path/name; omitted or bare --gp searches PATH')
    p.add_argument('--device',type=int,choices=(1,),default=1)
    args=p.parse_args();out=args.output.resolve();db=connect(args.db)
    if args.prepare_only:prepare(db,out,args.exponents,args.allow_cofactors);return
    gp = gp_path(args.gp) if args.factorize_hits else None
    plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    for name,want in plan['tools'].items():
        if digest(ROOT/name)!=want:raise RuntimeError('Preparation tool changed: '+name)
    exe1,exe2=args.stage1.resolve(),args.stage2.resolve();sha1,sha2=digest(exe1),digest(exe2)
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')};env['NTT_NO_PROGRESS']='1'
    summary_path=out/'runs.json';completed=json.loads(summary_path.read_text()) if args.resume and summary_path.exists() else []
    if completed and any(r['binary_sha256']!=sha2 for r in completed):raise RuntimeError('Cannot resume with another binary')
    def run(cmd,stdin,where,log):
        if digest(exe1)!=sha1 or digest(exe2)!=sha2:raise RuntimeError('Production binary changed')
        t=time.monotonic();r=subprocess.run(cmd,input=stdin,capture_output=True,cwd=where,env=env,timeout=180)
        log.write_bytes(r.stdout+r.stderr)
        if digest(exe1)!=sha1 or digest(exe2)!=sha2:raise RuntimeError('Production binary changed')
        return r,time.monotonic()-t
    for case in plan['cases']:
        n=case['exponent'];where=out/f'm{n}';where.mkdir(exist_ok=True)
        ini=where/'ecm.ini';ini.write_text('[gpu]\ndevice=1\n[queue]\nworktodo=unused.txt\ntmp_dir=.\n',encoding='utf-8')
        save=where/'stage1.save'
        if not args.resume or not save.exists():
            if save.exists():raise RuntimeError('Use a fresh directory or --resume')
            prefix=[str(exe1),'-ini',str(ini),'--method','gpu','--gpu-param','0','--exponent','lcm','--exp-cache','off',
                 '-d','1','-sigma',str(case['sigma']),'-gpucurves','1','--ckpt','0','-save',str(save),str(case['B1'])]
            current=(1<<n)-1;strips=[]
            for index,g in enumerate(case.get('removed_gcds',[])):
                cmd=prefix.copy();cmd[cmd.index('-save')+1]=str(where/f'strip_{index}.save')
                r,strip_seconds=run(cmd,f'{current}\n'.encode(),where,where/f'strip_{index}.log')
                hits=re.findall(r'factor\[\d+\]=(\d+)',(r.stdout+r.stderr).decode(errors='replace'))
                if r.returncode or hits!=[g]:raise RuntimeError(f'Native Stage1 reduction mismatch M{n}')
                strips.append(dict(N_hex=f'{current:x}',sigma=case['sigma'],B1=case['B1'],B2=0,param=0,
                    mersenne_exponent=n,factors=hits,bad_factors=0,status='stage1_completed',seconds=strip_seconds))
                current//=int(g)
            (where/'stage1_strips.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in strips),encoding='utf-8')
            cmd=prefix
            r,seconds=run(cmd,f'{current}\n'.encode(),where,where/'stage1.log')
            if r.returncode or not save.exists():raise RuntimeError(f'Production Stage1 failed M{n}')
            (where/'stage1_manifest.json').write_text(json.dumps(dict(command=cmd,seconds=seconds,binary_sha256=sha1),indent=2))
        text=save.read_text();x=int(re.search(r'\bX=(0x[0-9a-fA-F]+)',text)[1],16)
        checksum=int(re.search(r'\bCHECKSUM=(\d+)',text)[1])
        if f'{x:x}'!=case['X_hex'] or checksum!=case['checksum']:raise RuntimeError(f'GPU Stage1/CPU reference mismatch M{n}')
        save_sha=digest(save)
        for kind,b2 in [('positive',case['B2']),('target_negative',case['negative_B2'])]:
            if any(r['exponent']==n and r['case_kind']==kind for r in completed):continue
            result=where/(kind+'.jsonl');log=where/(kind+'.log')
            if result.exists():raise RuntimeError('Unaccounted result exists; inspect before resume')
            cmd=[str(exe2),'--save',str(save),'--b2',str(b2),'--d',str(case['D']),'--device','1',
                 '--arena-mb','1024','--results',str(result),'--log',str(log)]
            if args.factorize_hits:cmd+=['--factorize-hits','--gp',str(gp)]
            r,seconds=run(cmd,None,where,where/(kind+'_driver.log'))
            if r.returncode or not result.exists():raise RuntimeError(f'Production Stage2 failed M{n}/{kind}')
            record=json.loads(result.read_text());factors=[int(f) for f in record['factors']]
            modulus=int(case['N_hex'],16);target=int(case['expected_factor'])
            if int(record['N_hex'],16)!=modulus:raise RuntimeError('Native result modulus mismatch')
            valid=record['bad_factors']==0 and all(1<f<modulus and modulus%f==0 for f in factors)
            target_hit=any(f%target==0 for f in factors)
            if not valid or target_hit!=(kind=='positive'):raise RuntimeError(f'Unexpected target outcome M{n}/{kind}')
            if args.factorize_hits and not record.get('factorization_complete'):raise RuntimeError('Native decomposition incomplete')
            if digest(save)!=save_sha:raise RuntimeError('Save changed during Stage2')
            raw=log.read_text(errors='replace')
            if not all(t in raw for t in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')):
                raise RuntimeError('Mandatory native checks missing')
            row=dict(exponent=n,sigma=str(case['sigma']),b1=str(case['B1']),b2=str(b2),case_kind=kind,
                     expected_factor=case['expected_factor'],actual_factors=record['factors'],status='passed',
                     seconds=seconds,binary_sha256=sha2,stage1_binary_sha256=sha1,save_sha256=save_sha,
                     log_path=str(log),result_path=str(result),command_json=cmd)
            completed.append(row);summary_path.write_text(json.dumps(completed,indent=2),encoding='utf-8')
            print(json.dumps({'M':n,'kind':kind,'target':str(target),'factors':record['factors'],'seconds':seconds}),flush=True)
    (out/'provenance.json').write_text(json.dumps(dict(stage1_binary=str(exe1),stage1_sha256=sha1,stage2_binary=str(exe2),
        stage2_sha256=sha2,runner_sha256=digest(__file__),plan_sha256=digest(out/'plan.json'),passed=len(completed),failed=0),indent=2))
    db.close()


if __name__=='__main__':main()
