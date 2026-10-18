"""Freeze exact-N Edwards/PARAM0 inputs and compare complete GPU Stage2 builds.

All native checks stay enabled. Each invocation has fresh output paths; accepted
rows are durable, while an unrecorded invocation requires manual inspection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"ecm_report"))
from result_format import normalize_result
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from bench_stage2_production import fields,freeze,sha
from bench_stage2_n_scaling import Monitor

if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
DEFAULT_CELLS=('1009:low:normal','10267:low:normal','10267:high:normal',
               '16319:low:normal','10267:low:carrier','16319:low:carrier')


def read(path):return normalize_result(json.loads(Path(path).read_text(encoding='utf-8-sig')))
def write(path,value):
    temporary=Path(str(path)+'.new')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def carrier_proof(plan):
    proof=plan.get('carrier_proof')
    if not proof:return []
    if sha(proof['path'])!=proof['sha256']:raise ValueError('carrier proof changed')
    report=read(proof['path'])
    if not report['complete'] or report['performance_samples']!=0:raise ValueError('unfinished carrier proof')
    if not plan.get('reused_inputs') or report['study_sha256']!=plan['reused_inputs']['sha256']:
        raise ValueError('carrier proof must use the same frozen Stage1 input study')
    if sha(Path(plan['reused_inputs']['path'])/'plan.json')!=plan['reused_inputs']['sha256']:
        raise ValueError('carrier proof source study changed')
    if report['binaries']!={k:v['sha256'] for k,v in plan['binaries'].items()}:
        raise ValueError('carrier proof uses different binaries')
    for row in report['runs']:
        for path,digest in row['files'].items():
            if sha(path)!=digest:raise ValueError('carrier oracle evidence changed')
    return report['runs']


def check_leaf(plan,row,proof):
    cell=next(c for c in plan['cells'] if c['id']==row['cell'])
    peers=[r for r in plan['runs'] if r['cell']==row['cell'] and r['model']==row['model']]
    if cell['carrier']:
        case=plan['cases'][str(cell['exponent'])]
        witness=next(r for r in proof if r['cell']==row['cell'] and r['model']==row['model'] and r['build']==row['build'])
        if any(witness[k]!=expected for k,expected in dict(N_hex=case['N_hex'],sigma=case['sigma'],b1=case['b1'],
                b2=cell['b2'],d=cell['d'],carrier=cell['carrier']).items()):
            raise ValueError('carrier oracle work identity changed')
        if witness['raw']!=row['stats']['descent_values']:raise ValueError('carrier representation differs from target-ring oracle run')
        matches=[r for r in proof if r['cell']==row['cell'] and r['model']==row['model']]
        if len(matches)!=2 or len({json.dumps(r['target'],sort_keys=True) for r in matches})!=1:
            raise ValueError('cross-build target-ring leaf mismatch')
        peers=[r for r in peers if r['build']==row['build']]
    if any(r['stats']['descent_values']!=row['stats']['descent_values'] for r in peers):
        raise ValueError('complete leaf fingerprint mismatch')


def freeze_binary(source,destination):
    source=source.resolve();identity=freeze(source)
    expected=read(source.parent/'build_manifest.json')['gmp_hashes']['bin/gmp-10.dll'].lower()
    if sha(source.parent/'gmp-10.dll')!=expected:raise ValueError('GMP runtime differs from compiled dependency')
    for name,digest in identity['sources'].items():
        if sha(source.parent/'sources'/name)!=digest:raise ValueError('frozen source changed: '+name)
    destination.mkdir(parents=True)
    for name in (source.name,'gmp-10.dll','build_manifest.json','frozen_sources_manifest.json'):
        shutil.copy2(source.parent/name,destination/name)
    shutil.copytree(source.parent/'sources',destination/'sources')
    runtime={name:sha(destination/name) for name in ('gmp-10.dll','build_manifest.json','frozen_sources_manifest.json')}
    return dict(path=str(destination/source.name),sha256=sha(source),identity=identity,runtime_files=runtime)


def prepare(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('use a fresh study directory')
    reference=read(a.reference_study);inputs=out/'inputs';inputs.mkdir()
    reused=read(a.input_study/'plan.json') if a.input_study else None
    binaries={key:freeze_binary(path,out/'bin'/key) for key,path in
              (('baseline',a.baseline),('candidate',a.candidate))}
    ini=out/'bench.ini';ini.write_text('user=\ncomputer=\n',encoding='utf-8')
    requested=[]
    for cell in a.cells:
        exponent,bound,mode=cell.split(':')
        if bound not in ('low','high') or mode not in ('normal','carrier'):raise ValueError('invalid cell '+cell)
        requested.append((int(exponent),bound,mode))
    cases={}
    for exponent in sorted({c[0] for c in requested}):
        case=next(c for c in reference['cases'] if c['exponent']==exponent).copy()
        n=int(case['N_hex'],16)
        if ((1<<exponent)-1)//math.prod(map(int,case['known']))!=n or ((1<<exponent)-1)%math.prod(map(int,case['known'])):
            raise ValueError('target/known-factor mismatch')
        if sha(case['save'])!=case['save_sha256']:raise ValueError('verified Edwards save changed')
        ed=inputs/f'm{exponent}_ed.save';shutil.copy2(case['save'],ed)
        mont=inputs/f'm{exponent}_mont.save'
        if reused:
            prior=reused['cases'][str(exponent)]
            if any(prior[key]!=case[key] for key in ('N_hex','sigma','b1')):raise ValueError('reused curve identity differs')
            if sha(prior['saves']['mont'])!=prior['save_hashes']['mont']:raise ValueError('reused PARAM0 save changed')
            shutil.copy2(prior['saves']['mont'],mont)
            shutil.copy2(a.input_study/'inputs'/f'm{exponent}_mont_prepare.log',inputs/f'm{exponent}_mont_prepare.log')
        else:
            command=[a.gmp_ecm.resolve(),'-param',0,'-sigma',case['sigma'],'-save',mont,case['b1'],case['b1']]
            proc=subprocess.run(list(map(str,command)),input=str(n)+'\n',text=True,capture_output=True,cwd=out,timeout=1800)
            (inputs/f'm{exponent}_mont_prepare.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
            if proc.returncode or not mont.is_file():raise ValueError('PARAM0 preparation failed; raw log retained')
        # The reference library starts from exactly this integer; native workers
        # also validate every save and its checksum before performing arithmetic.
        tokens=dict(item.strip().split('=',1) for item in mont.read_text().split(';') if '=' in item)
        if int(tokens['N'],0)!=n or int(tokens['SIGMA'])!=case['sigma'] or int(float(tokens['B1']))!=case['b1']:
            raise ValueError('PARAM0 saved identity mismatch')
        case.update(saves=dict(ed=str(ed),mont=str(mont)),save_hashes=dict(ed=sha(ed),mont=sha(mont)))
        cases[exponent]=case
        print('PREPARED',exponent,case['bits'],'bits',flush=True)
    cells=[]
    for exponent,bound,mode in requested:
        case=cases[exponent]
        refs=[r for r in reference['runs'] if r['case']==case['id'] and r['engine']=='gpu' and r['full_sample']]
        b2=sorted({r['B2'] for r in refs})[0 if bound=='low' else -1]
        r=next(r for r in refs if r['B2']==b2)
        cells.append(dict(id=f'm{exponent}_{bound}_{mode}',exponent=exponent,bound=bound,mode=mode,
                          b2=b2,d=r['D'],carrier=exponent if mode=='carrier' else 0))
    if len({c['id'] for c in cells})!=len(cells):raise ValueError('duplicate cell')
    plan=dict(schema=1,complete=False,created_utc=datetime.now(timezone.utc).isoformat(),
        source_reference=dict(path=str(a.reference_study.resolve()),sha256=sha(a.reference_study)),
        native_stage1=reused['native_stage1'] if reused else dict(path=str(a.gmp_ecm.resolve()),sha256=sha(a.gmp_ecm)),
        reused_inputs=dict(path=str(a.input_study.resolve()),sha256=sha(a.input_study/'plan.json')) if a.input_study else None,
        collector_sha256=sha(__file__),device=a.device,repeats=a.repeats,warmups=1,binaries=binaries,
        carrier_proof=dict(path=str(a.carrier_proof.resolve()),sha256=sha(a.carrier_proof)) if a.carrier_proof else None,
        cases={str(k):v for k,v in cases.items()},cells=cells,runs=[],protocol=dict(
            timing='stage2_full_wall init+main includes in-engine checks/GCD; Stage1/startup/planning/preflight Mont-xADD checks excluded',
            leaves='ordinary: identical raw leaves across builds; carrier: per-build raw leaves match independently checked, identical target-ring oracle leaves',
            process='parent launch through observed exit, 0.2-second polling',
            order='one predeclared warmup per arm, then cyclic arm order per repetition; all runs retained',
            geometry='same exact N/B1/B2/D for both curve families and builds; scalar construction differs by curve',
            budgets_mib=dict(arena=6300,owner=640,batch=256),clocks='observe only; no settings changed'))
    if any(c['carrier'] for c in cells):
        if not a.carrier_proof:raise ValueError('carrier cells require an independent projection proof')
        proof=carrier_proof(plan)
        for cell in cells:
            if cell['carrier']:
                for model in ('ed','mont'):
                    for build in binaries:
                        witness=next(r for r in proof if r['cell']==cell['id'] and r['model']==model and r['build']==build)
                        check_leaf(plan,dict(cell=cell['id'],model=model,build=build,stats=dict(descent_values=witness['raw'])),proof)
    shutil.copy2(__file__,out/'collector_snapshot.py')
    write(out/'plan.json',plan)
    print('PREPARED',len(cells),'cells; no timing samples yet')


def run(a):
    out=a.output.resolve();plan=read(out/'plan.json')
    if plan['collector_sha256']!=sha(__file__):raise ValueError('collector changed after preparation')
    for binary in plan['binaries'].values():
        if sha(binary['path'])!=binary['sha256']:raise ValueError('frozen binary changed')
        for name,digest in binary['runtime_files'].items():
            if sha(Path(binary['path']).parent/name)!=digest:raise ValueError('frozen runtime changed')
    for case in plan['cases'].values():
        for model in ('ed','mont'):
            if sha(case['saves'][model])!=case['save_hashes'][model]:raise ValueError('Stage1 save changed')
    for row in plan['runs']:
        for name,digest in row['files'].items():
            if sha(name)!=digest:raise ValueError('accepted evidence changed')
    proof=carrier_proof(plan)
    for row in plan['runs']:check_leaf(plan,row,proof)
    done={r['id'] for r in plan['runs']};env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',CUDA_LAUNCH_BLOCKING='0')
    monitor=Monitor(plan['device']);plan['device_identity']=monitor.identity
    if plan.get('imported_device_identity') and plan['imported_device_identity']!=monitor.identity:
        raise ValueError('imported warmups came from a different device')
    arms=[('baseline','ed'),('candidate','ed'),('baseline','mont'),('candidate','mont')]
    completed=0
    for rep in range(plan['repeats']+1):
        category='warmup' if rep==0 else 'timing'
        order=arms[rep%4:]+arms[:rep%4]
        for cell in plan['cells']:
            case=plan['cases'][str(cell['exponent'])]
            for build,model in order:
                identity=f'{cell["id"]}_{category}{rep}_{build}_{model}'
                if identity in done:continue
                folder=out/'runs'/identity
                if folder.exists():raise ValueError('unrecorded invocation exists; inspect before continuing: '+identity)
                folder.mkdir(parents=True)
                exe=plan['binaries'][build]['path']
                command=[exe,'--ini',out/'bench.ini','--save',case['saves'][model],
                    '--b2',cell['b2'],'--d',cell['d'],'--device',plan['device'],'--curves',1,
                    '--carrier-exponent',cell['carrier'],'--short-calibration',0,'--tune-runtime',0,
                    '--arena-mb',6300,'--owner-budget-mb',640,'--batch-mb',256,'--factor-only',
                    '--log-level','curve','--results',folder/'internal.jsonl','--submission',folder/'results.json.txt',
                    '--log',folder/'engine.log','--debug-log-file',folder/'debug.log',
                    '--exponent',case['exponent']]
                for factor in case['known']:command+=['--known-factor',factor]
                command=list(map(str,command));print('RUN',identity,flush=True)
                write(folder/'command.json',dict(command=command,environment=dict(NTT_NO_PROGRESS='1',CUDA_LAUNCH_BLOCKING='0')))
                observations=[];start=time.perf_counter()
                with (folder/'driver.log').open('wb') as log:
                    proc=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=log)
                    while proc.poll() is None:
                        observations.append(monitor.sample())
                        if time.perf_counter()-start>1800:
                            proc.kill();proc.wait();raise TimeoutError(identity)
                        time.sleep(.2)
                wall=time.perf_counter()-start
                write(folder/'telemetry.json',observations)
                if proc.returncode:raise ValueError('native failure; preserve raw evidence: '+identity)
                result=read(folder/'internal.jsonl')
                if result['N_hex']!=case['N_hex'] or result['B1']!=case['b1'] or result['B2']!=cell['b2'] or result['curve']!=model:
                    raise ValueError('result identity mismatch')
                if result['bad_factors'] or result['factors'] or result['status']!='stage2_completed':
                    raise ValueError('not a complete NF timing sample; preserve separately')
                text=(folder/'engine.log').read_text()+'\n'+(folder/'debug.log').read_text()
                labels=('stage2_full_wall','real_shape','real_baby','real_batched_split','real_batched_wall',
                        'device_gleaf','s4_multiply_stats','descent_values')
                stats={key:fields(text,key) for key in labels};ledger=stats['stage2_full_wall'];s4=stats['s4_multiply_stats']
                if ledger['clean']!='1' or abs(float(ledger['total'])-float(ledger['init'])-float(ledger['main']))>.003:
                    raise ValueError('incomplete timing ledger')
                if not int(s4['gmp_selftest_cases']) or not int(s4['gmp_checked']) or int(s4['gmp_selftest_bad']) or int(s4['gmp_check_bad']):
                    raise ValueError('missing arithmetic checks')
                if int(stats['real_shape']['D'])!=cell['d'] or result['carrier_exponent']!=cell['carrier']:
                    raise ValueError('geometry changed')
                row=dict(id=identity,cell=cell['id'],model=model,build=build,category=category,repeat=rep,
                         process_seconds=wall,engine_seconds=float(ledger['total']),stats=stats,
                         version=result['program']['version'],files={str(p):sha(p) for p in folder.iterdir() if p.is_file()})
                check_leaf(plan,row,proof)
                plan['runs'].append(row);done.add(identity);write(out/'plan.json',plan)
                print('DONE',identity,round(row['engine_seconds'],6),flush=True)
                completed+=1
                if a.max_runs and completed>=a.max_runs:return
    plan['complete']=len(plan['runs'])==len(plan['cells'])*4*(plan['repeats']+1)
    write(out/'plan.json',plan);monitor.close()
    print('COMPLETE',len(plan['runs']),'runs; all raw samples retained')


def import_warmups(a):
    """Carry completed, predeclared warmups through a corrected collection gate.

    The old study and rejected/unrecorded invocations remain untouched. This is
    only allowed before formal sampling, with identical arithmetic and protocol.
    """
    out=a.output.resolve();plan=read(out/'plan.json');old=read(a.from_study/'plan.json')
    if plan['collector_sha256']!=sha(__file__) or plan['runs']:raise ValueError('fresh prepared study required')
    if any(c not in plan['cells'] for c in old['cells']) or plan['repeats']!=old['repeats'] or plan['device']!=old['device']:
        raise ValueError('collection scope changed')
    if plan['protocol']['budgets_mib']!=old['protocol']['budgets_mib']:raise ValueError('warmup budgets changed')
    if len({r['id'] for r in old['runs']})!=len(old['runs']):raise ValueError('duplicate old invocation')
    if any(r['category']!='warmup' or r['repeat']!=0 for r in old['runs']):raise ValueError('formal samples cannot be imported')
    if {k:v['sha256'] for k,v in plan['binaries'].items()}!={k:v['sha256'] for k,v in old['binaries'].items()}:
        raise ValueError('builds changed')
    for key,case in plan['cases'].items():
        if case['N_hex']!=old['cases'][key]['N_hex'] or case['save_hashes']!=old['cases'][key]['save_hashes']:
            raise ValueError('curve input changed')
    proof=carrier_proof(plan)
    for row in old['runs']:
        check_leaf(plan,row,proof)
        for path,digest in row['files'].items():
            if sha(path)!=digest:raise ValueError('old warmup evidence changed')
        source=a.from_study/'runs'/row['id'];destination=out/'runs'/row['id']
        shutil.copytree(source,destination)
        row['files']={str(destination/Path(path).name):digest for path,digest in row['files'].items()}
        row['imported_from']=dict(study=str(a.from_study.resolve()),sha256=sha(a.from_study/'plan.json'))
        plan['runs'].append(row)
    plan['imported_device_identity']=old['device_identity']
    write(out/'plan.json',plan);print('IMPORTED',len(plan['runs']),'verified warmups; no formal samples')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--reference-study',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--gmp-ecm',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--carrier-proof',type=Path)
    p.add_argument('--input-study',type=Path,help='reuse identical frozen Stage1 inputs and their native preparation evidence')
    p.add_argument('--cells',nargs='+',default=DEFAULT_CELLS);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--device',type=int,default=1)
    p=sub.add_parser('run');p.add_argument('--output',type=Path,required=True);p.add_argument('--max-runs',type=int,default=0)
    p=sub.add_parser('import-warmups');p.add_argument('--output',type=Path,required=True);p.add_argument('--from-study',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='prepare':
        if args.repeats<1 or args.device<0:parser.error('invalid repeat/device')
        prepare(args)
    elif args.action=='import-warmups':import_warmups(args)
    else:run(args)


if __name__=='__main__':main()
