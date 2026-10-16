"""Compare dual result output with local Prime95 field shapes, without submission.

Private samples are read in memory. Only aggregate pass/fail information is
printed; fixtures and generated identities stay under the evidence directory.
"""
import argparse
import importlib.util
import shutil
import json
import os
from pathlib import Path
import secrets
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def contract(row, allowed):
    assert set(row) <= allowed, 'non-submission fields escaped into results.json.txt'
    assert row['worktype'] == 'ECM' and row['status'] in ('F', 'NF')
    assert isinstance(row['fft-length'], int) and row['fft-length'] > 0
    assert isinstance(row['program']['port'], int)
    assert row['program']['name'] == 'ECM-ELF'
    if row['status'] == 'NF':
        assert row['Edwards'] == {} and row['curves'] >= 1 and row['b2'] > row['b1']
        assert 'stage' not in row and 'factors' not in row
    else:
        assert isinstance(row['Edwards']['sigma'], int)
        assert row['stage'] in (1, 2) and row['factors']
        if row['stage'] == 1:
            assert 'b2' not in row
    assert 'checksum' not in row and 'security-code' not in row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-stage2', type=Path, required=True)
    p.add_argument('--official-results', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--cuda-stage2', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    official = [json.loads(s) for s in args.official_results.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    ecm = [r for r in official if r.get('worktype') == 'ECM']
    assert any(r['status'] == 'NF' and r['Edwards'] == {} for r in ecm)
    assert any(r.get('stage') == 1 for r in ecm) and any(r.get('stage') == 2 for r in ecm)
    allowed = set().union(*(set(r) for r in ecm))
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):raise ValueError('use a fresh evidence directory')
    identity = {key: secrets.token_hex(12) for key in ('user', 'computer')}
    ini = out/'ecm.local.ini'
    ini.write_text('\n'.join(f'{k}={v}' for k, v in identity.items())+'\n', encoding='utf-8')
    internal, submission = out/'internal.jsonl', out/'results.json.txt'
    checks=[]
    def run(name, extra=(), save=None, good=True, gpu=False, destination=None):
        inside, outside=destination or (out/(name+'.jsonl'),out/(name+'.json.txt'))
        command=[args.cuda_stage2.resolve() if gpu else args.cpu_stage2.resolve(),
                 '--save', save or args.fixture.resolve(), '--b2', 40000, '--curves', 1,
                 '--results', inside, '--submission', outside, '--ini', ini, *extra]
        if gpu:command+=['--device',1,'--short-calibration',0,'--tune-runtime',0,'--d',210,'--factor-only','--log',out/(name+'_engine.log')]
        proc=subprocess.run(list(map(str,command)),cwd=out,capture_output=True,timeout=180)
        (out/(name+'.log')).write_bytes(proc.stdout+proc.stderr)
        assert (proc.returncode==0)==good, name+' failed; inspect ignored evidence log'
        if not good:
            checks.append(name);return inside,outside,[]
        rows=[json.loads(line) for line in outside.read_text(encoding='utf-8').splitlines() if line.strip()] if outside.exists() else []
        for r in rows:contract(r,allowed)
        checks.append(name)
        return inside,outside,rows
    internal,submission,rows=run('cpu',extra=['--exponent',127],destination=(internal,submission))
    a=json.loads(internal.read_text());b=rows[0]
    assert a['Edwards']['sigma']==17 and 'stage2-engine' in a
    assert all(b.get(k)==v for k,v in identity.items())
    assert 'aid' not in b and b['exponent']==127 and b['fft-length']==128
    # Multiple NF curves produce one immutable count, and replay adds no credit.
    inside,outside,rows=run('batch',extra=['--curves',4])
    assert len(rows)==1 and rows[0]['curves']==4
    before=outside.read_bytes()
    run('batch_replay',extra=['--curves',4],destination=(inside,outside))
    assert outside.read_bytes()==before
    # Damaged exported text can be reconstructed from private immutable entries.
    shutil.copy2(outside,out/'export_before_fault.json.txt')
    outside.write_bytes(b'broken export\n')
    shutil.copy2(outside,out/'export_fault_snapshot.json.txt')
    run('batch_recover',extra=['--curves',4],destination=(inside,outside))
    assert outside.read_bytes()==before
    outside.write_bytes(b'broken export\n')
    recovered=subprocess.run(list(map(str,[args.cpu_stage2.resolve(),'--export-results','--submission',outside,'--ini',ini])),cwd=out,capture_output=True)
    (out/'export_only.log').write_bytes(recovered.stdout+recovered.stderr)
    assert recovered.returncode==0 and outside.read_bytes()==before
    checks.append('export_without_arithmetic')
    # A crash can leave completed curves without a sealed batch. Exercise that
    # state on a copy, retaining the original evidence and its published count.
    pending=out/'pending.json.txt'
    pending_ledger=Path(str(pending)+'.ecm-ledger')
    shutil.copytree(Path(str(outside)+'.ecm-ledger'),pending_ledger)
    for batch in pending_ledger.glob('*.batch'):batch.unlink()
    recovered=subprocess.run(list(map(str,[args.cpu_stage2.resolve(),'--export-results','--submission',pending,'--ini',ini])),cwd=out,capture_output=True)
    (out/'pending_recovery.log').write_bytes(recovered.stdout+recovered.stderr)
    assert recovered.returncode==0
    pending_rows=[json.loads(s) for s in pending.read_text().splitlines()]
    assert len(pending_rows)==1 and pending_rows[0]['curves']==4
    contract(pending_rows[0],allowed);checks.append('recover_unsealed_completed_curves')
    # Previously emitted NF batches remain immutable when new curves finish.
    inside,outside,rows=run('incremental_first',extra=['--curves',2])
    original=set(outside.read_text().splitlines())
    _,_,rows=run('incremental_second',extra=['--skip',2,'--curves',2],destination=(inside,outside))
    assert sum(r['curves'] for r in rows)==4 and len(rows)==2
    assert original <= set(outside.read_text().splitlines()) and len(set(outside.read_text().splitlines()))==2
    # Worker overrides globals; CLI still overrides the selected INI projection.
    scoped={k:secrets.token_hex(12) for k in identity}
    ini.write_text('\n'.join(f'{k}={v}' for k,v in identity.items())+'\n[Worker #2]\n'+
                   '\n'.join(f'{k}={v}' for k,v in scoped.items())+'\n',encoding='utf-8')
    _,_,rows=run('worker',extra=['--worker',2])
    assert all(rows[0][k]==v for k,v in scoped.items())
    override=secrets.token_hex(12)
    _,_,rows=run('cli_override',extra=['--worker',2,'--user',override,'--computer',''])
    assert rows[0]['user']==override and 'computer' not in rows[0]
    _,_,rows=run('anonymous',extra=['--user','','--computer',''])
    assert all(k not in rows[0] for k in ('user','computer','aid'))
    # Stage1 factors and real Stage2 factors are generated locally, not copied
    # from the private Prime95 corpus or attributed to someone else's account.
    spec=importlib.util.spec_from_file_location('oracle',ROOT/'tools/test/test_edwards_gpu_pipeline.py')
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    def make_save(name,n,sigma,b1,factor=None):
        if factor is None:
            _,x,z=oracle.stage1(n,sigma,b1);x=x*pow(z,-1,n)%n;status='POINT'
        else:x=factor;status='FACTOR'
        path=out/(name+'.save')
        path.write_text(f'METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM; SIGMA={sigma}; B1={b1}; N=0x{n:x}; X=0x{x:x}; STATUS={status}; CHECKSUM={oracle.checksum(n,x,sigma,b1,status)};\n')
        return path
    large_save=make_save('fft_large',(1<<11213)-1,17,2)
    _,_,rows=run('fft_large_mersenne',save=large_save,extra=['--b2',30])
    assert rows[0]['exponent']==11213 and rows[0]['fft-length']==512
    factor_save=make_save('saved_factor',(1<<29)-1,17,2,233)
    _,_,rows=run('stage1_factor',save=factor_save)
    assert rows[0]['stage']==1 and 'b2' not in rows[0]
    stage2_found=False
    for exponent in (29,37,41,43,47,53,59,67):
        try:fixture=make_save('factor_input_'+str(exponent),(1<<exponent)-1,17,2)
        except (ValueError,TypeError):continue
        _,_,rows=run('factor_'+str(exponent),save=fixture)
        if any(r.get('stage')==2 for r in rows):stage2_found=True;break
    assert stage2_found
    if args.cuda_stage2:
        _,_,rows=run('cuda',gpu=True,extra=['--curves',4])
        assert len(rows)==1 and rows[0]['curves']==4 and all(rows[0][k]==v for k,v in identity.items())
        _,_,rows=run('cuda_stage1_factor',gpu=True,save=factor_save)
        assert rows[0]['stage']==1
        _,_,rows=run('cuda_stage2_factor',gpu=True,save=fixture)
        assert any(r.get('stage')==2 for r in rows)
    # Existing non-owned exports and aliases cannot be silently overwritten.
    foreign=out/'foreign.json.txt';foreign.write_text('foreign data\n')
    run('foreign_reject',destination=(out/'foreign_internal.jsonl',foreign),good=False)
    assert foreign.read_text()=='foreign data\n'
    run('alias_reject',extra=['--submission',args.fixture.resolve()],good=False)
    (out/'summary.json').write_text(json.dumps(dict(complete=True,official_ecm_rows=len(ecm),checks=checks))+'\n')
    print('PASS:',len(checks),'dual-output checks; no server request; private samples not copied')


if __name__ == '__main__':main()
