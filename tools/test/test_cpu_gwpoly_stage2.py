"""Saved-point GW polynomial Stage2: independent arithmetic and publication gates."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import time
import zlib

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('edwards_oracle',ROOT/'tools/test/test_edwards_gpu_pipeline.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)


def verify_crc(row):
    target=str(row['exponent']) if 'exponent' in row else 'N='+str(row['N'])
    slots=[target,'ECM',','.join(row.get('factors',[])),','.join(row.get('known-factors',[])),str(row['b1']),str(row.get('b2','')),
           'E'+str(row['Edwards'].get('sigma','')),str(row['fft-length']),'invariant:0',row['program']['name'],row['program']['version'],
           '','','Windows','x86_64',row['timestamp']]
    assert row['checksum']==dict(version=1,checksum=f'{zlib.crc32(";".join(slots).encode()):08X}')

def verify(row,n):
    assert row['program']['name']=='ECM-ELF' and re.fullmatch(r'\d+\.\d+\.\d+',row['program']['version'])
    assert row['stage2-engine']['name']=='gwnum-polymult' and row['stage2-engine']['backend']=='gwpoly'
    assert row['errors']['invariant']==0 and 'security-code' not in row
    verify_crc(row)
    if row['status']=='F':assert all(1<int(f)<n and n%int(f)==0 for f in row['factors'])
    else:assert row['status']=='NF' and 'factors' not in row
    if row.get('stage')!=1:
        assert row['d']>=6 and row['poly-size']>0 and row['fft-length']>0
        assert row['stage2-engine']['max-roundoff']<=.4
        keys=list(row);assert keys[keys.index('b2')+1:keys.index('b2')+3]==['d','poly-size']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1',type=Path,required=True);p.add_argument('--stage2',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',action='store_true')
    p.add_argument('--reference-study',type=Path,help='Optional frozen exact-N study for 10232/16272-bit GMP projection gates')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    s1=a.stage1.resolve();s2=a.stage2.resolve();ini=out/'ecm.toml'
    ini.write_text('config_version = 4\n[stage2]\nb2_round_chain = false\n',encoding='utf-8')
    default_ini=out/'defaults.toml';default_ini.write_text('config_version = 4\n',encoding='utf-8')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report=dict(complete=False,stage1_sha256=sha(s1),stage2_sha256=sha(s2),test_sha256=sha(__file__),checks=[])
    def call(name,cmd,data=None,good=True):
        r=subprocess.run(list(map(str,cmd)),input=data,capture_output=True,cwd=out,timeout=300)
        (out/(name+'.log')).write_bytes(r.stdout+r.stderr)
        assert (r.returncode==0)==good,(name,r.returncode,(r.stdout+r.stderr)[-1500:])
        report['checks'].append(dict(name=name,exit=r.returncode));(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PASS',name,flush=True);return r
    for threads in (1,4):call(f'math_t{threads}',[s2,'--selftest-gwpoly','--threads',threads])
    def save(name,target,b1=1000,sigma=17,count=1,gpu=False):
        path=out/(name+'.save')
        call(name,[s1,'--curve','ed','--target','gpu' if gpu else 'cpu','--impl','auto' if gpu else 'scalar',
            '-d',1,'-sigma',sigma,'-gpucurves',count,'--tmp-dir',out/(name+'_tmp'),'-save',path,b1,b1],str(target).encode())
        records=oracle.records(path);assert len(records)==count
        return path
    def run(name,path,n,b2=40000,extra=(),good=True,threads=4,config=None):
        result=out/(name+'.jsonl');submission=out/(name+'.json.txt')
        call(name,[s2,'--config',config or ini,'--save',path,'--backend','gwpoly','--threads',threads,'--b2',b2,
            '--results',result,'--submission',submission,'--gw-check',*extra],good=good)
        rows=[json.loads(line) for line in result.read_text().splitlines()] if result.exists() else []
        for row in rows:verify(row,n)
        if not good:assert not rows and not submission.exists()
        return rows
    n=(1<<127)-1;cpu=save('cpu_m127',n,count=2)
    rows=run('batch',cpu,n,extra=['--exponent',127]);assert len(rows)==2 and all(r['status']=='NF' for r in rows)
    public=json.loads((out/'batch.json.txt').read_text());assert public['curves']==2 and public['stage2-fft-length']>0
    verify_crc(public)
    assert 'security-code' not in public
    auto=out/'automatic.jsonl'
    call('automatic',[s2,'--config',ini,'--save',cpu,'--backend','auto','--curves',1,'--b2',40000,
        '--results',auto,'--submission',out/'automatic.json.txt'])
    verify(json.loads(auto.read_text()),n)
    plain=out/'plain_default.jsonl'
    call('default_libecm',[s2,'--config',default_ini,'--backend','plain','--save',cpu,'--b2',40000,
        '--curves',1,'--results',plain,'--submission',out/'plain_public.json.txt'])
    assert json.loads(plain.read_bytes())['b2']==40000
    call('reject_explicit_libecm_round',[s2,'--config',default_ini,'--save',cpu,'--b2',40000,
        '--backend','plain','--b2-round-chain',1],good=False)
    for d in (6,30,60,210):
        rows=run(f'wheel_{d}',cpu,n,extra=['--d',d,'--curves',1,'--exponent',127]);assert rows[0]['status']=='NF'
    # M127 at B2=40000 and D=30 has original last=1333. For m=7 the first
    # retained centre is 191; no terminal point may move beyond that grid.
    log=(out/'wheel_30.log').read_text()
    assert 'gwpoly-range: original_first=32 first=191 last=1333' in log
    assert 'giants=1143 ' in log and 'replay original grid' not in log
    rows=run('packed_threaded',cpu,n,b2=10**8,extra=['--d',60060,'--curves',1,'--exponent',127]);assert rows[0]['stage2-engine']['checked-values']==3
    full_log=(out/'packed_threaded.log').read_text()
    full_mib=float(re.search(r'planned_mib=([\d.]+)',full_log)[1])
    assert 'ftree=full' in full_log
    assert 'ftree_rebuild_seconds=0.000000' in full_log
    constrained=['--d',60060,'--curves',1,'--exponent',127,'--memory-mib',int(full_mib*0.85)]
    rebuilt=run('segmented_threaded',cpu,n,b2=10**8,extra=constrained)[0]
    segment_log=(out/'segmented_threaded.log').read_text()
    assert 'ftree=segmented' in segment_log and float(re.search(r'ftree_rebuild_seconds=([\d.]+)',segment_log)[1])>0
    assert rebuilt['status']==rows[0]['status']=='NF' and rebuilt['stage2-engine']['checked-values']==3
    run('skip',cpu,n,extra=['--skip',1,'--curves',1])
    if a.gpu:
        gpu=save('gpu_m127',n,gpu=True);assert oracle.records(gpu)[0][:4]==oracle.records(cpu)[0][:4]
        assert run('gpu_save',gpu,n,extra=['--exponent',127])[0]['status']=='NF'
    wider=(1<<2203)-1;wide=save('cpu_m2203',wider)
    assert run('wider_carrier',wide,wider,b2=10**6,extra=['--exponent',2203])[0]['status']=='NF'
    for exponent, factor, bits in [(8011,80111,7995),(16319,270811405193137,16272)]:
        modulus, remainder=divmod((1<<exponent)-1,factor)
        assert remainder==0 and modulus.bit_length()==bits
        path=save(f'cofactor_m{exponent}',f'(2^{exponent}-1)/{factor}',sigma=26)
        for threads in (1,4):
            name=f'cofactor_m{exponent}_t{threads}'
            row=run(name,path,modulus,b2=70000000,threads=threads,
                    extra=['--exponent',exponent,'--known-factor',factor,'--d',13860])[0]
            assert row['status']=='NF' and row['stage2-engine']['checked-values']==3
            log=(out/(name+'.log')).read_text()
            blocks=[(int(n),int(w)) for n,w in re.findall(r'gwpoly-gtree: roots=(\d+) tile=(\d+)',log)]
            assert blocks and blocks[0][0]==row['poly-size'] and blocks[-1][0]<row['poly-size']
            assert all(width>2 if threads==1 else width==2 for _,width in blocks)
    # GW add/subtract may mutate source FFT states: shared point constants
    # caused a four-thread failure on this large wheel despite a serial pass.
    m347=(1<<347)-1;large_bounds=save('cpu_m347',m347)
    checked=run('large_bounds_threaded',large_bounds,m347,b2=2600000000000,
                extra=['--memory-mib',4096,'--exponent',347])[0]
    assert checked['status']=='NF' and checked['stage2-engine']['checked-values']==3
    if a.reference_study:
        plan=json.loads((a.reference_study/'plan.json').read_bytes())
        for exponent in (10267,16319):
            case=next(c for c in plan['cases'] if c['exponent']==exponent)
            assert sha(case['save'])==case['save_sha256']
            reference=min((r for r in plan['prime95_rows'] if r['curve_type']=='Edwards' and r['benchmark_eligible'] and r['exponent']==exponent),key=lambda r:r['b2'])
            assert case['N_hex']==reference['N_hex'] and case['sigma']==int(reference['s'])
            extra=['--curves',1,'--memory-mib',4096,'--exponent',exponent]
            for factor in case['known']:extra+=['--known-factor',factor]
            row=run(f'exact_reference_m{exponent}',case['save'],int(case['N_hex'],16),b2=reference['b2'],extra=extra)[0]
            assert row['status']=='NF' and row['stage2-engine']['checked-values']==3
    large_sigma=(1<<64)-19;large=save('large_sigma',n,sigma=large_sigma)
    assert run('sigma64',large,n,extra=['--exponent',127])[0]['Edwards']['sigma']==large_sigma
    hit_n=10007*n;hit=save('factor_fixture',hit_n,b1=20)
    a24,x,z=oracle.stage1(hit_n,17,20);product=1
    for prime in range(21,40001):
        if any(prime%d==0 for d in range(2,math.isqrt(prime)+1)):continue
        _,rz=oracle.mont.ladder(prime,x,z,a24,hit_n);product=product*rz%hit_n
    expected=math.gcd(product,hit_n);assert 1<expected<hit_n
    r=run('stage2_factor',hit,hit_n)[0];assert r['stage']==2 and int(r['factors'][0])%expected==0
    # Force the shortened grid on a genuine hit. The canonical replay must
    # recover the same factor as the independent prime-by-prime ladder.
    recovered=run('suffix_factor_replay',hit,hit_n,extra=['--d',30])[0]
    assert recovered['factors']==r['factors'] and 'replay original grid' in (out/'suffix_factor_replay.log').read_text()
    # The saved point has prime order 127 modulo 10007. At D=30 it is
    # first represented by 4*30+7, beyond the two centres for B2=45.
    assert oracle.mont.ladder(127,x,z,a24,hit_n)[1] % 10007 == 0
    assert run('before_rounded_hit',hit,hit_n,45,['--d',30])[0]['status']=='NF'
    rounded=run('rounded_hit',hit,hit_n,45,['--d',30],config=default_ini)[0]
    assert (rounded['b2'],rounded['factors'])==(134,['10007'])
    assert rounded['stage2-engine']['requested-B2']==45 and rounded['stage2-engine']['actual-B2']==134
    rounded_public=json.loads((out/'rounded_hit.json.txt').read_bytes())
    verify_crc(rounded_public)
    assert rounded_public['b2']==134
    explicit=run('explicit_rounded_hit',hit,hit_n,134,['--d',30])[0]
    assert explicit['factors']==rounded['factors']
    run('round_over_limit',cpu,n,b2=10**15,extra=['--d',210,'--b2-round-chain',1],good=False)
    # Small primes represented in the baby wheel have already passed the
    # denominator check. Prime divisors of D require a separate ladder.
    for prime, sigma, order in [(131, 31, 3), (67, 19, 5), (97, 26, 7)]:
        modulus = prime * n
        path = save(f'small_q{order}', modulus, b1=2, sigma=sigma)
        a24, x, z = oracle.stage1(modulus, sigma, 2)
        assert math.gcd(oracle.mont.ladder(order, x, z, a24, modulus)[1], modulus) == prime
        for wheel in (2 * order**2, 2 * (order + 1)):
            row = run(f'small_q{order}_d{wheel}', path, modulus, b2=order, extra=['--d', wheel])[0]
            assert row['stage'] == 2 and str(prime) in row['factors']
    stored=save('stored_factor',1009*n)
    assert run('cached_stage1_factor',stored,1009*n)[0]['stage']==1
    damaged=out/'damaged.save';damaged.write_text(cpu.read_text().replace('CHECKSUM=','CHECKSUM=9',1))
    run('reject_checksum',damaged,n,good=False)
    for name,args in [('metadata',['--exponent',131]),('wheel',['--d',5]),('memory',['--d',20000000,'--memory-mib',1]),('bounds',['--b2',10**15+1])]:
        run('reject_'+name,cpu,n,extra=args,good=False)
    # An actual asynchronous stop must return 130 and publish no current NF.
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    class Security(ctypes.Structure):_fields_=[('length',wintypes.DWORD),('descriptor',wintypes.LPVOID),('inherit',wintypes.BOOL)]
    kernel.CreateEventW.argtypes=[ctypes.POINTER(Security),wintypes.BOOL,wintypes.BOOL,wintypes.LPCWSTR];kernel.CreateEventW.restype=wintypes.HANDLE
    kernel.SetEvent.argtypes=[wintypes.HANDLE];kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    sa=Security(ctypes.sizeof(Security),None,True);event=kernel.CreateEventW(ctypes.byref(sa),True,False,None)
    try:
        result=out/'stopped.jsonl'
        with (out/'stopped.log').open('wb') as log:
            proc=subprocess.Popen(list(map(str,[s2,'--config',ini,'--save',cpu,'--curves',1,'--backend','gwpoly','--b2',10**12,
                '--threads',4,'--results',result,'--submission',out/'stopped.json.txt','--stop-handle',event])),
                close_fds=False,stdout=log,stderr=subprocess.STDOUT,cwd=out)
            time.sleep(.35);kernel.SetEvent(event);assert proc.wait(timeout=60)==130
        assert not result.exists() and not (out/'stopped.json.txt').exists()
        report['checks'].append(dict(name='stop_inside_gwpoly',exit=130))
    finally:kernel.CloseHandle(event)
    report['complete']=True;(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('ALL PASS',flush=True)


if __name__=='__main__':main()
