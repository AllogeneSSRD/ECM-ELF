#!/usr/bin/env python3
"""Check reset-window partial products against an independent integer Montgomery ladder."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stat'))
from bench_stage1_prac_windows import sample
from ecm_prac_plan import prac_counts
from test_cuda_prac import run, rows


def double(point, a24, n):
    x, z = point
    aa = (x + z)**2 % n; bb = (x - z)**2 % n; e = (aa - bb) % n
    return aa * bb % n, e * (bb + a24 * e) % n


def add(p, q, difference, n):
    u = (p[0] + p[1]) * (q[0] - q[1]) % n
    v = (p[0] - p[1]) * (q[0] + q[1]) % n
    return difference[1] * (u + v)**2 % n, difference[0] * (u - v)**2 % n


def multiply(sigma, scalar, n):
    u = (sigma*sigma - 5) % n; v = 4*sigma % n
    point = pow(u, 3, n), pow(v, 3, n)
    a24 = pow(v-u, 3, n) * (3*u+v) * pow(16*pow(u, 3, n)*v % n, -1, n) % n
    left, right = point, double(point, a24, n)
    for bit in bin(scalar)[3:]:
        summed = add(left, right, point, n)
        if bit == '0': left, right = double(left, a24, n), summed
        else: left, right = summed, double(right, a24, n)
    return left


def prime(p):
    return p >= 2 and (p == 2 or p % 2 != 0 and all(p % d for d in range(3, math.isqrt(p)+1, 2)))


def product(cache, result):
    torsion = 12 if result['exponent'] == 'choose12' else 1
    file = cache / f"prac_v1_b{result['B1']}_t{torsion}_s7.bin"
    with file.open('rb') as source:
        magic, b1, scalar_id, records, full_work, checksum, version, search, t, endian = struct.unpack('<6Q4I',source.read(64))
        assert magic == 0x31504e414c504345 and b1 == result['B1'] and t == torsion
        assert (version, search, endian) == (1, 7, 0x12345678)
        assert file.stat().st_size == 64+16*records and full_work == result['full_work']
        count = min(result['requested_count'],records)
        first = 0 if result['position']=='prefix' else (records-count)//2 if result['position']=='middle' else records-count
        assert (result['first'],result['count']) == (first,count)
        source.seek(64+16*first)
        entries = [struct.unpack('<4I',source.read(16)) for _ in range(count)]
    scalar = 1; work = 0
    for p, d, repeats, cost in entries:
        assert prime(p) and (p <= b1 or b1==2 and torsion==12 and p==3)
        power = p; expected = 1
        while power <= b1//p: power *= p; expected += 1
        if torsion==12: expected += 2 if p==2 else 1 if p==3 and b1>=3 else 0
        assert repeats == expected
        dbl, dadd = (1,0) if p==2 else prac_counts(p,d)
        assert cost == 5*dbl+6*dadd
        scalar *= p**repeats; work += cost*repeats
    assert work == result['work']
    assert (entries[0][0],entries[-1][0]) == (result['p_first'],result['p_last'])
    return scalar


def verify(cache, folder, result):
    scalar = product(cache,result); n = 2**result['bits']-1
    with (folder/'window_q.csv').open(encoding='ascii',newline='') as source:
        actual = list(csv.DictReader(source))
    assert len(actual) == result['curves']
    for i,row in enumerate(actual):
        assert row['kind']=='partial_product' and int(row['b1'])==result['B1']
        assert tuple(map(int,(row['first'],row['count'],row['work']))) == (result['first'],result['count'],result['work'])
        sigma = int(row['sigma']); assert sigma == result['sigma']+i
        x,z = int(row['x'],16),int(row['z'],16)
        assert 0 <= x < n and 0 <= z < n and (x or z), 'Noncanonical/degenerate GPU coordinates'
        ox,oz = multiply(sigma,scalar,n)
        assert (ox or oz) and x*oz%n == ox*z%n, f'Window Q mismatch: {folder}, sigma={sigma}'
    projected = result['full_work']*result['kernel_ms']/(result['work']*result['measured'])/1000/result['curves']
    # Both native fields print six decimals; a nanosecond rounding error is
    # amplified when a tiny gate window projects a production-size total W.
    tolerance = 0.5e-6 * result['full_work']/(result['work']*result['measured'])/1000/result['curves'] + 0.5e-6
    assert abs(projected-result['projected_s_per_curve']) <= tolerance + 1e-12
    return len(actual)


def rejected(exe, folder, cache, settings, message, device):
    folder.mkdir()
    env = dict(os.environ, ECM_GPU_STAGE1_ALGO='prac',ECM_PRAC_REG_TARGET='255',
        ECM_PRAC_VARIANT='baseline',ECM_STAGE1_TPI='0',ECM_PRAC_WINDOW='tail',
        ECM_PRAC_WINDOW_COUNT='4',ECM_PRAC_WINDOW_WARMUP='1',ECM_PRAC_WINDOW_DUMP='1',
        ECM_GPU_STAGE1_SAMPLE_SECONDS='0.001',ECM_PRAC_PLAN_CACHE=str(cache))
    env.update(settings); env.pop('ECM_GPU_DUMP',None)
    cmd = [str(exe),'-gpu','-d',str(device),'--gpu-param','0','-sigma','0:26','-gpucurves','8',
        '--ckpt','0','--exp-cache',str(cache),'-savea','completed.save','1000','0']
    proc = subprocess.run(cmd,input=b'(2^4423-1)\n',env=env,cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    text=proc.stdout.decode('utf-8',errors='replace');(folder/'run.log').write_text(text,encoding='utf-8')
    assert proc.returncode==1 and message in text and 'PRAC_WINDOW_DONE' not in text
    assert not list(folder.glob('.ecm_ckpt_*')) and not (folder/'window_q.csv').exists()
    assert not rows(folder)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,default=Path('build_cuda_cmake/prac/ecm_cuda.exe'))
    p.add_argument('--cache',type=Path)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--production',action='store_true',help='Also verify 16-record B1=10m/260m subproducts using existing caches')
    p.add_argument('--production-counts',type=int,nargs='+',default=[16],help='Production window lengths, each 1..32')
    p.add_argument('--production-chunks',type=int,nargs='+',default=[0],help='Per-launch record counts, 0 means unsplit')
    a=p.parse_args();exe=a.exe.resolve(strict=True);cache=(a.cache or exe.parent).resolve();root=a.output.resolve()
    if any(not 1 <= count <= 32 for count in a.production_counts):p.error('Production window counts must be 1..32')
    if any(not 0 <= chunk <= min(a.production_counts) for chunk in a.production_chunks):p.error('Chunks must be in 0..smallest production count')
    root.mkdir(parents=True,exist_ok=False);results=[]
    for n in (2203,4423,8191):
        configs=[('baseline',0,255)]
        if n in (2203,4423):configs += [('baseline',32,255)]
        if n==4423:configs += [('compact',16,255),('compact',16,168)]
        for variant,tpi,registers in configs:
            for exponent in ('lcm','choose12'):
                for position in ('prefix','middle','tail'):
                    folder=root/f'n{n}_{variant}_t{tpi}_reg{registers}_{exponent}_{position}'
                    r=sample(exe,folder,n,1000,8,a.device,tpi,registers,variant,position,4,0.001,1,cache,True,exponent=exponent)
                    r['Q_compared']=verify(cache,folder,r);r['passed']=True;results.append(r)
    for b1,sigma,exponent in [(2,26,'choose12'),(10000,4611686018427511360,'choose12')]:
        for position in ('prefix','middle','tail'):
            folder=root/f'boundary_b{b1}_{position}'
            r=sample(exe,folder,4423,b1,8,a.device,16,168,'compact',position,16,0.001,1,cache,True,sigma=sigma,exponent=exponent)
            r['Q_compared']=verify(cache,folder,r);r['passed']=True;results.append(r)
    slicing_signatures={};bitwise_slice_checks=0
    if a.production:
        for b1 in (10000000,260000000):
            for variant,registers in [('baseline',255),('baseline',168),('compact',255),('compact',168)]:
                for count in a.production_counts:
                    for chunk in a.production_chunks:
                        for position in ('prefix','middle','tail'):
                            folder=root/f'production_b{b1}_{variant}_reg{registers}_c{count}_chunk{chunk}_{position}'
                            r=sample(exe,folder,4423,b1,8,a.device,16,registers,variant,position,count,0.001,1,cache,True,chunk=chunk)
                            signature=hashlib.sha256((folder/'window_q.csv').read_bytes()).hexdigest()
                            key=(b1,variant,registers,count,position)
                            if key in slicing_signatures:
                                assert signature==slicing_signatures[key], 'Slicing changed projective output bytes'
                                bitwise_slice_checks+=1
                            else:slicing_signatures[key]=signature
                            r['Q_compared']=verify(cache,folder,r);r['passed']=True;results.append(r)
    # A real compact checkpoint remains byte-identical during window runs and
    # can subsequently resume via the baseline kernel at the same TPI.
    settings=('(2^4423-1)',1000,8,4611686018427511360,'choose12')
    folder=root/'checkpoint_isolation'
    text=run(exe,folder,*settings,'prac',a.device,sample=0.000001,tpi=16,registers=168,variant='compact')
    assert 'sample limit reached' in text and len(list(folder.glob('.ecm_ckpt_*')))==1
    r=sample(exe,folder,4423,1000,8,a.device,16,168,'compact','tail',4,0.001,1,cache,True,sigma=settings[3],exponent='choose12')
    r['Q_compared']=verify(cache,folder,r);r['passed']=True;results.append(r)
    run(exe,root/'checkpoint_reference',*settings,'cpu',a.device)
    text=run(exe,folder,*settings,'prac',a.device,tpi=16,registers=255,variant='baseline')
    assert 'checkpoint resumed' in text and rows(folder)==rows(root/'checkpoint_reference')
    assert not list(folder.glob('.ecm_ckpt_*'))
    failures=[({'ECM_PRAC_WINDOW':'bad'},'must be prefix, middle or tail'),
              ({'ECM_PRAC_WINDOW_COUNT':'0'},'count must be 1..32'),
              ({'ECM_PRAC_WINDOW_COUNT':'-1'},'invalid ECM_PRAC_WINDOW_COUNT'),
              ({'ECM_GPU_STAGE1_SAMPLE_SECONDS':'nan'},'sample seconds must be finite'),
              ({'ECM_GPU_STAGE1_SAMPLE_SECONDS':'0'},'sample seconds must be finite'),
              ({'ECM_PRAC_WINDOW_CHUNK':'-1'},'invalid ECM_PRAC_WINDOW_CHUNK'),
              ({'ECM_PRAC_WINDOW_CHUNK':'5'},'chunk must not exceed selected count'),
              ({'ECM_PRAC_VARIANT':'compact','ECM_STAGE1_TPI':'32'},'policy is unavailable'),
              ({'ECM_PRAC_VARIANT':'compact','ECM_PRAC_REG_TARGET':'0'},'register policy 255 or 168'),
              ({'ECM_GPU_STAGE1_ALGO':'ladder'},'requires ECM_GPU_STAGE1_ALGO=prac')]
    for i,(settings,message) in enumerate(failures):rejected(exe,root/f'reject{i}',cache,settings,message,a.device)
    report=dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),results=results,
        window_Q_compared=sum(r['Q_compared'] for r in results),resume_Q_compared=8,
        rejection_cases=len(failures),production_windows=a.production,
        production_counts=a.production_counts if a.production else [],
        production_chunks=a.production_chunks if a.production else [],
        bitwise_slice_checks=bitwise_slice_checks,passed=True)
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
