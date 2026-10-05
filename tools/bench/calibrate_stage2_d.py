"""Collect serial Stage2 phase measurements at explicit D values using a verified A/B environment.

Does not modify ini/worktodo or choose a production default. An output directory must be fresh.
Every curve keeps mandatory arithmetic checks; different D values have different leaf/sample sets.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from functools import lru_cache

GL_P=(1<<64)-(1<<32)+1
# Frozen field-convolution ratios measured before the shape-policy D fit.
# They weight only the empirical NTT feature, not elapsed time or GPU cycles.
SHAPE_WEIGHTS={24:0.9386473792217225,25:0.8672913453433596,
               26:0.885360571656581,27:0.9302319270266772}
# Frozen before any short-reducer D fit: shape weights times same-binary
# short/long convolution ratios (8 ABBA+BAAB rows per size, GPU1, 2026-10-05).
# Unmeasured smaller sizes retain weight 1; this is an empirical ranking feature.
SHORT_WEIGHTS={16:0.7929270230627675,17:0.7355812422499033,
               18:0.7235194676663127,19:0.6392864988260281,
               20:0.631309088558015,21:0.6211608661393889,
               22:0.6778661412185687,23:0.8096173502640038,
               24:0.5948253971194828,25:0.5687842601012635,
               26:0.5722769388206116,27:0.6134229919008334}
FIXED_PTX_WEIGHTS=None

def load_fixed_ptx_weights(data):
    """Use immutable measured weights, independently of fitted full curves."""
    global FIXED_PTX_WEIGHTS
    if data.get('feature_profile')!=5 or data.get('gl_fixed_mode')!=3:
        raise ValueError('Weights must describe the measured fixed PTX backend')
    evidence=Path(data['measurements'])
    if hashlib.sha256(evidence.read_bytes()).hexdigest()!=data['measurements_sha256']:
        raise ValueError('Frozen NTT weight evidence changed')
    values={int(k):float(v) for k,v in data['weights'].items()}
    if set(values)!=set(range(16,28)) or any(not math.isfinite(v) or v<=0 for v in values.values()):
        raise ValueError('Need twelve positive finite NTT weights')
    measured=json.loads(evidence.read_text(encoding='utf-8'))
    reference=[r for r in measured['comparisons'] if r['base']=='reference_short']
    if len(reference)!=12 or measured.get('passed')!=16 or measured.get('failed')!=0:
        raise ValueError('Incomplete frozen NTT weight measurement')
    expected={r['k']:SHORT_WEIGHTS[r['k']]*r['means']['ptx']/r['means']['reference_short'] for r in reference}
    if any(values[k]!=expected[k] for k in values):
        raise ValueError('Weights differ from frozen convolution evidence')
    FIXED_PTX_WEIGHTS=values
    tree.cache_clear();inverse.cache_clear()

def command_bits(cmd):
    if '--n-hex' in cmd:return int(cmd[cmd.index('--n-hex')+1],16).bit_length()
    if '--save' in cmd:
        line=Path(cmd[cmd.index('--save')+1]).read_bytes().split(b'\n',1)[0].decode('ascii')
        if not re.search(r'\bN=\(2\^4423-1\)',line):raise ValueError('Save calibration currently requires exact M4423')
        return 4423
    raise ValueError('Cannot prove input modulus')

def command_b1(cmd):
    if '--b1' in cmd:return int(cmd[cmd.index('--b1')+1])
    line=Path(cmd[cmd.index('--save')+1]).read_bytes().split(b'\n',1)[0].decode('ascii')
    m=re.search(r'\bB1=(\d+)',line)
    if not m:raise ValueError('Cannot prove save B1')
    return int(m[1])

def phi(n):
    result=n;p=2
    while p*p<=n:
        if n%p==0:
            while n%p==0:n//=p
            result-=result//p
        p+=1
    if n>1:result-=result//n
    return result

@lru_cache(None)
def shape(p,bits):
    # Integer reproduction of choose_cfg's exact bound, used for the offline fit.
    # The integrated selector must query the real C++ backend; its plan gate compares these.
    slot=2*bits+max(1,(p-1).bit_length())
    for bpw in range(62,0,-1):
        sw=(slot+bpw-1)//bpw
        if p*sw*((1<<bpw)-1)**2<GL_P:break
    n=1<<(2*p*sw).bit_length()
    if n>1<<29:raise ValueError('NTT shape refused')
    return n,bpw,sw

def unit(p,bits,profile=0):
    n,_,_=shape(p,bits)
    k=n.bit_length()-1
    if profile==5 and FIXED_PTX_WEIGHTS is None:raise ValueError('Fixed PTX weights not loaded')
    weights=FIXED_PTX_WEIGHTS if profile==5 else SHORT_WEIGHTS if profile in (3,4) else SHAPE_WEIGHTS if profile==2 else {}
    return n*k*weights.get(k,1.0)

@lru_cache(None)
def tree(p,bits,profile=0):
    total=0;h=1
    while h<p:
        total+=((p+h)//(2*h))*unit(h+1,bits,profile)
        h*=2
    return total

@lru_cache(None)
def inverse(k,bits,profile=0):
    size=1;total=0
    while size<k:
        size=min(2*size,k)
        total+=2*unit(size,bits,profile)
    return total

def features(d,b2,bits=4423,profile=0):
    p=phi(d)//2;i=b2//d+2;g=(i+p-1)//p;q,r=divmod(i,p)
    top_child=(1<<((p-1).bit_length()-1)) if p>1 else 1
    return dict(D=d,P=p,I=i,G=g,B2=b2,bits=bits,profile=profile,fold_ntt=shape(p+1,bits)[0],tree_ntt=shape(top_child+1,bits)[0],
                baby=p*max(1,math.log2(d)-2),affine=p,
                ftree=tree(p,bits,profile),gtrees=q*tree(p,bits,profile)+(tree(r,bits,profile) if r else 0),
                fold=(g-1)*unit(p+1,bits,profile),descent=tree(p,bits,profile),inverse=inverse(p+1,bits,profile),
                giant=i*(6+22*math.log2(b2)/64),accum=p,owner_bytes=8*((bits+63)//64)*(9*p+8)+48)

def read_log(path):
    b=Path(path).read_bytes()
    return b.decode('utf-16' if b[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig',errors='replace')

def parse(text):
    wall=re.search(r'stage2_full_wall:.*?init=([\d.]+) main=([\d.]+) total=([\d.]+).*clean=(\d+)',text)
    if not wall or wall[4]!='1':raise RuntimeError('Missing clean full Stage2 timing')
    split=re.search(r'real_batched_split: (.*)',text)
    baby=re.search(r'real_baby: points=(\d+) ladder=([\d.]+) s affine=([\d.]+) s degenerate=(\d+)',text)
    if not split or not baby or baby[4]!='0':raise RuntimeError('Missing nondegenerate phase timing')
    row=dict(re.findall(r'(\w+)=([\d.]+)',split[1]))
    row={k:float(v) for k,v in row.items()}
    row.update(init=float(wall[1]),main=float(wall[2]),full=float(wall[3]),baby=float(baby[2]),affine=float(baby[3]))
    row['ftree']=max(0,row['init']-row['baby']-row['affine'])
    row['residual']=row['main']-sum(row[k] for k in ('giant','gtrees','fold','descent','inv','accum','name'))
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--provenance',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--d',type=int,nargs='+',required=True)
    p.add_argument('--device',type=int,default=1);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--b2',type=int,help='Override only the Stage2 upper bound; keep Stage1 Q identical')
    p.add_argument('--coop-mode',type=int,choices=(0,1,2),default=0,
                   help='0 original, 1 forced cooperative, 2 measured shape policy')
    p.add_argument('--short-reduce',type=int,choices=(0,1),default=0)
    p.add_argument('--baby-device',type=int,choices=(0,1),default=0)
    p.add_argument('--fixed-ptx-weights',type=Path,help='Frozen k16..27 full-convolution weights for fixed mode3')
    p.add_argument('--source-manifest',type=Path,help='Verify raw compiled dependencies before and after each curve')
    p.add_argument('--expected-q-sha256',required=True,help='SHA256 of lowercase affine Q hex from an independent reference')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise RuntimeError('Use a fresh output directory')
    prov=json.loads(a.provenance.read_text(encoding='utf-8-sig'));exe=a.exe.resolve()
    frozen_weights=json.loads(a.fixed_ptx_weights.read_text(encoding='utf-8')) if a.fixed_ptx_weights else None
    if frozen_weights:
        if prov.get('gl_fixed_mode')!=3 or not a.baby_device or not a.short_reduce or a.coop_mode!=2:
            raise ValueError('Fixed PTX calibration needs verified fixed3/GPU baby/short/shape')
        load_fixed_ptx_weights(frozen_weights)
    base={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    base.update({k:str(v) for k,v in prov['env'].items() if v is not None})
    wanted='short_fold' if a.short_reduce else 'shape_outer' if a.coop_mode==2 else '6_mont'
    control=next((x for x in prov['mode_controls'] if x['mode']==wanted),None)
    if control is None and not a.short_reduce:
        control=next(x for x in prov['mode_controls'] if x['mode']=='6_mont')
    if control is None:raise ValueError('Provenance has no matching reducer control')
    base.update({k:str(v) for k,v in control.items() if k!='mode'})
    if frozen_weights:base.update(NTT_GL_PTX_REDUCE='1',NTT_GL_SHIFT_SCALE='0')
    base.pop('NTT_FUSE_TRACE',None);base.update(NTT_XADD6_TEST='0',NTT_XADD6_TEST_BAD='0',NTT_D_MODEL='0',
        NTT_FUSE_COOP_OUTER=str(a.coop_mode),NTT_FUSE_COOP_TEST='0',NTT_FUSE_COOP_BAD='0',
        NTT_GL_SHORT_REDUCE=str(a.short_reduce))
    if a.baby_device:
        if not a.short_reduce or a.coop_mode!=2:raise ValueError('GPU baby fit requires short reducer and shape policy')
        base.update(NTT_BABY_DEVICE='1',NTT_BABY_DEVICE_CHECK='0',NTT_BABY_DEVICE_TEST='0',
                    NTT_BABY_DEVICE_TEST_BAD='0',NTT_BABY_DEVICE_ALLOC_FAIL='0',NTT_BABY_DEVICE_MAX_MB='512')
    else:
        base['NTT_BABY_DEVICE']='0'
    if a.short_reduce and a.coop_mode!=2:
        raise ValueError('Short-reducer fit requires measured shape policy mode 2')
    cmd0=[str(exe)]+[str(x) for x in prov['args']]
    if a.b2 is not None:cmd0[cmd0.index('--b2')+1]=str(a.b2)
    # Replace only values in the observed argv, retaining the exact N/Q/sigma/bounds.
    b2=int(cmd0[cmd0.index('--b2')+1]);bits=command_bits(cmd0)
    save=Path(cmd0[cmd0.index('--save')+1]) if '--save' in cmd0 else None
    save_sha=hashlib.sha256(save.read_bytes()).hexdigest() if save else None
    if a.coop_mode==2 and (bits!=4423 or command_b1(cmd0)!=1000 or
        base.get('NTT_FUSE_T')!='12' or base.get('NTT_FUSE_M','4')!='4' or
        base.get('NTT_FUSE_WARP_TAIL')!='1' or base.get('NTT_FUSE_COMPACT_SCRATCH')!='1'):
        raise RuntimeError('Shape-policy calibration needs the measured M4423/B1/tile/warp/compact scope')
    rows=[];expected_q=None;sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    if sha!=prov['sha256'].lower():raise RuntimeError('Binary differs from provenance')
    manifest=json.loads(a.source_manifest.read_text()) if a.source_manifest else None
    repo=Path(__file__).resolve().parents[2]
    def verify():
        if hashlib.sha256(exe.read_bytes()).hexdigest()!=sha:raise RuntimeError('Binary changed')
        if save and hashlib.sha256(save.read_bytes()).hexdigest()!=save_sha:raise RuntimeError('Stage1 save changed')
        if manifest:
            if manifest['sha256'].lower()!=sha:raise RuntimeError('Source manifest binary differs')
            for rel,want in manifest['sources'].items():
                if hashlib.sha256((repo/rel).read_bytes()).hexdigest()!=want.lower():raise RuntimeError('Compiled source changed: '+rel)
    verify()
    for repeat in range(a.repeats):
        for d in (a.d if repeat%2==0 else a.d[::-1]):
            cmd=cmd0.copy();cmd[cmd.index('--d')+1]=str(d);cmd[cmd.index('--device')+1]=str(a.device)
            name=f'{repeat+1}_{d}';log=out/(name+'.log');start=time.perf_counter()
            result=out/(name+'.jsonl')
            if '--curve-worker' in cmd and '--results' not in cmd:
                cmd+=['--results',str(result)]
            print('RUN',name,flush=True)
            verify()
            with log.open('wb') as f:r=subprocess.run(cmd,env=base,stdout=f,stderr=subprocess.STDOUT,timeout=900)
            text=read_log(log)
            verify()
            if r.returncode:raise RuntimeError(f'{name} exit={r.returncode}; see {log}')
            if '--curve-worker' in cmd:
                result_file=Path(cmd[cmd.index('--results')+1])
                completed=json.loads(result_file.read_text(encoding='utf-8').splitlines()[-1])
                if completed['bad_factors']!=0:raise RuntimeError('Native worker reported invalid factors')
            if '[trace]' in text:raise RuntimeError('Synchronized trace invalidates timing')
            if (a.short_reduce or 'ntt_gl_reduce_mode:' in text) and \
               f'ntt_gl_reduce_mode: device={a.device} short={a.short_reduce}' not in text:
                raise RuntimeError('Actual Goldilocks reducer differs from requested mode')
            if frozen_weights and f'ntt_gl_reduce_mode: device={a.device} short=1 ptx=1 fixed=3' not in text:
                raise RuntimeError('Actual fixed PTX backend differs from frozen weights')
            for token in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1','point_arithmetic: xadd6=1'):
                if token not in text:raise RuntimeError('Missing '+token)
            q=re.search(r'real_setup_Q_full: hex=([0-9a-f]+)',text)
            if not q or hashlib.sha256(q[1].encode()).hexdigest()!=a.expected_q_sha256.lower():
                raise RuntimeError('Stage1 Q differs from independent reference')
            qline=q[0]
            if expected_q is None:expected_q=qline
            elif qline!=expected_q:raise RuntimeError('Stage1 Q changed across D')
            if a.coop_mode==2:
                if 'RTX 4060 Laptop' not in text or base.get('NTT_FUSE_T')!='12' or \
                   base.get('NTT_FUSE_WARP_TAIL')!='1' or base.get('NTT_FUSE_COMPACT_SCRATCH')!='1':
                    raise RuntimeError('Shape-policy fit requires the measured device/tile/warp/compact scope')
            if a.baby_device and f'baby_device: requested=1 enabled=1' not in text:
                raise RuntimeError('GPU baby fell back; cannot fit GPU preparation costs')
            profile=5 if frozen_weights else 4 if a.baby_device else 3 if a.short_reduce else 2 if a.coop_mode==2 else 0
            f=features(d,b2,bits,profile)
            fd=re.search(r'real_batched_folddevice:.*enabled=(\d+)',text)
            if not fd or (f['G']>1 and fd[1]!='1'):
                raise RuntimeError('Multiple G polynomials require GPU resident fold')
            oracle=re.search(r's4_oracle_stats:.*selected=(\d+) queued=(\d+) compared=(\d+) samples=(\d+) pending=(\d+)',text)
            if not oracle or oracle[1]!=oracle[3] or oracle[1]!=oracle[2] or oracle[5]!='0':
                raise RuntimeError('Oracle coverage incomplete')
            phases=parse(text)
            rows.append(dict(name=name,command=cmd,seconds=time.perf_counter()-start,phases=phases,features=f,log=str(log)))
            controls={k:v for k,v in base.items() if k.startswith('NTT_')}
            (out/'measurements.json').write_text(json.dumps(dict(exe=str(exe),sha256=sha,device=a.device,
                sources=manifest['sources'] if manifest else {},env=controls|{'NTT_FUSE_TRACE':None},
                Q_line=expected_q,runs=rows,feature_profile=profile,ntt_weights=frozen_weights,
                gl_fixed_mode=prov.get('gl_fixed_mode',-1),save_sha256=save_sha,
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
            print(name,'full=',phases['full'],flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
