"""Independently audit a frozen Edwards relocation timing matrix and raw logs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fields(text,label):
    rows=re.findall('^'+re.escape(label)+r': (.*)$',text,re.M)
    assert rows,label
    return dict(re.findall(r'(\w+)=([^\s]+)',rows[-1]))


def verify_order(c,n,x):
    """Direct prime-order multiplication independently checks the BSGS fixture."""
    sys.path.insert(0,str(ROOT/'tools/test'))
    from test_edwards_gpu_pipeline import stage1
    p,r=int(c['known_factor']),c['residual_order']
    for v in (p,r):assert v>1 and all(v%d for d in range(2,math.isqrt(v)+1))
    a,u,z=stage1(p,c['sigma'],c['b1']);u=u*pow(z,-1,p)%p
    assert x%p==u and n%p==0 and r>c['d']//2 and r>c['b2']//c['d']+2
    A,B=(4*a-2)%p,4*a%p;y2=(u**3+A*u*u+u)*pow(B,-1,p)%p
    y=pow(y2,(p+1)//4,p);assert y*y%p==y2
    Q=(B*u%p,B*B*y%p);a2,a4=A*B%p,B*B%p
    def add(P,Q):
        if P is None:return Q
        if Q is None:return P
        x,y=P;u,v=Q
        if x==u and (y+v)%p==0:return None
        slope=((3*x*x+2*a2*x+a4)*pow(2*y,-1,p) if x==u else (v-y)*pow(u-x,-1,p))%p
        w=(slope*slope-a2-x-u)%p
        return w,(slope*(x-w)-y)%p
    R=None;P=Q;k=r
    while k:
        if k&1:R=add(R,P)
        P=add(P,P);k>>=1
    assert R is None and Q is not None


def audit(study,partial=False):
    path=study/'plan.json';p=read(path)
    assert (p['complete'] or partial) and p['warmups']==1 and p['repeats']>=3
    b=p['binary'];exe=Path(b['path']);assert sha(exe)==b['sha256']
    for name,h in b['runtime_files'].items():assert sha(exe.parent/name)==h
    for name,h in b['identity']['sources'].items():assert sha(exe.parent/'sources'/name)==h
    version=re.search(r'version\[\]\s*=\s*"([0-9.]+)"',
                      (exe.parent/'sources/src/core/ecm_program_version.h').read_text())[1]
    for name,h in p['helpers'].items():assert sha(study/'helpers'/name)==h and sha(ROOT/name)==h
    assert sha(study/'collector_snapshot.py')==p['collector_sha256']
    assert sha(p['gates']['path'])==p['gates']['sha256']
    gates=read(p['gates']['path']);assert gates['complete'] and gates['engine_sha256']==b['sha256']
    for c in gates['checks']:assert sha(c['log'])==c['sha256']
    zero_root=Path(p['gates']['path']).parent/'baby_zero';zero=read(zero_root/'summary.json')
    assert zero['complete'] and not zero['expected_old_miss'] and zero['probe_sha256']==gates['probe_sha256']
    assert len(zero['runs'])==4
    for r in zero['runs']:
        assert sha(zero_root/r['log'])==r['sha256']
        if r['relocation']:assert r['result']['replayed'] and r['result']['probe_seconds']==0
    assert sha(p['ini'])==p['ini_sha256']
    assert sha(p['source_reference']['path'])==p['source_reference']['sha256']
    assert len(p['cells'])==6 and len({c['id'] for c in p['cells']})==6
    if not partial:assert len(p['runs'])==len(p['cells'])*2*(p['repeats']+1)
    assert len({r['id'] for r in p['runs']})==len(p['runs'])
    checked=0;ntt=0
    for c in p['cells']:
        assert sha(c['save'])==c['save_sha256']
        tokens=dict(v.strip().split('=',1) for v in Path(c['save']).read_text().split(';') if '=' in v)
        n=int(tokens['N'],0);assert n.bit_length()==c['bits'] and int(tokens['B1'])==c['b1']
        if c['id']=='wide_factor_unit':verify_order(c,n,int(tokens['X'],0))
        rows=[r for r in p['runs'] if r['cell']==c['id']]
        expected={(k,m) for k in range(p['repeats']+1) for m in (0,1)}
        observed={(r['repeat'],r['mode']) for r in rows}
        assert observed<=expected and len(observed)==len(rows)
        if not partial:assert observed==expected
        if not rows:continue
        assert len({tuple(r['factors']) for r in rows})==1
        for mode in (0,1):assert len({json.dumps(r['leaf'],sort_keys=True) for r in rows if r['mode']==mode})<=1
        if c['kind']=='F':
            assert len({json.dumps(r['leaf'],sort_keys=True) for r in rows})==1
            assert n%int(c['known_factor'])==0
        last=c['b2']//c['d']+2
        m=next(m for m in range(2,c['d']) if math.gcd(m,c['d'])==1)
        first=max(1,last//m)
        ntt_by_mode={0:[],1:[]}
        for r in rows:
            out=Path(r['raw'])
            for name,h in r['raw_hashes'].items():assert sha(out/name)==h
            assert set(r['raw_hashes'])=={str(f.relative_to(out)) for f in out.rglob('*') if f.is_file()}
            row=read(out/'internal.jsonl')
            assert row['program']==dict(name='ECM-ELF',version=version)
            assert row['curve']=='ed' and row['param'] is None and row['bad_factors']==0
            assert row['B1']==c['b1'] and row['B2']==c['b2'] and row['d']==c['d'] and int(row['N_hex'],16)==n
            assert sorted(row['factors'])==r['factors'] and row['relocation']==bool(r['mode'])
            assert all(1<int(f)<n and n%int(f)==0 for f in row['factors'])
            assert r['warmup']==(r['repeat']==0)
            text=(out/'engine.log').read_text()+'\n'+(out/'debug.log').read_text()
            t=fields(text,'stage2_full_wall');s=fields(text,'s4_multiply_stats')
            assert t['clean']=='1' and abs(float(t['init'])+float(t['main'])-float(t['total']))<3e-6
            for k in ('init','main','shape','total'):assert float(t[k])==r['timing'][k]
            assert int(s['gmp_checked'])>0 and int(s['gmp_selftest_cases'])>0
            assert int(s['gmp_check_bad'])==int(s['gmp_selftest_bad'])==0
            checked+=int(s['gmp_checked'])
            assert fields(text,'descent_values')==r['leaf']
            assert row['giant_last']==last
            if c['kind']=='NF':
                assert not row['factors'] and not row['relocation_replayed']
                assert row['giant_first']==(first if r['mode'] else 1)
            else:
                assert row['factors'] and row['giant_first']==1
                assert row['relocation_replayed']==bool(r['mode'])
            assert row['giant_points']==last-row['giant_first']+1
            published=[json.loads(line) for line in (out/'results.json.txt').read_text().splitlines() if line.strip()]
            assert len(published)==1
            sub=published[0];assert sub['status']==c['kind'] and sub['worktype']=='ECM'
            assert sub['b1']==c['b1'] and sub['b2']==c['b2'] and sub['d']==c['d'] and sub['fft-length']>0
            keys=list(sub);assert keys[keys.index('b2')+1:keys.index('b2')+3]==['d','poly-size']
            assert sub['program']['name']=='ECM-ELF' and sub['program']['version']==version
            assert not any(k.startswith(('relocation','giant_')) for k in sub)
            assert 'checksum' not in sub and 'security-code' not in sub
            if c['kind']=='NF':assert sub['curves']==1 and sub['Edwards']=={}
            else:assert sorted(sub['factors'])==row['factors'] and sub['stage']==2
            probe=row['relocation_probe_seconds'];replay=row['relocation_replay_seconds']
            assert probe>=0 and replay>=0 and probe+replay<=float(t['main'])+3e-6
            if not r['mode']:assert probe==replay==0 and row['relocation_points']==0
            elif c['id']=='wide_factor_baby_nonunit':assert probe==0 and abs(replay-float(t['main']))<3e-6
            else:assert row['relocation_points']==last-first+1
            # The counter table must describe actual transforms, including replay.
            countrows=re.findall(r'^ntt_count_row:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$',text,re.M)
            assert countrows,'Missing NTT counter table'
            lengths=set();total=0;convolutions=0
            for L,k,conv,fwd,inv,allcalls in countrows:
                L,k,conv,fwd,inv,allcalls=map(int,(L,k,conv,fwd,inv,allcalls))
                assert L==1<<k and L not in lengths and fwd==2*conv and inv==conv and allcalls==3*conv
                lengths.add(L);total+=allcalls;convolutions+=conv
            work=fields(text,'ntt_work')
            assert int(work['transforms'])==total and int(work['convolutions'])==convolutions
            ntt+=total
            ntt_by_mode[r['mode']].append(tuple(tuple(map(int,v)) for v in countrows))
        for tables in ntt_by_mode.values():assert len(set(tables))<=1,'NTT arithmetic changed across repetitions'
        if c['id']=='wide_factor_baby_nonunit':
            assert len(set(ntt_by_mode[0]+ntt_by_mode[1]))==1,'Baby nonunit did extra polynomial work'
    return dict(complete=p['complete'] and not partial,version=version,study=str(path.resolve()),study_sha256=sha(path),binary_sha256=b['sha256'],
                audit_source_sha256=sha(__file__),runs=len(p['runs']),formal=sum(not r['warmup'] for r in p['runs']),
                gmp_checked=checked,ntt_total=ntt,gates=len(gates['checks']))


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--study',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--partial',action='store_true',help='Audit completed invocations only; output is never a complete plotting gate')
    args=a.parse_args()
    result=audit(args.study.resolve(),args.partial);args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise ValueError('Use a fresh audit path')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
