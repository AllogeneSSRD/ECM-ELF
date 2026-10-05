"""Scoped phase cost model and Prime95-style relative curve value.

Profiles are measured seconds, not instruction counts or absolute factor odds.
"""
import math
from calibrate_stage2_d import features

PHASE_FEATURES = dict(baby='baby',affine='affine',ftree='ftree',
    gtrees='gtrees',fold='fold',descent='descent',inv='inverse',accum='accum',glue='G')


def kruppa_value(b1,b2):
    if not 2 <= b1 <= b2: raise ValueError('Need 2 <= B1 <= B2')
    a = 1.96617-0.06781*math.log10(b1)
    return 0.11343+0.88657*(math.log10(b2/b1)/2)**a


def observed(row):
    p = row['phases']
    out = {key:p[key] for key in PHASE_FEATURES if key!='glue'}
    out['giant'] = p['giant']
    # Parse's residual excludes naming. The complete model must include it.
    out['glue'] = p['residual']+p['name']
    if any(not math.isfinite(v) or v < 0 for v in out.values()):
        raise ValueError('Negative/unusable phase decomposition')
    return out


def fit(rows):
    rates = {}
    for phase,feature in PHASE_FEATURES.items():
        denominator = sum(r['features'][feature]**2 for r in rows)
        if denominator <= 0: raise ValueError('Uncovered phase: '+phase)
        rates[phase] = sum(r['features'][feature]*observed(r)[phase] for r in rows)/denominator
    chain=[(giant_work(r['features']),observed(r)['giant']) for r in rows if giant_work(r['features'])['ladder']==0]
    ladder=[(giant_work(r['features']),observed(r)['giant']) for r in rows if giant_work(r['features'])['chain']==0]
    if not chain or not ladder: raise ValueError('Need measured chain and ladder paths')
    # Two nonnegative coefficients: chain point work and per-chunk fixed overhead.
    xx=sum(w['chain']**2 for w,y in chain);xz=sum(w['chain']*w['chain_chunks'] for w,y in chain)
    zz=sum(w['chain_chunks']**2 for w,y in chain)
    xy=sum(w['chain']*y for w,y in chain);zy=sum(w['chain_chunks']*y for w,y in chain)
    det=xx*zz-xz*xz
    choices=[(max(0,xy/xx),0),(0,max(0,zy/zz))]
    if det>xx*zz*1e-12:
        alpha,beta=(xy*zz-zy*xz)/det,(zy*xx-xy*xz)/det
        if alpha>=0 and beta>=0: choices.append((alpha,beta))
    alpha,beta=min(choices,key=lambda c:sum((c[0]*w['chain']+c[1]*w['chain_chunks']-y)**2 for w,y in chain))
    gamma=sum(w['ladder']*y for w,y in ladder)/sum(w['ladder']**2 for w,y in ladder)
    rates['giant']=dict(chain=alpha,chain_chunk=beta,ladder=gamma)
    return rates


def giant_work(f):
    # Mirrors run_stage2_batched: ceil-to-P chunks from its 256MiB coordinate budget.
    # Default chain minimum 32768 and block64 are part of this measured configuration.
    w=(f['bits']+63)//64;p=f['P'];i=f['I']
    k=max(p,(256<<20)//(16*w));chunk=p*((k+p-1)//p)
    full,tail=divmod(i,chunk)
    chain_points=full*chunk if chunk>=32768 else 0
    chain_chunks=full if chunk>=32768 else 0
    ladder_points=0 if chunk>=32768 else full*chunk
    if tail:
        if tail>=32768:chain_points+=tail;chain_chunks+=1
        else:ladder_points+=tail
    per_point=6+22*math.log2(f['B2'])/64
    return dict(chain=chain_points*per_point,ladder=ladder_points*per_point,chain_chunks=chain_chunks)


def predict(bits,d,b2,rates):
    f = features(d,b2,bits)
    stages = {phase:rates[phase]*f[feature] for phase,feature in PHASE_FEATURES.items()}
    work=giant_work(f);g=rates['giant']
    stages['giant']=g['chain']*work['chain']+g['chain_chunk']*work['chain_chunks']+g['ladder']*work['ladder']
    stages['init'] = stages['baby']+stages['affine']+stages['ftree']
    stages['main'] = sum(stages[k] for k in ('giant','gtrees','fold','descent','inv','accum','glue'))
    stages['full'] = stages['init']+stages['main']
    return stages,f
