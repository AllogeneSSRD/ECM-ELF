"""Scoped phase cost model and Prime95-style relative curve value.

Profiles are measured seconds, not instruction counts or absolute factor odds.
"""
import math
from functools import lru_cache
from itertools import combinations
from calibrate_stage2_d import features as base_features,unit,inverse

FEATURE_PROFILE=7


@lru_cache(None)
def tree_stats(n,bits):
    work=pairs=groups=copies=coeffs=0;h=1
    while h<n:
        q,r=divmod(n,2*h);count=q+(r>h)
        work+=count*unit(h+1,bits);pairs+=count
        groups+=(q>0)+(r>h)
        if 0<r<=h:copies+=1;coeffs+=r+1
        h*=2
    return dict(work=work,pairs=pairs,groups=groups,copies=copies,copy_coeffs=coeffs)


def features(d,b2,bits=4423,chain_min=32768):
    f=base_features(d,b2,bits);p=f['P'];q,r=divmod(f['I'],p)
    full=tree_stats(p,bits);tail=tree_stats(r,bits)
    f.update(profile=FEATURE_PROFILE,chain_min=chain_min,ftree=full['work'],descent=full['work'],
        gtrees=q*full['work']+tail['work'],g_tree_pairs=q*full['pairs']+tail['pairs'],
        gtrees_groups=q*full['groups']+tail['groups'],
        gtrees_copy_words=(q*full['copy_coeffs']+tail['copy_coeffs'])*((bits+63)//64),
        g_tree_copies=q*full['copies']+tail['copies'],
        inverse=inverse(p+1,bits) if f['G']>1 else 0,
        local_inverse=inverse(p,bits) if f['G']==1 else 0,
        root_reduction=(unit(1,bits)+unit(p+1,bits)) if f['G']==1 and f['I']==p else 0)
    return f


def scope_id(s):
    return ':'.join(map(str,(s['bits'],s['B1'],s['arena_mb'],s['owner_mb'],s['g_min'],s['g_max'],
        s['b2_min'],s['b2_max'],','.join(map(str,sorted(s['d_values']))))))


def admits(s,f):
    return (f['bits']==s['bits'] and f['D'] in s['d_values'] and s['b2_min']<=f['B2']<=s['b2_max'] and
            s['p_min']<=f['P']<=s['p_max'] and s['g_min']<=f['G']<=s['g_max'])


def nnls(x,y):
    """Small exact active-set NNLS, with normalized columns and rank checks."""
    if not x or len(x)!=len(y):raise ValueError('Empty/mismatched fit')
    n=len(x[0]);scales=[max(abs(row[j]) for row in x) for j in range(n)]
    z=[[row[j]/scales[j] if scales[j] else 0 for j in range(n)] for row in x]
    active=[j for j in range(n) if scales[j]];best=[0.0]*n;error=sum(v*v for v in y)
    for count in range(1,len(active)+1):
        for indices in combinations(active,count):
            a=[[sum(row[j]*row[k] for row in z) for k in indices]+[sum(row[j]*v for row,v in zip(z,y))] for j in indices]
            valid=True
            for c in range(count):
                pivot=max(range(c,count),key=lambda r:abs(a[r][c]));a[c],a[pivot]=a[pivot],a[c]
                if abs(a[c][c])<1e-12:valid=False;break
                div=a[c][c];a[c]=[v/div for v in a[c]]
                for r in range(count):
                    if r!=c:
                        mul=a[r][c];a[r]=[u-mul*v for u,v in zip(a[r],a[c])]
            if not valid or any(a[j][-1]<-1e-10 for j in range(count)):continue
            rates=[0.0]*n
            for j,c in enumerate(indices):rates[c]=max(0,a[j][-1])/scales[c]
            residual=sum((sum(u*v for u,v in zip(row,rates))-value)**2 for row,value in zip(x,y))
            if residual<error:best,error=rates,residual
    return best

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
        if denominator <= 0:
            if any(observed(r)[phase]>0.001 for r in rows):raise ValueError('Uncovered active phase: '+phase)
            rates[phase]=0;continue
        rates[phase] = sum(r['features'][feature]*observed(r)[phase] for r in rows)/denominator
    for phase,keys,outputs in [('gtrees',('gtrees','gtrees_groups','gtrees_copy_words'),('gtrees','gtrees_group','gtrees_copy_word')),
            ('descent',('descent','local_inverse','root_reduction'),('descent','descent_local_inverse','descent_root_reduce'))]:
        x=[[r['features'].get(k,0) for k in keys] for r in rows]
        for key,rate in zip(outputs,nnls(x,[observed(r)[phase] for r in rows])):rates[key]=rate
    x=[list(giant_work(r['features']).values()) for r in rows]
    rates['giant']=dict(zip(('chain','ladder','chain_chunk','ladder_launch'),nnls(x,[observed(r)['giant'] for r in rows])))
    rates['giant_coverage']=dict(zip(('chain','ladder','chain_chunk'),(any(row[j]>0 for row in x) for j in range(3))))
    return rates


def giant_work(f):
    # Mirrors run_stage2_batched: ceil-to-P chunks from its 256MiB coordinate budget.
    # Default chain minimum 32768 and block64 are part of this measured configuration.
    w=(f['bits']+63)//64;p=f['P'];i=f['I'];minimum=f.get('chain_min',32768)
    k=max(p,(256<<20)//(16*w));chunk=p*((k+p-1)//p)
    full,tail=divmod(i,chunk)
    chain_points=full*chunk if chunk>=minimum else 0
    chain_chunks=full if chunk>=minimum else 0
    ladder_points=0 if chunk>=minimum else full*chunk
    ladder_launches=0 if chunk>=minimum else full*((chunk+8191)//8192)
    if tail:
        if tail>=minimum:chain_points+=tail;chain_chunks+=1
        else:ladder_points+=tail;ladder_launches+=(tail+8191)//8192
    per_point=6+22*math.log2(f['B2'])/64
    return dict(chain=chain_points*per_point,ladder=ladder_points*per_point,chain_chunks=chain_chunks,ladder_launches=ladder_launches)


def predict(bits,d,b2,rates,chain_min=32768):
    f = features(d,b2,bits,chain_min)
    stages = {phase:rates[phase]*f[feature] for phase,feature in PHASE_FEATURES.items()}
    work=giant_work(f);g=rates['giant']
    stages['giant']=g['chain']*work['chain']+g['chain_chunk']*work['chain_chunks']+g['ladder']*work['ladder']+g.get('ladder_launch',0)*work['ladder_launches']
    stages['gtrees']+=rates.get('gtrees_group',0)*f['gtrees_groups']+rates.get('gtrees_copy_word',0)*f['gtrees_copy_words']
    stages['descent']+=rates.get('descent_local_inverse',0)*f['local_inverse']+rates.get('descent_root_reduce',0)*f['root_reduction']
    stages['init'] = stages['baby']+stages['affine']+stages['ftree']
    stages['main'] = sum(stages[k] for k in ('giant','gtrees','fold','descent','inv','accum','glue'))
    stages['full'] = stages['init']+stages['main']
    return stages,f
