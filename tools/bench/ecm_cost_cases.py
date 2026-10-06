"""Integer-only calibration coverage for G1, G2 and the gap to large B2.

This is a case schedule, not a performance model. A point count n occupies
B2 in [D*(n-2), D*(n-1)-1]. Adjacent regimes must include both endpoints.
"""
import math
from calibrate_stage2_d import phi


def point_b2(d,n,end=False):
    if d<6 or d%2 or n<2:raise ValueError('Invalid point geometry')
    return d*(n-2)+(d-1 if end else 0)


def bridge_range(d,large_min):
    return point_b2(d,phi(d)+1),large_min-1


def low_cases(d,large_min,chain_min,g1=False,g2=False,bridge=False):
    """Return (regime, B2, train|holdout), with no duplicate observations."""
    p=phi(d)//2;out=[]
    if p<12:raise ValueError('Need P>=12')
    def add(regime,b2,kind):
        row=(regime,b2,kind)
        if row not in out:out.append(row)
    if g1:
        counts={p//4,p//2,3*p//4,p}
        if g2:counts|={n for n in (chain_min-2,chain_min,chain_min+2) if 2<=n<=p}
        for n in sorted(counts):add('g1',point_b2(d,n),'train')
        if g2:add('g1',point_b2(d,p,True),'train')
        add('g1',point_b2(d,3*p//8),'holdout')
    if g2:
        counts={p+1,p+p//4,p+p//2,2*p}
        counts|={n for n in (chain_min-2,chain_min,chain_min+2) if p<n<=2*p}
        for n in sorted(counts):add('g2',point_b2(d,n),'train')
        add('g2',point_b2(d,2*p,True),'train')
        add('g2',point_b2(d,p+3*p//8),'holdout')
    if bridge:
        lo,hi=bridge_range(d,large_min)
        if lo<=hi:
            mid=math.isqrt(lo*hi);anchors={lo,mid,hi}
            for n in (chain_min-2,chain_min,chain_min+2):
                if n>=2 and lo<=point_b2(d,n)<=hi:anchors.add(point_b2(d,n))
            for b in sorted(anchors):add('bridge',b,'train')
            held=math.isqrt(lo*mid)
            if held in anchors:raise ValueError('Bridge too narrow for an independent holdout')
            add('bridge',held,'holdout')
    return out


def regime_intervals(d,large_min):
    p=phi(d)//2
    return dict(g1=(point_b2(d,p//4),point_b2(d,p,True)),
        g2=(point_b2(d,p+1),point_b2(d,2*p,True)),bridge=bridge_range(d,large_min))


def validation_points(scope,d,chain_min,large_b2):
    """Blind interior/boundary points; callers label training-shape replays."""
    p=phi(d)//2;targets={};regime=scope['regime'];lo=scope['b2_min'];hi=scope['b2_max']
    def add(b,kind):
        if lo<=b<=hi:targets.setdefault(b,kind)
    if regime=='g1':
        for n,kind in ((5*p//8,'blind'),(p-1,'boundary_blind'),(p,'root_replay')):add(point_b2(d,n),kind)
    elif regime=='g2':
        for n in (p+2,p+p//8,p+5*p//8,2*p-1):add(point_b2(d,n),'g2_blind')
    elif regime=='bridge':
        mid=math.isqrt(lo*hi);add(math.isqrt(mid*hi),'bridge_blind')
        group=(mid//d+2)//p
        for n in (group*p-1,group*p,group*p+1):
            if n>=2:add(point_b2(d,n),'group_boundary_blind')
    elif regime=='multiple':
        if not lo<=large_b2<=hi:raise ValueError('Large blind point outside scope')
        add(large_b2,'blind')
    else:raise ValueError('Unknown calibration regime')
    for n in (chain_min-1,chain_min,chain_min+1):
        if n>=2:add(point_b2(d,n),'crossover_blind')
    if regime!='g1':
        w=(scope['bits']+63)//64;capacity=max(p,(256<<20)//(16*w));chunk=p*((capacity+p-1)//p)
        for n in (chunk-1,chunk,chunk+1,chunk+chain_min-1,chunk+chain_min,chunk+chain_min+1):
            if n>=2:add(point_b2(d,n),'chunk_boundary_blind')
    if not targets:raise ValueError('Scope has no independent validation points')
    return sorted(targets.items())
