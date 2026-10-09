"""Independent integer verification of native NTT component event plans.

Checks descriptor allocation order, successful arena prefixes, exact physical
allocation/free counters and retained component peak. No CUDA calls. Does not
model owner admission, cold trim by real free memory, or allocation failures.
"""
import copy


def verify_ntt_events(plan):
    m=plan['ntt_memory']
    if m['version']<2:return
    if not m['exact_allocation_events'] or m['counters']['grouped_events']:
        raise ValueError('individual NTT events required')
    if not plan['request_program']['valid']:return
    layouts={x['N']:x for x in m['fuse_layouts']}
    if len(layouts)!=len(m['fuse_layouts']):raise ValueError('duplicate fuse layout')
    retained={x['N']:x for x in plan['request_program']['cache_shapes']}
    for n,x in layouts.items():
        base,table=x['base'],x['table']
        if x['grouped'] or n<=0 or n&(n-1) or len(base)<2:
            raise ValueError('invalid individual fuse descriptor')
        if any(a['bytes']<=0 or a['bytes']%8 for a in base+table):raise ValueError('invalid allocation byte count')
        if [(a['site'],a['index']) for a in base[:2]]!=[(0,0),(1,0)] or base[0]['bytes']!=base[1]['bytes']:
            raise ValueError('tile base allocation order differs')
        tile=base[0]['bytes']//8
        if tile<=0 or tile&(tile-1) or tile>n:raise ValueError('invalid tile words')
        k,t=n.bit_length()-1,tile.bit_length()-1
        widths={a['index']:(a['bytes']//8).bit_length()-1 for a in table if a['site']==5}
        if sorted(widths)!=list(range(len(widths))) or sum(widths.values())!=k-t:
            raise ValueError('outer pass coverage differs')
        expected=[];L=0
        for p,M in widths.items():
            if not 1<=M<=8:raise ValueError('invalid outer radix')
            expected.extend([dict(site=4,index=p,bytes=8*(n>>(L+M))),dict(site=5,index=p,bytes=8*(1<<M))]);L+=M
        for p in reversed(widths):
            M=widths[p];L-=M
            expected.extend([dict(site=6,index=p,bytes=8*(1<<(k-L-M))),dict(site=7,index=p,bytes=8*(1<<M))])
        if table!=expected:raise ValueError('forward/inverse allocation order differs')
        scratch=max((a['bytes']//8 for a in table if a['site'] in (4,6)),default=0)
        radix=max((1<<w for w in widths.values()),default=0)
        compact=base[:2]+([dict(site=2,index=0,bytes=8*scratch)] if scratch else [])+([dict(site=3,index=0,bytes=8*radix)] if radix else [])
        # Legacy cooperative scratch uses configured m_max, which may exceed
        # every actual pass width (including a zero-outer-pass shape). The layout
        # does not export m_max; validate all legal fixed/coop radix capacities.
        radices={64}|({r for r in (32,64,128,256) if r>=radix} if t>=5 else set())
        legacy=[base[:2]+[dict(site=2,index=0,bytes=8*(n//2+64)),dict(site=3,index=0,bytes=8*r)] for r in radices]
        if base!=compact and base not in legacy:raise ValueError('base scratch descriptor differs')
        if n not in retained or sum(a['bytes'] for a in base)!=retained[n]['base_bytes'] or sum(a['bytes'] for a in table)!=retained[n]['table_bytes']:
            raise ValueError('descriptor totals differ from independent cache query')
    s4=plan['s4_memory']
    if not s4['valid']:raise ValueError('independent transform descriptors required')
    shapes={x['operand']:x for x in s4['shapes']}
    counters={k:0 for k in m['counters']}
    live={k:0 for k in m['final_payload']};peak=0
    fuses={};bigs={};smalls={};work=carry=0
    buffers=2 if m['pool'] and m['reuse_bq'] else 3
    cap=m['cap_bytes'];v=plan['tree_workspace']
    class Refusal(Exception):pass
    def emit(part,delta):
        nonlocal peak
        if not delta:return
        live[part]+=delta;live['total_bytes']+=delta
        if live[part]<0 or live['total_bytes']<0:raise ValueError('independent event underflow')
        counters['allocations' if delta>0 else 'frees']+=1
        peak=max(peak,live['total_bytes'])
    def fit(extra):return not cap or live['total_bytes']+extra<=cap
    def snapshot():return copy.deepcopy((fuses,bigs,smalls,work,carry))
    def call(n,slots,slices):
        nonlocal work,carry
        counters['calls']+=1
        if n not in fuses:
            x=layouts[n]
            if not fit(sum(a['bytes'] for a in x['base']+x['table'])):raise Refusal('fuse_cap_refusal')
            for part,entries in [('base_bytes',x['base']),('table_bytes',x['table'])]:
                for a in entries:emit(part,a['bytes'])
            fuses[n]=True;counters['fuse_builds']+=1
        else:counters['fuse_hits']+=1
        key=(n,slices);need=8*buffers*n*slices if m['pool'] else 24*n*slices
        grow=work<n*slices if m['pool'] else key not in bigs
        if grow:
            if m['pool']:
                for _ in range(buffers):emit('big_bytes',-8*work)
                work=0
            if not fit(need):
                emit('digit_bytes',-carry);carry=0;freed=0
                for size in fuses:
                    if size==n or not fuses[size]:continue
                    for a in layouts[size]['table']:emit('table_bytes',-a['bytes']);freed+=a['bytes']
                    fuses[size]=False
                for entry in reversed(list(bigs)):
                    if entry==key:continue
                    size=bigs.pop(entry)
                    for _ in range(3):emit('big_bytes',-size//3)
                    freed+=size
                for entry in reversed(list(smalls)):
                    if entry==key:continue
                    out=smalls.pop(entry);batch=entry[1]
                    emit('digit_bytes',-8*out*batch);emit('digit_bytes',-16*batch);freed+=8*(out+2)*batch
                if freed:counters['cap_evictions']+=1;counters['cap_evicted_bytes']+=freed
            if not fit(need):raise Refusal('big_cap_refusal')
            for _ in range(buffers if m['pool'] else 3):emit('big_bytes',need//(buffers if m['pool'] else 3))
            if m['pool']:work=n*slices;counters['workspace_grows']+=1
            else:bigs[key]=need
        if key not in smalls:
            if not fit(8*(slots+2)*slices):raise Refusal('digits_cap_refusal')
            emit('digit_bytes',8*slots*slices);emit('digit_bytes',16*slices);smalls[key]=slots
        elif slots>smalls[key]:
            if not fit(8*(slots-smalls[key])*slices):raise Refusal('digits_cap_refusal')
            emit('digit_bytes',-8*smalls[key]*slices);emit('digit_bytes',8*slots*slices);smalls[key]=slots
        if m['carry_check']:
            for batch,times in [(65535,slices//65535),(slices%65535,int(bool(slices%65535)))]:
                if not times or n*batch<1<<20:continue
                scratch=8*((n+255)//256)*batch
                if scratch>carry:
                    if not fit(scratch-carry):counters['carry_refusals']+=times
                    else:emit('digit_bytes',-carry);emit('digit_bytes',scratch);carry=scratch;counters['carry_grows']+=1
    stop=None;checkpoints=[]
    try:
        for bi,b in enumerate(plan['request_program']['blocks']):
            rep=0
            while rep<b['repeat']:
                before=snapshot();counts=counters.copy()
                for ri,r in enumerate(b['requests']):
                    d=shapes[max(r['ma'],r['mb'])];n,slots=d['N'],d['slots'];c=r['pairs']
                    while c and 8*c*((buffers if v['physical_chunks'] else 3)*n+slots+(2 if v['physical_chunks'] else 0))>v['batch_bytes']:c//=2
                    c=max(1,c)
                    if v['chunk_max']:c=min(c,v['chunk_max'])
                    for slices,times in [(c,r['pairs']//c),(r['pairs']%c,int(bool(r['pairs']%c)))]:
                        while times:
                            call_before=snapshot();call_counts=counters.copy()
                            call(n,slots,slices);times-=1
                            if snapshot()==call_before:
                                for key in counters:counters[key]+=(counters[key]-call_counts[key])*times
                                break
                rep+=1
                if snapshot()==before:
                    for key in counters:counters[key]+=(counters[key]-counts[key])*(b['repeat']-rep)
                    break
            checkpoints.append(dict(repeat=b['repeat'],peak_bytes=peak,payload=live.copy(),counters=counters.copy()))
    except Refusal as exc:
        stop=dict(block=bi,repeat_index=rep,request_index=ri,phase=['ftree','gtrees','fold','descent','inverse'][r['phase']],
                  ma=r['ma'],mb=r['mb'],pairs=r['pairs'],N=n,slots=slots,slices=slices)
        if m['reason']!=str(exc):raise ValueError('independent refusal reason differs')
    if stop!=m['stopped_at'] or bool(stop)==m['finished'] or live!=m['final_payload'] or counters!=m['counters'] or peak!=m['peak_bytes'] or checkpoints!=m['checkpoints']:
        raise ValueError('independent NTT allocation event program differs')
