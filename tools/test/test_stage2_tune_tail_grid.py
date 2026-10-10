"""Check adaptive B2 bounds, actual route provenance and default effort capacity."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import tomllib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from stage2_tune_route_cost import annotate


def phi(n):
    result=n
    q=2
    while q*q<=n:
        if n%q==0:
            result-=result//q
            while n%q==0:n//=q
        q+=1
    return result-result//n if n>1 else result


def run_grids(fixture,cases):
    text=''.join(' '.join(str(int(x)) for x in [*c[:-1],len(c[-1]),*c[-1]])+'\n' for c in cases)
    proc=subprocess.run([str(fixture),'--tail-grid'],input=text,text=True,capture_output=True,timeout=60)
    assert proc.returncode==0,proc.stderr
    result=[json.loads(line) for line in proc.stdout.splitlines()]
    assert len(result)==len(cases)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--valid-profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    fixture=a.fixture.resolve();cases=[];rng=random.Random(7301)
    # b1,D,P,C,min,force,maxG,n,bounds. Moderate intervals can be enumerated
    # independently; the reference does not reproduce the sampling algorithm.
    for _ in range(1400):
        d=rng.choice([2,6,30,60]);p0=rng.randint(1,12);chunk=p0*rng.randint(1,14)
        low=rng.randint(3,1200);high=low+rng.randint(0,1600)
        bounds=[low,high,(low+high)//2]
        cases.append((rng.randint(2,100),d,p0,chunk,rng.randint(0,chunk+6),rng.randrange(8)==0,
            rng.choice([0,2,8,64]),rng.choice([0,1,3,16]),bounds))
    # Exact full-chunk endpoint, threshold endpoints, duplicate I, singleton,
    # no G2, all-ladder and the original 5872/6011-bit short-tail ranges.
    cases += [(2,6,2,20,8,0,0,3,[108,114,180]),
              (2,6,2,20,8,0,0,16,[108,114,120]),
              (2,6,2,20,8,0,0,3,[108,109,110,180]),
              (2,6,2,20,8,0,0,3,[114,114]),
              (2,6,20,40,8,0,1,3,[12,180]),
              (2,6,2,20,21,0,0,3,[108,180]),
              (2,6,2,20,8,1,0,3,[108,180])]
    small=len(cases);results=run_grids(fixture,cases)
    (out/'cases.json').write_text(json.dumps(dict(cases=cases,results=results),indent=2)+'\n',encoding='utf-8')
    for c,result in zip(cases,results):
        b1,d,p0,chunk,minimum,force,batches,n,bounds=c
        assert result['valid'],(c,result)
        low=max(min(bounds),b1+1,(p0-1)*d)
        high=min(max(bounds),(p0*batches-1)*d-1) if batches else max(bounds)
        if not n or low>high or low==high or force or chunk<minimum:
            assert not result['points'],(c,result)
            continue
        assert (result['low'],result['high'])==(low,high)
        lo,hi=low//d+2,high//d+2
        # Direct enumerate each possible integer I and classify its residual.
        original={b//d+2 for b in bounds}
        chain={i for i in range(lo,hi+1) if i%chunk==0 or i%chunk>=minimum}
        tails=set(range(lo,hi+1))-chain
        available_tails=tails-original
        used=set(original);actual_chain=actual_tail=0
        assert result['chain_anchors']==len(chain&original),(c,result)
        for point in result['points']:
            b2=point['b2'];i=b2//d+2
            assert low<=b2<=high and i not in used and i>p0
            assert not batches or (i+p0-1)//p0<=batches
            used.add(i)
            if point['source']=='ladder_tail':
                actual_tail+=1;assert i in tails
            else:
                actual_chain+=1;assert point['source']=='chain_anchor' and i in chain
        ladder=len(tails&original)+n
        target=max(3,7-ladder) if ladder>=3 else 3
        assert actual_chain==min(max(0,target-result['chain_anchors']),len(chain-original)),(c,result)
        assert actual_tail==min(n,len(available_tails)),(c,result)
    # All ten native effort catalogues, using the current default 256 MiB X/Z
    # chunk formula as an independent input policy. Runtime obtains C from plan.
    levels=[];default_cases=[]
    for level in range(1,11):
        get=lambda kind:list(map(int,subprocess.check_output([str(fixture),kind,str(level)],text=True).split()))
        exponents,ds,bounds=get('--effort-exponents'),get('--effort-d'),get('--effort-b2')
        tail=get('--effort-tail')[0];assert tail==(3 if level>=3 else 0)
        groups=[]
        for bits in exponents:
            nw=(bits+63)//64
            for d in ds:
                p0=phi(d)//2;k=(256<<20)//(16*nw);chunk=p0*max(1,(k+p0-1)//p0)
                groups.append((20,d,p0,chunk,32768,0,64+16*(level-1),tail,bounds))
        grids=run_grids(fixture,groups)
        assert all(g['valid'] for g in grids)
        base=len(exponents)*len(ds)*len(bounds)
        extra=sum(len(g['points']) for g in grids)
        assert base+extra<=4096,(level,base,extra)
        levels.append(dict(level=level,base=base,extra=extra,total=base+extra,
            ladder_tail=sum(p['source']=='ladder_tail' for g in grids for p in g['points']),
            chain_anchor=sum(p['source']=='chain_anchor' for g in grids for p in g['points'])))
        default_cases.extend(groups)
    large_cases=[(20,60060,5760,c,32768,0,0,3,[10400000000,14707821049,20800000000,29415642097,41600000000]) for c in [184320,178560]]
    large=run_grids(fixture,large_cases)
    for c,g in zip(large_cases,large):
        assert g['valid'] and sum(p['source']=='ladder_tail' for p in g['points'])==3
        for p in g['points']:
            s=annotate(dict(b2=p['b2'],d=c[1],p=c[2],giant_points=p['b2']//c[1]+2),c[3],c[4],False)
            assert bool(s['giant_ladder_steps'])==(p['source']=='ladder_tail')
    clipped_case=(20,120120,11520,184320,32768,0,96,3,[2600000000,8221921916,26000000000])
    clipped=run_grids(fixture,[clipped_case])[0]
    tails=[(p['b2']//clipped_case[1]+2)%clipped_case[3] for p in clipped['points'] if p['source']=='ladder_tail']
    (out/'clipped_tail.json').write_text(json.dumps(dict(case=clipped_case,result=clipped,tails=tails),indent=2)+'\n',encoding='utf-8')
    assert tails==[2047,16383,30719],('clipping must not collapse feasible tail fractions',tails)
    invalid=[(20,6,2,0,8,0,0,3,[100,200]),(20,6,2,21,8,0,0,3,[100,200]),
             (20,6,2,20,8,0,0,17,[100,200])]
    assert all(not r['valid'] for r in run_grids(fixture,invalid))
    maximum=(1<<64)-1
    edges=[(maximum,6,2,20,8,0,0,3,[100,200]),(20,1,1,1,0,0,0,3,[maximum-1,maximum]),
           (20,6,2,20,8,0,maximum,3,[maximum-1000,maximum-10])]
    edge_results=run_grids(fixture,edges)
    assert not edge_results[0]['points'] and not edge_results[1]['points']
    assert edge_results[2]['valid'] and edge_results[2]['points']
    for point in edge_results[2]['points']:
        b2=point['b2'];assert maximum-1000<=b2<=maximum-10
        work=annotate(dict(b2=b2,d=6,p=2,giant_points=b2//6+2),20,8,False)
        assert work['giant_ladder_steps']<=maximum and (b2//6+2)*6<=maximum
    # New metadata is optional for legacy profiles, but never partly accepted.
    original=a.valid_profile.read_text(encoding='utf-8')
    metadata='sampling_model = "giant_tail_grid_v1"\ntail_samples = 3\n'
    valid=original.replace('[device]',metadata+'[device]').replace('[ecm.sample_0]','[ecm.sample_0]\nsampling_source = "base"')
    samples=[valid,
        valid.replace('tail_samples = 3','tail_samples = 17'),
        valid.replace('giant_tail_grid_v1','unknown_grid'),
        valid.replace(metadata,''),
        valid.replace('tail_samples = 3\n',''),
        valid.replace('sampling_source = "base"','sampling_source = "other"'),
        valid.replace('sampling_source = "base"','sampling_source = "ladder_tail"')]
    for i,text in enumerate(samples):
        f=out/f'metadata_{i}.toml';f.write_text(text,encoding='utf-8')
        proc=subprocess.run([str(fixture),'--load',str(f)],capture_output=True)
        assert (proc.returncode==0)==(i==0),(i,proc.stderr)
    tagged=tomllib.loads(valid)
    def write(name,data):
        sections=[('profile',data['profile']),('device',data['device']),
            ('policy',{k:v for k,v in data['policy'].items() if k!='environment'}),
            ('policy.environment',data['policy']['environment'])]
        sections += [('ecm.'+k,s) for k,s in data['ecm'].items()]
        sections += [('summary',data['summary'])]
        path=out/(name+'.toml')
        path.write_text(''.join('\n['+name+']\n'+''.join(k+' = '+json.dumps(v)+'\n' for k,v in f.items())
            for name,f in sections),encoding='utf-8')
        return path
    route_checks=0
    for source,chunk,minimum,accepted in [('ladder_tail',138240,32768,True),('chain_anchor',17280,1000,True),
        ('chain_anchor',138240,32768,False),('ladder_tail',17280,1000,False)]:
        data=json.loads(json.dumps(tagged));s=data['ecm']['sample_0']
        data['ecm']['sample_0']=annotate(s,chunk,minimum,False)
        data['ecm']['sample_0']['sampling_source']=source
        path=write('route_'+str(route_checks),data)
        proc=subprocess.run([str(fixture),'--load',str(path)],capture_output=True)
        assert (proc.returncode==0)==accepted,(source,proc.stderr)
        route_checks+=1
    legacy=out/'legacy.toml';legacy.write_text(original,encoding='utf-8')
    tagged['ecm']['sample_0']['b2']+=1
    tagged['profile']['tail_samples']=16
    newer=write('newer',tagged)
    merged=subprocess.check_output([str(fixture),'--merge',str(legacy),str(newer)],text=True)
    data=tomllib.loads(merged);assert data['profile']['tail_samples']==16 and len(data['ecm'])==2
    merged_path=out/'merged.toml';merged_path.write_text(merged,encoding='utf-8')
    assert subprocess.run([str(fixture),'--load',str(merged_path)],capture_output=True).returncode==0
    report=dict(complete=True,enumerated_cases=small,large_cases=large,default_levels=levels,
        invalid_policies=len(invalid),boundary_cases=edge_results,metadata_accept=1,metadata_refusals=len(samples)-1,
        route_provenance_checks=route_checks,mixed_sampling_merge=True,clipped_tail_fractions=tails)
    (out/'cases.json').write_text(json.dumps(dict(cases=cases,results=results),indent=2)+'\n',encoding='utf-8')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
