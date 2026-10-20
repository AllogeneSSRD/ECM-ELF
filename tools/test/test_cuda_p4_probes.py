"""P4 module probe contracts; point values, SDK call counts and GMP/fault gates."""
import argparse,hashlib,itertools,json,os,re,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('host-baseline','host','pack','giant','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',type=int,default=1);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exes=[a.host_baseline.resolve(),a.host.resolve(),a.pack.resolve(),a.giant.resolve()]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    identities={str(e):sha(e) for e in exes};result=dict(complete=False,binaries=identities,host_pairs=[],giant_cases=[],pack=[])
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    def run(name,exe,args,code=0):
        r=subprocess.run(list(map(str,[exe,*args])),env=env,capture_output=True,timeout=180)
        text=(r.stdout+r.stderr).decode('utf-8','replace');(out/(name+'.log')).write_text(text)
        assert r.returncode==code,(name,r.returncode,text[-1000:]);return text
    volatile={'wall_seconds','cpu_seconds','cpu_cores','gpu_seconds','dictionary_ms','event_intervals_ms'}
    for flags,blocking,order in itertools.product((0,4),(0,1),(0,1,2)):
        name=f'host_{flags}_{blocking}_{order}';pair=[]
        for side,exe in zip(('old','new'),exes[:2]):
            points=out/(name+'_'+side+'.points')
            text=run(name+'_'+side,exe,[a.device,flags,blocking,129,3,points,order,1279,32,0])
            row=json.loads([s for s in text.splitlines() if s.startswith('{')][-1])
            assert row['invariant_ok'] and row['event_creations']==2
            stable={k:v for k,v in row.items() if k not in volatile|{'api'}}
            stable['api_calls']={k:v['calls'] for k,v in row['api'].items()}
            stable['event_interval_count']=len(row['event_intervals_ms'])
            pair.append(dict(fields=stable,points=sha(points)))
        assert pair[0]==pair[1],(name,pair)
        result['host_pairs'].append(dict(flags=flags,blocking=blocking,order=order,points_equal=True,calls_equal=True))
    for side,exe in zip(('old','new'),exes[:2]):
        text=run('host_invalid_'+side,exe,[a.device,0,0,129,1,out/'unused.points',3],1)
        assert 'invalid wait order' in text
    for width,nw in ((64,5),(128,65),(160,129),(192,161),(256,193)):
        for rotate in (0,1):
            for points,block,threads in ((1,2,32),(2,64,128),(31,17,128),(31,7,128)):
                name=f'giant_{width}_{rotate}_{points}_{block}'
                text=run(name,exes[3],[nw,width,points,block,threads,rotate,a.device])
                fields=dict(re.findall(r'(\w+)=(\S+)',text.split('giant_probe:')[-1]))
                # Each comparison imports all nw limbs. The existing probe
                # checks every point in first/last blocks, and the last point
                # in each middle block. Cases with <=2 blocks cover all points.
                blocks=(points+block-1)//block
                covered=points if blocks<=2 else block+(points-(blocks-1)*block)+(blocks-2)
                assert fields['bad']=='0' and int(fields['checked'])==2*covered,(name,fields)
                result['giant_cases'].append(dict(width=width,nw=nw,rotate=rotate,points=points,
                    block=block,full_points=blocks<=2,gmp_coordinates=2*covered))
    for flag,code in (('--check',0),('--fault',3)):
        text=run('pack_'+flag[2:],exes[2],[a.device,flag],code)
        if not code:
            assert 'bad=0' in text and 'cases=663' in text and '1048576' in text
        else:assert re.search(r'bad=[1-9]|mismatch|fault=1',text)
        result['pack'].append(dict(mode=flag,expected_exit=code))
    assert identities=={str(e):sha(e) for e in exes}
    result['complete']=True
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,host_pairs=len(result['host_pairs']),giant_cases=len(result['giant_cases']),pack_gates=len(result['pack']))))


if __name__=='__main__':main()
