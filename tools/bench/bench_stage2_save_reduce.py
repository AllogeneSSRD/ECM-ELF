"""Serial same-binary reducer or baby normalization A/B on an immutable Stage1 save and fixed D.

Uses production defaults, GPU1 unless explicitly selected, and mandatory checks.
Writes only fresh experiment logs/results; never edits the save, ini or queue.
"""
import argparse,hashlib,json,os,re,statistics,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--save',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    p.add_argument('--b2',type=int,default=2011326186870);p.add_argument('--d',type=int,default=1381380)
    p.add_argument('--runs',type=int,choices=(4,8),default=4)
    p.add_argument('--toggle',choices=('short','baby'),default='short')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    if a.d<=0:raise ValueError('The arithmetic comparison requires a fixed positive D')
    exe=a.exe.resolve();save=a.save.resolve();repo=Path(__file__).resolve().parents[2]
    manifest=json.loads((exe.parent/'build_manifest.json').read_text(encoding='utf-8-sig'))
    sources={m[1]:m[2].lower() for value in manifest['sources']
             if (m:=re.fullmatch(r'([^=]+\.(?:cu|cuh|cpp|h|ps1))=([A-Fa-f0-9]{64})',value))}
    sha=hashlib.sha256(exe.read_bytes()).hexdigest();save_sha=hashlib.sha256(save.read_bytes()).hexdigest()
    assert sources and sha==manifest['sha256'].lower()
    def verify():
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha
        assert hashlib.sha256(save.read_bytes()).hexdigest()==save_sha
        for name,want in sources.items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==want,name
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_ARENA_CAP_KB='6451200')
    toggle='NTT_BABY_DEVICE' if a.toggle=='baby' else 'NTT_GL_SHORT_REDUCE'
    if a.toggle=='baby':env['NTT_GL_SHORT_REDUCE']='1'
    order=[0,1,1,0] if a.runs==4 else [0,1,1,0,1,0,0,1]
    data=dict(exe=str(exe),sha256=sha,save=str(save),save_sha256=save_sha,sources=sources,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),env=env|{toggle:'per run'},toggle=a.toggle,
              device=a.device,D=a.d,B2=a.b2,order=order,runs=[])
    # Store only explicit experiment controls; inherited unrelated environment may be private.
    data['env']={k:v for k,v in data['env'].items() if k.startswith('NTT_')}
    reference=None
    for number,mode in enumerate(order,1):
        verify();name=f'{number}_{mode}';engine=out/(name+'_engine.log');result=out/(name+'.jsonl')
        cmd=[str(exe),'--save',str(save),'--b2',str(a.b2),'--d',str(a.d),'--device',str(a.device),
             '--results',str(result),'--log',str(engine)]
        print('RUN',name,flush=True);started=time.monotonic()
        with (out/(name+'_driver.log')).open('wb') as output:
            r=subprocess.run(cmd,env=env|{toggle:str(mode)},stdout=output,stderr=subprocess.STDOUT,timeout=300)
        verify();text=engine.read_text(encoding='utf-8',errors='replace') if engine.exists() else ''
        assert r.returncode==0,(name,r.returncode,text[-1500:])
        for token in ('stage1_skipped=1',f'ntt_gl_reduce_mode: device={a.device} short={1 if a.toggle=="baby" else mode}',
                      'point_arithmetic: xadd6=1','gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1'):
            assert token in text,(name,token)
        assert '[trace]' not in text and 'prime_powers=' not in text
        if a.toggle=='baby':
            assert f'baby_device: requested={mode} enabled={mode}' in text,(name,'baby fallback')
            assert 'small_prime_reuse: requested=1 available=1 matched=1' in text
        wall=re.search(r'stage2_full_wall:.*?init=([\d.]+) main=([\d.]+) total=([\d.]+)',text)
        oracle=re.search(r's4_oracle_stats:.*selected=(\d+) queued=(\d+) compared=(\d+) samples=(\d+) pending=(\d+).*signature=(\S+)',text)
        leaf=re.search(r'descent_values: (.*)',text)
        assert wall and oracle and oracle[1]==oracle[2]==oracle[3] and oracle[5]=='0' and leaf
        row=json.loads(result.read_text().splitlines()[-1]);assert row['bad_factors']==0
        signature=(leaf[1],oracle[1],oracle[4],oracle[6],row['factors'])
        if reference is None:reference=signature
        else:assert signature==reference,(name,signature,reference)
        item=dict(name=name,short=mode,command=cmd,driver_seconds=time.monotonic()-started,
                  mode=mode,init=float(wall[1]),main=float(wall[2]),full=float(wall[3]),leaf=leaf[1],oracle_signature=oracle[6])
        if a.toggle=='baby':
            item.pop('short');item['baby_stats']=re.search(r'baby_device: (.*)',text)[1]
            times=re.search(r'real_baby:.*ladder=([\d.]+) s affine=([\d.]+) s',text)
            assert times;item.update(baby_ladder=float(times[1]),baby_affine=float(times[2]))
        data['runs'].append(item)
        (out/'measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(name,'full=',item['full'],flush=True)
    means={str(mode):statistics.mean(r['full'] for r in data['runs'] if r['mode']==mode) for mode in (0,1)}
    data.update(means=means,gain_percent=100*(1-means['1']/means['0']))
    (out/'measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8');print(json.dumps(means))
    return 0


if __name__=='__main__':raise SystemExit(main())
