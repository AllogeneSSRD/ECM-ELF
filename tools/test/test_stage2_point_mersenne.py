"""Serial exact Mersenne Montgomery gates or paired primitive timings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--bench',action='store_true')
    a=p.parse_args();exe=a.exe.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    manifest=json.loads((exe.parent/'manifest.json').read_text(encoding='utf-8-sig'))
    root=Path(__file__).resolve().parents[2];sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    def verify():
        assert sha==manifest['sha256'].lower()==hashlib.sha256(exe.read_bytes()).hexdigest()
        for name,h in manifest['sources'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h.lower(),name
    runs=[]
    def run(bits,mode,alias=0,count=128,repeats=3,fault=0,threads=128):
        verify();cmd=[str(exe),*map(str,(bits,count,repeats,mode,alias,threads,a.device,fault))]
        result=subprocess.run(cmd,capture_output=True,timeout=120)
        text=(result.stdout+result.stderr).decode('utf-8','replace');(out/f'{len(runs)+1}_{bits}_{mode}_{alias}.log').write_text(text)
        match=re.search(r'point_mersenne: (.*)',text);assert match,text[-2000:]
        data=dict(re.findall(r'(\w+)=([^ ]+)',match[1]));bad=int(data['bad'])
        assert result.returncode==(1 if fault else 0) and ((bad>0) if fault else bad==0),text
        assert all(int(data[k])==v for k,v in [('bits',bits),('mode',mode),('alias',alias),('count',count),('repeats',repeats),('device',a.device),('threads',threads)])
        runs.append(dict(command=cmd,exit=result.returncode,data=data,fault=fault));verify()
        print(bits,mode,alias,'bad',bad,'ms',data['event_ms'],flush=True)
    if a.bench:
        for mode in (0,1,1,0,1,0,0,1):run(4423,mode,alias=1,count=8192,repeats=16)
    else:
        for bits in (2,3,31,32,63,64,65,127,128,129,257,513,1025,2049,4097,4423,5261,8192):
            for mode in (0,1):
                for alias in (0,1,2):run(bits,mode,alias)
        run(4423,1,fault=1)
    summary=dict(exe=str(exe),sha256=sha,manifest=manifest,device=a.device,runs=runs,passed=len(runs),failed=0,
                 scope='Exact raw canonical products vs GMP, repeated Montgomery recurrence; independent primitive event timing, not Stage2 speedup.')
    if a.bench:
        means={str(mode):statistics.mean(float(r['data']['event_ms']) for r in runs if int(r['data']['mode'])==mode) for mode in (0,1)}
        summary.update(means=means,gain_percent=100*(1-means['1']/means['0']))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('manifest','runs')}))


if __name__=='__main__':main()
