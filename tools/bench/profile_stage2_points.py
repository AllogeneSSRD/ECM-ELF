"""Freeze and profile a native Stage2 curve; summarize own GPU events and API gaps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)  # Bounded 16384-bit native factor records.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--b2', type=int, required=True)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--point-mersenne',type=int,choices=(0,1),default=0)
    p.add_argument('--carry-check-fused',type=int,choices=(0,1),default=0)
    p.add_argument('--chain-min',type=int,default=None)
    p.add_argument('--seed-pair',type=int,choices=(0,1),default=0)
    p.add_argument('--base-cpu',type=int,choices=(0,1),default=0)
    p.add_argument('--gscale-device',type=int,choices=(0,1),default=None,
                   help='Development GPU Gamma correction; omit for older binaries')
    p.add_argument('--cuda-launch-blocking',type=int,choices=(0,1),default=None,
                   help='Explicitly bind the capture launch mode instead of inheriting it')
    p.add_argument('--chain-block',type=int,default=None,help='Explicit chain points per thread; 4..2^20')
    p.add_argument('--short-chain-block',type=int,default=None,help='Opt-in short-chunk points per thread: 0 or 4..64')
    p.add_argument('--short-chain-max',type=int,default=None,help='Exclusive short-chunk point limit, at most 2^20')
    p.add_argument('--owner-mb',type=int,default=None)
    p.add_argument('--owner-reuse',type=int,choices=(0,1,2,3),default=None,
                   help='Development owner layout; omit for historical binaries')
    p.add_argument('--arena-mb',type=int,default=None)
    p.add_argument('--factor-only',action='store_true')
    p.add_argument('--log-level', choices=('quiet','curve','phases','batches','debug'),
                   help='New production logger; profiling requires debug. Omit for historical binaries.')
    p.add_argument('--cuda-event-trace',action='store_true',help='Collect CUDA event completion/correlation; diagnostic overhead may change scheduling')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--analyze-existing', action='store_true', help='Validate and analyze the existing command/log/trace without relaunching')
    p.add_argument('--nsys', type=Path, default=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe'))
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not a.analyze_existing: raise ValueError('Use an empty output directory')
    exe = a.exe.resolve(); save = a.save.resolve()
    build = json.loads((exe.parent / 'build_manifest.json').read_text(encoding='utf-8-sig'))
    raw_sources = build['sources']
    sources = raw_sources if isinstance(raw_sources, dict) else {s.split('=',1)[0]:s.split('=',1)[1].lower() for s in raw_sources if '=' in s and (root/s.split('=',1)[0]).is_file()}
    exe_sha = hashlib.sha256(exe.read_bytes()).hexdigest()
    save_sha = hashlib.sha256(save.read_bytes()).hexdigest()
    frozen_manifest=exe.parent/'frozen_sources_manifest.json'
    source_root=root
    if frozen_manifest.exists():
        frozen=json.loads(frozen_manifest.read_text(encoding='utf-8'))
        assert frozen['binary_sha256']==exe_sha, 'Frozen build binary differs'
        assert {n:d.lower() for n,d in frozen['sources'].items()}=={n:d.lower() for n,d in sources.items()}, 'Frozen dependencies differ'
        source_root=exe.parent/'sources'
    manifest_sha=hashlib.sha256(frozen_manifest.read_bytes()).hexdigest() if frozen_manifest.exists() else None
    def verify():
        assert hashlib.sha256(exe.read_bytes()).hexdigest() == exe_sha == build['sha256'].lower()
        assert hashlib.sha256(save.read_bytes()).hexdigest() == save_sha
        if manifest_sha:assert hashlib.sha256(frozen_manifest.read_bytes()).hexdigest()==manifest_sha, 'Frozen manifest changed'
        for name, digest in sources.items(): assert hashlib.sha256((source_root/name).read_bytes()).hexdigest() == digest.lower(), name
    verify()
    env = {k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1', NTT_ARENA_CAP_KB='6451200', NTT_D_MODEL='0', NTT_STAGE1_Q_DUMP='1',NTT_POINT_MERSENNE=str(a.point_mersenne))
    env['NTT_CARRY_CHECK_FUSED']=str(a.carry_check_fused)
    if a.chain_min is not None:
        if not 0<=a.chain_min<=100000000:raise ValueError('Invalid chain minimum')
        env['NTT_GIANT_CHAIN_MIN']=str(a.chain_min)
    env['NTT_GIANT_SEED_PAIR']=str(a.seed_pair)
    env['NTT_GIANT_BASE_CPU']=str(a.base_cpu)
    if a.gscale_device is not None:env['NTT_GSCALE_DEVICE']=str(a.gscale_device)
    if a.cuda_launch_blocking is not None:env['CUDA_LAUNCH_BLOCKING']=str(a.cuda_launch_blocking)
    if a.chain_block is not None:
        if not 4<=a.chain_block<=1<<20:raise ValueError('Invalid chain block')
        env['NTT_GIANT_CHAIN_BLOCK']=str(a.chain_block)
    if a.short_chain_block is not None:
        if a.short_chain_block!=0 and not 4<=a.short_chain_block<=64:raise ValueError('Invalid short chain block')
        env['NTT_GIANT_CHAIN_SMALL_BLOCK']=str(a.short_chain_block)
    if a.short_chain_max is not None:
        if not 0<=a.short_chain_max<=1<<20:raise ValueError('Invalid short chain maximum')
        env['NTT_GIANT_CHAIN_SMALL_MAX']=str(a.short_chain_max)
    if a.owner_mb is not None:
        if a.owner_mb<0:raise ValueError('Invalid owner budget')
        env['NTT_FOLD_DEVICE_MAX_MB']=str(a.owner_mb)
    if a.owner_reuse is not None:
        env['NTT_FOLD_OWNER_REUSE']=str(a.owner_reuse)
    command = [str(exe), '--save', str(save), '--b2', str(a.b2), '--d', str(a.d), '--device', str(a.device), '--results', str(out/'results.jsonl'), '--log', str(out/'engine.log')]
    if a.arena_mb is not None:
        if a.arena_mb<=0:raise ValueError('Invalid arena budget')
        command+=['--arena-mb',str(a.arena_mb)]
    if a.factor_only:command+=['--factor-only']
    if a.log_level is not None:
        if a.log_level!='debug':raise ValueError('Profiling needs debug arithmetic and identity logs')
        command+=['--log-level',a.log_level]
    assert not any(any(c in token for c in '&|<>%!^\r\n"') for token in command)
    wrapper = out/'run.cmd'
    wrapper_bytes=('@echo off\r\n'+' '.join('"'+x+'"' for x in command)+' > "'+str(out/'app.log')+'" 2>&1\r\nexit /b %errorlevel%\r\n').encode()
    if a.analyze_existing: assert wrapper.read_bytes()==wrapper_bytes, 'Recorded application command differs'
    else: wrapper.write_bytes(wrapper_bytes)
    trace = out/'trace'
    profile = [str(a.nsys), 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none', '--cuda-memory-usage=true', '--force-overwrite=true', '-o', str(trace), 'C:/Windows/System32/cmd.exe', '/d', '/c', str(wrapper)]
    if a.cuda_event_trace:profile.insert(2,'--cuda-event-trace=true')
    if not a.analyze_existing:
        with (out/'profile.log').open('wb') as log:
            subprocess.run(profile, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    else: assert trace.with_suffix('.nsys-rep').is_file()
    verify()
    text = (out/'engine.log').read_text(encoding='utf-8',errors='replace')
    for token in ('stage1_skipped=1', 'gmp_check_bad=0', 'gmp_selftest_bad=0', 'pending=0', 'clean=1', 'fixed=3', 'point_arithmetic: xadd6=1'):
        assert token in text, token
    result = json.loads((out/'results.jsonl').read_text(encoding='utf-8').splitlines()[-1])
    if a.gscale_device is not None:
        scale=dict(re.findall(r'(\w+)=(\S+)',re.search(r'real_gscale_device: (.*)',text)[1]))
        assert int(scale['requested'])==a.gscale_device, 'Gamma request differs'
        assert scale['checked_words']=='0' and scale['check_d2h_bytes']=='0', 'Diagnostic GMP copies in capture'
    if a.owner_reuse is not None:
        owner=dict(re.findall(r'(\w+)=(\S+)',re.search(r'real_batched_folddevice: (.*)',text)[1]))
        assert int(owner['reuse'])==a.owner_reuse, 'Actual owner layout differs'
    n=int(result['N_hex'],16)
    if a.point_mersenne:
        enabled=int(n>1 and (n & (n+1))==0)
        assert 'point_mersenne_mode: requested=1 enabled='+str(enabled) in text
    if a.carry_check_fused or any(n.endswith('/ntt_carry_partial.cuh') for n in sources):
        stats=dict(re.findall(r'(\w+)=(\d+)',re.search(r'ntt_carry_check_stats: (.*)',text)[1]))
        assert int(stats['requested'])==a.carry_check_fused
        assert int(stats['fused_calls'])>0 if a.carry_check_fused else int(stats['fused_calls'])==0
    q = re.search(r'real_setup_Q_full: hex=([0-9a-f]+)', text)[1]
    saved_x = re.search(rb'\bX=(?:0x)?([0-9a-fA-F]+)', save.read_bytes().splitlines()[0])[1].decode().lower().lstrip('0') or '0'
    assert q == saved_x
    assert result['bad_factors'] == 0
    if a.factor_only:
        assert all(1<int(f)<n and n%int(f)==0 for f in result['factors'])
    else:assert result['factors'] == []
    export = [str(a.nsys), 'export', '--type=sqlite', '--force-overwrite=true', '--output', str(trace)+'.sqlite', str(trace)+'.nsys-rep']
    if not a.analyze_existing or not trace.with_suffix('.sqlite').is_file():
        with (out/'export.log').open('wb') as log:
            subprocess.run(export, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    manifest = dict(exe=str(exe), sha256=exe_sha, save=str(save), save_sha256=save_sha, sources=sources,source_root=str(source_root),frozen_manifest_sha256=manifest_sha,collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), command=command, env={k:v for k,v in env.items() if k.startswith('NTT_')}, profile=profile, export=export, Q_sha256=hashlib.sha256(q.encode()).hexdigest(), leaf=re.search(r'descent_values: (.*)',text)[1], full_wall=re.search(r'stage2_full_wall: (.*)',text)[1], oracle=re.search(r's4_oracle_stats: (.*)',text)[1])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    c = sqlite3.connect('file:'+trace.with_suffix('.sqlite').as_posix()+'?mode=ro', uri=True)
    c.row_factory = sqlite3.Row; c.text_factory = lambda b:b.decode('utf-8','replace')
    def rows(sql, args=()): return [dict(r) for r in c.execute(sql,args)]
    devices = rows('select deviceId,count(*) count from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId')
    assert len(devices)==1 and devices[0]['deviceId']==a.device
    kernels = rows('select s.value name,count(*) count,sum(k.end-k.start)/1e9 seconds,k.registersPerThread regs,k.localMemoryPerThread local,k.blockX threads from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName group by k.shortName,k.registersPerThread,k.localMemoryPerThread,k.blockX order by seconds desc')
    copies = rows('select e.label kind,count(*) count,sum(m.bytes) bytes,sum(m.end-m.start)/1e9 seconds from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind group by m.copyKind')
    apis = rows('select s.value name,count(*) count,sum(r.end-r.start)/1e9 seconds from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId group by r.nameId order by seconds desc')
    intervals = []
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        intervals += [(r[0],r[1]) for r in c.execute('select start,end from '+table+' where deviceId=?',(a.device,))]
    intervals.sort(); left,right=intervals[0]; first=left; union=0; gaps=[]
    for start,end in intervals[1:]:
        if start>right:
            union+=right-left; gaps.append(dict(start=right,end=start,seconds=(start-right)/1e9)); left,right=start,end
        else: right=max(right,end)
    union+=right-left; span=right-first
    largest=sorted(gaps,key=lambda r:r['seconds'],reverse=True)[:12]
    for g in largest:
        g['runtime_in_gap']=rows('select s.value name,r.start,r.end,(r.end-r.start)/1e9 seconds from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.start>=? and r.end<=? order by r.start limit 24',(g['start'],g['end']))
        g['previous_copy']=rows('select m.bytes,e.label kind,m.start,m.end from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.deviceId=? and m.end<=? order by m.end desc limit 1',(a.device,g['start']))
        g['next_copy']=rows('select m.bytes,e.label kind,m.start,m.end from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.deviceId=? and m.start>=? order by m.start limit 1',(a.device,g['end']))
    summary=dict(devices=devices,kernels=kernels,copies=copies,apis=apis,largest_gaps=largest,window=dict(span=span/1e9,event_union=union/1e9,no_own_gpu_event=(span-union)/1e9,percent_no_own_gpu_event=100*(span-union)/span),scope='All own GPU events in a native saved Stage2 process, including startup selftests and tail; approximate GPU window, not exact full-wall or whole-card idle. Runtime duration includes waits; no CPU sampling or hardware counters.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); c.close(); verify()
    print(json.dumps(dict(window=summary['window'],top_kernels=kernels[:8],copies=copies)))


if __name__=='__main__': main()
