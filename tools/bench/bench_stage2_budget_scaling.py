"""Serial B2/bit-size/owner study with an external big-workspace shape filter.

The big limit is a planner constraint, verified against actual arena statistics;
it is not a new allocator hard limit. ArenaMB is the existing cache budget.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from functools import lru_cache
from plan_stage2_d import candidates
from calibrate_stage2_d import shape

MIB = 1 << 20
GPU_UUID = 'GPU-8a67b1f8-ef1c-3177-a822-813a7ac2224d'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(None)
def geometry(p, bits):
    nf, _, _ = shape(p + 1, bits)
    nt, _, _ = shape(p // 2 + 1, bits)
    return dict(n_fold=nf, n_tree=nt, big_bytes=24 * nf,
                arena_est_bytes=8 * (3 * nf + 2 * p + 1 + 2 * (3 * nt + 2 * (p // 2 + 1) - 1)),
                owner_bytes=8 * ((bits + 63) // 64) * (9 * p + 8) + 48)


def legacy_cost(d, p, b2):
    i = b2 // d + 2
    size = min(p, i)
    batches = (i + size - 1) // size
    return (265.756 / (3400110 * math.log2(51840)) * i * max(1, math.log2(size))
            + 50.537 / 51840 * p + 50.655 / 3400110 * i + 30.339 / 66 * batches + 25)


class MemorySample(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]


class Utilization(ctypes.Structure):
    _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]


class Nvml:
    def __init__(self):
        self.lib = ctypes.WinDLL('nvml.dll')
        assert self.lib.nvmlInit_v2() == 0
        self.handle = ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByUUID(GPU_UUID.encode(), ctypes.byref(self.handle)) == 0
        name = ctypes.create_string_buffer(128)
        assert self.lib.nvmlDeviceGetName(self.handle, name, len(name)) == 0
        self.name = name.value.decode()
        assert 'RTX 4060 Laptop' in self.name

    def sample(self):
        memory, util = MemorySample(), Utilization()
        mc = self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory))
        uc = self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util))
        # Monitoring failures are missing observations, not arithmetic failures.
        return dict(total=memory.total if mc == 0 else None,
                    free=memory.free if mc == 0 else None, used=memory.used if mc == 0 else None,
                    gpu=util.gpu if uc == 0 else None, memory=util.memory if uc == 0 else None,
                    memory_status=mc, utilization_status=uc)


def fields(text, label):
    match = re.search(re.escape(label) + r': (.*)', text)
    assert match, label
    return dict(re.findall(r'(\w+)=([^\s]+)', match[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--shape-probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, choices=(1,), default=1)
    parser.add_argument('--big-mb', type=int, default=3072, help='High external big-shape budget (MiB)')
    parser.add_argument('--small-big-mb', type=int, default=768, help='Low external big-shape budget (MiB)')
    parser.add_argument('--owner-mb', type=int, default=1024, help='High owner budget (MiB; not a residency guarantee)')
    parser.add_argument('--small-owner-mb', type=int, default=128, help='Low owner budget (MiB; 0 forces budget fallback)')
    parser.add_argument('--arena-mb', type=int, default=6300, help='Existing arena cache/planner budget (MiB)')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gmp-ecm', type=Path, default=Path('D:/code/GIMPS/gmp-ecm/ecm-7.0.5-znver3/ecm.exe'))
    a = parser.parse_args()
    if not (0 < a.small_big_mb <= a.big_mb and 0 <= a.small_owner_mb <= a.owner_mb
            and a.owner_mb > 0 and a.arena_mb > 0):
        parser.error('Need 0<small-big<=big, 0<=small-owner<=owner, owner>0 and arena>0')
    repo = Path(__file__).resolve().parents[2]
    out, exe, src = a.output.resolve(), a.exe.resolve(), a.sources.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert a.resume or not any(out.iterdir()), 'Use a fresh output directory'
    build = json.loads((exe.parent / 'build_manifest.json').read_text(encoding='utf-8-sig'))
    deps = {m[1]: m[2].lower() for s in build['sources']
            if (m := re.fullmatch(r'([^=]+\.(?:cu|cuh|cpp|h|ps1))=([A-Fa-f0-9]{64})', s))}
    assert len(deps) == 19 and build['gl_fixed_mode'] == 3 and build['architecture'] == 'sm_89'
    assert digest(exe) == build['sha256'].lower()
    probe = a.shape_probe.resolve()
    pm = json.loads((probe.parent / 'manifest.json').read_text(encoding='utf-8-sig'))
    assert all(pm['sources'][name].lower() == deps[name] for name in pm['sources'])
    tools = [Path(__file__).relative_to(repo).as_posix(), 'tools/bench/plan_stage2_d.py',
             'tools/bench/calibrate_stage2_d.py', 'tools/bench/fit_stage2_d.py', 'tools/stat/suyama_mont_ref.py']
    tool_hashes = {name: digest(repo / name) for name in tools}

    def verify():
        assert digest(exe) == build['sha256'].lower() and digest(probe) == pm['sha256'].lower()
        for name, want in deps.items():
            assert digest(src / name) == want, name
        for name, want in tool_hashes.items():
            assert digest(repo / name) == want, name
        for name, want in pm['generated_sources'].items():
            assert digest(probe.parent / name) == want.lower(), name
        for name, want in pm['local_sources'].items():
            assert digest(repo / name) == want.lower(), name

    verify()
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1', NTT_D_MODEL='0', NTT_STAGE1_Q_DUMP='1',
               NTT_CARRY_CHECK_FUSED='0', NTT_POINT_MERSENNE='1')
    configs = {'large_resident': dict(big_mb=a.big_mb, arena_mb=a.arena_mb, fold_mb=a.owner_mb),
               'large_owner128': dict(big_mb=a.big_mb, arena_mb=a.arena_mb, fold_mb=a.small_owner_mb),
               'small_big': dict(big_mb=a.small_big_mb, arena_mb=a.arena_mb, fold_mb=a.owner_mb)}
    plan_path = out / 'plan.json'
    if a.resume:
        data = json.loads(plan_path.read_text())
        configs = data['configs']  # Resume the frozen plan, never reinterpret new CLI budgets.
        assert data['exe_sha256'] == digest(exe)
        assert digest(out/'preparation_driver.py')==data['tool_hashes'][tools[0]]
        assert all(data['tool_hashes'][name]==tool_hashes[name] for name in tools[1:])
        if data['runs']:
            assert digest(out/'runtime_driver.py')==data['runtime_tool_hashes'][tools[0]]
            assert all(data['runtime_tool_hashes'][name]==tool_hashes[name] for name in tools[1:])
            prior=out/('runtime_driver_'+data['runtime_tool_hashes'][tools[0]]+'.py')
            if not prior.exists():prior.write_bytes((out/'runtime_driver.py').read_bytes())
    else:
        (out/'preparation_driver.py').write_bytes(Path(__file__).read_bytes())
        # Stage1 is computed before GPU timings, and compared with independent GMP-ECM.
        spec = importlib.util.spec_from_file_location('mont_ref', repo / tools[-1])
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        saves = {}
        assert a.gmp_ecm.is_file()
        gmp_sha = digest(a.gmp_ecm)
        for bits in (2203, 4423, 8191):
            n = (1 << bits) - 1
            res = ref.stage1(26, 1000, n)
            assert res['gcd'] == 1
            gmp_save = out / f'gmp_m{bits}.save'
            cmd = [str(a.gmp_ecm.resolve()), '-param', '0', '-sigma', '26', '-c', '1', '-save', str(gmp_save), '1000', '1000']
            r = subprocess.run(cmd, input=str(n).encode(), capture_output=True, cwd=out, timeout=120)
            (out / f'gmp_m{bits}.log').write_bytes(r.stdout + r.stderr)
            gx = int(re.search(r'\bX=(0x[0-9a-fA-F]+)', gmp_save.read_text())[1], 16)
            assert gx == res['x'] and digest(a.gmp_ecm) == gmp_sha, bits
            checksum = 1000 * 26 * n * gx % 4294967291
            saved = out / f'm{bits}.save'
            saved.write_text(f'METHOD=ECM; SIGMA=26; B1=1000; N=(2^{bits}-1); X=0x{gx:x}; CHECKSUM={checksum};\n')
            q_sha = hashlib.sha256(f'{gx:x}'.encode()).hexdigest()
            saves[str(bits)] = dict(path=str(saved),sha256=digest(saved),Q_sha256=q_sha,gmp_command=cmd,gmp_exit=r.returncode)
            print('STAGE1',bits,'CPU/GMP Q matches',flush=True)
        pool = list(candidates(200000000))

        def select(bits, b2, cfg):
            best = None
            # Even the maximal bpw62 requires L >= 4*P*bits/62.
            # Reject impossible P before evaluating the exact shape.
            max_p=cfg['big_mb']*MIB*62//(96*bits)
            for d, p in pool:
                if d < 6 or d % 2 or p>max_p:
                    continue
                try:
                    geo = geometry(p, bits)
                except ValueError:
                    continue
                if geo['big_bytes'] > cfg['big_mb'] * MIB or geo['arena_est_bytes'] > cfg['arena_mb'] * MIB:
                    continue
                cost = legacy_cost(d, p, b2)
                if best is None or cost < best['legacy_prediction']:
                    best = dict(D=d,P=p,legacy_prediction=cost,**geo)
            assert best
            return best

        plans = {}
        for bits in (2203,4423,8191):
            for b2 in (80000000000,260000000000,800000000000):
                for name,cfg in configs.items():
                    plans[(bits,b2,name)] = select(bits,b2,cfg)
        for b2 in (2600000000000,8000000000000):
            for name,cfg in configs.items():
                plans[(4423,b2,name)] = select(4423,b2,cfg)
        fixed = plans[(4423,800000000000,'large_resident')]
        # Verify every chosen P/bit geometry with the actual immutable C++ packing planner.
        queries = sorted({(row['P'],bits) for (bits,_,_),row in plans.items()})
        r = subprocess.run([str(probe),str(a.device)],input=''.join(f'{p} {b}\n' for p,b in queries).encode(),capture_output=True,env=env,timeout=60)
        (out/'shape_queries.log').write_bytes(r.stdout+r.stderr)
        assert r.returncode==0
        actual = [fields('budget_shape: '+line,'budget_shape') for line in re.findall(r'budget_shape: (.*)',r.stdout.decode())]
        assert len(actual)==len(queries)
        for row,(p,bits) in zip(actual,queries):
            assert int(row['P'])==p and int(row['bits'])==bits
            for key in ('n_fold','n_tree','big_bytes','arena_est_bytes'):
                assert int(row[key])==geometry(p,bits)[key],(p,bits,key)
        cases=[]

        def case(bits,b2,name,group='shape_policy',rep=0,fixed_d=None):
            row=dict(bits=bits,B2=b2,config=name,group=group,rep=rep,**configs[name])
            row['plan'] = fixed_d or plans[(bits,b2,name)]
            row['name']=f'{len(cases)+1:02d}_{group}_m{bits}_b{b2}_{name}_r{rep}'
            cases.append(row)

        for bits in (2203,4423,8191):
            case(bits,80000000000,'large_resident','warmup')
        for j,b2 in enumerate((80000000000,260000000000,800000000000)):
            sizes=(2203,4423,8191)
            for bits in sizes[j:]+sizes[:j]:
                names=list(configs)
                for name in names[j:]+names[:j]:
                    case(bits,b2,name)
        for b2 in (2600000000000,8000000000000):
            for name in configs:
                case(4423,b2,name)
        for b2 in (260000000000,800000000000,2600000000000):
            for rep,name in enumerate(('large_resident','large_owner128','large_owner128','large_resident'),1):
                case(4423,b2,name,'fixed_D',rep,fixed)
        data=dict(exe=str(exe),exe_sha256=digest(exe),build=build,sources=deps,tool_hashes=tool_hashes,
                  shape_probe=pm,saves=saves,gmp_sha256=gmp_sha,configs=configs,cases=cases,runs=[],
                  device=1,gpu_uuid=GPU_UUID,environment=env,
                  scope='External legacy ranking with big/arena geometry filters; owner budget deliberately absent from ranking; sampled whole-card memory, not process peak; default mandatory checks; exploratory unreplicated grid plus fixed-D ABBA')
        plan_path.write_text(json.dumps(data,indent=2))
        for row in cases:
            print('PLAN',row['name'],'D',row['plan']['D'],'P',row['plan']['P'],'ownerMiB',round(row['plan']['owner_bytes']/MIB,2),flush=True)
    if a.prepare_only:
        return
    history=data.setdefault('runtime_tool_hashes_history',[])
    if data.get('runtime_tool_hashes') and data['runtime_tool_hashes'] not in history:
        history.append(data['runtime_tool_hashes'])
    if tool_hashes not in history:history.append(tool_hashes)
    data['runtime_tool_hashes']=tool_hashes
    (out/'runtime_driver.py').write_bytes(Path(__file__).read_bytes())
    (out/('runtime_driver_'+tool_hashes[tools[0]]+'.py')).write_bytes(Path(__file__).read_bytes())
    nvml=Nvml()
    for row in data['cases'][len(data['runs']):]:
        verify()
        saved=Path(data['saves'][str(row['bits'])]['path'])
        assert digest(saved)==data['saves'][str(row['bits'])]['sha256']
        name=row['name'];log=out/(name+'_engine.log');results=out/(name+'.jsonl')
        assert not log.exists() and not results.exists(),name
        command=[str(exe),'--save',str(saved),'--b2',str(row['B2']),'--d',str(row['plan']['D']),
                 '--device','1','--arena-mb',str(row['arena_mb']),'--batch-mb','64',
                 '--results',str(results),'--log',str(log)]
        before=nvml.sample();samples=[dict(seconds=0,**before)]
        start=time.perf_counter()
        with (out/(name+'_driver.log')).open('wb') as f:
            child=subprocess.Popen(command,env=env|{'NTT_FOLD_DEVICE_MAX_MB':str(row['fold_mb'])},stdout=f,stderr=subprocess.STDOUT)
            while child.poll() is None:
                samples.append(dict(seconds=time.perf_counter()-start,**nvml.sample()))
                if time.perf_counter()-start>1800:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                    raise RuntimeError(f'{name}: own process tree timed out')
                time.sleep(.2)
        elapsed=time.perf_counter()-start
        samples.append(dict(seconds=elapsed,**nvml.sample()))
        (out/(name+'_gpu.json')).write_text(json.dumps(dict(uuid=GPU_UUID,name=nvml.name,sample_interval_seconds=.2,samples=samples,scope='Whole-card samples including startup/tail/other apps; observed max is not guaranteed peak'),indent=2))
        verify();assert child.returncode==0,(name,child.returncode)
        text=log.read_text(encoding='utf-8',errors='replace')
        for token in ('RTX 4060 Laptop','stage1_skipped=1','gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1','point_arithmetic: xadd6=1','fixed=3','d_model: requested=0 enabled=0 version=legacy_56_1'):
            assert token in text,(name,token)
        q=re.search(r'real_setup_Q_full: hex=([0-9a-f]+)',text)[1]
        assert hashlib.sha256(q.encode()).hexdigest()==data['saves'][str(row['bits'])]['Q_sha256']
        wall={k:float(v) for k,v in fields(text,'stage2_full_wall').items() if k in ('shape','init','main','total')}
        fd=fields(text,'real_batched_folddevice');ntt=fields(text,'ntt_workspace_stats')
        assert int(ntt['big_peak_bytes'])<=row['big_mb']*MIB,(name,ntt['big_peak_bytes'])
        assert int(fd['peak_bytes'])<=row['fold_mb']*MIB
        expected_resident=row['plan']['owner_bytes']<=row['fold_mb']*MIB
        assert bool(int(fd['enabled']))==expected_resident,(name,fd)
        if not expected_resident:assert fd['fallback']=='budget'
        shape_row=fields(text,'real_shape')
        shape_row['P']=shape_row['P'].rsplit('=',1)[-1]
        assert int(shape_row['D'])==row['plan']['D'] and int(shape_row['P'])==row['plan']['P']
        result=json.loads(results.read_text().splitlines()[-1]);assert result['bad_factors']==0
        valid_memory=[s['used'] for s in samples if s['used'] is not None]
        record=dict(case=row,command=command,runtime_tool_hashes=tool_hashes,
                    process_seconds=elapsed,wall=wall,fold=fd,ntt=ntt,
                    shape=shape_row,leaf=fields(text,'descent_values'),oracle=fields(text,'s4_oracle_stats'),
                    s4=fields(text,'s4_multiply_stats'),gmemory=fields(text,'real_batched_gmemory'),
                    gleaf=fields(text,'device_gleaf'),output=fields(text,'real_batched_outputwindow'),
                    phases=fields(text,'real_batched_split'),arena=fields(text,'real_batched_breakdown'),result=result,
                    device_samples=dict(before_used=before['used'],observed_max_used=max(valid_memory,default=None),
                                        count=len(samples),valid_memory_count=len(valid_memory),
                                        missing_utilization_count=sum(s['gpu'] is None for s in samples)))
        key=(row['bits'],row['B2'],row['plan']['D'])
        for old in data['runs']:
            c=old['case']
            if (c['bits'],c['B2'],c['plan']['D'])==key:
                assert old['leaf']['hash']==record['leaf']['hash'] and old['result']['factors']==result['factors']
        data['runs'].append(record)
        (out/'measurements.json').write_text(json.dumps(data,indent=2))
        plan_path.write_text(json.dumps(data,indent=2))
        print('RUN',name,'full',wall['total'],'owner',fd['enabled'],fd['fallback'],'bigMiB',int(ntt['big_peak_bytes'])/MIB,flush=True)
    data.update(passed=len(data['runs']),failed=0)
    (out/'measurements.json').write_text(json.dumps(data,indent=2))


if __name__=='__main__':
    main()
