"""Compare Auto candidate construction with frozen and current native headers.

Links the same verified production CUDA object for its pure packing query.
Does not query the device or execute curves. Each request constructs a fresh
estimator; no warmed estimator is reused between measured selections.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--baseline-evidence', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--baseline-only', action='store_true')
    p.add_argument('--vcvars', type=Path, default=Path(
        r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    a = p.parse_args()
    assert 1 <= a.rounds <= 100 and 3 <= a.repeats <= 20
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build, evidence = a.build.resolve(), a.baseline_evidence.resolve()
    original = json.loads((evidence/'result.json').read_text(encoding='utf-8-sig'))
    assert original['complete'] and not original.get('failure')
    manifest = json.loads((build/'build_manifest.json').read_text(encoding='utf-8-sig'))
    assert manifest['engine'] == 'production' and manifest['gl_fixed_mode'] == 3
    assert manifest['sha256'].lower() == original['source_identities'][str(build/'ecm_cuda_stage2.exe')]
    assert sha(build/'ecm_cuda_stage2.exe') == manifest['sha256'].lower()
    identities = {}

    def freeze(path, dest):
        identity = sha(path)
        identities[str(path)] = identity
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        assert sha(dest) == identity
        return dest

    freeze(Path(__file__), out/'inputs'/Path(__file__).name)
    fixture = freeze(ROOT/'tools/bench/stage2_tune_component_planning_fixture.cpp', out/'inputs/fixture.cpp')
    profile_path = freeze(a.profile.resolve(), out/'inputs/profile.toml')
    profile = tomllib.loads(profile_path.read_text(encoding='utf-8-sig'))
    samples = list(profile['ecm'].values())
    assert len({s['target_bits'] for s in samples}) == len({s['b1'] for s in samples}) == 1
    assert profile['summary']['complete'] == 1 and profile['profile']['format'] == 4
    # Reconstruct the baseline source tree exclusively from frozen evidence.
    for relative, expected in manifest['source_hashes'].items():
        source = ROOT/relative
        assert original['source_identities'][str(source)] == expected.lower()
        old = evidence/'inputs'/expected.lower()/source.name
        assert sha(old) == expected.lower()
        freeze(old, out/'baseline'/relative)
        if not a.baseline_only:
            freeze(source, out/'current'/relative)
    gmp = ROOT/'third_party/gmp-zen3/dist'
    for name, expected in manifest['gmp_hashes'].items():
        assert sha(gmp/name) == expected.lower()
        freeze(gmp/name, out/'gmp'/name)
    freeze(build/'build_manifest.json', out/'inputs/build_manifest.json')
    objects = []
    for stem, expected in manifest['objects'].items():
        if stem == 'ecm_cuda_stage2_main':
            continue
        objects.append(freeze(build/'_objects'/(stem+'.obj'), out/'objects'/(stem+'.obj')))
        assert sha(objects[-1]) == expected.lower()
    # CUDA object and all of its recorded dependencies must be unchanged.
    cuda_dependencies = [name for name in manifest['source_hashes'] if name.startswith('src/cuda/')]
    assert all(sha(ROOT/name) == manifest['source_hashes'][name].lower() for name in cuda_dependencies)
    b1 = samples[0]['b1']
    low, high = min(s['b2'] for s in samples), max(s['b2'] for s in samples)
    t1 = original['t1_seconds']
    requests = [[b1,0,0,t,1,0,-1] for t in (t1,3.,44.)]
    requests += [[b1,0,0,t1,r,0,-1] for r in (.5,2.)]
    requests += [[b1,0,0,t1,1,d,c] for d in sorted({s['d'] for s in samples})
                 for c in sorted({s['carrier_exponent'] for s in samples})]
    requests += [[b1,int(low+(high-low)*.3),int(low+(high-low)*.7),t1,1,0,-1],
                 [b1,low,low,t1,1,0,-1], [b1,high,high,t1,1,0,-1]]
    inputs = out/'requests.txt'
    inputs.write_text(str(len(requests))+'\n'+'\n'.join(' '.join(map(str,r)) for r in requests)+'\n')
    nvcc = Path(manifest['cuda_root'])/'bin/nvcc.exe'
    compiler = manifest['host_compiler']
    assert nvcc.is_file() and Path(compiler).is_file()
    binaries = {}
    for variant in (['baseline'] if a.baseline_only else ['baseline','current']):
        exe = out/(variant+'.exe')
        obj = out/(variant+'.obj')
        common = f'"{nvcc}" -ccbin "{compiler}" -std=c++17 -O3 -arch={manifest["architecture"]}'
        command = out/(variant+'.cmd')
        command.write_text('@echo off\ncall "'+str(a.vcvars)+'" >nul 2>&1\nif errorlevel 1 exit /b 1\n'+
            common+f' -I "{out/variant/"src/core"}" -I "{out/"gmp/include"}" -Xcompiler /utf-8 '+
            f'-c "{fixture}" -o "{obj}"\nif errorlevel 1 exit /b 1\n'+common+' '+
            ' '.join('"'+str(path)+'"' for path in [obj]+objects)+
            f' -L "{out/"gmp/lib"}" -lgmp -o "{exe}"\n', encoding='utf-8')
        proc = subprocess.run(['cmd','/d','/c',str(command)],capture_output=True,text=True,errors='replace',timeout=120)
        (out/(variant+'.compile.log')).write_text(proc.stdout+proc.stderr)
        assert proc.returncode == 0, proc.stdout+proc.stderr
        binaries[variant] = exe
        freeze(exe, out/'inputs'/(variant+'.exe'))
    shutil.copy2(out/'gmp/bin/gmp-10.dll', out/'gmp-10.dll')
    runs = {v: [] for v in binaries}
    reference = None
    for iteration in range(a.repeats+1):
        order = list(binaries)
        if iteration % 2:
            order.reverse()
        for variant in order:
            proc = subprocess.run([str(binaries[variant]),str(profile_path),str(inputs),str(a.rounds)],
                                  capture_output=True,text=True,errors='replace',timeout=120)
            (out/f'{variant}_{iteration}.json').write_text(proc.stdout)
            (out/f'{variant}_{iteration}.stderr').write_text(proc.stderr)
            assert proc.returncode == 0, proc.stderr
            value = json.loads(proc.stdout)
            candidates = [r['candidates'] for r in value['cases']]
            if reference is None:
                reference = candidates
            assert candidates == reference, ('candidate prediction/order differs',variant,iteration)
            if iteration:
                runs[variant].append(value['cases'])
            print(f'{variant} repeat={iteration} cases={len(candidates)} first_seconds={value["cases"][0]["seconds"]:.6f}', flush=True)
    summary = {}
    for variant, repeats in runs.items():
        summary[variant] = [{
            'request': request,
            'candidate_count': len(repeats[0][i]['candidates']),
            'seconds': [r[i]['seconds'] for r in repeats],
            'median_seconds': statistics.median(r[i]['seconds'] for r in repeats),
            'query_seconds_median': statistics.median(r[i]['query_seconds'] for r in repeats),
            'query_calls': [r[i]['query_calls'] for r in repeats],
        } for i, request in enumerate(requests)]
    assert all(sha(Path(path)) == expected for path, expected in identities.items())
    result = dict(complete=True, baseline_only=a.baseline_only, rounds=a.rounds,repeats=a.repeats,
                  source_identities=identities, summary=summary,
                  candidates_per_repeat=sum(map(len,reference)), candidates_identical=True,
                  device_queries=0, curves=0)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('complete','candidates_per_repeat','candidates_identical')}))


if __name__ == '__main__':
    main()
