"""Verify native full-ECM tune inputs, effort grids and versioned TOML reader."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--vcvars', type=Path, default=Path(
        r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    gmp = ROOT/'third_party/gmp-zen3/dist'
    exe = out/'fixture.exe'
    cmd = out/'compile.cmd'
    cmd.write_text('@echo off\ncall "'+str(a.vcvars)+'" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
        'cl /nologo /std:c++17 /EHsc /O2 /utf-8 /I"'+str(gmp/'include')+'" "'+
        str(ROOT/'tools/test/stage2_tune_ecm_fixture.cpp')+'" /Fe:"'+str(exe)+'" /Fo:"'+
        str(out/'fixture.obj')+'" /link /LIBPATH:"'+str(gmp/'lib')+'" gmp.lib\n', encoding='utf-8')
    proc = subprocess.run(['cmd', '/c', str(cmd)], capture_output=True, text=True, errors='replace', timeout=60)
    (out/'compile.log').write_text(proc.stdout+proc.stderr, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(proc.stdout+proc.stderr)
    for dll in (gmp/'bin').glob('*.dll'):
        shutil.copy2(dll, out/dll.name)
    spec = importlib.util.spec_from_file_location('ref', ROOT/'tools/stat/suyama_mont_ref.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    primes = [107,127,521,607,1279,2203,2281,3217,4253,4423,9689,9941,11213]
    for exponent in primes:
        modulus = (1 << exponent)-1
        residue = 4
        for _ in range(exponent-2):
            residue = (residue*residue-2) % modulus
        assert residue == 0, exponent  # Independent Lucas-Lehmer check of the catalogue.
        n, x = subprocess.check_output([str(exe),'--prime',str(exponent)], text=True).split()
        expected = ref.stage1(26,20,(1 << exponent)-1)
        assert int(n,16) == (1 << exponent)-1
        assert int(x,16) == expected['x']
    assert subprocess.run([str(exe),'--prime','503'], capture_output=True).returncode != 0
    assert subprocess.run([str(exe),'--prime','8191'], capture_output=True).returncode != 0
    prev = (0,0,0,0,0)
    previous_bounds = set()
    previous_ds = set()
    for level in range(1,11):
        grid = tuple(map(int, subprocess.check_output([str(exe),'--effort',str(level)], text=True).split()))
        subdivisions = 1 if level<3 else 4
        assert grid == (min(13,level+3),level+2+(2 if level>=7 else 0),1+((level-1)//2)*subdivisions,2*level+1,64+16*(level-1))
        assert all(x >= y for x,y in zip(grid,prev)) and grid[-1] > prev[-1]
        bounds = list(map(int, subprocess.check_output([str(exe),'--effort-b2',str(level)],text=True).split()))
        assert bounds == sorted(set(bounds)) and len(bounds) == grid[2]
        assert previous_bounds.issubset(bounds) and bounds[0] == 2600000000
        if level in (3,4):
            assert len(bounds)==5 and 4000000000<bounds[1]<6000000000
            assert bounds[2]==8221921916 and 14000000000<bounds[3]<16000000000
        assert grid[0]*grid[1]*grid[2] <= 4096
        ds = list(map(int, subprocess.check_output([str(exe),'--effort-d',str(level)],text=True).split()))
        assert ds == sorted(set(ds)) and len(ds) == grid[1]
        assert previous_ds.issubset(ds)
        if level>=7:
            assert {810810,1021020}.issubset(ds)
        previous_ds = set(ds)
        previous_bounds = set(bounds)
        prev = grid
    for bad in [0,11]:
        assert subprocess.run([str(exe),'--effort',str(bad)], capture_output=True).returncode != 0
    profile = '''[profile]
format = 3
unit = "full_stage2"
algorithm_revision = 1
effort_level = 1
repeats = 3
warmups = 1
max_batches = 64
[device]
uuid_hex = "0123456789abcdef0123456789abcdef"
sm_major = 8
sm_minor = 9
cuda_runtime = 13030
cuda_driver = 13030
gl_fixed_mode = 3
outer_unroll_u = 0
add_sub_mask = 1
[policy]
batch_mb = 256
arena_mb = 6300
fold_mb = 640
[policy.environment]
xadd6 = 1
[ecm.sample_0]
target_bits = 318
arithmetic_bits = 503
carrier_exponent = 503
modulus_kind = "mersenne"
b1 = 20
b2 = 26000000000
d = 180180
p = 17280
giant_points = 144302
clean = 1
hits = 0
bad = 0
selftest_cases = 2016
checked = 2992
fold_resident = 1
frontier_resident = 1
required_free_bytes = 2000000000
repeats = 3
seconds = [2.0, 3.0, 4.0]
median_seconds = 3.0
mad_seconds = 1.0
init_seconds = 1.0
main_seconds = 2.0
giant_seconds = 0.1
gtrees_seconds = 0.2
fold_seconds = 0.3
descent_seconds = 0.4
inverse_seconds = 0.5
accum_seconds = 0.01
[summary]
complete = 1
measured = 1
skipped = 0
failed = 0
'''
    valid = out/'valid.toml'
    valid.write_text(profile, encoding='utf-8')
    assert subprocess.check_output([str(exe),'--load',str(valid)], text=True).strip() == '1'
    bom = out/'bom.toml'
    bom.write_text(profile, encoding='utf-8-sig')
    assert subprocess.check_output([str(exe),'--load',str(bom)], text=True).strip() == '1'
    assert tomllib.loads(profile)['ecm']['sample_0']['median_seconds'] == 3
    mutations = [(f'{key} = {old}', f'{key} = {new}') for key,old,new in [
        ('format','3','1'), ('algorithm_revision','1','2'), ('complete','1','0'), ('measured','1','2'),
        ('max_batches','64','1'),
        ('bad','0','1'), ('hits','0','1'), ('clean','1','0'), ('checked','2992','0'),
        ('selftest_cases','2016','0'), ('median_seconds','3.0','3.5'), ('mad_seconds','1.0','0.5'),
        ('carrier_exponent','503','502'), ('target_bits','318','504'), ('p','17280','17281'),
        ('giant_points','144302','144303'), ('repeats','3','2'), ('init_seconds','1.0','nan'),
        ('seconds','[2.0, 3.0, 4.0]','[2.0, 3.0, -4.0]')]]
    for index,(old,new) in enumerate(mutations):
        path = out/f'bad_{index}.toml'
        assert old in profile
        path.write_text(profile.replace(old,new), encoding='utf-8')
        assert subprocess.run([str(exe),'--load',str(path)], capture_output=True).returncode != 0, new
    for index,tail in enumerate(['\n[summary]\ncomplete = 1\n', '\ncomplete = 1\n']):
        path = out/f'duplicate_{index}.toml'
        path.write_text(profile+tail, encoding='utf-8')
        assert subprocess.run([str(exe),'--load',str(path)], capture_output=True).returncode != 0
    legacy = out/'legacy.toml'
    legacy.write_text(profile.replace('format = 3','format = 2').replace('[policy.environment]\nxadd6 = 1','environment = "NTT_XADD6=1;"'), encoding='utf-8')
    assert subprocess.check_output([str(exe),'--load',str(legacy)], text=True).strip() == '1'
    result = dict(prime_point_oracles=len(primes), prime_certificates=len(primes), effort_levels=10, rejected_profiles=len(mutations)+2,
                  device_policy_mismatch_checks=3, native_toml_roundtrip=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
