"""Native portable-grid, scalar summary, incremental update and Stage1 CSV checks."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tomllib
from stage2_tune_tables import summaries

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--vcvars', type=Path, default=Path(
        r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    gmp = ROOT/'third_party/gmp-zen3/dist'
    exe = out/'fixture.exe'
    cmd = out/'compile.cmd'
    cmd.write_text('@echo off\ncall "'+str(a.vcvars)+'" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
        'cl /nologo /std:c++17 /EHsc /O2 /utf-8 /I"'+str(gmp/'include')+'" "'+
        str(ROOT/'tools/test/stage2_portable_fixture.cpp')+'" /Fe:"'+str(exe)+'" /Fo:"'+
        str(out/'fixture.obj')+'" /link /LIBPATH:"'+str(gmp/'lib')+'" gmp.lib\n', encoding='utf-8')
    proc = subprocess.run(['cmd', '/c', str(cmd)], capture_output=True, text=True, errors='replace', timeout=60)
    (out/'compile.log').write_text(proc.stdout+proc.stderr, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(proc.stdout+proc.stderr)
    for dll in (gmp/'bin').glob('*.dll'):
        shutil.copy2(dll, out/dll.name)
    def run(*args):
        return subprocess.check_output([str(exe), *map(str, args)], text=True)
    preset = [(3,4096,4,3,10,2),(4,3072,6,3,10,2),(5,2048,8,4,10,2),
              (5,2048,10,4,10,2),(5,2048,10,6,5,3),(8,1536,12,6,5,3),
              (8,1536,12,6,5,3),(10,1024,12,8,5,3),(15,1024,12,8,5,5),(15,1024,14,8,5,5)]
    for level, (count,step,ds,bc,ratio,reps) in enumerate(preset, 1):
        grid = json.loads(run('--grid', level))
        assert grid['widths'] == [512]+[step*i for i in range(1,count+1)]
        assert len(set(grid['ds'])) == ds and grid['ds'][0] == 30030 and grid['ds'][-1] == 2282280
        assert grid['repeats'] == reps and len(grid['b2s']) == bc
        assert grid['b2s'][-1] == 2600000000000
        assert all(high//ratio == low for low, high in zip(grid['b2s'], grid['b2s'][1:]))
    base = '''[profile]
format = 6
unit = "full_stage2"
algorithm_revision = 1
revision = 0
[condition.0]
uuid_hex = "0123456789abcdef0123456789abcdef"
sm_major = 8
sm_minor = 9
cuda_runtime = 13030
cuda_driver = 13030
gl_fixed_mode = 3
outer_unroll_u = 0
add_sub_mask = 1
env_condition_tag = "baseline"
[width.512.table.0]
condition = 0
target_bits = 512
arithmetic_bits = 512
carrier_exponent = 0
modulus_kind = "generic"
b1 = 20
b2 = 26000000000
d = 30030
p = 2880
fold_resident = 1
frontier_resident = 1
execution_path = "fixture_resident"
source = "measured"
repeats = 2
clean = 1
bad = 0
hits = 0
selftest_cases = 1
checked = 1
columns = ["median_seconds", "mad_seconds"]
rows = [
    [2, 0.1],
]
'''
    original = out/'base.toml'
    original.write_text(base, encoding='utf-8')
    assert summaries(tomllib.loads(run('--roundtrip', original)))[0]['median_seconds'] == 2
    for value in [1, 2, 3]:
        other = out/f'new_{value}.toml'
        other.write_text(base.replace('[2, 0.1]', f'[{value}, 0.1]'), encoding='utf-8')
        result = tomllib.loads(run('--update', original, other))
        assert summaries(result)[0]['median_seconds'] == min(2, value)
    other = out/'other_condition.toml'
    other.write_text(base.replace('0123456789abcdef0123456789abcdef','1123456789abcdef0123456789abcdef'), encoding='utf-8')
    result = tomllib.loads(run('--update', original, other))
    assert len(result['condition']) == 2 and len(summaries(result)) == 2
    assert run('--matches',original,original,'')=='true'
    assert run('--matches',original,other,'')=='false'
    default_ignore='gpu,driver,cuda,backend,environment'
    assert run('--matches',original,other,default_ignore)=='true'
    altered=out/'altered_environment.toml'
    altered.write_text(base.replace('13030','12060').replace('gl_fixed_mode = 3','gl_fixed_mode = 1')
                       .replace('"baseline"','"different"'),encoding='utf-8')
    assert run('--matches',original,altered,default_ignore)=='true'
    assert run('--matches',original,altered,'gpu')=='false'
    assert subprocess.run([str(exe),'--matches',str(original),str(altered),'unknown'],capture_output=True).returncode!=0
    modeled = out/'model.toml'
    modeled.write_text(base.replace('source = "measured"','source = "model"').replace('[2, 0.1]','[0.1, 0.1]'), encoding='utf-8')
    assert summaries(tomllib.loads(run('--update', original, modeled)))[0]['median_seconds'] == 2
    corruptions = [base+'seconds = [2,2]\n', base.replace('clean = 1','clean = 0'),
                   base.replace('p = 2880','p = 2881'), base+'validation_max_relative_error = 0\n',
                   base.replace('condition = 0','condition = 1')]
    for i, text in enumerate(corruptions):
        bad = out/f'bad_{i}.toml'
        bad.write_text(text, encoding='utf-8')
        assert subprocess.run([str(exe),'--roundtrip',str(bad)], capture_output=True).returncode != 0
    csv = out/'stage1.csv'
    csv.write_text('container_bits,b1,mhz,tpi,curves,seconds_per_curve\n'
                   '1792,110e6,2000,8,768,9.5\n'
                   '2560,110e6,2000,16,384,19.6\n'
                   '3072,110e6,2000,16,384,26.6\n'
                   '3584,110e6,2000,16,384,\n'
                   '8192,110e6,2000,16,384,140.4\n', encoding='utf-8')
    exact = json.loads(run('--csv', csv, 1786, 110e6, 2000))
    assert abs(exact['seconds']-9.5) < 1e-12 and exact['container_bits'] == 1792
    upper = json.loads(run('--csv', csv, 8186, 110e6, 2000))
    next_tier = json.loads(run('--csv', csv, 8192, 110e6, 2000))
    assert upper['container_bits'] == 8192 and upper['tpi'] == 16
    assert next_tier['container_bits'] == 9216 and next_tier['tpi'] == 32
    scaled = json.loads(run('--csv', csv, 1786, 220e6, 1000))
    assert abs(scaled['seconds']-38) < 1e-10  # already seconds per curve; no batch division.
    interpolated = json.loads(run('--csv', csv, 3500, 110e6, 2000))
    assert interpolated['source'] == 'interpolated' and interpolated['container_bits'] == 3584
    large = json.loads(run('--csv', csv, 15360, 110e6, 2000))
    assert large['crosses_tpi'] and large['source'] == 'extrapolated' and large['tpi'] == 32
    assert abs(large['seconds']-561.6) < 1e-9
    across_b1=json.loads(run('--predict',original,512,10000000,26000000000))
    across_width=json.loads(run('--predict',original,4423,10000000,2600000000000))
    assert across_b1['seconds']>0 and across_width['seconds']>across_b1['seconds']
    assert not across_width['validated'] and across_width['rank']>across_width['seconds']
    legacy_result=[]
    for legacy in [ROOT/'data/experiments/ecm_tune_n318_prod_20261010/stage2.toml',
                   ROOT/'data/experiments/ntt_phase_tune_20261010/supplement/ntt_0.toml',
                   ROOT/'data/experiments/stage2_portable_tune_20261010/gpu_06/resident.toml']:
        if legacy.is_file():
            p=subprocess.run([str(exe),'--roundtrip',str(legacy)],capture_output=True,text=True)
            assert p.returncode!=0 and 'unsupported tune format' in p.stderr
            legacy_result.append(str(legacy))
    for count in range(1,15):
        for center in range(count):
            assert json.loads(run('--neighbors',center,count))==list(range(max(0,center-1),min(count,center+2)))
    # Compact tables must preserve all named scalar values through native reload.
    compact=run('--roundtrip',original)
    packed=out/'compact.toml';packed.write_text(compact,encoding='utf-8')
    assert summaries(tomllib.loads(run('--roundtrip',packed)))==summaries(tomllib.loads(compact))
    # All final rows share proof/context fields; only the B2 and timing vary.
    expanded=[]
    for i in range(20):
        record=base.split('[width.512.table.0]',1)[1]
        record=record.replace('b2 = 26000000000',f'b2 = {26000000000+i}')
        record=record.replace('[2, 0.1]',f'[{2+i/100}, 0.1]')
        expanded.append(f'[width.512.table.{i}]'+record)
    many=out/'many.toml'
    many.write_text(base.split('[width.512.table.0]',1)[0]+''.join(expanded),encoding='utf-8')
    many_compact=run('--roundtrip',many)
    many_packed=out/'many_compact.toml';many_packed.write_text(many_compact,encoding='utf-8')
    assert len(summaries(tomllib.loads(many_compact)))==20
    assert len(tomllib.loads(many_compact)['width']['512']['table'])==1
    assert summaries(tomllib.loads(run('--roundtrip',many_packed)))==summaries(tomllib.loads(many_compact))
    assert len(many_compact.encode('utf-8')) < len(many.read_bytes())/2
    table_bad=[
        base.replace('[2, 0.1]', '[2]'),
        base.replace('"mad_seconds"', '"median_seconds"'),
        base.replace('[width.512.table.0]', '[width.513.table.0]'),
        base.replace('    [2, 0.1],', '    [2, 0.1],\n    [2, 0.1],'),
        base.replace('    [2, 0.1],', '    [2, [0.1]],'),
        base.replace('    [2, 0.1],',''),
    ]
    for i,text in enumerate(table_bad):
        damaged=out/f'bad_table_{i}.toml';damaged.write_text(text,encoding='utf-8')
        assert subprocess.run([str(exe),'--roundtrip',str(damaged)],capture_output=True).returncode!=0
    report = dict(passed=True, grid_levels=10, update_cases=5, corrupt_profiles=len(corruptions),
                  csv=dict(exact=exact,scaled=scaled,interpolated=interpolated,extrapolated=large),
                  predictions=dict(across_b1=across_b1,across_width=across_width),legacy_rejected=legacy_result,table_rejections=len(table_bad),neighbor_cases=105,
                  compact_rows=20,compact_bytes=len(many_compact.encode('utf-8')),expanded_bytes=len(many.read_bytes()),
                  container_boundary=dict(upper=upper,next=next_tier))
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
