"""Compile native tune serializer; verify its output with a TOML reader."""
import argparse
import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vcvars', type=Path, default=Path(
        r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    exe = out/'fixture.exe'
    script = out/'compile.cmd'
    script.write_text('@echo off\ncall "'+str(args.vcvars)+'" >nul 2>&1\n'
        'if errorlevel 1 exit /b 1\ncl /nologo /std:c++17 /EHsc /utf-8 "'+
        str(ROOT/'tools/test/stage2_tune_format_fixture.cpp')+'" /Fe:"'+str(exe)+
        '" /Fo:"'+str(out/'fixture.obj')+'"\n', encoding='utf-8')
    build = subprocess.run(['cmd','/c',str(script)], capture_output=True, text=True, timeout=60)
    (out/'compile.log').write_text(build.stdout+build.stderr, encoding='utf-8')
    if build.returncode:
        raise RuntimeError(build.stdout+build.stderr)
    records = [dict(type='device', device_index=0, name='GPU "test"', uuid_hex='abc',
                    sm_major=8, sm_minor=9),
               dict(type='sample', status='measured', length=65536, log2_length=16,
                    median_seconds=0.005, conv_iter_per_s=200.0, seconds=[0.004,0.006],
                    outer_radix_bits=[6,5], compact_scratch=True, bad=0),
               dict(type='sample', status='skipped_memory', length=131072, log2_length=17),
               dict(type='complete', measured=1, skipped=1, failed=0, usable=True)]
    result = subprocess.run([str(exe)], input='\n'.join(map(json.dumps,records)),
                            capture_output=True, text=True, check=True)
    (out/'profile.toml').write_text(result.stdout, encoding='utf-8')
    parsed = tomllib.loads(result.stdout)
    assert parsed['device']['name'] == records[0]['name']
    assert 'device_index' not in parsed['device']
    for record in records[1:3]:
        assert parsed['ntt'][f'length_{record["length"]}'] == {k:v for k,v in record.items() if k!='type'}
    assert parsed['summary']['usable'] is True
    batches=[dict(type='sample',status='measured',length=64,batch=b,
                  median_seconds=.001*b,conv_iter_per_s=1000.0) for b in [1,3,16]]
    result=subprocess.run([str(exe),'--batched'],input='\n'.join(map(json.dumps,batches)),
                          capture_output=True,text=True,check=True)
    grouped=tomllib.loads(result.stdout)['ntt']['length_64']
    assert set(grouped)=={'slices_1','slices_3','slices_16'}
    assert all(grouped['slices_'+str(b)]['batch']==b for b in [1,3,16])
    for bad in [dict(type='sample',length=64),dict(type='sample',length=64,batch=0),
                dict(type='sample',length=64,batch='1.2')]:
        assert subprocess.run([str(exe),'--batched'],input=json.dumps(bad),capture_output=True,text=True).returncode
    for bad in ['{"type":"sample"}', '{"type":"device","type":"device"}',
                '{"type":"device",}', '{"type":"device"}garbage',
                '{"type":"device","value":null}', '{"type":"unknown"}',
                '{"type":"sample","length":"a.b"}', '{"type":"device","x":[{}]}']:
        assert subprocess.run([str(exe)], input=bad, capture_output=True, text=True).returncode != 0
    previous = 0
    for level in range(1,11):
        lines = subprocess.check_output([str(exe),str(level)], text=True).splitlines()
        values = list(map(int, lines[0].split()))
        assert values == [3,20+min(level-1,7),2*level*level+1]
        assert list(map(int, lines[1].split())) == [1,4,16,64,256,1024,4096,16384,65535][:min(level,9)]
        assert values[2] > previous
        previous = values[2]
    for level in [0,11]:
        assert subprocess.run([str(exe),str(level)], capture_output=True).returncode != 0
    print('PASS: native TOML roundtrip, rejected callbacks, effort levels 1..10')

if __name__ == '__main__':
    main()
