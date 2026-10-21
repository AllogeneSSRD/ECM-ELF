"""Verify the production OpenCL payload and unchanged actual Stage1 results."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from source_layout import ROOT, LAYOUT
from test_result_output_consistency import crc_valid


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic(row):
    row=dict(row)
    assert crc_valid(row),'result CRC mismatch'
    row['program']=dict(row['program'])
    row['program'].pop('version')
    for name in ['timestamp','checksum','source-build']:
        row.pop(name,None)
    if 'stage1-engine' in row:
        row['stage1-engine']=dict(row['stage1-engine'])
        row['stage1-engine'].pop('batch-seconds',None)
    return row


def points(path):
    rows=[]
    for line in path.read_text(encoding='utf8',errors='replace').splitlines():
        fields=dict(re.findall(r'(\w+)=([^;]+);',line))
        assert {'METHOD','PARAM','SIGMA','B1','N','X','CHECKSUM'}<=fields.keys()
        rows.append({k:fields[k] for k in ['METHOD','PARAM','SIGMA','B1','N','X','CHECKSUM','X0','Y0'] if k in fields})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--cache-hit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.build/'build_manifest.json').read_text(encoding='utf-8-sig'))
    assert manifest['complete'] and not manifest['cuda_compiler_enabled']
    assert manifest['executable_sha256'].lower()==sha(a.build/'ecm.exe')
    resources={new.split('/resources/',1)[1]:ROOT/new for old,new in LAYOUT['moved'].items() if new.startswith('src/ecm/backend/opencl/resources/')}
    assert len(resources)==172 and set(manifest['resources'])==set(resources)
    for name,source in resources.items():
        assert sha(source)==sha(a.build/'opencl_resources'/name)==manifest['resources'][name].lower(),name
    assert 'diagnostics/mont.cl' in resources and not (ROOT/'kernels/opencl/bench/mont.cl').exists()
    commands=json.loads((a.build/'compile_commands.json').read_text(encoding='utf8'))
    assert commands and all(Path(row['file']).suffix=='.cpp' and 'nvcc' not in row['command'].lower() for row in commands)
    baseline_points=points(a.baseline/'points.save');assert len(baseline_points)==4
    baseline_cache={p.name:sha(p) for p in (a.baseline/'.opencl_cache').glob('*.bin')};assert len(baseline_cache)==2
    for folder in [a.candidate,a.cache_hit]:
        call=json.loads((folder/'invocation.json').read_text())
        assert call['returncode']==0 and call['binary_sha256']==sha(a.build/'ecm.exe')
        assert points(folder/'points.save')==baseline_points
        for name in ['stage1_results.jsonl','results.json.txt']:
            before=[semantic(json.loads(v)) for v in (a.baseline/name).read_text().splitlines()]
            after=[semantic(json.loads(v)) for v in (folder/name).read_text().splitlines()]
            assert before==after,(folder,name)
        assert {p.name:sha(p) for p in (folder/'.opencl_cache').glob('*.bin')}==baseline_cache
    hits=(a.cache_hit/'run.log').read_text(encoding='utf8')
    assert hits.count('OpenCL cache hit: ')==2 and 'OpenCL cache miss' not in hits
    # The loader retains Android's logical asset root. Source/registry/cache
    # assembly is covered by the exhaustive relocation-body gate separately.
    gradle=(ROOT/'Android/ECM/app/build.gradle.kts').read_text(encoding='utf8')
    assert 'from(mpaRoot.resolve("src/ecm/backend/opencl/resources"))' in gradle
    assert 'src/main/assets/kernels/opencl' in gradle and 'include("bench/**")' in gradle
    result=dict(complete=True,resources=172,ordinary_cxx_commands=len(commands),cuda_compiler_invocations=0,curves=4,actual_runs_compared=2,cache_files=2,cache_hits=2,save_points_equal=True,results_equal=True,android_asset_contract_only=True,binary_sha256=sha(a.build/'ecm.exe'))
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
