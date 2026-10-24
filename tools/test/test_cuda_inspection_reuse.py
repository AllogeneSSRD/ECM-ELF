"""Real-tool cold/warm inspection probe, without publishing or changing builds.

Checks two representative Stage1 objects and all Stage2 objects plus both EXEs.
This tests inspection reuse, not current-source or GPU runtime qualification.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/packaging'))
import package_release as packaging
from cuda_inspection_cache import InspectionCache, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--profile',choices=packaging.TARGETS,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    ptx,sass=packaging.TARGETS[args.profile];runs=[]
    for stage in ('stage1','stage2'):
        record=packaging.read(args.build_root/'components'/f'{args.profile}.{stage}.json')
        manifest=packaging.read(record['manifest']);binary=Path(record['exe']);build=Path(record['build']);cuda=Path(manifest['cuda_root'])
        assert sha(binary).lower()==manifest.get('executable_sha256',manifest.get('sha256')).lower()
        if stage=='stage1':
            subset=out/'stage1';subset.mkdir()
            units=[unit for unit in packaging.read(build/'compile_commands.json') if Path(unit['file']).name in ('edwards_512.cu','mont0_512.cu')]
            assert len(units)==2
            (subset/'compile_commands.json').write_text(json.dumps(units),encoding='utf-8');build=subset
        report=out/(stage+'.modules.json')
        cache=InspectionCache(out/'cache',cuda);start=time.perf_counter()
        images=packaging.images(cuda,binary,ptx,sass,cache)
        packaging.inspect_modules(cuda,build,stage,report,cache)
        cold=time.perf_counter()-start;modules=packaging.read(report)
        if stage=='stage2':packaging.verify_stage2_membership(modules,manifest['kernel_layout'],packaging.stage2_catalog())
        assert all(module['ptx_targets']==['.target sm_'+ptx] for module in modules['modules'])
        before=sha(report);warm=InspectionCache(out/'cache',cuda);start=time.perf_counter()
        with patch.object(packaging.subprocess,'Popen',side_effect=AssertionError('Warm inspection launched cuobjdump')),\
             patch.object(packaging.subprocess,'check_output',side_effect=AssertionError('Warm inspection launched cuobjdump')):
            assert packaging.images(cuda,binary,ptx,sass,warm)==images
            packaging.inspect_modules(cuda,build,stage,report,warm)
        assert sha(report)==before and warm.misses==0 and warm.hits==1+len(modules['modules'])
        runs.append(dict(stage=stage,cold_seconds=cold,warm_seconds=time.perf_counter()-start,
            cold_inspections=cache.misses,warm_hits=warm.hits,warm_tool_calls=0,modules=len(modules['modules']),
            executable_sha256=sha(binary),report_sha256=before,cuobjdump_sha256=sha(cuda/'bin/cuobjdump.exe')))
        print(f'PASS {stage}: warm hits={warm.hits}, cuobjdump calls=0',flush=True)
    (out/'result.json').write_text(json.dumps(dict(complete=True,profile=args.profile,runs=runs,
        qualification='inspection cache only; representative Stage1 subset; no GPU execution'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
