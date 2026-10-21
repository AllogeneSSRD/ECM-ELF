"""Verify shared selection and generated defaults retain their exact behavior."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from test_stage1_host_modules import normalize,body

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=lambda name:subprocess.check_output(['git','show','72b6cd0:'+name],cwd=ROOT).decode('utf-8-sig')
    read=lambda name:(ROOT/name).read_text(encoding='utf-8-sig')
    previous=old('src/ecm/application/configuration/ecm_stage1_selection.h')
    current=read('src/ecm/core/stage1/selection.cpp')
    names=['lower_ascii','parse_curve_model','parse_execution_target','parse_cpu_implementation','parse_simd_reduction','parse_scalar_construction','valid_route','valid_gpu_param','resolve_gpu_param','curve_model_name','execution_target_name']
    hashes={}
    for name in names:
        expected,actual=body(previous,name),body(current,name);assert expected==actual,name
        hashes[name]=hashlib.sha256(actual.encode()).hexdigest()
    public=read('include/ecm/core/stage1/selection.h')
    before=old('include/ecm/core/stage1/selection.h')
    field=lambda s:s[s.index('struct Selection {'):s.index('};',s.index('struct Selection {'))+2]
    assert normalize(field(before))==normalize(field(public))
    generated='src/ecm/application/configuration/generated/ecm_config_generated.h'
    before=old(generated);start=before.index('namespace defaults {');end=before.index('} // namespace defaults',start)+len('} // namespace defaults')
    expected=before[:start]+before[end:]
    actual=read(generated).replace('#include "../../../platform/generated/configuration_defaults.h"','')
    assert normalize(expected)==normalize(actual),'schema, binding or parser implementation changed'
    defaults=read('src/ecm/platform/generated/configuration_defaults.h')
    original=before[start+len('namespace defaults {'):before.index('} // namespace defaults',start)]
    actual=defaults[defaults.index('namespace ecm_config { namespace defaults {')+len('namespace ecm_config { namespace defaults {'):defaults.index('} } // namespace ecm_config::defaults')]
    assert normalize(original)==normalize(actual),'generated defaults changed'
    unchanged=['config/ecm_options.json','config/ecm.toml.default','config/ecm.toml.example','docs/ECM_TOML_REFERENCE.md','src/ecm/application/configuration/generated/ecm_toml_template.h','src/ecm/application/configuration/generated/ecm_toml_schema.h']
    for name in unchanged:assert old(name)==read(name),name
    for name in ['task.cpp','preparation.cpp']:
        source=read('src/ecm/service/stage1/'+name)
        assert 'application/configuration/' not in source and 'ecm_stage1_selection.h' not in source,name
    assert 'ecm_toml' not in defaults and '#include <array>' in defaults
    assert not (ROOT/'src/ecm/application/configuration/ecm_stage1_selection.h').exists()
    inventory=a.output/'inventory.json'
    subprocess.run(['cmake','-DECM_CUDA_SOURCES_OUTPUT='+str(inventory),'-P',str(ROOT/'tools/build/internal/cuda_sources.cmake')],check=True)
    groups=json.loads(inventory.read_text())['groups']
    for group in ['STAGE1_COMMON','STAGE2_HOST']:assert groups[group].count('src/ecm/core/stage1/selection.cpp')==1
    assert 'src/ecm/core/stage1/selection.cpp' not in groups['STAGE1_CUDA']
    subprocess.run(['python',str(ROOT/'tools/gen/generate_ecm_config.py'),'--check'],cwd=ROOT,check=True)
    result=dict(complete=True,baseline='72b6cd0',selection_functions=hashes,selection_layout_unchanged=True,generated_defaults_unchanged=True,schema_bindings_unchanged=True,unchanged_schema_template_reference_files=unchanged,service_application_configuration_includes=0,new_service_classes=0,cuda_runs=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='selection_functions'}))


if __name__=='__main__':main()
