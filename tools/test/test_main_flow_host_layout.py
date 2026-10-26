"""Check relocated host contracts, unchanged generated values and libecm ownership."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from source_layout import ROOT, LAYOUT, current_source

BASE='8c6ba76'


def frozen(path):
    return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sha=lambda data:hashlib.sha256(data).hexdigest()
    before=json.loads(frozen('config/ecm_config.generated.json'))
    after=json.loads((ROOT/'config/ecm_config.generated.json').read_text(encoding='utf8'))
    paths={old:LAYOUT['moved'].get(old,old) for old in before['sha256']}
    assert set(after['sha256'])==set(paths.values())
    checked=[]
    for old,new in paths.items():
        digest=sha((ROOT/new).read_bytes());assert digest==after['sha256'][new]
        if old!='tools/gen/generate_ecm_config.py':assert digest==before['sha256'][old],('default/schema/reference changed',old)
        checked.append(new)
    generator=frozen('tools/gen/generate_ecm_config.py').decode('utf8')
    for old,new in sorted(LAYOUT['moved'].items(),key=lambda pair:-len(pair[0])):generator=generator.replace(old,new)
    assert generator==(ROOT/'tools/gen/generate_ecm_config.py').read_text(encoding='utf8'),'generator policy changed beyond output paths'
    # These contracts must remain usable by ordinary host code. An actual MSVC
    # dependency-record test separately checks the real task collaborators.
    public=['include/ecm/stage1/task.h','include/ecm/stage1/selection.h',
            'include/ecm/stage1/scalar.h','include/ecm/core/stage2/probability.h',
            'include/ecm/platform/logging.h','include/ecm/platform/program_version.h',
            'include/ecm/application/configuration/runtime.h','include/ecm/stage1/gpu_backend.h']
    for name in public:
        text=(ROOT/name).read_text(encoding='utf-8-sig')
        code=re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"','',text,flags=re.S)
        assert not re.search(r'\b(?:__global__|__device__|dim3|cudaStream_t|cudaEvent_t|cgbn_t)\b',code),name
        for inc in re.findall(r'(?m)^\s*#\s*include\s*"([^"]+)"',text):
            assert (ROOT/'include'/inc).is_file() and '..' not in Path(inc).parts,(name,inc)
    main=current_source('src/core/ecm_cpu_stage2_main.cpp').read_text(encoding='utf-8-sig')
    assert '#include <ecm.h>' in main and not (ROOT/'include/ecm.h').exists()
    helper=(ROOT/'tools/build/internal/ecm_cpu_stage2.cmake').read_text(encoding='utf8')
    assert 'ecm_params_init.cpp' not in helper and 'service/execution/legacy' not in helper
    names=['ecm_internal_result.h','ecm_result_submission.h','ecm_result_json.h','ecm_result_paths.h','ecm_submission_writer.h','ecm_program_version.h']
    loop=re.search(r'foreach\(_shared\s+([^)]*)\)',helper)[1].split()
    assert loop==[LAYOUT['moved']['src/core/'+name] for name in names],'CPU identity list membership/order changed'
    assert '${_ecm_repo}/${_shared}' in helper and 'src/core/${_shared}' not in helper
    assert 'target_include_directories(ecm_cpu_stage2 PRIVATE "${_ecm_repo}/include")' in helper
    result=dict(complete=True,generated_inputs=checked,public_host_headers=public,cpu_libecm_header_preserved=True,cpu_identity_inputs=loop,defaults_unchanged=True,gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(complete=True,generated_inputs=len(checked),public_host_headers=len(public),cpu_identity_inputs=len(loop))))


if __name__=='__main__':main()
