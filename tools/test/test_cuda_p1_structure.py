"""Check Stage1 module direction and preserve the frozen device arithmetic bodies."""
from source_layout import current_source,current_files,relocation_text
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from instance_refactor import verify_handoff

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', default='c0c4c74')
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    inventory = out/'cuda_sources.json'
    subprocess.run(['cmake', '-DECM_CUDA_SOURCES_OUTPUT='+str(inventory), '-P',
                    str(ROOT/'tools/build/internal/cuda_sources.cmake')], check=True)
    groups = json.loads(inventory.read_text())['groups']
    handoff = verify_handoff()
    sources = groups['STAGE1_CUDA']+groups['STAGE1_PRAC']
    assert len(sources) == len(set(sources))
    assert all((ROOT/s).is_file() for s in sources)
    generated = out/'generated'
    subprocess.run(['cmake', '-DECM_STAGE1_MODULES_OUTPUT='+generated.resolve().as_posix(),
                    '-DECM_CUDA_FULL_BUILD=ON', '-DECM_CUDA_ENABLE_PRAC=ON',
                    '-DCMAKE_CUDA_COMPILER_VERSION=13.3', '-P',
                    str(ROOT/'tools/build/internal/cuda_modules.cmake')], check=True)
    modules = json.loads((generated/'cuda_modules/stage1/modules.json').read_text())['modules']
    assert len(modules) == 161 and len({m['label'] for m in modules}) == 161
    assert sum(m['curve'] == 'edwards' for m in modules) == 32
    assert sum(m['family'] == 'ladder' for m in modules) == 93
    assert sum(m['family'].startswith('prac_') for m in modules) == 36
    sources += groups['STAGE1_TEMPLATES']+groups['STAGE1_PRAC_TEMPLATES']
    sources += [m['source'] for m in modules]
    visited = set()

    def text(path): return path.read_text(encoding='utf-8-sig')
    def walk(path):
        path = path.resolve()
        if path in visited: return
        visited.add(path); body = text(path)
        if path == current_source('include/ecm/cuda/cgbn_types.cuh').resolve():
            # Upstream requires GMP only while parsing its host templates.
            assert '#if !defined(__CUDA_ARCH__)\n#include <gmp.h>\n#endif' in body
        else:
            assert not re.search(r'#\s*include\s*[<"]gmp\.h', body), path
        code = re.sub(r'/\*.*?\*/|//[^\n]*', '', body, flags=re.S)
        assert not re.search(r'\bmpz_\w*|EcmEdwardsProgress|ecm_stage1_checkpoint', code), path
        for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"', body, re.M):
            candidate = next((v for v in (path.parent/name, ROOT/'include'/name)
                              if v.is_file()), None)
            assert candidate is not None, (path, name)
            relative = candidate.resolve().relative_to(ROOT).as_posix()
            assert candidate.resolve() in device_files, (path, relative)
            walk(candidate)
    device_files={p.resolve() for p in current_files('kernels/cuda')+current_files('include/ecm/cuda')}
    device_files.update((ROOT/s).resolve() for s in groups['STAGE1_TEMPLATES']+groups['STAGE1_PRAC_TEMPLATES'])
    device_files.update(Path(m['source']).resolve() for m in modules)
    # The current registry owns host dispatch, occupancy and a launch bridge.
    # Its historical kernels/ location does not make the GMP-facing entry
    # declarations a pure device-arithmetic dependency. Check that TU separately.
    registry=current_source('kernels/cuda/stage1/montgomery/registry.cu').resolve()
    registry_code=re.sub(r'/\*.*?\*/|//[^\n]*','',text(registry),flags=re.S)
    assert not re.search(r'__global__|__device__|\bmpz_\w*|EcmEdwardsProgress|ecm_stage1_checkpoint',registry_code)
    device_files.remove(registry)
    dispatcher=ROOT/'src/ecm/stage1/montgomery/cuda/prac/dispatch.cu'
    dispatcher_code=re.sub(r'/\*.*?\*/|//[^\n]*','',text(dispatcher),flags=re.S)
    assert not re.search(r'__global__|__device__|\bmpz_\w*',dispatcher_code)
    device_files.remove(dispatcher.resolve())
    # Extracted dispatch values were already part of launch.h/constants.h.
    # This single data-only header joins the closure; host policy bodies do not.
    domain=ROOT/'src/ecm/stage1/montgomery/cuda/prac/domain.h'
    code=re.sub(r'/\*.*?\*/|//[^\n]*','',text(domain),flags=re.S)
    assert not re.search(r'#\s*include|\b(?:struct|class|inline|static)\b|\(',code)
    device_files.add(domain.resolve())
    for s in sources:
        if (ROOT/s).resolve() in device_files:walk(ROOT/s)
    for path in current_files('src/cuda/stage1'):
        body = re.sub(r'/\*.*?\*/|//[^\n]*', '', text(path), flags=re.S)
        assert not re.search(r'__global__|__device__|<<<', body), path
    comparisons = [
        ('kernels/cuda/cgbn_stage1_kernel.h', 'kernels/cuda/stage1/montgomery/arithmetic.cuh',
         'template<uint32_t tpi, uint32_t bits>', '// Types come from'),
        ('kernels/cuda/cgbn_edwards_point.cuh', 'kernels/cuda/stage1/edwards/point.cuh', '', ''),
        ('kernels/cuda/cgbn_edwards_naf.cuh', 'kernels/cuda/stage1/edwards/naf.cuh', '', ''),
        ('kernels/cuda/cgbn_stage1_prac_kernel.cuh', 'kernels/cuda/optional/prac/cgbn_stage1_prac_kernel.cuh',
         '// Outputs may overwrite', ''),
    ]
    evidence = []
    for old, new, start, stop in comparisons:
        before = subprocess.check_output(['git', 'show', a.baseline+':'+old], cwd=ROOT).decode('utf-8-sig')
        after = relocation_text(new)
        if start: before = before[before.index(start):]; after = after[after.index(start):]
        if stop:
            before = before[:before.index('// ── kernel param typedefs')]
            after = after[:after.index(stop)]
        normalize = lambda s: re.sub(r'\s+', '', re.sub(r'/\*.*?\*/|//[^\n]*', '', s, flags=re.S))
        assert normalize(before) == normalize(after), new
        evidence.append(dict(source=new, arithmetic_sha256=hashlib.sha256(normalize(after).encode()).hexdigest()))
    result = dict(complete=True, device_closure_files=len(visited), host_dispatch_files=2, arithmetic_bodies=evidence,
                  generated_modules=len(modules), retired_binding_handoff=handoff, gpu_runs=0)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
