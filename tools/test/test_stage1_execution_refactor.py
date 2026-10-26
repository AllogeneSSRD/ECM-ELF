"""Current execution/IO boundaries and frozen arithmetic compatibility gate.

Numerical and persistence contracts run separately with the real workers. This
check permits explicit checkpoint borrowing in CUDA host orchestration, while
requiring every remaining device/core token to match the pre-migration source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASELINE = '0ba1462'
METHODS = {
    'set_backup_limit':'ecm_stage1_set_backup_limit', 'backup_limit':'ecm_stage1_backup_limit',
    'candidates':'ecm_stage1_checkpoint_candidates', 'mark_bad_primary':'ecm_stage1_mark_bad_primary',
    'commit':'ecm_stage1_commit_checkpoint', 'remove':'ecm_stage1_remove_checkpoints',
    'set_path':'ecm_stage1_set_checkpoint_path', 'path':'ecm_stage1_checkpoint_path',
    'begin':'ecm_stage1_begin_batch', 'pending':'ecm_stage1_checkpoint_pending',
    'validate_pending':'ecm_stage1_validate_pending_checkpoint', 'saved_sigma':'ecm_stage1_saved_sigma',
    'sigma':'ecm_stage1_batch_sigma', 'read_checkpoint_blob':'ecm_stage1_read_checkpoint_blob',
    'read_blob':'ecm_stage1_read_blob', 'write_blob':'ecm_stage1_write_blob',
    'import_payload':'ecm_stage1_import_gpu_payload', 'close':'ecm_stage1_end_batch'}
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\s]', re.S)

def tokens(text):
    text = re.sub(r'^\s*#\s*include[^\n]*', '', text, flags=re.M)
    return [m.group() for m in TOKEN.finditer(text) if not m.group().startswith(('//', '/*'))]

def borrowed(text):
    text = re.sub(r'ecm_stage1_checkpoint::Context\s*&\s*checkpoint\s*,\s*', '', text)
    text = re.sub(r'ecm_stage1_checkpoint::Context\s+checkpoint\s*;', '', text)
    for method, original in METHODS.items():
        text = re.sub(r'checkpoint\.' + method + r'\s*\(', original + '(', text)
    for name in ('checkpoint_record','save_checkpoint','run_mont_resident','run_mont_runtime','execute_mont_batch'):
        text = re.sub(r'\b'+name+r'\(checkpoint\s*,\s*', name+'(', text)
        text = text.replace(name+'(checkpoint)', name+'()')
    return text

def definition(text, marker):
    start = text.index(marker)
    opening = text.index('{', start)
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == '{': depth += 1
        elif text[index] == '}':
            depth -= 1
            if not depth:
                end = index + 1
                if text[end:end+1] == ';': end += 1
                return text[start:end]
    raise AssertionError(marker)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    old = lambda name: subprocess.check_output(['git','show',BASELINE+':'+name],cwd=ROOT).decode('utf-8-sig')
    read = lambda name: (ROOT/name).read_text(encoding='utf-8-sig')
    names = subprocess.check_output(['git','ls-tree','-r','--name-only',BASELINE],cwd=ROOT,text=True).splitlines()
    hashes = {}; device = []; core = []
    for name in names:
        path = ROOT/name
        if path.suffix in ('.cu','.cuh','.cl') and name.startswith(('src/ecm/backend/','kernels/')):
            assert tokens(borrowed(read(name))) == tokens(old(name)), ('device arithmetic changed',name)
            device.append(name)
        if name.startswith('src/ecm/core/') and path.suffix in ('.cpp','.h'):
            expected = old(name)
            if name.endswith('curve/ecm_edwards_curve.h'):
                owner = read('include/ecm/integer.h')
                for marker in ('struct Integer {', 'class Array {'):
                    part = definition(expected,marker)
                    assert tokens(part) == tokens(definition(owner,marker)), ('integer owner changed',marker)
                    expected = expected.replace(part,'')
            assert tokens(read(name)) == tokens(expected), ('core arithmetic changed',name)
            core.append(name)
        if path.is_file() and name.startswith('src/ecm/backend/cpu/arithmetic/'):
            assert tokens(read(name)) == tokens(old(name)), ('CPU arithmetic changed',name)
        if path.is_file() and name.startswith('src/ecm/backend/cpu/stage1/'):
            current = read(name)
            current = re.sub(r', const std::string& cache_dir(?: = \{\})?', '', current)
            current = current.replace('torsion, cache_dir, &status',
                                      'torsion, ecm_exp_cache_get_dir(), &status')
            assert tokens(current) == tokens(old(name)), ('CPU scalar/SIMD arithmetic changed',name)
        if path.is_file() and name.startswith(('src/ecm/core/','src/ecm/backend/')):
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for curve in ('edwards','montgomery'):
        for backend in ('cpu','cuda'):
            workflow = read(f'src/ecm/backend/{backend}/stage1/{curve}/workflow.cpp')
            assert 'LegacyExecutionResult' not in workflow and 'take_computation_result' not in workflow
            assert 'ComputationResult' in workflow
            assert 'publish_save(' not in workflow and 'append_save_lines' not in workflow
    task = read('src/ecm/stage1/task.cpp')
    assert 'ecm_stage1_checkpoint::Context session;' in task
    assert 'deps.execute(request, execution, session)' in task
    assert 'struct Context::State' in read('src/io/stage1_checkpoint.cpp')
    for text in (read('src/io/stage1_checkpoint.cpp'),read('src/io/exponent_cache.cpp')):
        assert 'thread_local' not in text and 'g_cache_dir' not in text
    assert not list((ROOT/'src/ecm/storage').rglob('*'))
    assert not list((ROOT/'include/ecm/storage').rglob('*'))
    assert read('config/ecm_options.json') == old('config/ecm_options.json'), 'configuration schema changed'
    subprocess.run(['python','tools/gen/generate_ecm_config.py','--check'],cwd=ROOT,check=True)
    result = dict(complete=True,baseline=BASELINE,device_files=len(device),core_files=len(core),
                  explicit_context_only_in_device_orchestration=True,core_arithmetic_unchanged=True,
                  owned_cpu_cuda_results=True,storage_removed=True,source_hashes=hashes,gpu_runs=0)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))

if __name__ == '__main__': main()
