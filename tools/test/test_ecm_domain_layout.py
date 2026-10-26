"""Audit the approved domain migration without changing frozen source identities.

Runtime tests cover publication, naming and checkpoint import. This gate checks
the complete file inventory and unchanged computation tokens in relocated files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from test_stage1_host_modules import normalize
from test_stage2_planning_extraction import balanced

ROOT = Path(__file__).resolve().parents[2]
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*|\s+|.', re.S)


def tokens(text):
    text = re.sub(r'(?m)^\s*#\s*include[^\n]*', '', text)
    return ''.join(m[0] for m in TOKEN.finditer(text)
                   if not m[0].isspace() and not m[0].startswith(('//', '/*')))


def definition_body(text, name):
    # Skip the complete parameter list, including a default std::function {}.
    # Its initializer braces are not the implementation body.
    start = re.search(r'\b'+re.escape(name)+r'\s*\(', text).end()-1
    end = balanced(text, start, '(', ')')
    opening = text.index('{', end)
    return normalize(text[opening:balanced(text, opening, '{', '}')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/'docs/development/ECM_DOMAIN_REFACTOR_MANIFEST.json').read_text(encoding='utf8'))
    baseline = manifest['baseline']
    checked = {}
    # Explicitly approved function moves/merges are covered through their real
    # interfaces. All remaining moved implementations retain computation tokens.
    exceptions = {
        'src/ecm/platform/stage1_progress.cpp': 'absorbs monotonic clock and TTY operations',
        'include/ecm/platform/stage1_progress.h': 'declares those shared host operations',
        'src/ecm/backend/cuda/stage1/montgomery/checkpoint.cpp': 'absorbs validated V5 importer',
        'src/ecm/backend/cuda/stage1/montgomery/checkpoint.h': 'declares validated V5 importer',
        'src/ecm/backend/opencl/stage1/montgomery/engine.cpp': 'absorbs unchanged C entry wrapper',
        'src/ecm/backend/opencl/stage1/montgomery/workflow.cpp': 'removes unreachable CUDA-only importer',
    }
    for item in manifest['files']:
        source, targets, action = item['source'], item['targets'], item['action']
        for target in targets:
            assert (ROOT/target).is_file(), ('missing approved target', source, target)
        if action in ('move', 'move_split', 'merge'):
            assert not (ROOT/source).is_file(), ('old path retained', source)
        if action != 'move' or source in exceptions or Path(source).suffix not in {'.h', '.hpp', '.cuh', '.cu', '.cpp', '.inl', '.cl'}:
            continue
        frozen = subprocess.check_output(['git', 'show', baseline+':'+source], cwd=ROOT)
        current = (ROOT/targets[0]).read_bytes()
        assert frozen.startswith(b'\xef\xbb\xbf') == current.startswith(b'\xef\xbb\xbf'), ('BOM changed', source)
        before = frozen.decode('utf-8-sig')
        after = current.decode('utf-8-sig')
        # Only the save/host-operation seam names changed in the CPU workflows.
        before = before.replace('using ecm_stage1::ensure_dir;', 'using ecm_io::ensure_dir;')
        before = before.replace('using ecm_stage1::mont_format_save_name;', 'using ecm_save::format_name;')
        before = before.replace('mont_format_save_name(', 'format_name(')
        # Source-tree fallback follows the moved payload. Portable deployment
        # still uses opencl_resources and is checked by package-layout tests.
        if source == 'src/ecm/backend/opencl/runtime/runtime.cpp':
            before = before.replace('src/ecm/backend/opencl/resources', 'src/ecm/opencl/resources')
        assert tokens(before) == tokens(after), ('computation tokens changed', source, targets[0])
        checked[targets[0]] = hashlib.sha256(after.encode()).hexdigest()
    # The two merged entry bodies keep their checks, ABI-facing arguments,
    # ordering, checksum policy and return values exactly as in the baseline.
    for old, new, name in [
        ('src/ecm/service/stage1/legacy_gpu_checkpoint.h',
         'src/ecm/montgomery_stage1/cuda/checkpoint.cpp', 'ecm_stage1_import_legacy_gpu'),
        ('src/ecm/backend/opencl/stage1/montgomery/entry.cpp',
         'src/ecm/montgomery_stage1/opencl/engine.cpp', 'opencl_ecm_stage1'),
    ]:
        frozen = subprocess.check_output(['git', 'show', baseline+':'+old], cwd=ROOT).decode('utf-8-sig')
        current = (ROOT/new).read_text(encoding='utf-8-sig')
        assert definition_body(frozen, name) == definition_body(current, name), ('merged function changed', name)
    result = dict(complete=True, baseline=baseline, inventory_files=len(manifest['files']),
                  unchanged_computation_files=len(checked), function_move_exceptions=exceptions,
                  sources=checked)
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'function_move_exceptions')}))


if __name__ == '__main__':
    main()
