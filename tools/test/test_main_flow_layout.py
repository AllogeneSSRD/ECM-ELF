"""Verify the domain layout and unchanged CMake compile/link policy.

Approved merges are not one-to-one renames. Computation tokens are audited by
test_ecm_domain_layout.py; this gate retains exact build membership and flags.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/'docs/development/ECM_DOMAIN_REFACTOR_MANIFEST.json').read_text(encoding='utf8'))
    mapping = {entry['source']: entry['targets'][0] for entry in manifest['files']
               if entry['action'] in ('move', 'move_split', 'merge')}
    for old, new in mapping.items():
        assert not (ROOT/old).is_file(), old
        assert (ROOT/new).is_file(), new
    def frozen(name):
        return subprocess.check_output(['git', 'show', manifest['baseline']+':'+name], cwd=ROOT).decode('utf-8-sig')
    checked = []
    for name in ['tools/build/internal/cuda_sources.cmake', 'CMakeLists.txt']:
        expected = frozen(name)
        if name == 'CMakeLists.txt':
            # M8 removes only the independent entry TU and its empty target.
            start = expected.index('add_library(opencl_ecm_entry STATIC')
            end = expected.index('# The Stage1 driver', start)
            expected = expected[:start]+expected[end:]
            expected = expected.replace('    opencl_ecm_entry\n', '')
        for old, new in sorted(mapping.items(), key=lambda pair: -len(pair[0])):
            expected = expected.replace(old, new)
        for old, new in [
            ('src/ecm/backend/cuda', 'src/ecm/cuda'),
            ('src/ecm/backend/opencl', 'src/ecm/opencl'),
            ('src/ecm/service/diagnostics', 'src/ecm/diagnostics'),
        ]:
            expected = expected.replace(old, new)
        current = (ROOT/name).read_text(encoding='utf-8-sig')
        assert expected.split() == current.split(), ('compile/link policy changed', name)
        checked.append(name)
    report = dict(complete=True, baseline=manifest['baseline'], migrated_files=len(mapping),
                  approved_merges=8, tu_order_unchanged=True, source_properties_unchanged=True,
                  compile_link_inputs=checked, gpu_runs=0)
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
