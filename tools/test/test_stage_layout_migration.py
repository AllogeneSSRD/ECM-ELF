"""Verify exact Stage1/Stage2 relocation and unchanged include dependency edges."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = re.compile(r'(?m)^(\s*#\s*include\s*["<])([^">\r\n]+)([">])')
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*|\s+|.', re.S)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tokens(text):
    return ''.join(m[0] for m in TOKEN.finditer(text)
                   if not m[0].isspace() and not m[0].startswith(('//', '/*')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/'docs/development/STAGE_LAYOUT_MIGRATION_MANIFEST.json').read_text(encoding='utf-8'))
    baseline = manifest['baseline']
    moves = {item['source']: item['target'] for item in manifest['files']}
    assert len(moves) == len(set(moves.values())) == len(manifest['files']) == 112
    frozen_paths = set(subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', baseline], cwd=ROOT, text=True).splitlines())

    def resolve(owner, name, quoted, frozen):
        candidates = ([Path(owner).parent/name] if quoted else []) + [Path(p)/name for p in manifest['include_roots']]
        for candidate in candidates:
            key = os.path.normpath(candidate).replace('\\', '/')
            if (key in frozen_paths if frozen else (ROOT/key).is_file()):
                return key
        return None

    paths = dict(moves)
    paths.update({item['source']: item['target'] for item in manifest['include_consumers']})
    checked, edges = {}, []
    for old, new in paths.items():
        before_bytes = subprocess.check_output(['git', 'show', baseline+':'+old], cwd=ROOT)
        after_bytes = (ROOT/new).read_bytes()
        assert before_bytes.startswith(b'\xef\xbb\xbf') == after_bytes.startswith(b'\xef\xbb\xbf'), ('BOM changed', old)
        before, after = before_bytes.decode('utf-8-sig'), after_bytes.decode('utf-8-sig')
        original, current = list(INCLUDE.finditer(before)), list(INCLUDE.finditer(after))
        assert len(original) == len(current), ('include count changed', old)
        restored = after
        for a, b in reversed(list(zip(original, current))):
            assert a[1].strip() == b[1].strip() and a[3] == b[3], ('include directive changed', old)
            quoted = a[1].rstrip().endswith('"')
            source = resolve(old, a[2], quoted, True)
            target = resolve(new, b[2], quoted, False)
            if source is not None:
                expected = moves.get(source, source)
                assert target == expected, ('include edge changed', old, a[2], b[2], expected, target)
                edges.append({'source': old, 'target': new, 'before': source, 'after': target})
            else:
                assert a[2] == b[2], ('unresolved external include changed', old, a[2], b[2])
            restored = restored[:b.start(2)] + a[2] + restored[b.end(2):]
        assert tokens(before) == tokens(restored), ('tokens changed beyond include paths', old)
        checked[new] = {'source': old, 'before_sha256': digest(before_bytes),
                        'after_sha256': digest(after_bytes), 'token_sha256': digest(tokens(before).encode())}
    for old, new in moves.items():
        assert not (ROOT/old).exists() and (ROOT/new).is_file(), ('move incomplete', old, new)
    for old in ('src/ecm/edwards_stage1', 'src/ecm/montgomery_stage1',
                'src/ecm/application/cpu_stage2', 'src/ecm/application/cuda_stage2'):
        assert not (ROOT/old).exists(), ('old directory remains', old)
    from source_layout import current_source, frozen_source
    previous_layout = json.loads(subprocess.check_output(
        ['git', 'show', baseline+':tools/test/main_flow_layout.json'], cwd=ROOT))
    previous = {**previous_layout['moved'], **previous_layout.get('current_overrides', {})}
    affected_aliases = 0
    for alias, target in previous.items():
        expected = moves.get(target, target)
        assert current_source(alias) == ROOT/expected, ('alias changed identity', alias, expected)
        if target in moves:
            assert current_source(alias).is_file(), ('moved alias is unresolved', alias)
            affected_aliases += 1
    for old, new in moves.items():
        assert current_source(old) == ROOT/new, ('migration alias missing', old, new)
    # The OpenCL engine absorbed an entry TU in an earlier refactor. Reverse
    # lookup must retain the original implementation owner instead of choosing
    # between merged aliases. A newer frozen revision has the intermediate name.
    domain = json.loads((ROOT/'docs/development/ECM_DOMAIN_REFACTOR_MANIFEST.json').read_text(encoding='utf-8'))
    reverse_checks = [
        (domain['baseline'], 'src/ecm/stage1/montgomery/opencl/engine.cpp',
         'src/ecm/backend/opencl/stage1/montgomery/engine.cpp'),
        (baseline, 'src/ecm/stage1/edwards/cuda/runtime.cpp',
         'src/ecm/edwards_stage1/cuda/runtime.cpp'),
    ]
    for revision, current, original in reverse_checks:
        frozen = subprocess.check_output(['git', 'show', revision+':'+original], cwd=ROOT).decode('utf-8-sig')
        assert frozen_source(revision, current) == frozen, ('reverse frozen identity changed', revision, current)
    result = {'complete': True, 'baseline': baseline, 'moved_files': len(moves),
              'unchanged_computation_files': len(checked), 'include_edges': len(edges),
              'business_exceptions': 0, 'affected_historical_aliases': affected_aliases,
              'frozen_reverse_identity_checks': len(reverse_checks),
              'sources': checked, 'edges': edges}
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'edges')}))


if __name__ == '__main__':
    main()
