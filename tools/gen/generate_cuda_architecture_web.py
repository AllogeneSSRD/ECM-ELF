"""Generate the offline CUDA architecture snapshot from the authoritative sources.

Editorial explanations live in docs/web/cuda-architecture.content.json; source members,
geometry, symbols, snippets and fingerprints are extracted, never hand copied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'web/architecture.data.js'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def build():
    content = json.loads((ROOT / 'docs/web/cuda-architecture.content.json').read_text(encoding='utf-8'))
    cache = ROOT / 'data/experiments/cuda_architecture_web'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='inventory-', dir=cache) as directory:
        assert Path(directory).resolve().is_relative_to(cache.resolve())
        inventory_path = Path(directory) / 'sources.json'
        subprocess.run(['cmake', f'-DECM_CUDA_SOURCES_OUTPUT={inventory_path.as_posix()}',
                        '-P', str(ROOT / 'tools/build/internal/cuda_sources.cmake')],
                       cwd=ROOT, check=True, capture_output=True)
        groups = json.loads(inventory_path.read_text())['groups']
    tier_source = (ROOT / 'src/ecm/backend/cuda/dispatch/tiers.def').read_text()
    tiers = [dict(bits=int(row[0]), tpi=int(row[1]), group=int(row[2]),
                  montgomery=int(row[3]), development=int(row[4]))
             for row in re.findall(r'^ECM_CUDA_TIER\((\d+),(\d+),(\d+),(\d+),(\d+),', tier_source, re.M)]
    capacities = [int(n) for n in re.findall(r'ROW\((\d+),FN,', tier_source)]
    assert len(tiers) == len({row['bits'] for row in tiers}) and tiers
    assert capacities == sorted(set(capacities)) and capacities
    module_ids = {module['id'] for module in content['modules']}
    assert len(module_ids) == len(content['modules'])
    refs = {}

    def add_reference(path, find=''):
        source = ROOT / path
        assert source.is_file(), f'Missing source: {path}'
        lines = source.read_text(encoding='utf-8-sig').splitlines()
        line = 1
        if find:
            matches = [i + 1 for i, text in enumerate(lines) if find in text]
            assert matches, f'Missing symbol: {path}: {find}'
            line = matches[0]
        start = max(1, line - 2)
        key = path + ':' + str(line)
        refs[key] = dict(path=path, line=line, start=start,
                         excerpt=lines[start - 1:min(len(lines), line + 13)])
        return key

    for module in content['modules']:
        assert set(module['depends']) <= module_ids, module['id']
        module['ref'] = add_reference(module['path'], module.get('find', ''))
    for tier in tiers:
        tier['ref'] = add_reference('src/ecm/backend/cuda/dispatch/tiers.def', f"ECM_CUDA_TIER({tier['bits']},")
    for flow in content['flows'].values():
        for step in flow['steps']:
            assert set(step['modules']) <= module_ids, step['title']
    for step in content['buildSteps']:
        add_reference(step[2])
    for change in content['changes']:
        for path in change['paths']:
            add_reference(path)
    for _, path, _ in content['reading']:
        assert (ROOT / path).is_file(), path

    catalog = []
    descriptions = {module['path']: module for module in content['modules']}
    for folder, zone in [('include/ecm', 'interface'), ('src/ecm', 'host'),
                         ('tools/experiments/cuda', 'experiment')]:
        for file in sorted((ROOT / folder).rglob('*')):
            if file.suffix not in {'.h', '.hpp', '.cuh', '.cu', '.cpp', '.inl', '.inc', '.def'}:
                continue
            path = file.relative_to(ROOT).as_posix()
            member = [name for name, sources in groups.items() if path in sources]
            module = descriptions.get(path)
            file_zone=zone
            if zone == 'host':
                file_zone='device' if file.suffix in ('.cu','.cuh') else 'host'
            catalog.append(dict(path=path, zone=file_zone, groups=member,
                                lines=len(file.read_text(encoding='utf-8-sig').splitlines()),
                                sha256=sha(file.read_bytes()), module=module['id'] if module else None,
                                ref=add_reference(path, module.get('find', '') if module else '')))
    for sources in groups.values():
        for path in sources:
            assert (ROOT / path).is_file(), path
            add_reference(path)
    # Include the referenced core/build files as well as the CUDA directory
    # trees. This is a browsing index, not a substitute for compiler depfiles.
    known = {file['path'] for file in catalog}
    for path in sorted({ref['path'] for ref in refs.values()} - known):
        file = ROOT / path
        module = descriptions.get(path)
        catalog.append(dict(path=path, zone='build' if path.startswith('tools/') or path == 'CMakeLists.txt' else 'core',
                            groups=[name for name, sources in groups.items() if path in sources],
                            lines=len(file.read_text(encoding='utf-8-sig').splitlines()),
                            sha256=sha(file.read_bytes()), module=module['id'] if module else None,
                            ref=add_reference(path, module.get('find', '') if module else '')))
    version = re.search(r'version\[\]\s*=\s*"([^"]+)"',
                        (ROOT / 'include/ecm/platform/program_version.h').read_text())[1]
    signature_paths = sorted({file['path'] for file in catalog} |
                             {ref['path'] for ref in refs.values()} |
                             {'tools/build/internal/cuda_sources.cmake', 'include/ecm/platform/program_version.h'})
    signature = sha('\n'.join(path + ':' + sha((ROOT / path).read_bytes()) for path in signature_paths).encode())
    revision = subprocess.check_output(['git', 'log', '-1', '--format=%h', '--',
        'src/ecm', 'include/ecm',
        'tools/build/internal/cuda_sources.cmake'], cwd=ROOT).decode().strip()
    data = dict(meta=dict(version=version, sourceRevision=revision, sourceSignature=signature,
                          sourceFiles=len(signature_paths), schema=1),
                groups=groups, tiers=tiers, capacities=capacities, catalog=catalog, refs=refs, **content)
    return '// Generated by tools/gen/generate_cuda_architecture_web.py; do not edit.\n' + \
        'window.CUDA_ARCHITECTURE = ' + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + ';\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    generated = build()
    if args.check:
        assert OUTPUT.is_file() and OUTPUT.read_text(encoding='utf-8') == generated, \
            'CUDA architecture snapshot is stale; run this generator without --check'
        print('PASS: CUDA web snapshot, source paths, symbols, groups and geometry are current')
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(generated, encoding='utf-8', newline='\n')
        print('Generated ' + str(OUTPUT))


if __name__ == '__main__':
    main()
