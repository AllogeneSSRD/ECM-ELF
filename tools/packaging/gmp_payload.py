"""Corresponding GMP distribution materials, selected from the actual prefix."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def distribution(prefix):
    prefix = Path(prefix).resolve()
    spdx = prefix / 'share/gmp/vcpkg.spdx.json'
    if not spdx.exists():
        if prefix != (ROOT / 'third_party/gmp-zen3/dist').resolve():
            raise ValueError(f'Unknown GMP distribution: {prefix}')
        files = [(p, 'third_party/gmp-zen3/src/' + p.relative_to(ROOT / 'third_party/gmp-zen3/src').as_posix())
                 for p in sorted((ROOT / 'third_party/gmp-zen3/src').rglob('*')) if p.is_file()]
        files += [(ROOT / 'third_party/gmp-zen3/build' / name, 'third_party/gmp-zen3/build/' + name)
                  for name in ['config.status', 'config.h', 'config.m4'] if (ROOT / 'third_party/gmp-zen3/build' / name).is_file()]
        return files, [(ROOT / 'third_party/gmp-zen3/src' / n, 'GMP-' + n + '.txt') for n in ['COPYING.LESSERv3', 'COPYINGv3']]
    value = json.loads(spdx.read_text(encoding='utf-8'))
    if not value['name'].startswith('gmp:x64-windows@'):
        raise ValueError('Release GMP requires the vcpkg generic x64-windows triplet')
    vcpkg = prefix.parent.parent
    files = [(spdx, 'build-dependencies/gmp/vcpkg.spdx.json')]
    for package in value.get('packages', []):
        if package.get('name', '').startswith('gmp-') and package['name'].endswith('.tar.xz'):
            archive = vcpkg / 'downloads' / package['name']
            checksums = {c['algorithm']: c['checksumValue'].lower() for c in package['checksums']}
            if hashlib.sha512(archive.read_bytes()).hexdigest() != checksums['SHA512']:
                raise ValueError(f'GMP source archive identity mismatch: {archive}')
            files.append((archive, 'third_party/gmp/' + archive.name))
    if not any(name.endswith('.tar.xz') for _, name in files):
        raise ValueError('vcpkg GMP SPDX contains no matching source archive')
    # SPDX pins the actual port files used for this installation.
    port = vcpkg / 'ports/gmp'
    for entry in value.get('files', []):
        if not entry.get('SPDXID', '').startswith('SPDXRef-port-file-'):
            continue
        name = entry['fileName'].replace('\\', '/')
        if name.startswith('./'):
            name = name[2:]
        if '/' not in name:
            name = 'ports/gmp/' + name
        if name.startswith('ports/gmp/'):
            path = vcpkg / name
            for check in entry.get('checksums', []):
                algorithm = check['algorithm'].replace('-', '').lower()
                if hashlib.new(algorithm, path.read_bytes()).hexdigest() != check['checksumValue'].lower():
                    raise ValueError(f'vcpkg GMP port identity mismatch: {path}')
            files.append((path, 'third_party/vcpkg/' + name))
    if not port.is_dir():
        raise ValueError('vcpkg GMP build recipe missing')
    return files, [(prefix / 'share/gmp/copyright', 'GMP-COPYRIGHT.txt')]
