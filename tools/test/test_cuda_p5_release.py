"""Inspect all default CUDA release images and exact isolated ZIP payloads.

This is a static build/image/package gate. It never labels an architecture as
having passed native hardware or performance qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ARCHS = (60, 70, 75, 86, 89, 120)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def profile_hash(fields):
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in fields.items())
    # Match the .NET UTF-16 string lengths used by the build helper's framing.
    framed = ''.join(f'{len(v.encode("utf-16-le"))//2}:{v}'
                     for k in sorted(fields, key=lambda k: k.encode('utf-16-be')) for v in (k, fields[k]))
    return hashlib.sha256(framed.encode('utf-8')).hexdigest()


def release_leaf(root, stage, arch, explicit=None):
    """Use an explicit map or one compatible profile; never pick first/latest."""
    root = Path(root)
    toolkit = 'cuda12_6' if arch < 75 else 'cuda13_3'
    architectures = ('60,70' if arch < 75 else '75,86,89,120') if stage == 'stage1' else str(arch)
    arch_label = 'sm' + architectures.replace(',', '-sm')
    def check_marker(leaf):
        marker = load(leaf/'profile.json'); fields = marker['fields']
        assert marker['schema'] == 1 and profile_hash(fields) == marker['profile_sha256']
        assert fields['role'] == 'release' and fields['stage'] == stage and fields['engine'] == 'production'
        assert fields['architectures'] == architectures
        return fields
    if explicit is not None:
        candidates = [Path(explicit)]
    elif (root/'profile.json').exists():
        candidates = [root]
    else:
        markers = list(root.glob(f'{stage}/{toolkit}/{arch_label}/*/profile.json'))
        candidates = []
        for path in markers:
            fields = check_marker(path.parent)
            defaults = (fields.get('option/tiers') == '' and fields.get('option/enable_prac') == '0') if stage == 'stage1' else all(
                fields.get('option/'+k) == str(v) for k, v in
                dict(gl_backend='ptx', split_compile=8, add_sub_mask=1, outer_unroll_u=0, gl_fixed_mode=3).items())
            if defaults:
                candidates.append(path.parent)
        if not markers:
            # Frozen legacy directories retain their exact old layout.
            legacy = root/(toolkit if stage == 'stage1' else f'sm_{arch}')
            if not (legacy/'profile.json').exists():
                return legacy
            candidates = [legacy]
    assert len(candidates) == 1, 'Ambiguous release profiles; pass --build-leaves with explicit architecture-to-leaf paths'
    leaf = candidates[0]
    check_marker(leaf)
    return leaf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('stage1-build', 'stage1-packages', 'stage2-build', 'stage2-packages', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--archs', default=','.join(map(str, ARCHS)),
                   help='Completed architecture subset; defaults to the complete six-target matrix')
    p.add_argument('--build-leaves', type=Path,
                   help='Optional JSON: stage1/stage2 maps architecture strings to exact profile leaves')
    a = p.parse_args()
    leaf_map = load(a.build_leaves) if a.build_leaves else None
    def leaf(stage, arch, root):
        explicit = leaf_map[stage][str(arch)] if leaf_map is not None else None
        if explicit is not None and not Path(explicit).is_absolute():
            explicit = a.build_leaves.resolve().parent/explicit
        return release_leaf(root, stage, arch, explicit)
    archs = list(map(int, a.archs.split(',')))
    assert archs and len(archs) == len(set(archs)) and set(archs) <= set(ARCHS)
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    result = dict(complete=False, collector_sha256=sha(__file__), architectures=archs,
                  images=[], gpu_runs=0, scope='static source/object/image/package identities')
    def persist(): (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    def sources(manifest):
        for path, digest in manifest['source_hashes'].items():
            assert sha(ROOT/path) == digest.lower(), ('source changed', path)
    def host_supported(manifest):
        header = (Path(manifest['cuda_root'])/'include/crt/host_config.h').read_text()
        limits = re.search(r'#if\s+_MSC_VER\s*<\s*(\d+)\s*\|\|\s*_MSC_VER\s*>=\s*(\d+)', header)
        assert limits and manifest['host_version'].startswith('14.')
        msc = 1900+int(manifest['host_version'].split('.')[1])
        assert int(limits[1]) <= msc < int(limits[2]), 'unsupported CUDA host compiler'
    def stage1_defaults(group, arch):
        cache = dict(re.findall(r'^([^:#\n]+):[^=\n]+=(.*)$',
                               (group/'CMakeCache.txt').read_text(), re.M))
        expected = dict(ECM_CUDA_FULL_BUILD='ON', ECM_CUDA_ENABLE_PRAC='OFF',
            ECM_CUDA_COMPRESS='OFF', ECM_CUDA_EMBED_PTX='OFF', ECM_TPB='128',
            ECM_MAX_ROTATION='1', ECM_NO_PARAM2='0', ECM_MERS_FOLD='0',
            ECM_MAXRREG='0', ECM_MAXRREG_SMALL='0', ECM_MAXRREG_SUYAMA='0',
            ECM_REG_TARGET_FORCE='0', ECM_PROBE_ADD_DENSITY='1', ECM_PROBE_CHAIN_W='0',
            ECM_TIERS='', ECM_CUDA_PTX_ARCH='' if arch<75 else '75')
        assert all(cache.get(k)==v for k,v in expected.items()), ('non-default release profile', group)
    def inspect(stage, arch, exe, toolkit):
        dumps = {}
        for kind in ('elf', 'ptx'):
            text = subprocess.check_output([str(Path(toolkit)/'bin/cuobjdump.exe'), '--list-'+kind, str(exe)])
            log = out/f'{stage}_sm{arch}_{kind}.log'; log.write_bytes(text)
            dumps[kind] = text.decode('utf-8', 'replace')
        cubins = set(re.findall(r'\.sm_(\d+)\.cubin', dumps['elf']))
        ptx = set(re.findall(r'\.sm_(\d+)\.ptx', dumps['ptx']))
        assert cubins == {str(arch)}, (stage, arch, cubins)
        wanted_ptx = {str(arch)} if stage == 'stage2' or arch == 75 else set()
        assert ptx == wanted_ptx, (stage, arch, 'PTX policy', ptx)
        return dict(cubins=sorted(cubins), ptx=sorted(ptx),
                    cubin_entries=len(re.findall(r'\.cubin', dumps['elf'])),
                    ptx_entries=len(re.findall(r'\.ptx', dumps['ptx'])))
    def package(folder, manifest, exe):
        archive = Path(str(folder)+'.zip')
        assert Path(str(archive)+'.sha256').read_text().split()[0].lower() == sha(archive)
        expected = set(manifest['hashes']) | {'ecm.toml', 'worktodo.txt'}
        assert len(expected) == 8
        with zipfile.ZipFile(archive) as z:
            assert set(z.namelist()) == expected, ('archive payload', archive, z.namelist())
            for name, digest in manifest['hashes'].items():
                assert sha(folder/name) == digest.lower()
                assert hashlib.sha256(z.read(name)).hexdigest() == digest.lower()
            assert z.read('ecm.toml') == z.read('ecm.toml.example')
            assert z.read('worktodo.txt').decode('utf-8').startswith('# Add ')
            assert len(z.read('worktodo.txt').decode('utf-8').splitlines()) == 1
            assert hashlib.sha256(z.read(exe.name)).hexdigest() == sha(exe)
        return dict(zip=str(archive.resolve()), sha256=sha(archive), entries=sorted(expected))
    persist()
    for arch in archs:
        group = leaf('stage1', arch, a.stage1_build)
        path = group/'stage1_build_manifest.json'
        manifest = load(path); split = load(group/'split_manifest.json')
        if (group/'profile.json').exists():
            assert manifest['role'] == 'release' and manifest['profile_sha256'] == load(group/'profile.json')['profile_sha256']
        assert manifest['tiers'] == '' and not manifest['enable_prac']
        assert manifest['architectures'] == ('60,70' if arch < 75 else '75,86,89,120')
        stage1_defaults(group, arch)
        host_supported(manifest)
        assert manifest['executable_sha256'].lower() == sha(group/'ecm_cuda.exe')
        assert split['build_manifest_sha256'].lower() == sha(path)
        sources(manifest)
        for obj, digest in manifest['objects'].items(): assert sha(group/obj) == digest.lower()
        exe = a.stage1_packages/f'ecm_cuda_sm{arch}.exe'
        assert sha(exe) == split['executables'][exe.name].lower()
        payload = a.stage1_packages/f'cuda-stage1-sm{arch}'
        assert sha(payload/'ecm_cuda.exe') == sha(exe)
        row = dict(stage='stage1', architecture=arch, exe_sha256=sha(exe),
            manifest_sha256=sha(path), toolkit=manifest['cuda_root'], host_version=manifest['host_version'],
            images=inspect('stage1', arch, exe, manifest['cuda_root']),
            package=package(payload, load(group/f'package_sm{arch}_manifest.json'), payload/'ecm_cuda.exe'))
        result['images'].append(row); persist(); print('PASS static Stage1', arch, flush=True)
        group = leaf('stage2', arch, a.stage2_build)
        path = group/'build_manifest.json'; manifest = load(path)
        if (group/'profile.json').exists():
            assert manifest['role'] == 'release' and manifest['profile_sha256'] == load(group/'profile.json')['profile_sha256']
        assert manifest['architecture'] == f'sm_{arch}' and manifest['engine'] == 'production'
        assert manifest['gl_backend'] == 'ptx' and manifest['split_compile'] == 8
        assert manifest['add_sub_mask'] == 1 and manifest['outer_unroll_u'] == 0 and manifest['gl_fixed_mode'] == 3
        host_supported(manifest)
        sources(manifest)
        for unit in manifest['compile_units'].values():
            assert sha(unit['entry']['output']) == unit['object_sha256'].lower()
        payload = a.stage2_packages/f'cuda-stage2-sm{arch}'
        exe = payload/'ecm_cuda_stage2.exe'
        assert sha(exe) == manifest['sha256'].lower() == sha(group/exe.name)
        row = dict(stage='stage2', architecture=arch, exe_sha256=sha(exe),
            manifest_sha256=sha(path), toolkit=manifest['cuda_root'], host_version=manifest['host_version'],
            images=inspect('stage2', arch, exe, manifest['cuda_root']),
            package=package(payload, load(group/'package_manifest.json'), exe))
        result['images'].append(row); persist(); print('PASS static Stage2', arch, flush=True)
    result['complete'] = True; persist()


if __name__ == '__main__': main()
