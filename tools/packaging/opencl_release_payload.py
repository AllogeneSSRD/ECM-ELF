"""Freeze the shared OpenCL release worker, resources and host dependencies."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import shutil
from cuda_inspection_cache import sha

ROOT=Path(__file__).resolve().parents[2]
RUNTIMES=('gmp-10.dll','libcrypto-3-x64.dll','libssl-3-x64.dll')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sources():
    # Conservative repository closure also detects added/deleted source files.
    roots=['src','include','config','tools/build/internal','tools/gen','CMakeLists.txt',
        'tools/build/build_opencl_local.ps1','tools/packaging/opencl_release_payload.py',
        'tools/packaging/gmp_payload.py','tools/packaging/cuda_inspection_cache.py','tools/packaging/diagnostics_runtime.py']
    names=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard','--',*roots],cwd=ROOT).decode('utf-8').split('\0')
    return {name:sha(ROOT/name) for name in sorted(set(names)) if name and (ROOT/name).is_file() and Path(name).suffix.lower()!='.md' and '__pycache__' not in Path(name).parts}


def compiler_inputs(build):
    # NMake/MSVC writes one UTF-8 header path per line, including SDK headers.
    # Freeze the actual native dependency files and their corresponding objects.
    paths=set()
    for depfile in (build/'CMakeFiles').rglob('*.obj.d'):
        obj=depfile.with_suffix('')
        if not obj.is_file():raise ValueError(f'OpenCL object missing: {obj}')
        paths.update((depfile.resolve(),obj.resolve()))
        paths.update(Path(line.strip()).resolve() for line in depfile.read_text(encoding='utf-8-sig').splitlines() if line.strip())
    if paths:paths.update((build/'CMakeCache.txt',build/'compile_commands.json'))
    return {str(path):sha(path) for path in sorted(paths)}


def snapshot_inputs(build):
    # Previously known headers must remain stable even if the new compiler
    # dependency set no longer contains them. Objects/depfiles are build outputs.
    headers={path:digest for path,digest in compiler_inputs(build).items() if not Path(path).is_relative_to(build.resolve())}
    return dict(sources=sources(),headers=headers)


def profile_identity(build):
    profile=read(build/'profile.json');fields=profile['fields']
    framed=''.join(str(len(str(v).encode('utf-16-le'))//2)+':'+str(v) for name in sorted(fields) for v in (name,fields[name]))
    if hashlib.sha256(framed.encode()).hexdigest()!=profile['profile_sha256']:raise ValueError('OpenCL profile digest mismatch')
    if fields['role']!='release' or fields['backend']!='opencl':raise ValueError('OpenCL worker requires a release profile')
    for name,digest in fields.items():
        path=Path(name.removeprefix('dependency/')) if name.startswith('dependency/') else None
        if name.startswith('gmp/'):path=Path(fields['gmp_root'])/name[4:]
        if path and sha(path).lower()!=digest.lower():raise ValueError(f'OpenCL dependency changed: {path}')
    for name,key in [('compiler','compiler_sha256'),('cmake','cmake_sha256')]:
        if sha(fields[name]).lower()!=fields[key].lower():raise ValueError(f'OpenCL tool changed: {name}')
    return profile


def freeze(build,snapshot):
    current=sources();before=read(snapshot)
    if current!=before['sources']:raise ValueError('OpenCL sources changed during build')
    for path,digest in before['headers'].items():
        if sha(path)!=digest:raise ValueError(f'OpenCL header changed during build: {path}')
    profile=profile_identity(build);fields=profile['fields']
    cache=(build/'CMakeCache.txt').read_text(encoding='utf-8-sig')
    if not re.search(r'^ECM_OPENCL_PORTABLE_RESOURCES:[^=]+=ON$',cache,re.M) or not re.search(r'^ECM_ENABLE_CUDA:[^=]+=OFF$',cache,re.M):raise ValueError('OpenCL release requires portable resources and CUDA disabled')
    commands=read(build/'compile_commands.json')
    if not commands or any(Path(unit['file']).suffix!='.cpp' or 'nvcc' in unit['command'].lower() for unit in commands):raise ValueError('OpenCL release contains CUDA compile commands')
    resources=ROOT/'src/ecm/opencl/resources'
    expected={p.relative_to(resources).as_posix():sha(p) for p in resources.rglob('*') if p.is_file()}
    actual={p.relative_to(build/'opencl_resources').as_posix():sha(p) for p in (build/'opencl_resources').rglob('*') if p.is_file()}
    if not expected or actual!=expected:raise ValueError('OpenCL resource deployment differs from source')
    payload={name:sha(build/name) for name in ('ecm.exe',*RUNTIMES)}
    payload.update({'opencl_resources/'+name:value for name,value in actual.items()})
    license=build/'licenses/OpenSSL.txt';license.parent.mkdir(exist_ok=True)
    shutil.copy2(Path(fields['openssl_root'])/'share/openssl/copyright',license)
    payload['licenses/OpenSSL.txt']=sha(license)
    for name in RUNTIMES:
        prefix=fields['gmp_root'] if name=='gmp-10.dll' else fields['openssl_root']
        if payload[name]!=sha(Path(prefix)/'bin'/name):raise ValueError(f'OpenCL runtime differs: {name}')
    native_inputs=compiler_inputs(build)
    if not native_inputs:raise ValueError('OpenCL compiler dependency/object records missing')
    identity=dict(schema=1,complete=True,role='release',backend='opencl',cuda_compiler_enabled=False,
        profile_sha256=profile['profile_sha256'],source_hashes=current,files_sha256=payload,resources=actual,
        compiler_inputs_sha256=native_inputs,executable_sha256=payload['ecm.exe'])
    (build/'build_manifest.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')


def verify(build,gmp):
    build=Path(build);identity=read(build/'build_manifest.json');profile=profile_identity(build)
    if identity.get('schema')!=1 or not identity.get('complete') or identity.get('role')!='release' or identity.get('backend')!='opencl' or identity.get('cuda_compiler_enabled') is not False:raise ValueError('Invalid OpenCL release manifest')
    if identity['source_hashes']!=sources() or identity['profile_sha256']!=profile['profile_sha256']:raise ValueError('OpenCL source/profile identity changed; rebuild the release worker')
    if not identity.get('compiler_inputs_sha256') or identity['compiler_inputs_sha256']!=compiler_inputs(build):raise ValueError('OpenCL compiler dependency/object identity changed')
    if Path(profile['fields']['gmp_root']).resolve()!=Path(gmp).resolve():raise ValueError('OpenCL GMP prefix differs from release')
    files=identity['files_sha256']
    if not {'ecm.exe',*RUNTIMES}<=files.keys() or not identity['resources']:raise ValueError('Incomplete OpenCL payload')
    resource_root=ROOT/'src/ecm/opencl/resources'
    resources={p.relative_to(resource_root).as_posix():sha(p) for p in resource_root.rglob('*') if p.is_file()}
    if identity['resources']!=resources or {name[17:]:digest for name,digest in files.items() if name.startswith('opencl_resources/')}!=resources:raise ValueError('Incomplete OpenCL resource membership')
    for name,digest in files.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe OpenCL payload path')
        if sha(build/name)!=digest:raise ValueError(f'OpenCL payload changed: {name}')
    if sha(build/'gmp-10.dll')!=sha(Path(gmp)/'bin/gmp-10.dll'):raise ValueError('OpenCL runtime GMP differs from release')
    return identity


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--verify',type=Path,metavar='GMP')
    args=parser.parse_args()
    if args.snapshot:args.snapshot.write_text(json.dumps(snapshot_inputs(args.build),indent=2),encoding='utf-8')
    elif args.freeze:freeze(args.build,args.freeze)
    elif args.verify:verify(args.build,args.verify);print('Verified OpenCL release worker and resources')
    else:parser.error('select --snapshot, --freeze or --verify')
