"""Assemble and inspect complete, independently usable CUDA release bundles."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path
from gmp_payload import distribution

ROOT=Path(__file__).resolve().parents[2]
TARGETS={'cu11.8':('50','60'),'cu13.3':('75','80')}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def release_paths(out,version,profile):
    folder=out/f'ecm-elf-{version}-{profile}'
    archive=out/f'{folder.name}-win-x64.zip'
    return folder,archive,archive.with_suffix('.zip.sha256')


def assert_outputs_available(paths):
    for path in paths:
        if path.exists():raise ValueError(f'Release output already exists; preserve user configuration and select a fresh OutDir: {path}')


def write_payload_archive(folder,archive,files):
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as payload:
        for filename in files:payload.write(folder/filename,folder.name+'/'+filename)
    digest=sha(archive)
    archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
    return digest


def source_paths():
    # Respect repository ignores for application/tools: local test queues and
    # results can live below tools/test, even when they have source-like suffixes.
    roots=['src','include','config','tools/build','tools/test','tools/packaging','tools/gen']
    checkout=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT).decode('utf-8').strip()).resolve()
    if checkout!=ROOT.resolve():raise ValueError('Release packaging requires this source root to be a Git checkout; initialize the extracted source directory first')
    names=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard','--',*roots],cwd=ROOT).decode('utf-8').split('\0')
    paths={ROOT/name for name in names if name and (ROOT/name).is_file()}
    # These explicitly pinned vendor sources are intentionally Git-ignored.
    for base in ['third_party/cgbn','third_party/imgui','third_party/imgui_vulkan','third_party/gui_v2_vulkan','third_party/gui_v2_pet']:
        paths.update(path for path in (ROOT/base).rglob('*') if path.is_file())
    extensions={'.c','.cpp','.cu','.h','.hpp','.cuh','.inl','.def','.ps1','.py','.bat','.cmake','.txt','.json','.xml','.md','.rc','.manifest','.in','.patch','.rs'}
    notices={'LICENSE','COPYING','COPYRIGHT','NOTICE'}
    def allowed(path):
        parts=path.relative_to(ROOT).parts
        private_test=parts[:2]==('tools','test') and any(part.startswith('_') for part in parts[2:-1])
        return not private_test and '__pycache__' not in parts and (path.suffix.lower() in extensions or path.name in notices)
    return sorted((p for p in paths if allowed(p)),key=lambda p:p.as_posix())


def verify_inputs(values):
    for name,digest in values.items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if sha(path).lower()!=digest.lower():raise ValueError(f'Frozen source changed: {name}')


def images(cuda,binary,ptx,sass):
    inspector=cuda/'bin/cuobjdump.exe'
    result={}
    for kind,flag,expected,suffix in [('sass','--list-elf',sass,'cubin'),('ptx','--list-ptx',ptx,'ptx')]:
        value=subprocess.check_output([str(inspector),flag,str(binary)],stderr=subprocess.STDOUT).decode(errors='replace')
        found=set(re.findall(r'\.sm_(\d+)\.[^\s]*'+suffix,value))
        if found!={expected}:raise ValueError(f'{binary}: unexpected {kind} images {found}, expected {expected}')
        result[kind]=sorted(found)
    return result


def stage2_catalog():
    spec=importlib.util.spec_from_file_location('stage2_modules',ROOT/'tools/build/internal/generate_cuda_modules.py')
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    # Read the authoritative entry/instantiation inventory without generating
    # isolated sources for the default grouped build.
    return generator.generate(ROOT,emit=False)['modules']


def verify_stage2_membership(report,layout,catalog):
    modules=report['modules']
    if layout=='isolated':
        expected={m['label']:m['kernel'] for m in catalog}
        if len(modules)!=len(expected) or {m['label'] for m in modules}!=set(expected):raise ValueError('Incomplete isolated Stage2 module membership')
        for module in modules:
            if len(module['entries'])!=1 or expected[module['label']] not in module['entries'][0]:raise ValueError('Wrong isolated Stage2 entry')
    else:
        if len(modules)!=2 or {m['label'] for m in modules}!={'arithmetic_kernels','transform_kernels'}:raise ValueError('Wrong grouped Stage2 object membership')
        for module in modules:
            group='arithmetic' if module['label']=='arithmetic_kernels' else 'ntt'
            expected=Counter(m['kernel'] for m in catalog if m['label'].startswith(group+'_'))
            actual=Counter()
            for entry in module['entries']:
                # NVCC anonymous namespace hashes can end in digits directly
                # before the Itanium name-length field.
                matches=[name for name in expected if re.search(str(len(name))+re.escape(name)+r'(?:I|[A-Za-z_])',entry)]
                if len(matches)!=1:raise ValueError(f'Unknown grouped Stage2 entry: {entry}')
                actual[matches[0]]+=1
            if actual!=expected:raise ValueError(f'Incomplete grouped Stage2 entry membership: {module["label"]}')


def inspect_modules(cuda,build,stage,report):
    records=[]
    if stage=='stage1':
        units=read(build/'compile_commands.json')
        units=[u for u in units if '/cuda_modules/stage1/' in u['file'].replace('\\','/')]
        for unit in units:
            match=re.search(r'(edwards|mont[023])_(\d+)\.cu$',unit['file'])
            if not match:raise ValueError('Unexpected isolated Stage1 module')
            obj=re.search(r'(?:^|\s)(?:-o\s+|/Fo)(?:"([^"]+)"|(\S+))',unit['command'])
            if not obj:raise ValueError('Module object path missing')
            path=Path(obj[1] or obj[2]);path=path if path.is_absolute() else Path(unit['directory'])/path
            records.append(dict(label=Path(unit['file']).stem,object=str(path),curve=match[1],capacity=int(match[2])))
    else:
        manifest=read(build/'build_manifest.json')
        if manifest['module_directory']:
            modules=read(Path(manifest['module_directory'])/'modules.json')['modules']
            for module in modules:
                unit=manifest['compile_units'][module['source']]
                records.append(dict(label=module['label'],object=unit['entry']['output'],kernel=module['kernel'],arguments=module['arguments']))
        else:
            for source in manifest['cuda_sources']:
                unit=manifest['compile_units'][source]
                records.append(dict(label=Path(source).stem,object=unit['entry']['output']))
    inspector=cuda/'bin/cuobjdump.exe'
    for index,record in enumerate(records,1):
        if index==1 or index%10==0 or index==len(records):
            print(f'  Inspect {stage} PTX: {index}/{len(records)} {record["label"]}',flush=True)
        # Stream only entry declarations out of the potentially large PTX dump.
        proc=subprocess.Popen([str(inspector),'--dump-ptx',record['object']],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        entries=[];versions=set();targets=set()
        for raw in proc.stdout:
            line=raw.decode(errors='replace').strip()
            if '.entry ' in line:entries.append(line)
            if line.startswith('.version '):versions.add(line)
            if line.startswith('.target '):targets.add(line)
        errors=proc.stderr.read();code=proc.wait()
        if code:raise ValueError(f'Module inspection failed: {record["label"]}: {errors!r}')
        if not entries:raise ValueError(f'Isolated device module has no entry: {record["label"]}')
        if stage=='stage1':
            for entry in entries:
                if record['curve']=='edwards':
                    width=re.search(r'ParamsILj(\d+)ELj\d+E',entry)
                    valid=width and int(width[1])==record['capacity']
                else:
                    width=re.search(r'cgbn_params_tILj\d+ELj(\d+)E',entry)
                    param=int(record['curve'][-1])
                    valid=width and int(width[1])==record['capacity']
                    valid=valid and (('kernel_double_add_suyama' in entry and ('ELb1E' in entry)==(param==2)) if param in (0,2) else ('kernel_double_add' in entry and 'suyama' not in entry))
                if not valid:raise ValueError(f'Unexpected curve/parameter/width inside {record["label"]}: {entry}')
        elif 'kernel' in record and (len(entries)!=1 or record['kernel'] not in entries[0]):
            raise ValueError(f'Unexpected entry membership inside {record["label"]}: {entries}')
        record.update(entries=entries,ptx_versions=sorted(versions),ptx_targets=sorted(targets),sha256=sha(record['object']))
    report.write_text(json.dumps(dict(schema=1,stage=stage,modules=records),indent=2),encoding='utf-8')
    return sha(report)


def gui_identity(gui,gmp):
    identity=read(gui/'gui_v2_build_manifest.json')
    if not identity.get('base_pet_only'):raise ValueError('Complete release requires the base-pet-only GUI build')
    if not identity.get('gmp_root') or Path(identity['gmp_root']).resolve()!=gmp.resolve():raise ValueError('GUI GMP prefix differs')
    verify_inputs(identity['source_catalog_sha256'])
    if not identity.get('gmp_hashes'):raise ValueError('GUI build lacks coherent GMP identity')
    for filename,digest in identity['gmp_hashes'].items():
        if sha(gmp/filename).lower()!=digest.lower():raise ValueError(f'GUI GMP input changed: {filename}')
    for name,key in [('ecm_gui_v2.exe','exe_sha256'),('ecm_group_order_ui.exe','group_order_helper_sha256'),
                     ('ecm_diagnostics_host.exe','diagnostics_host_sha256'),('gmp-10.dll','group_order_runtime_sha256')]:
        if sha(gui/name)!=identity[key]:raise ValueError(f'GUI component changed: {name}')
    if sha(gui/'gmp-10.dll')!=sha(gmp/'bin/gmp-10.dll'):raise ValueError('GUI runtime DLL differs from vcpkg')
    cache=(gui/'CMakeCache.txt').read_text(encoding='utf-8-sig')
    for key,path in [('ECM_GROUP_ORDER_GMP_INCLUDE',gmp/'include'),('ECM_GROUP_ORDER_GMP_LIBRARY',gmp/'lib/gmp.lib')]:
        match=re.search(r'^'+key+r':[^=]+=(.+)$',cache,re.M)
        if not match or Path(match[1].strip()).resolve()!=path.resolve():raise ValueError(f'GUI GMP cache mismatch: {key}')
    return identity


def main(plan):
    build=Path(plan['build_root']).resolve();out=Path(plan['out']).resolve();gui=Path(plan['gui']).resolve();gmp=Path(plan['gmp']).resolve()
    print('== Package preflight: Stage2 entry catalogue ==',flush=True)
    catalog=stage2_catalog()
    gui_identity(gui,gmp)
    gmp_sources,gmp_licenses=distribution(gmp)
    version=re.search(r'version\[\]\s*=\s*"([0-9.]+)"',(ROOT/'include/ecm/platform/program_version.h').read_text())[1]
    outputs=[path for profile in plan['profiles'] for path in release_paths(out,version,profile)]
    assert_outputs_available(outputs)
    staging=build/'_packages'/uuid.uuid4().hex;staging.mkdir(parents=True)
    qualified=dict(schema=2,status='build_and_static_images_verified',tests_run=False,version=version,profiles={},gmp_sha256=sha(gmp/'bin/gmp-10.dll'))
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    # A source snapshot includes new/uncommitted implementation files too.
    source_archive=staging/'release-source.zip'
    with zipfile.ZipFile(source_archive,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths():archive.write(path,path.relative_to(ROOT).as_posix())
        for rel in ['.gitignore','CMakeLists.txt','LICENSE','docs/ECM_TOML_REFERENCE.md','tools/diag/ensure_bom.ps1','assets/gui_v2/pet_ely.png','assets/gui_v2/pet_ely.json','assets/gui_v2/CLASSIC-ELY-NOTICE.md','assets/gui_v2/ELYTerminal-LICENSE.txt']:
            archive.write(ROOT/rel,rel)
        for path,rel in gmp_sources:archive.write(path,rel)
    for profile in plan['profiles']:
        ptx,sass=TARGETS[profile];folder,archive,_=release_paths(staging,version,profile);folder.mkdir()
        component_records={}
        for stage,exe in [('stage1','ecm_cuda.exe'),('stage2','ecm_cuda_stage2.exe')]:
            record=read(build/'components'/f'{profile}.{stage}.json');leaf=Path(record['build']);manifest=read(record['manifest'])
            if manifest['role']!='release' or manifest['ptx_architecture']!=ptx:raise ValueError('Wrong release component profile')
            if stage=='stage1':
                if manifest['architectures']!=sass or manifest['tiers'] or manifest['enable_prac']:raise ValueError('Stage1 is not a full production release')
            else:
                if manifest['architecture']!='sm_'+sass or manifest['engine']!='production' or manifest['gl_fixed_mode']!=3 or manifest['add_sub_mask']!=1 or manifest['outer_unroll_u']!=0:
                    raise ValueError('Stage2 is not a production release')
                layout=plan.get('stage2_kernel_layout','grouped')
                if layout not in ('grouped','isolated') or manifest.get('kernel_layout')!=layout or bool(manifest.get('module_directory'))!=(layout=='isolated'):
                    raise ValueError('Stage2 kernel layout differs from release plan')
            verify_inputs(manifest['source_hashes'])
            if sha(leaf/'gmp-10.dll')!=sha(gmp/'bin/gmp-10.dll'):raise ValueError(f'{stage} runtime differs from vcpkg')
            binary=Path(record['exe']);expected=manifest.get('executable_sha256',manifest.get('sha256'))
            if sha(binary).lower()!=expected.lower():raise ValueError(f'Worker changed: {binary}')
            banner=subprocess.check_output([str(binary),'--version'],stderr=subprocess.STDOUT).decode(errors='replace')
            if version not in banner:raise ValueError(f'Worker version differs from release: {banner}')
            print(f'== Package {profile}/{stage}: CUDA images and module entries ==',flush=True)
            cuda=Path(manifest['cuda_root']);image=images(cuda,binary,ptx,sass)
            report=build/'components'/f'{profile}.{stage}.modules.json'
            image['module_report_sha256']=inspect_modules(cuda,leaf,stage,report)
            module_report=read(report)
            if stage=='stage1':
                rows=re.findall(r'^ECM_CUDA_TIER\((\d+),\d+,\d+,([01]),',(ROOT/'src/ecm/backend/cuda/dispatch/tiers.def').read_text(),re.M)
                expected_modules={f'edwards_{width}' for width,_ in rows}|{f'mont{param}_{width}' for width,mont in rows if mont=='1' for param in (0,2,3)}
                if {m['label'] for m in module_report['modules']}!=expected_modules:raise ValueError('Incomplete Stage1 module membership')
            else:verify_stage2_membership(module_report,layout,catalog)
            if any(m['ptx_targets']!=['.target sm_'+ptx] for m in module_report['modules']):raise ValueError('Unexpected module PTX target')
            copy(binary,folder/exe)
            component_records[stage]=dict(executable_sha256=sha(binary),manifest_sha256=sha(record['manifest']),images=image)
            if stage=='stage2':component_records[stage]['kernel_layout']=layout
        for filename in ['ecm_gui_v2.exe','ecm_group_order_ui.exe','gmp-10.dll']:
            copy(gui/filename,folder/filename)
        diagnostic=read(gui/'diagnostics_build_manifest.json')
        if diagnostic.get('gmp_prefix') and Path(diagnostic['gmp_prefix']).resolve()!=gmp:raise ValueError('Diagnostic host GMP differs')
        verify_inputs(diagnostic['sources_sha256'])
        for filename,digest in diagnostic['files_sha256'].items():
            if Path(filename).is_absolute() or '..' in Path(filename).parts:raise ValueError('Unsafe diagnostic payload')
            if sha(gui/filename)!=digest:raise ValueError(f'Diagnostic payload changed: {filename}')
            copy(gui/filename,folder/filename)
        for filename in ['pet_ely.png','pet_ely.json','CLASSIC-ELY-NOTICE.md','ELYTerminal-LICENSE.txt']:
            copy(ROOT/'assets/gui_v2'/filename,folder/'assets'/filename)
        shutil.copytree(gui/'localization/v2',folder/'localization/v2')
        copy(gmp/'bin/gmp-10.dll',folder/'gmp-10.dll')
        for filename in ['ecm.toml','ecm.toml.example']:copy(ROOT/'config/ecm.toml.example',folder/filename)
        (folder/'worktodo.txt').write_text('# Add ECM or ECMSTAGE2 tasks here.\n',encoding='utf-8')
        copy(ROOT/'docs/ECM_TOML_REFERENCE.md',folder/'ECM_TOML_REFERENCE.md')
        copy(ROOT/'LICENSE',folder/'LICENSE')
        for path,filename in gmp_licenses:copy(path,folder/'licenses'/filename)
        for source,filename in [('third_party/imgui/LICENSE.txt','ImGui-MIT.txt'),('third_party/gui_v2_pet/LICENSE','Ely-Pet-MIT.txt'),
                                ('third_party/gui_v2_vulkan/volk/LICENSE.md','Volk-MIT.txt'),('third_party/gui_v2_vulkan/LICENSE.md','Vulkan-Headers-LICENSE.md')]:
            copy(ROOT/source,folder/'licenses'/filename)
        copy(source_archive,folder/'source'/source_archive.name)
        files={path.relative_to(folder).as_posix():sha(path) for path in sorted(folder.rglob('*')) if path.is_file()}
        archive_sha=write_payload_archive(folder,archive,files)
        qualified['profiles'][profile]=dict(directory=folder.name,archive=archive.name,components=component_records,files_sha256=files,archive_sha256=archive_sha)
    source_archive.unlink()  # Scratch copy; the corresponding source is inside every CUDA directory.
    (build/'components/package-manifest.json').write_text(json.dumps(qualified,indent=2),encoding='utf-8')
    out.mkdir(parents=True,exist_ok=True)
    assert_outputs_available(outputs)
    for final in outputs:(staging/final.name).rename(final)
    staging.rmdir()
    print(f'Complete packages: {out} ({", ".join(plan["profiles"])})')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    main(read(parser.parse_args().plan))
