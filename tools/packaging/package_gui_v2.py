"""Produce an independent GUI-only candidate; never overwrite an output directory."""
import argparse,hashlib,json,re,shutil,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();exe=a.exe.resolve();out=a.out.resolve()
    assert not out.exists(),'Use a fresh package output directory'
    banner=subprocess.check_output([str(exe),'--version']).decode();match=re.search(r'\b\d+\.\d+\.\d+\b',banner);assert match,banner
    version=match[0];package=out/f'ECM-ELF-GUI-v2-{version}';package.mkdir(parents=True)
    shutil.copy2(exe,package/'ecm_gui_v2.exe');(package/'assets').mkdir();shutil.copy2(ROOT/'assets/gui_v2/pet_ely.png',package/'assets/pet_ely.png');shutil.copy2(ROOT/'assets/gui_v2/pet_ely.json',package/'assets/pet_ely.json')
    for name in ['CLASSIC-ELY-NOTICE.md','ELYTerminal-LICENSE.txt']:
        shutil.copy2(ROOT/'assets/gui_v2'/name,package/'assets'/name)
    skin_catalog=json.loads((ROOT/'assets/gui_v2/skins/manifest.json').read_text(encoding='utf-8'))
    for skin in skin_catalog['skins']:
        for source in [(ROOT/'assets/gui_v2'/skin['atlas']),(ROOT/'assets/gui_v2'/skin['atlas']).with_suffix('.json')]:shutil.copy2(source,package/'assets'/source.name)
    shutil.copy2(ROOT/'assets/gui_v2/skins/README.md',package/'assets/ELY-skins-NOTICE.md')
    build_identity=exe.with_name('gui_v2_build_manifest.json');identity=json.loads(build_identity.read_text(encoding='utf-8-sig'))
    assert identity['exe_sha256']==hashlib.sha256(exe.read_bytes()).hexdigest(),'Build manifest does not match executable'
    shutil.copy2(build_identity,package/'build-manifest.json')
    shutil.copytree(exe.parent/'localization/v2',package/'localization/v2')
    licenses=package/'licenses';licenses.mkdir()
    sources={
        'ECM-ELF-LICENSE.txt':'LICENSE',
        'ImGui-MIT.txt':'third_party/imgui/LICENSE.txt','Volk-MIT.txt':'third_party/gui_v2_vulkan/volk/LICENSE.md',
        'Vulkan-Headers-LICENSE.md':'third_party/gui_v2_vulkan/LICENSE.md','Vulkan-Headers-Apache-2.0.txt':'third_party/gui_v2_vulkan/LICENSES/Apache-2.0.txt',
        'Vulkan-Headers-MIT.txt':'third_party/gui_v2_vulkan/LICENSES/MIT.txt','Ely-Pet-MIT.txt':'third_party/gui_v2_pet/LICENSE',
        'Ely-Pet-NOTICE.md':'third_party/gui_v2_pet/README.md','Ely-Design-NOTICE.md':'src/ecm/application/gui_v2/ui/NOTICE.md',
        'Vulkan-provenance.json':'third_party/gui_v2_vulkan/provenance.json','Vulkan-NOTICE.md':'third_party/gui_v2_vulkan/README.md',
        'acquire-before-use.patch':'third_party/imgui_vulkan/acquire-before-use.patch'}
    for name,source in sources.items():shutil.copy2(ROOT/source,licenses/name)
    (package/'docs').mkdir();shutil.copy2(ROOT/'docs/usage/GUI_V2.md',package/'docs/GUI_V2.md');shutil.copy2(ROOT/'docs/GUI_V2_STATE_REFERENCE.md',package/'docs/GUI_V2_STATE_REFERENCE.md')
    shutil.copy2(ROOT/'config/ecm.toml.example',package/'ecm.toml.example')
    # Keep the standalone target rebuildable alongside the binary, without private
    # ignored workflow/evidence files or the large compute SDK dependencies.
    roots=['src/ecm/application/gui_v2','src/ecm/application/gui','src/ecm/application/configuration',
           'src/ecm/platform/generated','src/ecm/storage','include/ecm','third_party/imgui',
           'third_party/imgui_vulkan','third_party/gui_v2_vulkan','third_party/gui_v2_pet',
           'assets/gui_v2','tools/test/gui','tools/test/gui_v2','tools/gen','tools/build','config',
           'LICENSE','docs/ECM_TOML_REFERENCE.md','docs/GUI_V2_STATE_REFERENCE.md','docs/usage/GUI_V2.md']
    tracked=subprocess.check_output(['git','ls-files','-z','--',*roots],cwd=ROOT).decode().split('\0')
    selected={name for name in tracked if name}
    # This is a local working-tree snapshot, not an export of HEAD. Include the
    # exact frozen build inputs even when optional art is deliberately ignored.
    selected.update(identity['source_catalog_sha256'])
    for name,digest in identity['source_catalog_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,f'Build input changed after compilation: {name}'
    # The original pinned core is intentionally an ignored local dependency in
    # this repository; include only its public source/header/license files.
    for folder in [ROOT/'third_party/imgui',ROOT/'third_party/imgui/backends']:
        for path in folder.iterdir():
            if path.is_file() and (path.suffix in ['.h','.cpp'] or path.name=='LICENSE.txt'):
                selected.add(path.relative_to(ROOT).as_posix())
    selected.update(['src/ecm/application/gui_v2/app.manifest','tools/packaging/package_gui_v2.py',
                     'tools/build/package_gui_v2.ps1','tools/test/gui_v2/test_package.py','tools/test/gui_v2/test_failure_shutdown.py'])
    selected.update(['.gitattributes','config/gui_v2_pet_skins.generated.json',
                     'config/gui_v2_localization_aliases.json','config/gui_v2_localization.generated.json','tools/gen/generate_gui_v2_localization.py',
                     'tools/test/gui_v2/language_probe.py','tools/test/gui_v2/test_localization_files.py',
                     'assets/gui_v2/CLASSIC-ELY-NOTICE.md','assets/gui_v2/ELYTerminal-LICENSE.txt',
                     'tools/gen/generate_gui_v2_pet_skins.py','tools/gen/render_ely_skins_preview.py',
                     'tools/test/gui_v2/test_pet_skins.py','tools/test/gui_v2/test_skin_assets.cjs',
                     'tools/test/gui_v2/test_skin_preview.cjs','tools/test/gui_v2/test_startup_dispatch.py','tools/test/gui_v2/test_presentation.py','tools/test/gui_v2/test_font_exit.py','tools/test/gui_v2/test_font_offsets.py','tools/test/gui_v2/test_settings_window.py','tools/test/gui_v2/test_table_typography.py','tools/test/gui_v2/test_gpu_page.py','tools/test/gui_v2/test_navigation.py','docs/web/ely-skins-20261022.html',
                     'docs/development/ELY_SKINS_DRAFT.md'])
    source_dir=package/'source';source_dir.mkdir();source_archive=source_dir/f'ECM-ELF-GUI-v2-{version}-source.zip'
    with zipfile.ZipFile(source_archive,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(selected):archive.write(ROOT/name,'ECM-ELF-GUI-v2-source/'+name)
        archive.writestr('ECM-ELF-GUI-v2-source/SOURCE_IDENTITY.json',json.dumps(identity,indent=2))
    readme=f'''ECM-ELF GUI v2 {version} - independent Windows x64 candidate

Run ecm_gui_v2.exe and choose a workflow in Settings, or:
  ecm_gui_v2.exe --config "D:\\workflow\\ecm.toml"

Requirements: Windows 10/11 x64, Direct3D 11 driver (system WARP fallback),
Microsoft Visual C++ x64 runtime matching the MSVC build. No development SDK,
validation layer, Electron or browser runtime is required.

This GUI-only package contains no compute worker and never replaces ecm_gui.exe.
Install updated Stage1/Stage2 workers separately, with their original dependencies
and complete GPU tiers. Do NOT ship the small-tier acceptance workers as a release.
Workers must report ECM_QUEUE_OWNER=1 for --gui-queue-protocol. An unchanged old
GUI participates in mutual exclusion only when it uses those updated workers.

Pet is optional and off by default. Enable in Settings or pass --pet.
Choose ELY skin in Settings, or --pet --pet-skin miss_pink_elf,
--pet --pet-skin herrscher_of_human_ego, or --pet --pet-skin hi_love_elf.
Skin PNGs are external files in assets/ beside the executable, not embedded.
Copy or replace a registered atlas and restart; see docs/GUI_V2.md for format.
Additional unregistered names/IDs still require a rebuilt skin registry.
UI preferences are independent in LocalAppData; --ui-state overrides the path.
UI translations are external UTF-8 XML files under localization/v2/.
Edit english.xml/chineseSimplified.xml and use Language > Reload localization.
The options.*.xml files are generated from the configuration schema.
Use --diagnostics to choose a writable log path if the current directory is read-only.
Review ecm.toml.example before using it; it is not loaded automatically.

System fonts, Vulkan loader/driver and Microsoft runtime are not redistributed.
Licenses and exact dependency provenance are included under licenses/.
The source/ archive contains the standalone GUI target, matching dependencies,
asset source and build scripts. Extract it and run tools/build/build_gui_v2.bat
inside ECM-ELF-GUI-v2-source to rebuild (MSVC/CMake required; no compute SDK).
This local snapshot includes uncommitted work and deliberately untracked art;
the source_head field identifies only the base Git commit. Nothing is uploaded.
See docs/GUI_V2.md for controls and boundaries. Candidate qualification is local;
different physical DPI/real IME/controlled co-load/total VRAM and a separate clean
machine are not implied by a clean-directory smoke test.

DX11 is the default renderer and does not load a Vulkan runtime. This candidate's
bounded RTX 4070 Ti dual-screen drag/resize/minimize test had no user-observed
freeze. This is not a long-term or cross-device stability guarantee.
Use --renderer vulkan only for explicit diagnosis; that optional path requires
Vulkan 1.1 and has an unresolved desktop-image freeze during panel dragging.
Clean Vulkan validation and CPU timings do not qualify actual desktop refresh.
'''
    (package/'README.txt').write_text(readme,encoding='utf-8')
    files={p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*')) if p.is_file()}
    manifest=dict(kind='independent-gui-only-candidate',version=version,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),source_has_uncommitted_changes=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),includes_local_untracked_assets=True,files_sha256=files,worker_protocol_required=1,development_sdk_required=False,default_renderer='dx11',optional_renderer='vulkan',pet_skins=['classic']+[skin['id'] for skin in skin_catalog['skins']],system_requirements=['Windows x64','Direct3D 11 driver or system WARP','MSVC x64 runtime'],optional_vulkan_requirement='Vulkan 1.1 loader/graphics/present/swapchain; desktop freeze unresolved',contains_compute_workers=False)
    (package/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');archive=out/(package.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(package.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(out))
    result=dict(package=str(package),archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files)+1,exe_sha256=files['ecm_gui_v2.exe']);(out/'package-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
