"""Exercise release assembly, ZIP roots and BAT dispatch without CUDA builds."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/packaging'))
import package_release as packaging


class ReleaseLayout(unittest.TestCase):
    def setUp(self):
        base=ROOT/'build_release_contract_tests';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='package space ',dir=base)
        self.root=Path(self.temp.name);self.repo=self.root/'repo'
        self.build=self.root/'build';self.out=self.root/'release'
        self.gui=self.root/'gui';self.gmp=self.root/'gmp'
        self.opencl=self.build/'opencl-release'
        self.cache_forces=[]

    def tearDown(self):self.temp.cleanup()

    def put(self,path,value='fixture\n'):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding='utf-8');return path

    def test_packaging_imports_real_stage2_catalog(self):
        # Exercise the production import/call boundary: assembly fixtures mock
        # this seam and cannot detect a reverted generator API.
        catalog=packaging.stage2_catalog()
        self.assertEqual(len(catalog),176)
        self.assertEqual(sum(m['label'].startswith('arithmetic_') for m in catalog),112)
        self.assertEqual(sum(m['label'].startswith('ntt_') for m in catalog),64)

    def test_catalog_failure_precedes_payload_work(self):
        plan=dict(build_root=str(self.build),out=str(self.out),gui=str(self.gui),gmp=str(self.gmp),profiles=['cu11.8'])
        with patch.object(packaging,'stage2_catalog',side_effect=RuntimeError('catalog unavailable')),\
             patch.object(packaging,'gui_identity') as gui:
            with self.assertRaisesRegex(RuntimeError,'catalog unavailable'):packaging.main(plan)
            gui.assert_not_called()

    def fixture(self,layout='grouped'):
        self.layout=layout
        self.put(self.repo/'include/ecm/platform/program_version.h','inline constexpr char version[] = "0.52.1";')
        self.put(self.repo/'src/ecm/backend/cuda/dispatch/tiers.def','ECM_CUDA_TIER(512,4,128,1,0)\n')
        for name in ['.gitignore','CMakeLists.txt','LICENSE','docs/ECM_TOML_REFERENCE.md','tools/diag/ensure_bom.ps1',
                     'config/ecm.toml.example','assets/gui_v2/pet_ely.png','assets/gui_v2/pet_ely.json',
                     'assets/gui_v2/CLASSIC-ELY-NOTICE.md','assets/gui_v2/ELYTerminal-LICENSE.txt',
                     'third_party/imgui/LICENSE.txt','third_party/gui_v2_pet/LICENSE',
                     'third_party/gui_v2_vulkan/volk/LICENSE.md','third_party/gui_v2_vulkan/LICENSE.md',
                     'tools/build/build_release.ps1']:
            self.put(self.repo/name)
        self.put(self.gmp/'bin/gmp-10.dll')
        for name in ['ecm.exe','gmp-10.dll','libcrypto-3-x64.dll','libssl-3-x64.dll','opencl_resources/test.cl','licenses/OpenSSL.txt']:
            self.put(self.opencl/name)
        self.opencl_files={p.relative_to(self.opencl).as_posix():packaging.sha(p) for p in self.opencl.rglob('*') if p.is_file()}
        self.put(self.opencl/'build_manifest.json','{}')
        for name in ['ecm_gui_v2.exe','ecm_group_order_ui.exe','gmp-10.dll','ecm_diagnostics_host.exe','localization/v2/en.xml']:
            self.put(self.gui/name)
        self.put(self.gui/'diagnostics_build_manifest.json',json.dumps(dict(sources_sha256={},files_sha256={'ecm_diagnostics_host.exe':packaging.sha(self.gui/'ecm_diagnostics_host.exe')})))
        for profile,(ptx,sass) in packaging.TARGETS.items():
            for stage,exe in [('stage1','ecm_cuda.exe'),('stage2','ecm_cuda_stage2.exe')]:
                leaf=self.build/profile/stage;binary=self.put(leaf/exe);self.put(leaf/'gmp-10.dll')
                manifest=dict(role='release',ptx_architecture=ptx,source_hashes={},cuda_root=str(self.root/'cuda'),sha256=packaging.sha(binary))
                if stage=='stage1':manifest.update(architectures=sass,tiers='',enable_prac=False)
                else:manifest.update(architecture='sm_'+sass,engine='production',gl_fixed_mode=3,add_sub_mask=1,outer_unroll_u=0,kernel_layout=layout,module_directory=('isolated' if layout=='isolated' else ''))
                path=self.put(leaf/'manifest.json',json.dumps(manifest))
                self.put(self.build/'components'/f'{profile}.{stage}.json',json.dumps(dict(build=str(leaf),manifest=str(path),exe=str(binary))))

    def inspect(self,cuda,leaf,stage,report,cache=None):
        self.cache_forces.append(cache.force)
        ptx=packaging.TARGETS[leaf.parent.name][0]
        if stage=='stage1':
            modules=[dict(label=label) for label in ['edwards_512','mont0_512','mont2_512','mont3_512']]
        elif self.layout=='isolated':
            modules=[dict(label=m['label'],entries=[f'.entry _Z{len(m["kernel"])}{m["kernel"]}ILi4EEv()']) for m in self.catalog()]
        else:
            modules=[dict(label=label,entries=[f'.entry _Z{len(m["kernel"])}{m["kernel"]}ILi4EEv()' for m in self.catalog() if m['label'].startswith(group+'_')])
                     for label,group in [('arithmetic_kernels','arithmetic'),('transform_kernels','ntt')]]
        for module in modules:module['ptx_targets']=['.target sm_'+ptx]
        self.put(report,json.dumps(dict(modules=modules)))
        return packaging.sha(report)

    def catalog(self):
        return [dict(label='arithmetic_test_4',kernel='s2g_test_kernel'),dict(label='ntt_test_4',kernel='tile_kernel')]

    def assemble(self,profiles,reinspect=False):
        plan=dict(build_root=str(self.build),out=str(self.out),gui=str(self.gui),gmp=str(self.gmp),profiles=profiles,stage2_kernel_layout=self.layout,reinspect_cuda=reinspect)
        with patch.object(packaging,'ROOT',self.repo),patch.object(packaging,'gui_identity'),\
             patch.object(packaging,'verify_opencl',return_value=dict(files_sha256=self.opencl_files)),\
             patch.object(packaging,'distribution',return_value=([],[])),\
             patch.object(packaging,'source_paths',return_value=[self.repo/'tools/build/build_release.ps1']),\
             patch.object(packaging,'images',return_value={}),\
             patch.object(packaging,'stage2_catalog',side_effect=self.catalog),\
             patch.object(packaging,'inspect_modules',side_effect=self.inspect),\
             patch.object(packaging.subprocess,'check_output',return_value=b'ECM-ELF 0.52.1'):
            packaging.main(plan)
        report=self.root/('report-'+ '-'.join(profiles))
        result=subprocess.run([sys.executable,str(ROOT/'tools/test/test_release_bundle.py'),
            '--release',str(self.out),'--manifest',str(self.build/'components/package-manifest.json'),
            '--output',str(report),'--static-only'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertTrue(json.loads((report/'result.json').read_text())['complete'])

    def test_complete_bundles_and_extracted_directory_names(self):
        self.fixture();self.assemble(['cu11.8','cu13.3'])
        self.assertFalse((self.out/'ecm-elf-0.52.1').exists())
        for profile in packaging.TARGETS:
            folder,archive,checksum=packaging.release_paths(self.out,'0.52.1',profile)
            self.assertTrue(folder.is_dir());self.assertTrue(checksum.is_file())
            extracted=self.root/('unzip-'+profile)
            with zipfile.ZipFile(archive) as payload:payload.extractall(extracted)
            self.assertEqual([p.name for p in extracted.iterdir()],[folder.name])
            self.assertTrue((extracted/folder.name/'ecm_gui_v2.exe').is_file())
            for name in self.opencl_files:self.assertEqual(packaging.sha(extracted/folder.name/name),self.opencl_files[name])
            before=packaging.sha(archive)
            with self.assertRaisesRegex(ValueError,'already exists'):self.assemble([profile])
            self.assertEqual(packaging.sha(archive),before)

    def test_profiles_can_be_published_separately(self):
        self.fixture();self.assemble(['cu11.8'])
        first=packaging.release_paths(self.out,'0.52.1','cu11.8')[1];digest=packaging.sha(first)
        self.assemble(['cu13.3']);self.assertEqual(packaging.sha(first),digest)

    def test_isolated_stage2_remains_available(self):
        self.fixture('isolated');self.assemble(['cu11.8','cu13.3'])

    def test_force_inspection_option_reaches_each_component(self):
        self.fixture();self.assemble(['cu11.8'],reinspect=True)
        self.assertEqual(self.cache_forces,[True,True])

    def test_grouped_membership_rejects_wrong_or_missing_entries(self):
        report=dict(modules=[dict(label='arithmetic_kernels',entries=['.entry _Z15s2g_test_kernelILi4EEv()']),
                             dict(label='transform_kernels',entries=['.entry _ZN53_GLOBAL__N__hash12311tile_kernelILi4EEv()'])])
        packaging.verify_stage2_membership(report,'grouped',self.catalog())
        report['modules'][1]['entries']=[]
        with self.assertRaisesRegex(ValueError,'entry membership'):packaging.verify_stage2_membership(report,'grouped',self.catalog())
        report['modules'][1]['entries']=['.entry _Z12wrong_kernelILi4EEv()']
        with self.assertRaisesRegex(ValueError,'Unknown'):packaging.verify_stage2_membership(report,'grouped',self.catalog())

    def test_existing_folder_zip_or_checksum_is_preserved(self):
        paths=packaging.release_paths(self.out,'0.52.1','cu11.8')
        for path in paths:
            with self.subTest(path=path.name):
                self.put(path,'user data')
                with self.assertRaisesRegex(ValueError,'already exists'):packaging.assert_outputs_available(paths)
                self.assertEqual(path.read_text(),'user data');path.unlink()

    @unittest.skipUnless(sys.platform=='win32','Windows BAT entry points')
    def test_bat_dispatch_profile_jobs_arguments_and_exit_code(self):
        self.put(self.root/'build_release.ps1',
            'param([string]$Profile,[int]$Jobs,[string]$ResultFile)\n'
            '@{profile=$Profile;jobs=$Jobs} | ConvertTo-Json | Set-Content -LiteralPath $ResultFile\nexit 23\n')
        for profile in packaging.TARGETS:
            with self.subTest(profile=profile):
                wrapper=ROOT/'tools/build'/f'build_release_{profile}.bat'
                target=self.root/wrapper.name;shutil.copy2(wrapper,target)
                result_file=self.root/f'{profile}.json'
                result=subprocess.run(f'cmd /d /c ""{target}" -ResultFile "{result_file}""',capture_output=True,text=True)
                self.assertEqual(result.returncode,23,result.stdout+result.stderr)
                self.assertEqual(json.loads(result_file.read_text(encoding='utf-8-sig')),dict(profile=profile,jobs=16))


if __name__=='__main__':unittest.main()
