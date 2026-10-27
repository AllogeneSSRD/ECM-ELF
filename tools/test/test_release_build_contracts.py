"""CPU regression gates for release profiles, module generation and phase timing."""
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
INTERNAL=ROOT/'tools/build/internal'


def module(name):
    spec=importlib.util.spec_from_file_location(name,INTERNAL/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


class Contracts(unittest.TestCase):
    def setUp(self):
        root=ROOT/'build_release_contract_tests';root.mkdir(exist_ok=True)
        (root/'.gitignore').write_text('*\n',encoding='utf-8')
        self.temporary=tempfile.TemporaryDirectory(dir=root);self.out=Path(self.temporary.name)
    def tearDown(self):self.temporary.cleanup()

    def test_definition_template_boundary(self):
        generator=module('generate_cuda_modules')
        text='template<int N>\nvoid first() {}\n\nvoid second() {}\n'
        start=text.index('void second')
        self.assertEqual(generator.prefix(text,start),start)
        self.assertEqual(generator.prefix(text,text.index('void first')),0)

    def test_module_membership_and_stable_generation(self):
        generator=module('generate_cuda_modules');generator.generate(self.out)
        inventory=json.loads((self.out/'modules.json').read_text())
        templates=sum(len(re.findall(r'^template void ',(ROOT/p).read_text(encoding='utf-8-sig'),re.M))
                      for p in ['src/ecm/stage2/cuda/device/arithmetic_kernels.cu',
                                'src/ecm/cuda/ntt/device/transform_kernels.cu'])
        self.assertEqual(sum(bool(v['arguments']) for v in inventory['modules']),templates)
        self.assertEqual(len({v['label'] for v in inventory['modules']}),len(inventory['modules']))
        before={p:p.stat().st_mtime_ns for p in self.out.rglob('*') if p.is_file()}
        generator.generate(self.out)
        self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})
        metadata=self.out/'metadata-only'
        catalog=generator.generate(metadata,emit=False)
        self.assertEqual([(v['label'],v['kernel'],v['arguments']) for v in catalog['modules']],
                         [(v['label'],v['kernel'],v['arguments']) for v in inventory['modules']])
        self.assertFalse(metadata.exists())
        for entry in inventory['modules']:
            text=Path(entry['source']).read_text(encoding='utf-8-sig')
            self.assertEqual(re.findall(r'^#define ECM_MODULE_(\w+)',text,re.M),[entry['kernel']])
            if entry['arguments']:self.assertIn('<'+entry['arguments']+'>(',text)
        chain=next(v for v in inventory['modules'] if v['label'].endswith('s2g_chain_kernel_4_true'))
        text=Path(chain['source']).read_text(encoding='utf-8-sig')
        self.assertLess(text.index('/point.cuh'),text.index('/giant_chain.cuh'))
        point=(self.out/'headers/stage2/cuda/device/point.cuh').read_text(encoding='utf-8-sig')
        self.assertIn('ecm_stage2_bind_module_constants();',point)
        pack=next(v for v in inventory['modules'] if v['label'].endswith('pack_zero_kernel_true_false'))
        self.assertIn('template void stage2_pack::ecm_launch_pack_zero_kernel<',Path(pack['source']).read_text(encoding='utf-8-sig'))

    def test_timing_launcher_failure_and_isolation(self):
        fake=self.out/'compiler.py'
        fake.write_text('import pathlib,sys\nargs=sys.argv[1:]\n'
                        'if "--missing" not in args:pathlib.Path(args[args.index("--time")+1]).write_text("phase,seconds\\ncompile,0.1\\n")\n'
                        'raise SystemExit(8 if "--fail" in args else 0)\n')
        def run(*args):
            return subprocess.run([sys.executable,str(INTERNAL/'timed_nvcc.py'),'--root',str(self.out),'--',sys.executable,str(fake),*args],capture_output=True)
        self.assertEqual(run('-arch=sm_60').returncode,0)
        self.assertEqual(run('-arch=sm_60').returncode,0)
        self.assertEqual(run('--fail').returncode,8)
        self.assertEqual(run('--missing').returncode,125)
        records=[json.loads(p.read_text()) for p in (self.out/'timing').glob('*/command.json')]
        self.assertEqual(len(records),4)
        self.assertEqual(len({v['csv'] for v in records}),4)
        self.assertEqual(sorted(v['status'] for v in records),['complete','complete','failed','timing_missing'])
        self.assertTrue(all('--time' not in v['semantic_command'] for v in records))

    @unittest.skipUnless(sys.platform=='win32','Windows compiler stdout handle contract')
    def test_compile_executor_uses_pipe_and_preserves_failure(self):
        executor=module('run_compile_jobs')
        compiler=self.out/'compiler_handle.py'
        compiler.write_text('import ctypes,ctypes.wintypes,sys\n'
            'kernel=ctypes.windll.kernel32\n'
            'kernel.GetStdHandle.restype=ctypes.wintypes.HANDLE\n'
            'kernel.GetFileType.argtypes=[ctypes.wintypes.HANDLE]\n'
            'kind=kernel.GetFileType(kernel.GetStdHandle(-11))\n'
            'print("compiler stdout type",kind,flush=True)\n'
            'raise SystemExit(int(sys.argv[1]) if kind==3 else 31)\n')
        batch=self.out/'compile.cmd';log=self.out/'compile.log'
        for expected in (0,8):
            batch.write_text(f'@echo off\n"{sys.executable}" "{compiler}" {expected}\n')
            result=executor.run(dict(source=str(compiler),command_file=str(batch),log=str(log)))
            self.assertEqual(result['exit_code'],expected,log.read_text(errors='replace'))
            self.assertIn('compiler stdout type 3',log.read_text())

    def test_stage1_tier_filter_preserves_single_modules(self):
        project=self.out/'CMakeLists.txt'
        project.write_text('cmake_minimum_required(VERSION 3.20)\nproject(modules NONE)\n'
            f'set(CMAKE_SOURCE_DIR "{ROOT.as_posix()}")\nset(CMAKE_CUDA_COMPILER_VERSION 13.3)\nset(ECM_TIERS "512,17408")\n'
            f'include("{INTERNAL.as_posix()}/cuda_modules.cmake")\necm_stage1_generate_modules()\n')
        result=subprocess.run(['cmake','-S',str(self.out),'-B',str(self.out/'build')],capture_output=True)
        self.assertEqual(result.returncode,0,(result.stdout+result.stderr).decode(errors='replace'))
        names={p.name for p in (self.out/'build/cuda_modules/stage1').glob('*.cu')}
        self.assertEqual(names,{'edwards_512.cu','mont0_512.cu','mont2_512.cu','mont3_512.cu','edwards_17408.cu'})

    def test_target_profiles_are_explicit(self):
        command=f". '{(INTERNAL/'release_profiles.ps1').as_posix()}'; @((Get-EcmReleaseTarget cu11.8),(Get-EcmReleaseTarget cu13.3)) | ConvertTo-Json"
        values=json.loads(subprocess.check_output(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-Command',command]))
        self.assertEqual([(v['Name'],v['Version'],v['Ptx'],v['Sass']) for v in values],
                         [('cu11.8','11.8','50','60'),('cu13.3','13.3','75','80')])
        self.assertEqual([v['SplitCompile'] for v in values],[1,8])
        command=f". '{(INTERNAL/'release_profiles.ps1').as_posix()}'; (Get-EcmReleaseTarget cu13.3 isolated).SplitCompile"
        self.assertEqual(subprocess.check_output(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-Command',command]).strip(),b'1')

    def test_source_archive_excludes_private_test_data(self):
        packaging=ROOT/'tools/packaging';sys.path.insert(0,str(packaging))
        import package_release
        subprocess.run(['git','init','-q',str(self.out)],check=True)
        (self.out/'.gitignore').write_text('/tools/test/_*\n/third_party/*\n')
        for name in ['tools/test/_run/worktodo.txt','tools/test/new_gate.py','third_party/cgbn/include/cgbn.h','src/ecm/opencl/resources/test.cl']:
            path=self.out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture\n')
        original=package_release.ROOT
        try:
            package_release.ROOT=self.out
            paths={p.relative_to(self.out).as_posix() for p in package_release.source_paths()}
            self.assertNotIn('tools/test/_run/worktodo.txt',paths)
            self.assertIn('tools/test/new_gate.py',paths)
            self.assertIn('third_party/cgbn/include/cgbn.h',paths)
            self.assertIn('src/ecm/opencl/resources/test.cl',paths)
            (self.out/'.gitignore').write_text('/third_party/*\n')
            paths={p.relative_to(self.out).as_posix() for p in package_release.source_paths()}
            self.assertNotIn('tools/test/_run/worktodo.txt',paths)
        finally:package_release.ROOT=original


if __name__=='__main__':unittest.main()
