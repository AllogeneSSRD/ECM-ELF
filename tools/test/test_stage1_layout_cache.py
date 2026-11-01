"""Run the real local entry against an intact legacy grouped cache, without CUDA builds."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools.test.test_cuda_p5_release import profile_hash

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == 'win32', 'Windows Stage1 entry')
class Stage1LayoutCacheTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / 'data/experiments/stage1_layout_cache_tests'
        base.mkdir(parents=True, exist_ok=True)
        (base / '.gitignore').write_bytes(b'*\n')
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.repo = Path(self.temp.name)
        internal = self.repo / 'tools/build/internal'
        internal.mkdir(parents=True)
        for name in ('cuda_build_profile.ps1', 'build_directory.ps1'):
            shutil.copy2(ROOT / 'tools/build/internal' / name, internal / name)
        shutil.copy2(ROOT / 'tools/build/build_stage1_local.ps1', internal.parent / 'build_stage1_local.ps1')
        (internal / 'cuda_build_state.ps1').write_text('')
        (internal / 'gmp_dependency.ps1').write_text('''
function Resolve-EcmGmp {
    return [pscustomobject]@{Root=(Join-Path $env:ECM_LAYOUT_FIXTURE 'gmp');Hashes=[ordered]@{'include/gmp.h'='h';'lib/gmp.lib'='l';'bin/gmp-10.dll'='d'}}
}
''')
        (internal / 'cuda_toolchain.ps1').write_text('''
function Resolve-EcmCudaToolkit {
    return [pscustomobject]@{Root=(Join-Path $env:ECM_LAYOUT_FIXTURE 'cuda');Nvcc=(Join-Path $env:ECM_LAYOUT_FIXTURE 'nvcc.exe');VersionText='release 13.3, fixture'}
}
function Resolve-EcmVcEnvironment {
    return [pscustomobject]@{Compiler=(Join-Path $env:ECM_LAYOUT_FIXTURE 'cl.exe');VcVars=(Join-Path $env:ECM_LAYOUT_FIXTURE 'vcvars.bat');Version='14.51.36231';Setup=('call "'+$env:ECM_LAYOUT_FIXTURE+'/vcvars.bat" >nul 2>&1')}
}
function Assert-EcmCmakeToolchain {}
''')
        for name in ('nvcc.exe', 'cl.exe'):
            (self.repo / name).write_text('compiler identity fixture')
        (self.repo / 'vcvars.bat').write_bytes(b'@exit /b 0\r\n')
        diag = self.repo / 'tools/diag'
        diag.mkdir()
        (diag / 'ensure_bom.ps1').write_text('exit 0')
        binary = self.repo / 'bin'
        binary.mkdir()
        (binary / 'cmake.cmd').write_bytes(
            ('@"'+sys.executable+'" "'+str(self.repo/'cmake_fixture.py')+'" %*\r\n').encode())
        (self.repo / 'cmake_fixture.py').write_text('''
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
if '--version' in args: print('cmake version 4.3.2');sys.exit(0)
if '-DECM_STAGE1_TIERS_QUERY=ON' in args: print('ECM_STAGE1_TIERS=');sys.exit(0)
root=Path(os.environ['ECM_LAYOUT_FIXTURE']);build=Path(args[args.index('-B')+1])
isolated='-DECM_CUDA_ISOLATE_TIERS=ON' in args
with (root/'configure.jsonl').open('a') as out:out.write(json.dumps(dict(build=str(build),isolated=isolated))+'\\n')
print('-- ecm_cuda: no PTX fixture configure' if isolated else 'CMake Error: Stage1 grouped layout was removed')
sys.exit(17 if isolated else 13)  # Stop before any project compilation.
''')
        self.env = {**os.environ, 'ECM_LAYOUT_FIXTURE': str(self.repo), 'PATH': str(binary)+os.pathsep+os.environ['PATH']}
        # Use the modules matching the Windows PowerShell executable, even when
        # launched from a host bundling a reduced PowerShell module collection.
        inherited = self.env.pop('PSMODULEPATH', self.env.pop('PSModulePath', ''))
        modules = Path(os.environ['WINDIR']) / 'System32/WindowsPowerShell/v1.0/Modules'
        self.env['PSModulePath'] = str(modules)+os.pathsep+inherited
        self.shell = str(Path(os.environ['WINDIR']) / 'System32/WindowsPowerShell/v1.0/powershell.exe')
        bootstrap_output = self.run_entry()
        self.leaf = self.repo / 'build_stage1_local'
        marker = self.leaf / 'profile.json'
        self.assertTrue(marker.is_file(), bootstrap_output)
        legacy = json.loads(marker.read_text(encoding='utf-8-sig'))
        legacy['fields']['option/configure'] = legacy['fields']['option/configure'].replace(' -DECM_CUDA_ISOLATE_TIERS=ON', '')
        legacy['profile_sha256'] = profile_hash(legacy['fields'])
        marker.write_text(json.dumps(legacy), encoding='utf-8')
        (self.leaf / 'CMakeCache.txt').write_text('ECM_CUDA_ISOLATE_TIERS:BOOL=OFF\n')
        self.before = {name: (self.leaf/name).read_bytes() for name in ('profile.json', 'CMakeCache.txt')}
        (self.repo / 'configure.jsonl').unlink()

    def tearDown(self):
        self.temp.cleanup()

    def run_entry(self, *args):
        result = subprocess.run([self.shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                 str(self.repo/'tools/build/build_stage1_local.ps1'), *args],
                                env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)  # Intentional fixture configure boundary.
        return result.stdout+result.stderr

    def test_default_forces_generated_layout_and_preserves_grouped_cache(self):
        self.run_entry()
        request = json.loads((self.repo/'configure.jsonl').read_text())
        self.assertTrue(request['isolated'], 'CMake must receive an explicit ON independent of its old cache')
        self.assertNotEqual(Path(request['build']), self.leaf)
        marker = json.loads((Path(request['build'])/'profile.json').read_text(encoding='utf-8-sig'))
        self.assertIn('-DECM_CUDA_ISOLATE_TIERS=ON', marker['fields']['option/configure'])
        for name, content in self.before.items():
            self.assertEqual((self.leaf/name).read_bytes(), content)

    def test_explicit_grouped_cache_is_rejected_before_configure(self):
        output = self.run_entry('-BuildDir', str(self.leaf))
        self.assertFalse((self.repo/'configure.jsonl').exists())
        self.assertIn('option/configure', output)


if __name__ == '__main__':
    unittest.main()
