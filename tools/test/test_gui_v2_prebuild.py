"""Exercise GUI v2 config synchronization through the actual PowerShell entry."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == 'win32', 'Windows GUI build entry')
class GuiPrebuildTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / 'data/experiments/gui_v2_prebuild_20261031/fixtures'
        base.mkdir(parents=True, exist_ok=True)
        (base / '.gitignore').write_bytes(b'*\n')
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.repo = Path(self.temp.name)
        build = self.repo / 'tools/build'
        (build / 'test').mkdir(parents=True)
        (build / 'internal').mkdir()
        shutil.copy2(ROOT / 'tools/build/test/build_gui_v2.ps1', build / 'test/build_gui_v2.ps1')
        (build / 'internal/gmp_dependency.ps1').write_text(
            'function Resolve-EcmGmp([string]$Repo,[string]$Gmp,[switch]$Release) {\n'
            'Add-Content -LiteralPath (Join-Path $Repo "visits.txt") -Value dependency\n'
            'throw "fixture dependency boundary"\n}\n', encoding='utf-8')
        gen = self.repo / 'tools/gen'
        gen.mkdir()
        for name in ('generate_ecm_config', 'generate_gui_v2_state'):
            (gen / (name + '.py')).write_text(
                'from pathlib import Path\nimport sys\n'
                'repo=Path(__file__).resolve().parents[2]\n'
                'name=Path(__file__).stem\n'
                'with (repo/"visits.txt").open("a") as log:log.write(name+"\\n")\n'
                'raise SystemExit(17 if (repo/(name+".fail")).exists() else 0)\n', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def run_entry(self, cwd):
        target = self.repo / 'build with spaces'
        result = subprocess.run([
            'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
            str(self.repo / 'tools/build/test/build_gui_v2.ps1'),
            '-BuildDir', str(target), '-GmpRoot', str(self.repo / 'gmp'),
        ], cwd=cwd, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)  # Fixture stops before dependency/compiler work.
        self.assertFalse(target.exists())
        return (self.repo / 'visits.txt').read_text(encoding='utf-8-sig').splitlines(), result.stdout + result.stderr

    def test_sync_precedes_dependency_resolution_and_source_freeze(self):
        visits, output = self.run_entry(self.repo)
        self.assertEqual(visits, ['generate_ecm_config', 'generate_gui_v2_state', 'dependency'])
        self.assertIn('synchronize generate_ecm_config.py', output)
        self.assertIn('synchronize generate_gui_v2_state.py', output)

    def test_sync_uses_repository_paths_from_build_working_directory(self):
        visits, _ = self.run_entry(self.repo / 'tools/build')
        self.assertEqual(visits, ['generate_ecm_config', 'generate_gui_v2_state', 'dependency'])

    def test_generator_failure_stops_remaining_build_work(self):
        for index, name in enumerate(('generate_ecm_config', 'generate_gui_v2_state')):
            with self.subTest(generator=name):
                (self.repo / 'visits.txt').unlink(missing_ok=True)
                failure = self.repo / (name + '.fail')
                failure.touch()
                visits, output = self.run_entry(self.repo)
                self.assertEqual(visits, ['generate_ecm_config', 'generate_gui_v2_state'][:index + 1])
                self.assertIn('GUI v2 configuration synchronization failed: ' + name + '.py', output)
                failure.unlink()


if __name__ == '__main__':
    unittest.main()
