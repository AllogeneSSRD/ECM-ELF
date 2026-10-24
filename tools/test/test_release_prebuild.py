"""Run the real release entry through config generation, before GPU/toolchain work."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform=='win32','PowerShell release entry')
class PrebuildTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'build_release_contract_tests';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.repo=Path(self.temp.name)
        build=self.repo/'tools/build';(build/'internal').mkdir(parents=True)
        shutil.copy2(ROOT/'tools/build/build_release.ps1',build/'build_release.ps1')
        for name in ('release_profiles','gmp_dependency','stage1_release_sources','cuda_build_profile'):
            (build/'internal'/f'{name}.ps1').write_text('')
        (build/'internal/gmp_dependency.ps1').write_text(
            'function Resolve-EcmGmp([string]$Repo,[string]$Gmp,[switch]$Release) {\n'
            'Add-Content -LiteralPath (Join-Path $Repo "visits.txt") -Value dependency\n'
            'throw "fixture dependency boundary"\n}\n')
        gen=self.repo/'tools/gen';gen.mkdir()
        for name in ('generate_ecm_config','generate_gui_v2_state'):
            (gen/(name+'.py')).write_text(
                'from pathlib import Path\nimport sys\n'
                'repo=Path(__file__).resolve().parents[2]\n'
                'name=Path(__file__).stem\n'
                'with (repo/"visits.txt").open("a") as log:log.write(name+":"+str(sys.argv[1:])+"\\n")\n'
                'raise SystemExit(17 if (repo/(name+".fail")).exists() else 0)\n')

    def tearDown(self):self.temp.cleanup()
    def run_entry(self,*args):
        result=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(self.repo/'tools/build/build_release.ps1'),*args],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)  # Stop at the fixture dependency boundary.
        return (self.repo/'visits.txt').read_text(encoding='utf-8-sig').splitlines(),result.stdout+result.stderr

    def test_generation_order_before_dependency_resolution(self):
        visits,output=self.run_entry()
        self.assertEqual(visits,['generate_ecm_config:[]','generate_gui_v2_state:[]','dependency'])
        self.assertIn('synchronize generate_ecm_config.py',output)
        self.assertIn('synchronize generate_gui_v2_state.py',output)

    def test_skip_build_checks_without_generation(self):
        visits,output=self.run_entry('-SkipBuild')
        self.assertEqual(visits,["generate_ecm_config:['--check']","generate_gui_v2_state:['--check']",'dependency'])
        self.assertIn('check generate_ecm_config.py',output)

    def test_generator_failure_stops_before_other_work(self):
        for name in ('generate_ecm_config','generate_gui_v2_state'):
            with self.subTest(generator=name):
                visits_file=self.repo/'visits.txt';visits_file.unlink(missing_ok=True)
                marker=self.repo/(name+'.fail');marker.touch()
                visits,output=self.run_entry()
                self.assertNotIn('dependency',visits)
                self.assertTrue(visits[-1].startswith(name+':'))
                self.assertIn('Release configuration synchronize failed: '+name+'.py',output)
                marker.unlink()


class GeneratedWriteTests(unittest.TestCase):
    def test_gui_generation_preserves_unchanged_outputs(self):
        base=ROOT/'build_release_contract_tests';base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            repo=Path(directory);(repo/'tools/gen').mkdir(parents=True);(repo/'config').mkdir()
            script=repo/'tools/gen/generate_gui_v2_state.py'
            shutil.copy2(ROOT/'tools/gen/generate_gui_v2_state.py',script)
            schema=repo/'config/ecm_options.json';shutil.copy2(ROOT/'config/ecm_options.json',schema)
            def run():subprocess.run([sys.executable,str(script)],check=True,capture_output=True)
            run()
            before={p.relative_to(repo):p.stat().st_mtime_ns for p in repo.rglob('*') if p.is_file()}
            run();self.assertEqual(before,{p.relative_to(repo):p.stat().st_mtime_ns for p in repo.rglob('*') if p.is_file()})
            values=json.loads(schema.read_text(encoding='utf-8'))
            next(option for option in values['options'] if option['owner']!='gui')['description'][0]+=' changed'
            schema.write_text(json.dumps(values),encoding='utf-8');run()
            header=Path('src/ecm/application/gui_v2/generated/ui_values.h')
            self.assertEqual((repo/header).stat().st_mtime_ns,before[header])
            xml=repo/'src/ecm/application/gui_v2/localization/options.english.xml'
            self.assertIn(' changed',xml.read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
