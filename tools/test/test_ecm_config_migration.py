"""Check explicit ECM-ELF v1 configuration export and preview."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest


TOOL = Path(__file__).resolve().parents[1] / "refactor" / "migrate_ecm_config.py"
SPEC = importlib.util.spec_from_file_location("migrate_ecm_config", TOOL)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)
SCHEMA = json.loads(migration.SCHEMA.read_text(encoding="utf-8"))

OLD = r"""# file introduction
[shared]
worktodo = 'queue#live.txt' # relative to this TOML
priority = 1
skip = 2
[stage1]
device = 0
impl = "gwnum"
auto_stage2 = +1
auto_stage2_worktodo = ""
exp_cache = '..\cache'
ntt_warp_tail = "0"
# independent device
[stage2]
stage2_device = 1
stage2_submission_file = ""
stage2_parallel = "auto"
stage2_parallel_workspace = 2e0_0
stage2_opencl_short_point_batch = 0
stage2_factor_only = 1
[gui]
NumWorkers = 2
priority = "normal"
[workers.2.stage1]
device = 2
ntt_length = 8192
[workers.2.stage2] # worker 2 Stage2
stage2_device = 3
[workers.2]
name = "Blue \"GPU#2\""
autostart = -2
[workers.2.shared]
worktodo = "secondary.txt"
# trailing comment
"""


class ConfigMigration(unittest.TestCase):
    def test_migration_and_worker_precedence_inputs(self):
        new = migration.migrate(OLD, SCHEMA, "ecm.toml")
        data = tomllib.loads(new)
        self.assertEqual(data["config_version"], 2)
        self.assertEqual(data["shared"], {
            "worktodo_file": "queue#live.txt",
            "queue_order": "stage1_first",
            "queue_skip": "stage2",
        })
        self.assertEqual(data["stage1"], {
            "gpu_device": 0,
            "cpu_implementation": "gwnum",
            "handoff": {"enabled": True, "worktodo_file": ""},
            "exponent_cache_dir": r"..\cache",
            "ntt": {"warp_tail": False},
        })
        self.assertEqual(data["stage2"], {
            "gpu_device": 1,
            "submission_file": "",
            "cuda": {"curve_concurrency": "auto", "workspace_policy": "leased"},
            "opencl": {"short_point_batch": False},
            "diagnostics": {"skip_prime_attribution": True},
        })
        self.assertEqual(data["gui"], {"worker_count": 2, "process_priority": "normal"})
        self.assertEqual(data["worker"]["2"], {
            "stage1": {"gpu_device": 2, "ntt": {"length": 8192}},
            "stage2": {"gpu_device": 3},
            "shared": {"worktodo_file": "secondary.txt"},
            "name": 'Blue "GPU#2"',
            "autostart": True,
        })
        for comment in ("file introduction", "relative to this TOML", "independent device",
                        "worker 2 Stage2", "trailing comment"):
            self.assertIn("# " + comment, new)
        self.assertIn("worktodo_file = 'queue#live.txt' #", new)
        self.assertEqual(migration.migrate(new, SCHEMA, "ecm.v2.toml"), new)

    def test_empty_input_explicit_v1_and_bad_input(self):
        self.assertEqual(tomllib.loads(migration.migrate("", SCHEMA, "empty")), {"config_version": 2})
        self.assertEqual(migration.migrate("config_version = 1 # pin\n", SCHEMA, "old"),
                         "config_version = 2 # pin\n")
        self.assertEqual(migration.migrate("'config_version' = +1\n", SCHEMA, "old"),
                         "config_version = 2\n")
        aliases = migration.migrate('[stage2]\nstage2_parallel = " 0.2e1 "\n', SCHEMA, "old")
        self.assertEqual(tomllib.loads(aliases)["stage2"]["cuda"]["curve_concurrency"], "dual")
        cases = (
            ("config_version = \"1\"\n", "config_version"),
            ("config_version = 7\n", "config_version"),
            ("config_version = 1\nconfig_version = 2\n", "line 2"),
            ("[stage1]\nconfig_version = 1\n", "no v2 mapping"),
            ("[workers.1000001]\nautostart=1\n", "unknown v1 table"),
            ("[stage2]\nstage2_device=1\nfuture_unknown=2\n", "no v2 mapping"),
            ("[stage1]\nauto_stage2 = \"\"\n", "expected v1 type int"),
            ("[stage1]\nauto_stage2 = 2\n", "expected 0 or 1"),
            ("[stage2]\nstage2_parallel = \"0_2\"\n", "expected auto or an integer"),
            ("[stage2]\nstage2_parallel = -0\n", "expected auto or an integer"),
            ("[stage2]\nstage2_parallel = 1.5\n", "expected legacy mode"),
        )
        for source, message in cases:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, message):
                migration.migrate(source, SCHEMA, "ecm.toml")

    def test_collision_at_worker_scope(self):
        schema = {"options": [
            {"owner": "stage1", "key": key, "type": "int", "parser": "legacy", "v2_key": "same"}
            for key in ("old_a", "old_b")
        ]}
        with self.assertRaisesRegex(ValueError, "also defined on line 2"):
            migration.migrate("[workers.2.stage1]\nold_a=1\nold_b=2\n", schema, "ecm.toml")

    def test_cli_preview_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "ecm.toml"
            output = source.with_name("ecm.v2.toml")
            original = ("\ufeff" + OLD.replace("\n", "\r\n")).encode("utf-8")
            source.write_bytes(original)

            def run(*options):
                return subprocess.run([sys.executable, str(TOOL), str(source), '--to', '2', *options],
                                      capture_output=True, check=False, timeout=10)

            preview = run("--preview")
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertFalse(output.exists())
            result = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            exported = output.read_bytes()
            self.assertEqual(exported.replace(b"\r\n", b"\n"), preview.stdout)
            self.assertEqual(source.read_bytes(), original)
            self.assertNotEqual(run().returncode, 0)
            self.assertEqual(output.read_bytes(), exported)
            self.assertNotEqual(run("--output", str(source)).returncode, 0)
            moved = Path(tmp) / "elsewhere" / "ecm.v2.toml"
            self.assertIn(b"preserve relative paths", run("--output", str(moved)).stderr)
            self.assertEqual(source.read_bytes(), original)

    def test_v3_upgrade_requires_output_and_route_choices(self):
        source = '''config_version = 2
[shared]
b2_ratio = 15
[stage1]
curve = 'mont'
gpu_montgomery_param = 2
log_file = 'legacy1.log'
handoff.enabled = true
prime95.worktodo_file = 'other.txt'
[stage2]
log_file = 'legacy2.log'
tune.profile = 'stage2.toml'
[worker.2]
shared.b2_ratio = 30
stage1.target = 'cpu'
'''
        with self.assertRaisesRegex(ValueError, 'paths differ'):
            migration.upgrade_v3(source, SCHEMA, 'ecm.toml', stage1_directory=Path.cwd())
        with self.assertRaisesRegex(ValueError, 'both routes'):
            migration.upgrade_v3(source, SCHEMA, 'ecm.toml', log_file='ecm.log')
        with self.assertRaisesRegex(ValueError, 'different CUDA tune paths'):
            migration.upgrade_v3(source, SCHEMA, 'ecm.toml', log_file='ecm.log', handoff_mode='ecm-elf')
        converted = migration.upgrade_v3(source, SCHEMA, 'ecm.toml', log_file='ecm.log',
                                         handoff_mode='ecm-elf', tune_file='combined.toml')
        data = tomllib.loads(converted)
        self.assertEqual(data['config_version'], 3)
        self.assertEqual(data['shared'], {'log_file': 'ecm.log', 'tune_file': 'combined.toml'})
        self.assertEqual(data['stage1']['param'], 'm2')
        self.assertEqual(data['stage1']['handoff']['mode'], 'ecm-elf')
        self.assertEqual(data['stage2'], {'b2_ratio': 15})
        self.assertEqual(data['worker']['2']['stage1']['param'], 'm0')
        self.assertEqual(data['worker']['2']['stage2']['b2_ratio'], 30)
        self.assertEqual(migration.upgrade_v3(converted, SCHEMA, 'ecm.toml'), converted)

    def test_v3_coalesces_identical_tune_paths(self):
        source = '''config_version = 2
[stage1]
ntt.tune_file = 'combined.toml'
[stage2]
tune.profile = 'combined.toml'
'''
        shared = tomllib.loads(migration.upgrade_v3(source, SCHEMA, 'ecm.toml'))
        self.assertEqual(shared['shared']['tune_file'], 'combined.toml')
        self.assertNotIn('tune_file', shared['stage1'])
        self.assertNotIn('tune_file', shared['stage2'])
        source += "\n[worker.2]\nstage2.tune.profile = 'gpu1.toml'\n"
        with self.assertRaisesRegex(ValueError, 'worker 2 has different CUDA tune paths'):
            migration.upgrade_v3(source, SCHEMA, 'ecm.toml')
        source += "stage1.ntt.tune_file = 'gpu1.toml'\n"
        separate = tomllib.loads(migration.upgrade_v3(source, SCHEMA, 'ecm.toml'))
        self.assertEqual(separate['shared']['tune_file'], 'combined.toml')
        self.assertEqual(separate['worker']['2']['shared']['tune_file'], 'gpu1.toml')

    def test_v3_legacy_tune_empty_and_auto_are_distinct(self):
        source = "config_version=2\n[stage1]\nntt.tune_file=''\n[stage2]\ntune.profile=''\n"
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / 'ecm.toml'
            with self.assertRaisesRegex(ValueError, r"Stage1 '', Stage2 'tune.toml'"):
                migration.upgrade_v3(source, SCHEMA, path)
            path.write_text(source, encoding='utf-8')
            preview = subprocess.run([sys.executable, str(TOOL), str(path), '--preview', '--tune-file='],
                                     capture_output=True, check=True, timeout=10)
            self.assertEqual(tomllib.loads(preview.stdout.decode('utf-8'))['shared']['tune_file'], '')
            (base / 'stage2_ecm_tune.toml').touch()
            with self.assertRaisesRegex(ValueError, 'stage2_ecm_tune.toml'):
                migration.upgrade_v3('config_version=2\n', SCHEMA, path)

    def test_v3_normalizes_previous_staged_tune_keys(self):
        source = '''config_version = 3
[stage1]
tune_file = 'combined.toml' # formerly separate
param = 'm0'
[stage2]
tune_file = './combined.toml'
b2_ratio = 15
[stage2.opencl]
tune_file = 'opencl.toml'
[worker.2.stage2]
tune_file = 'gpu1.toml'
[worker.2.stage1]
tune_file = 'gpu1.toml'
'''
        text = migration.upgrade_v3(source, SCHEMA, 'ecm.toml')
        data = tomllib.loads(text)
        self.assertEqual(data['shared']['tune_file'], 'combined.toml')
        self.assertEqual(data['stage1'], {'param': 'm0'})
        self.assertEqual(data['stage2'], {'b2_ratio': 15, 'opencl': {'tune_file': 'opencl.toml'}})
        self.assertEqual(data['worker']['2'], {'shared': {'tune_file': 'gpu1.toml'}})
        self.assertIn('# formerly separate', text)
        self.assertEqual(migration.upgrade_v3(text, SCHEMA, 'ecm.toml'), text)


if __name__ == "__main__":
    unittest.main()
