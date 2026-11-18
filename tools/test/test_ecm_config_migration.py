"""Exercise offline v3 to v4 export with flat worker keys and optional GUI preferences."""

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

V3 = r'''# workflow
config_version = 3 # explicitly selected

[shared]
worker_count = 3
worktodo_file = "queue.txt" # main queue
tune_file = ""

[stage1]
gpu_device = 0
gpu_backend = "cuda"
handoff.mode = "ecm-elf"
cpu_threads = 4
cpu.threads = 2
# ntt.allow_known_factors = true

[stage2]
device = "gpu"
gpu_device = 1
[stage2.cuda] # tuning
weighted_ntt = true

[gui]
worker_count = 2
stage2_exe = "ecm_stage2.exe"
pause_schedule = [
    "foobar, 3, Mon-Fri, 08:30-17:30", # preserve
]

[worker.3]
shared.tune_file = "worker3.toml"
stage2.device = "cPu"
stage2.cpu.threads = 3
[worker.3.stage1.ntt]
length = 1024
# end of file
'''

SIDECAR = r'''[state]
schema = 1
[ui]
theme = "dark"
language = "zh"
font_size = 18
dock = "layout"
worker = 3
stage2_tune_file = "timing.toml"
[window]
width = 1680
[pet]
enabled = false
[worker]
count = 1
stage1_exe = "ecm_cuda.exe"
stage2_exe = "missing.exe"
'''


class Migration(unittest.TestCase):
    def migrate(self, source, *, gui_state=None):
        return migration.migrate(source, SCHEMA, "ecm.toml", gui_state=gui_state)

    def test_business_and_gui_settings(self):
        output = self.migrate(V3, gui_state=("ui.toml", SIDECAR))
        parsed = tomllib.loads(output)
        self.assertEqual(parsed["config_version"], 4)
        self.assertEqual(parsed["shared"], {
            "worker_count": 3, "worktodo_file": "queue.txt", "tune_file": "",
        })
        self.assertEqual(parsed["stage1"], {
            "stage1_gpu_device": 0, "stage1_gpu_backend": "cuda", "handoff_mode": "ecm-elf",
            "stage1_cpu_threads": 2, "gpu_init_threads": 4,
        })
        self.assertEqual(parsed["stage2"], {"device": "gpu", "stage2_gpu_device": 1, "weighted_ntt": True})
        self.assertEqual(parsed["worker"]["3"], {
            "tune_file": "worker3.toml", "device": "cPu", "stage2_cpu_threads": 3, "ntt_length": 1024,
        })
        self.assertEqual(parsed["gui"], {
            "stage1_exe": "ecm_cuda.exe", "stage2_exe": "ecm_stage2.exe",
            "pause_schedule": ["foobar, 3, Mon-Fri, 08:30-17:30"], "theme": "dark",
            "display_language": "zh", "body_font_size": 18, "workspace_layout": "layout",
            "selected_worker": 3, "stage2_timing_profile": "timing.toml", "window_width": 1680,
            "pet_enabled": False,
        })
        for comment in ("# workflow", "# explicitly selected", "# main queue", "# tuning",
                        "# preserve", "# end of file", "# ntt_allow_known_factors = true"):
            self.assertIn(comment, output)
        self.assertEqual(self.migrate(output), output)

    def test_legacy_folds_follow_specific_values(self):
        source = """config_version=3
[stage1]
cpu_affinity='0-3'
cpu_threads=8
[shared]
cpu_affinity='4'
[worker.2]
stage1.cpu_affinity='0-1'
stage1.cpu_threads=4
stage1.gpu.init_threads=12
"""
        parsed = tomllib.loads(self.migrate(source))
        self.assertEqual(parsed["shared"], {"cpu_affinity": "4"})
        self.assertEqual(parsed["stage1"], {"stage1_cpu_threads": 8, "gpu_init_threads": 8})
        self.assertEqual(parsed["worker"]["2"], {
            "cpu_affinity": "0-1", "stage1_cpu_threads": 4, "gpu_init_threads": 12,
        })

    def test_invalid_input(self):
        for source in ("", "config_version=1", "config_version=2", "config_version=5",
                       'config_version="3"', "config_version=3\n[stage2]\nunknown=true\n"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.migrate(source)
        with self.assertRaisesRegex(ValueError, "unknown GUI preference ui.typo"):
            self.migrate(V3, gui_state=("ui.toml", "[ui]\ntypo=1"))

    def test_quoted_v3_names(self):
        source = '''"config_version" = +3
["stage2"]
"cuda" . "weighted_ntt" = true
["worker".'3']
"stage2".device = "cpu"
'''
        self.assertEqual(tomllib.loads(self.migrate(source)), {
            "config_version": 4, "stage2": {"weighted_ntt": True},
            "worker": {"3": {"device": "cpu"}},
        })

    def test_preview_and_export_keep_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "ecm.toml"
            output = source.with_name("ecm.v4.toml")
            original = ("\ufeff" + V3.replace("\n", "\r\n")).encode("utf-8")
            source.write_bytes(original)

            def cli(*flags):
                return subprocess.run([sys.executable, str(TOOL), str(source), *flags],
                                      capture_output=True, timeout=10)

            preview = cli("--preview")
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertFalse(output.exists())
            write = cli()
            self.assertEqual(write.returncode, 0, write.stderr)
            exported = output.read_bytes()
            self.assertEqual(exported.replace(b"\r\n", b"\n"), preview.stdout)
            self.assertNotEqual(cli().returncode, 0)
            self.assertNotEqual(cli("--output", str(source)).returncode, 0)
            self.assertIn(b"preserve relative paths",
                          cli("--output", str(source.parent / "elsewhere" / "ecm.toml")).stderr)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(output.read_bytes(), exported)


if __name__ == "__main__":
    unittest.main()
