"""Regression for the Linux content-verified Make build with a small native project."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build/internal"))
import verified_linux_build as verified


@unittest.skipUnless(sys.platform == "linux", "Linux Unix Makefiles contract")
class ReleaseIdentityTest(unittest.TestCase):
    def test_same_timestamp_edits_and_object_tampering(self):
        with tempfile.TemporaryDirectory(prefix="ecm-linux-identity-") as temp:
            root = Path(temp)
            source = root / "src"
            source.mkdir()
            build = root / "build"
            build.mkdir()
            (build / ".gitignore").write_text("*\n")
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\nproject(small C)\n"
                "add_executable(ecm_cuda first.c)\nadd_executable(ecm second.c)\n"
                "add_executable(ecm_cpu_stage2 third.c)\n"
                "set_target_properties(ecm_cpu_stage2 PROPERTIES OUTPUT_NAME ecm_stage2)\n"
                "add_library(cgbn_opencl STATIC library.c)\n"
                "target_link_libraries(ecm PRIVATE cgbn_opencl)\n"
            )
            header = source / "value header.h"
            header.write_text("#define NUMBER 3\n")
            (source / "first.c").write_text('#include "value header.h"\nint main(void) { return NUMBER; }\n')
            for name in ("second", "third", "library"):
                (source / (name + ".c")).write_text("int main(void) { return 0; }\n")
            subprocess.run(["cmake", "-S", source, "-B", build, "-G", "Unix Makefiles",
                            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], check=True)
            manifest = build / "verified-build.json"
            with (
                patch.object(verified, "source_files", side_effect=lambda: set(source.iterdir())),
                patch.object(verified, "tool_inputs",
                             return_value=([], {Path("/usr/include/stdc-predef.h")})),
            ):
                def build_or_check(skip=False):
                    verified.verify_build(build, Path("/usr/bin/cc"), 2, [], skip)
                    return json.loads(manifest.read_text())

                first = build_or_check()
                self.assertEqual(first, build_or_check(skip=True))
                before = header.stat()
                header.write_text("#define NUMBER 4\n")
                os.utime(header, ns=(before.st_atime_ns, before.st_mtime_ns))
                with self.assertRaisesRegex(ValueError, "do not match"):
                    build_or_check(skip=True)

                second = build_or_check()
                changed = [name for name in first["units"] if first["units"][name] != second["units"][name]]
                self.assertEqual(changed, ["CMakeFiles/ecm_cuda.dir/first.c.o"])
                self.assertEqual(subprocess.run([build / "ecm_cuda"]).returncode, 4)

                obj = build / changed[0]
                before = obj.stat()
                with obj.open("ab") as stream:
                    stream.write(b"changed")
                os.utime(obj, ns=(before.st_atime_ns, before.st_mtime_ns))
                with self.assertRaisesRegex(ValueError, "do not match"):
                    build_or_check(skip=True)
                self.assertEqual(second, build_or_check())


if __name__ == "__main__":
    unittest.main()
