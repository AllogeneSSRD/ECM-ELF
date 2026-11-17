"""Run in an MSVC x64 developer shell with CMake and NMake on PATH."""

import subprocess
import tempfile
from pathlib import Path


def run(command, expected=0):
    result = subprocess.run(command, capture_output=True)
    if result.returncode != expected:
        raise RuntimeError(
            f"{command}: expected {expected}, got {result.returncode}\n"
            + result.stdout.decode(errors="replace")
            + result.stderr.decode(errors="replace")
        )


def main():
    repo = Path(__file__).resolve().parents[3]
    experiments = repo / "data" / "experiments" / "nmake_host_dependencies"
    experiments.mkdir(parents=True, exist_ok=True)
    source = Path(tempfile.mkdtemp(dir=experiments))
    build = source / "build"
    build.mkdir()
    (build / ".gitignore").write_text("*\n", encoding="ascii")
    helper = repo / "tools" / "build" / "internal" / "nmake_host_dependencies.cmake"
    (source / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 3.20)
project(include_probe LANGUAGES CXX)
if(REPAIR)
    include("{helper.as_posix()}")
else()
    set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "unmatched prefix: ")
endif()
add_executable(probe main.cpp)
target_compile_options(probe PRIVATE /utf-8)
if(REPAIR)
    ecm_track_nmake_host_dependencies(probe)
endif()
""",
        encoding="utf-8",
    )
    header = source / "value.h"
    header.write_text("constexpr int value = 11;\n", encoding="ascii")
    (source / "main.cpp").write_text(
        '#include "value.h"\nint main() { return value; }\n', encoding="ascii"
    )
    configure = ["cmake", "-S", source, "-B", build, "-G", "NMake Makefiles",
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_TRY_COMPILE_CONFIGURATION=Release"]
    compile_target = ["cmake", "--build", build, "--target", "probe"]
    probe = [build / "probe.exe"]

    run([*configure, "-DREPAIR=OFF"])
    run(compile_target)
    run(probe, expected=11)
    header.write_text("constexpr int value = 12;\n", encoding="ascii")
    run(compile_target)
    run(probe, expected=11)

    run([*configure, "-DREPAIR=ON"])
    run(compile_target)
    run(probe, expected=12)
    header.write_text("constexpr int value = 13;\n", encoding="ascii")
    run(compile_target)
    run(probe, expected=13)
    obj = next(build.rglob("main.cpp.obj"))
    compiled = obj.stat().st_mtime_ns
    run(compile_target)
    assert obj.stat().st_mtime_ns == compiled, "unchanged host source rebuilt"
    run([*configure, "-DREPAIR=ON"])
    run(compile_target)
    assert obj.stat().st_mtime_ns == compiled, "unchanged configure rebuilt host source"
    print(f"PASS NMake host dependency migration and incremental rebuild ({source})")


if __name__ == "__main__":
    main()
