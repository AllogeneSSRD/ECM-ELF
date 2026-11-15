"""Batch source hashing and UTF-8 build output for the Windows GUI builder."""

from argparse import ArgumentParser
import base64
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


INCLUDES = re.compile(
    r"^\s*(?:Note\s*:\s*including file|注意\s*[:：]\s*包含文件|"
    r"娉ㄦ剰\s*[:：]\s*鍖呭惈鏂囦欢|\?敞鎰\?\s*[:：]?\s*鍖呭惈鏂囦欢)\s*[:：]",
    re.IGNORECASE,
)
BASE_ASSETS = {"pet_ely.png", "pet_ely.json", "CLASSIC-ELY-NOTICE.md", "ELYTerminal-LICENSE.txt"}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def snapshot(args):
    paths = set()
    for name in args.catalog:
        path = args.root / name
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            paths.update(child for child in path.rglob("*") if child.is_file())
        else:
            raise FileNotFoundError(path)
    if args.base_pet_only:
        assets = args.root / "assets/gui_v2"
        paths = {path for path in paths if not path.is_relative_to(assets)
                 or (path.parent == assets and path.name in BASE_ASSETS)}

    expected = {}
    if not args.without_local_pari:
        pari = args.root / "third_party/pari-2.17.3"
        provenance = pari / "provenance.json"
        if not provenance.is_file():
            raise RuntimeError("Prepare local PARI with tools/build/test/prepare_gui_v2_pari.py first")
        paths.add(provenance)
        expected = {pari / name: value for name, value in
                    json.loads(provenance.read_text(encoding="utf-8-sig"))["sha256"].items()}
        paths.update(expected)
    result = {}
    for path in sorted(paths):
        value = digest(path)
        if path in expected and value != expected[path]:
            raise RuntimeError(f"Local PARI identity mismatch: {path.relative_to(pari)}")
        result[path.relative_to(args.root).as_posix()] = value
    if args.verify:
        original = json.loads(args.snapshot.read_text(encoding="utf-8"))
        for name in sorted(result.keys() | original.keys()):
            if result.get(name) != original.get(name):
                raise RuntimeError(f"GUI v2 source catalog changed during build: {name}")
    else:
        args.snapshot.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"GUI v2 source catalog: {len(result)} files {'verified' if args.verify else 'checked'}")
    return 0


def decoded(raw, fallback):
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode(fallback, errors="replace")


def run_command(command, destination, environment, fallback):
    command = base64.b64decode(command).decode("utf-8")
    # Pass cmd its original quotes. Windows argv escaping is for CRT programs,
    # while cmd interprets the rest of its command line itself.
    # CMake decodes compiler output using the console code page. Give its
    # process tree a UTF-8 console without changing the caller's code page.
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    with destination.open("w", encoding="utf-8", newline="\n") as log:
        with subprocess.Popen("cmd.exe /d /c chcp 65001 >nul && " + command,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment,
                              creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup) as process:
            try:
                for raw in process.stdout:
                    line = decoded(raw, fallback).rstrip("\r\n")
                    log.write(line + "\n")
                    if not INCLUDES.match(line):
                        log.flush()
                        print(line, flush=True)
                return process.wait()
            except KeyboardInterrupt:
                subprocess.run(["taskkill.exe", "/pid", str(process.pid), "/t", "/f"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                raise


def refresh_empty_dependencies(cmake_files):
    # Recompile old objects whose localized include scan produced no dependencies.
    count = 0
    for dependencies in cmake_files.rglob("*.obj.d"):
        obj = dependencies.with_suffix("")
        if dependencies.stat().st_size == 0 and obj.is_file() and obj.resolve().is_relative_to(cmake_files):
            obj.unlink()
            count += 1
    if count:
        print(f"GUI v2: refreshing {count} objects with empty dependency lists", flush=True)


def run(args):
    print("== GUI v2 MSVC setup ==", flush=True)
    start = time.perf_counter()
    setup = subprocess.run(f'cmd.exe /d /u /c call "{args.vcvars}" >nul 2>&1 && set',
                           stdout=subprocess.PIPE)
    if setup.returncode:
        raise RuntimeError(f"MSVC environment setup failed ({setup.returncode}): {args.vcvars}")
    environment = {}
    for line in setup.stdout.decode("utf-16-le").splitlines():
        key, separator, value = line.partition("=")
        if key and separator:
            environment[key] = value

    fallback = f"cp{ctypes.windll.kernel32.GetOEMCP()}"
    if fallback == "cp65001":
        fallback = f"cp{ctypes.windll.kernel32.GetACP()}"
    migration = args.directory / ".cmake-msvc-utf8-v1"
    cmake_files = (args.directory / "CMakeFiles").resolve()
    if not cmake_files.is_relative_to(args.directory.resolve()):
        raise RuntimeError(f"CMakeFiles must be inside the GUI build directory: {cmake_files}")
    if not migration.exists():
        # The cached include prefix depends on the console encoding used when
        # CMake detected the compiler. Keep the cache and re-detect this metadata.
        for compiler in ("CMakeCCompiler.cmake", "CMakeCXXCompiler.cmake"):
            for info in cmake_files.glob(f"*/{compiler}"):
                if info.resolve().is_relative_to(cmake_files):
                    info.unlink()
    print(f"== GUI v2 CMake configure (MSVC setup {time.perf_counter() - start:.1f}s) ==", flush=True)
    start = time.perf_counter()
    for label, command, name in (("configure", args.configure, "configure.log"),
                                  ("build", args.build, "build.log")):
        if label == "build":
            print(f"== GUI v2 build (configure {time.perf_counter() - start:.1f}s) ==", flush=True)
            if not migration.exists():
                refresh_empty_dependencies(cmake_files)
        destination = args.directory / name
        code = run_command(command, destination, environment, fallback)
        if code:
            print(f"GUI v2 {label} failed ({code}). Log: {destination}", flush=True)
            return code
    if not migration.exists():
        migration.touch()
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    inputs = commands.add_parser("snapshot")
    inputs.add_argument("--root", type=Path, required=True)
    inputs.add_argument("--snapshot", type=Path, required=True)
    inputs.add_argument("--verify", action="store_true")
    inputs.add_argument("--base-pet-only", action="store_true")
    inputs.add_argument("--without-local-pari", action="store_true")
    inputs.add_argument("catalog", nargs="+")
    inputs.set_defaults(action=snapshot)
    output = commands.add_parser("run")
    output.add_argument("--directory", type=Path, required=True)
    output.add_argument("--vcvars", type=Path, required=True)
    output.add_argument("--configure", required=True)
    output.add_argument("--build", required=True)
    output.set_defaults(action=run)
    args = parser.parse_args()
    try:
        return args.action(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
