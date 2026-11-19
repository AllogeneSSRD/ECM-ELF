"""Validate CMake/Unix Makefiles compile units by content before reusing them."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOTS = ("src", "include", "kernels", "config", "tools/build", "tools/gen")
BINARY_TARGETS = {"ecm_cuda": "ecm_cuda", "ecm": "ecm", "ecm_cpu_stage2": "ecm_stage2"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(*args, **kwargs):
    return subprocess.check_output(args, cwd=ROOT, text=True, **kwargs).strip()


def source_files():
    names = run("git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
                "--", *SOURCE_ROOTS).split("\0")
    paths = {ROOT / name for name in names if name and (ROOT / name).is_file()}
    paths.update(ROOT / name for name in ("CMakeLists.txt", ".gitattributes", "LICENSE"))
    paths.update((ROOT / "docs").glob("ECM_TOML_REFERENCE.*.md"))
    paths.update((ROOT / "third_party/cgbn/include").rglob("*"))
    paths.add(ROOT / "third_party/cgbn/LICENSE")
    return {p for p in paths if p.is_file()}


def snapshot(paths):
    resolved = sorted({p.resolve(strict=True) for p in paths})
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(map(str, resolved), pool.map(sha, resolved), strict=True))


def tree_files(roots):
    return {file for root in roots for file in root.rglob("*") if file.is_file()}


def build_inventory(build):
    entries = json.loads((build / "compile_commands.json").read_text())
    units = {}
    for entry in entries:
        directory = Path(entry["directory"])
        words = shlex.split(entry["command"])
        output = (directory / words[words.index("-o") + 1]).resolve()
        relative = output.relative_to(build)
        if relative.parts[0] != "CMakeFiles" or output.suffix != ".o" or output in units:
            raise ValueError(f"Unexpected compile unit: {output}")
        # CMake may put flags into response files instead of the database.
        responses = set()
        pending = deque(words)
        while pending:
            word = pending.popleft()
            if word in ("--options-file", "-optf"):
                name = pending.popleft()
            elif word.startswith(("--options-file=", "-optf=")):
                name = word.split("=", 1)[1]
            else:
                name = word[1:] if word.startswith("@") else ""
            for item in name.split(",") if name else ():
                path = (directory / item).resolve(strict=True)
                if path not in responses:
                    responses.add(path)
                    pending.extend(shlex.split(path.read_text()))
        command = hashlib.sha256(entry["command"].encode())
        for response in sorted(responses):
            command.update(str(response).encode())
            command.update(sha(response).encode())
        units[output] = (directory, Path(entry["file"]).resolve(strict=True), command.hexdigest())
    if not units:
        raise ValueError("No compile units in CMake's database")
    return units


def resolved_input(path):
    return path.resolve(strict=True)


def dependencies(output, directory, source, resolve=resolved_input):
    depfile = output.with_suffix(".o.d")
    # GNU Make quoting: the compiler emits one target followed by a continued rule.
    text = depfile.read_text().replace("\\\n", " ").replace("$$", "$")
    target, separator, inputs = text.partition(":")
    if not separator or len(shlex.split(target)) != 1 or (directory / shlex.split(target)[0]).resolve() != output:
        raise ValueError(f"Dependency file does not identify {output}")
    paths = {resolve(directory / item) for item in shlex.split(inputs)}
    if source not in paths:
        raise ValueError(f"Dependency file does not list its source: {depfile}")
    return sorted(map(str, paths))


def tool_inputs(nvcc):
    cuda = nvcc.parent.parent
    roots = [Path("/usr/include"), Path("/usr/local/include"), cuda / "targets/x86_64-linux/include"]
    files = {nvcc, Path("/usr/bin/gcc"), Path("/usr/bin/g++"), Path("/usr/bin/cmake"), Path("/usr/bin/make")}
    roots += [Path(run("/usr/bin/gcc", "-print-file-name=include"))]
    for tool in ("cc1", "cc1plus", "collect2", "as", "ld"):
        name = run("/usr/bin/g++", f"-print-prog-name={tool}")
        path = Path(name) if "/" in name else Path("/usr/bin") / name
        files.add(path)
    for name in ("crtbeginS.o", "crtendS.o", "Scrt1.o", "crti.o", "crtn.o", "libstdc++.so",
                 "libc.so", "libgcc.a", "libgcc_s.so", "libgomp.so", "libm.so"):
        path = Path(run("/usr/bin/g++", f"-print-file-name={name}"))
        if path.is_absolute():
            files.add(path)
    for tool in ("ptxas", "cudafe++", "fatbinary", "nvlink", "cuobjdump"):
        files.add(nvcc.parent / tool)
    files.add(cuda / "nvvm/bin/cicc")
    files.update((cuda / "nvvm/libdevice").glob("*.bc"))
    for library in ("libcudart_static.a", "libcudadevrt.a", "libculibos.a"):
        path = cuda / "targets/x86_64-linux/lib" / library
        if path.exists():
            files.add(path)
    return roots, files


def record(output, unit, hashes, resolve):
    directory, source, command = unit
    names = dependencies(output, directory, source, resolve)
    try:
        inputs = {name: hashes[name] for name in names}
    except KeyError as error:
        raise ValueError(f"Input outside the build snapshot: {error.args[0]}") from error
    return dict(command=command, inputs=inputs, object_sha256=sha(output))


def read_state(path):
    try:
        state = json.loads(path.read_text())
        return state if state["schema"] == 1 else {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def freeze(build, nvcc, explicit):
    roots, tools = tool_inputs(nvcc)
    generated = tree_files([build / "cuda_modules"])
    sources = source_files()
    external = tree_files(roots)
    tool_names = {str(resolved_input(path)) for path in tools | set(explicit)}
    hashes = snapshot(sources | external | tools | set(explicit) | generated)
    return hashes, {name: hashes[name] for name in sorted(tool_names)}


def save_state(manifest, state):
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(manifest)


def completed_cuda_units(build, units, before_logs, hashes, resolve):
    for path in set((build / "timing").glob("*/command.json")) - before_logs:
        try:
            timing = json.loads(path.read_text())
            command = timing["semantic_command"]
            if timing["status"] != "complete" or "-o" not in command:
                continue
            output = (build / command[command.index("-o") + 1]).resolve()
            if output in units:
                yield str(output.relative_to(build)), record(output, units[output], hashes, resolve)
        except (OSError, ValueError, KeyError):
            continue


def verify_build(build, nvcc, jobs, explicit, skip):
    manifest = build / "verified-build.json"
    old = read_state(manifest)
    units = build_inventory(build)
    binaries = {name: build / binary for name, binary in BINARY_TARGETS.items()}
    link_plans = {build / "CMakeFiles" / f"{name}.dir" / "link.txt" for name in binaries}
    link_plans.add(build / "CMakeFiles/cgbn_opencl.dir/link.txt")
    linked = {Path(word) for plan in link_plans for word in shlex.split(plan.read_text())
              if word.startswith("/") and (word.endswith((".so", ".a")) or ".so." in Path(word).name)}
    explicit = set(explicit) | linked
    print(f"Hashing Linux build inputs for {len(units)} compile units", flush=True)
    before, toolchain = freeze(build, nvcc, explicit)
    revision = run("git", "rev-parse", "HEAD")
    # A new header can take precedence over a previously recorded include.
    names = hashlib.sha256("\n".join(before).encode()).hexdigest()
    context = dict(toolchain=toolchain, input_names=names,
                   compiler_environment={key: os.getenv(key) for key in (
                       "PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
                       "COMPILER_PATH", "GCC_EXEC_PREFIX", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")})
    valid = {}
    # Resolve repeated include paths once per phase. File content is hashed anew
    # for each snapshot so link and build phases still have independent evidence.
    resolve_before = cache(resolved_input)
    if old.get("context") == context:
        for output, unit in units.items():
            key = str(output.relative_to(build))
            previous = old.get("units", {}).get(key)
            if not previous or not output.is_file():
                continue
            try:
                if record(output, unit, before, resolve_before) == previous:
                    valid[key] = previous
            except (OSError, ValueError):
                pass
    links = snapshot(link_plans)
    products_match = (
        old.get("links") == links and all(
            path.is_file() and sha(path) == old.get("products", {}).get(path.name)
            for path in [*binaries.values(), build / "libcgbn_opencl.a"])
    )
    invalid = [path for path in units if str(path.relative_to(build)) not in valid]
    print(f"Linux object verification: {len(valid)} reusable, {len(invalid)} to rebuild", flush=True)
    if skip:
        if invalid or not products_match or old.get("revision") != revision or old.get("inputs") != before:
            raise ValueError("Existing release artifacts do not match the verified build")
        return
    for output in invalid:
        output.unlink(missing_ok=True)
    if not products_match or old.get("context") != context:
        for path in [*binaries.values(), build / "libcgbn_opencl.a"]:
            path.unlink(missing_ok=True)
    before_logs = set((build / "timing").glob("*/command.json"))
    failure = None
    try:
        subprocess.run(["cmake", "--build", str(build), "--target", *binaries,
                        "--parallel", str(jobs)], cwd=ROOT, check=True)
    except (subprocess.CalledProcessError, KeyboardInterrupt) as error:
        failure = error
    print("Rechecking Linux build inputs and products", flush=True)
    after, _ = freeze(build, nvcc, explicit)
    resolve_after = cache(resolved_input)
    if before != after or units != build_inventory(build) or revision != run("git", "rev-parse", "HEAD"):
        raise ValueError("Source, dependency or compile plan changed during the verified build")
    if links != snapshot(map(Path, links)):
        raise ValueError("Link commands changed during the verified build")
    state = dict(schema=1, revision=revision, context=context, inputs=after, links=links)
    if failure:
        valid.update(completed_cuda_units(build, units, before_logs, after, resolve_after))
        save_state(manifest, dict(state, units=valid, products={}))
        raise failure
    verified = {str(path.relative_to(build)): record(path, unit, after, resolve_after)
                for path, unit in units.items()}
    products = {path.name: sha(path) for path in [*binaries.values(), build / "libcgbn_opencl.a"]}
    save_state(manifest, dict(state, units=verified, products=products))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    verify_build(args.build.resolve(), args.nvcc.resolve(), args.jobs, args.input, args.skip_build)
