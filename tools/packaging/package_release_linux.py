"""Package verified Linux x86-64 ECM workers with matching sources."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
TOOL_ENV = dict(os.environ, LC_ALL="C")
sys.path.insert(0, str(ROOT / "tools/build/internal"))
from verified_linux_build import BINARY_TARGETS, read_state, sha, snapshot, source_files


def cuda_images(inspector, binary, arch, ptx):
    images = {}
    for kind, flag, extension, expected in (
        ("sass", "--list-elf", "cubin", [arch]),
        ("ptx", "--list-ptx", "ptx", [] if ptx == "none" else [ptx]),
    ):
        result = subprocess.run([inspector, flag, binary], text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False, env=TOOL_ENV)
        if result.returncode and not (not expected and "Could not find any ptx" in result.stdout):
            raise ValueError(f"{inspector} {flag}: {result.stdout.strip()}")
        images[kind] = sorted(set(re.findall(rf"\.sm_(\d+)\.[^\s]*{extension}\b", result.stdout)))
        if images[kind] != expected:
            raise ValueError(f"Unexpected {kind} images: {images[kind]}, expected {expected}")
    return images


def inspect_elf(path, workdir):
    header = subprocess.check_output(["readelf", "-h", path], text=True, env=TOOL_ENV)
    if "ELF64" not in header or "Advanced Micro Devices X86-64" not in header:
        raise ValueError(f"Not a Linux x86-64 ELF: {path.name}")
    dynamic = subprocess.check_output(["readelf", "-d", path], text=True, env=TOOL_ENV)
    if re.search(r"\((?:RPATH|RUNPATH)\)", dynamic):
        raise ValueError(f"Unexpected runtime search path: {path.name}")
    needed = re.findall(r"\(NEEDED\).*\[([^\]]+)\]", dynamic)
    text = subprocess.check_output([path, "--version"], cwd=workdir, text=True)
    version = re.search(r"^ECM-ELF (\d+\.\d+\.\d+)\b", text)
    if not version:
        raise ValueError(f"Version probe failed: {path.name}")
    return version[1], needed


def copy_verified(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha(destination) != expected:
        raise ValueError(f"Verified file changed: {source}")


def payload_permissions(member):
    relative = member.name.partition("/")[2]
    executable = relative in BINARY_TARGETS.values()
    member.mode = 0o755 if member.isdir() or executable else 0o644
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    return member


def stage1_modules(build, state, inspector, arch, ptx):
    cache = {key: value for key, value in re.findall(
        r"^(ECM_\w+):\w+=(.*)$", (build / "CMakeCache.txt").read_text(), re.M)}
    expected = dict(ECM_TIERS="", ECM_CUDA_ARCHITECTURES=arch,
                    ECM_CUDA_PTX_ARCH="" if ptx == "none" else ptx, ECM_CUDA_EMBED_PTX="OFF",
                    ECM_CUDA_FULL_BUILD="ON", ECM_CUDA_ENABLE_PRAC="ON",
                    ECM_BUILD_CPU_STAGE2="ON", ECM_OPENCL_PORTABLE_RESOURCES="ON")
    if any(cache.get(key) != value for key, value in expected.items()):
        raise ValueError("Release requires the full Stage1 configuration and portable OpenCL resources")
    catalog = build / "cuda_modules/stage1/modules.json"
    if sha(catalog) != state["inputs"].get(str(catalog.resolve())):
        raise ValueError("Stage1 module catalog changed since the verified build")
    modules = json.loads(catalog.read_text())["modules"]
    owners = {Path(output).name: output for output in state["units"]
              if "/cuda_modules/stage1/" in output}
    objects = []
    for module in modules:
        source = Path(module["source"]).resolve()
        output = owners.get(source.name + ".o")
        if not output or str(source) not in state["units"][output]["inputs"]:
            raise ValueError(f"Unverified Stage1 module: {source.name}")
        objects.append((build / output, state["units"][output]["object_sha256"]))

    def inspect_module(entry):
        path, digest = entry
        if sha(path) != digest:
            raise ValueError(f"Stage1 object changed: {path.name}")
        cuda_images(inspector, path, arch, ptx)
        if sha(path) != digest:
            raise ValueError(f"Stage1 object changed during inspection: {path.name}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(inspect_module, objects):
            pass
    return {family: sum(m["family"] == family for m in modules)
            for family in sorted({m["family"] for m in modules})}


def runtime_files(state, build, folder, libecm_source):
    for program in BINARY_TARGETS.values():
        copy_verified(build / program, folder / program, state["products"][program])
    resources = ROOT / "src/ecm/opencl/resources"
    for path in sorted(resources.rglob("*")):
        if path.is_file():
            copy_verified(path, folder / "opencl_resources" / path.relative_to(resources),
                          state["inputs"][str(path.resolve())])
    for source, destination in (
        ("LICENSE", "LICENSE"),
        ("third_party/cgbn/LICENSE", "licenses/CGBN-LICENSE.txt"),
        ("config/ecm.toml.default", "ecm.toml"),
        ("docs/ECM_TOML_REFERENCE.en.md", "ECM_TOML_REFERENCE.en.md"),
        ("docs/ECM_TOML_REFERENCE.zh-CN.md", "ECM_TOML_REFERENCE.zh-CN.md"),
    ):
        path = ROOT / source
        copy_verified(path, folder / destination, state["inputs"][str(path.resolve())])
    (folder / "worktodo.txt").touch()
    libecm_archive = folder / "sources/gmp-ecm.tar"
    copy_verified(libecm_source, libecm_archive, state["inputs"][str(libecm_source.resolve())])
    with tarfile.open(libecm_archive) as archive:
        for name in ("COPYING", "COPYING.LIB"):
            with archive.extractfile(name) as stream:
                (folder / "licenses" / f"GMP-ECM-{name}.txt").write_bytes(stream.read())
    package_source = Path(__file__).resolve()
    with tarfile.open(folder / "sources/ecm-elf.tar.gz", "w:gz", dereference=True) as archive:
        for path in sorted(source_files()):
            with path.open("rb") as stream:
                # Read exactly the bytes that go into the source archive.
                data = stream.read()
            if hashlib.sha256(data).hexdigest() != state["inputs"][str(path.resolve())]:
                raise ValueError(f"Corresponding source changed: {path}")
            archive.addfile(archive.gettarinfo(str(path), path.relative_to(ROOT).as_posix()), io.BytesIO(data))
        archive.add(package_source, arcname=package_source.relative_to(ROOT).as_posix())


def publish(args):
    profile = re.fullmatch(r"cu([0-9]+\.[0-9]+)-sm([0-9]+)-ptx([0-9]+|none)", args.profile)
    if not profile or profile.groups()[1:] != (args.arch, args.ptx_arch):
        raise ValueError(f"Invalid Linux release profile: {args.profile}")
    build = args.build.resolve()
    state = read_state(build / "verified-build.json")
    if not state.get("products"):
        raise ValueError("Linux release requires a completed verified build")
    inspector = args.nvcc.resolve().with_name("cuobjdump")
    nvcc = subprocess.check_output([args.nvcc, "--version"], text=True, env=TOOL_ENV)
    if f"release {profile[1]}," not in nvcc or sha(inspector) != state["inputs"].get(str(inspector.resolve())):
        raise ValueError("Release inspector or CUDA toolkit differs from the verified build")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".linux-release.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with TemporaryDirectory(prefix=".linux-release.", dir=out) as scratch:
            staging = Path(scratch)
            folder = staging / "payload"
            runtime_files(state, build, folder, args.libecm_source)
            results = {name: inspect_elf(folder / name, staging) for name in BINARY_TARGETS.values()}
            version, = {entry[0] for entry in results.values()}
            name = f"ecm-elf-{version}-{args.profile}-linux-x64"
            archive_name = name + ".tar.gz"
            outputs = [name, archive_name, archive_name + ".sha256"]
            for output in outputs:
                if (out / output).exists() or (out / output).is_symlink():
                    raise FileExistsError(f"Release output already exists: {out / output}")
            images = cuda_images(inspector, folder / "ecm_cuda", args.arch, args.ptx_arch)
            modules = stage1_modules(build, state, inspector, args.arch, args.ptx_arch)
            runtime = {name: entry[1] for name, entry in results.items()}
            (folder / "README.txt").write_text(
                f"ECM-ELF {version} | Linux x86-64 | {args.profile}\n"
                "ecm_cuda: CUDA Stage1\necm: OpenCL and CPU Stage1\necm_stage2: libecm Edwards CPU Stage2\n\n"
                "Build baseline: see release-manifest.json. Install the system libraries listed there.\n"
                "The CUDA Stage1 binary uses the NVIDIA driver. OpenCL GPU use requires a working ICD.\n"
                "Start with ./ecm_cuda --help or ./ecm_stage2 --help. The supplied ecm.toml is a template.\n"
                "Matching project and GMP-ECM sources are in sources/.\n"
            )
            baseline = subprocess.check_output(["getconf", "GNU_LIBC_VERSION"], text=True).strip()
            files = sorted(p for p in folder.rglob("*") if p.is_file())
            manifest = dict(schema=1, version=version, profile=args.profile, source_revision=state["revision"],
                            nvcc_version=nvcc.strip(),
                            glibc_build_host=baseline, images=images, stage1_modules=modules, needed=runtime,
                            files={p.relative_to(folder).as_posix(): sha(p) for p in files})
            (folder / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            archive = staging / archive_name
            with tarfile.open(archive, "w:gz") as payload:
                payload.add(folder, arcname=name, filter=payload_permissions)
            (staging / outputs[2]).write_text(f"{sha(archive)}  {archive_name}\n", encoding="ascii")
            if snapshot(map(Path, state["inputs"])) != state["inputs"]:
                raise ValueError("Verified build inputs changed during packaging")
            folder.rename(staging / name)
            published = []
            try:
                for output in outputs[1:] + outputs[:1]:
                    (staging / output).rename(out / output)
                    published.append(output)
            except OSError:
                for output in reversed(published):
                    (out / output).rename(staging / output)
                raise
            print(f"Linux release: {out / archive_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("build", "out", "nvcc", "libecm-source"):
        parser.add_argument("--" + argument, type=Path, required=True)
    for argument in ("profile", "arch", "ptx-arch"):
        parser.add_argument("--" + argument, required=True)
    publish(parser.parse_args())
