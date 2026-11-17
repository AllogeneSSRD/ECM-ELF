"""Compare Edwards one-thread-per-curve through native CUDA and OpenCL Stage1."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from bench_stage1_opencl_scaling import ED_TIME, SIGMA, probable_prime, read_points
from gpu_telemetry import TelemetryRecorder


# B1 keeps most device intervals between 0.5 and 1 second on RTX 4060 Laptop.
FIXTURES = {
    127: (12288, 100000, 128),
    191: (6144, 80000, 128),
    241: (6144, 64000, 32),
    319: (6144, 22000, 128),
    384: (6144, 20000, 128),
    385: (6144, 15000, 128),
    511: (6144, 14000, 128),
    767: (6144, 3500, 128),
    1021: (6144, 2000, 128),
}
DETAILS = re.compile(r"Stage1 execution: .*?backend=(\S+) (.*)")
PHASES = re.compile(r"construction_s=([\d.]+) init_s=([\d.]+) dictionary_gpu_s=([\d.]+)")


def run(exe, backend, root, device, bits, n, b1, curves, block, repeat, setup_threads, kernel_cache, env):
    case = root / f"bit{bits}_{backend}_r{repeat}"
    case.mkdir()
    conf = case / "ecm.toml"
    save = case / "point.save"
    algo = "thread" if backend == "cuda" else "cgbn"
    arithmetic = "fips_ptx" if bits <= 512 else "cios"
    options = (
        'config_version = 3\n[shared]\nlog_file = ""\n[stage1]\nparam = "e"\ntarget = "gpu"\n'
        f'gpu_backend = "{backend}"\ngpu_device = {device}\ngpu_algorithm = "{algo}"\n'
        'edwards_naf_window_bits = 6\nedwards_doubling = 1\nhandoff.mode = "off"\n'
        f'cpu_threads = {setup_threads}\n'
        f'checkpoint_interval_seconds = 0\nsave_dir = {json.dumps(case.as_posix())}\n'
        f'exponent_cache_dir = {json.dumps((root / "scalar_cache").as_posix())}\n'
    )
    if backend == "opencl":
        options += f'opencl.kernel_mul = "{arithmetic}"\nopencl.kernel_sqr = "{arithmetic}"\n'
    conf.write_text(options, encoding="ascii")
    argv = [str(exe), "--config", str(conf), "--wg", str(block), "--tmp-dir", str(case),
            "-sigma", str(SIGMA), "-gpucurves", str(curves), "-savea", str(save), str(b1), "0"]
    if backend == "opencl":
        argv += ["--kernel-cache-dir", str(kernel_cache), "--compile-verbose"]
    start = time.perf_counter()
    proc = subprocess.run(argv, cwd=case, input=f"{n}\n", env=env, text=True, encoding="utf-8",
                          errors="replace", capture_output=True, timeout=450)
    wall = time.perf_counter() - start
    log = proc.stdout + proc.stderr
    (case / "run.log").write_text(log, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"{case}:\n{log[-2500:]}")
    reported, details = DETAILS.search(log).groups()
    assert reported == ("cuda-thread" if backend == "cuda" else "opencl"), log[-2500:]
    shape = dict(item.split("=", 1) for item in details.split())
    assert shape["field_mul"] == arithmetic and shape["tpi"] == "1", shape
    assert shape["field_sqr"] == ("fips_ptx_square" if arithmetic == "fips_ptx" else "cios"), shape
    assert int(shape["tpb" if backend == "cuda" else "workgroup"]) == block, shape
    columns, unit, value = ED_TIME.search(log).groups()
    construction, init, dictionary = map(float, PHASES.search(log).groups())
    threads = re.search(r"\bconstruction_threads=(\d+)", log)
    row = dict(backend=backend, repeat=repeat, wall=wall, columns=int(columns),
               device=float(value) / (1000 if unit == "ms" else 1), construction=construction,
               init=init, dictionary=dictionary, execution=shape,
               construction_threads=int(threads[1]) if threads else 1)
    print(f"{bits} bit {backend} r{repeat} block={block}: {row['device']:.6f} s device, "
          f"{construction:.3f} s construction, {wall:.3f} s wall", flush=True)
    return row, read_points(save, curves)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda", type=Path, required=True)
    p.add_argument("--opencl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--widths", type=int, choices=FIXTURES, nargs="+", default=list(FIXTURES))
    p.add_argument("--block", type=int, choices=(32, 64, 128, 256))
    p.add_argument("--setup-threads", type=int, default=0, help="host curve construction threads, 0 for auto")
    p.add_argument("--kernel-cache-dir", type=Path, help="reuse an existing OpenCL binary cache")
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    if a.repeats < 1 or not 0 <= a.setup_threads < 2**32:
        p.error("--repeats must be positive and --setup-threads must fit in u32")
    root = a.out.resolve()
    kernel_cache = a.kernel_cache_dir.resolve() if a.kernel_cache_dir else root / "kernel_cache"
    exes = {"opencl": a.opencl.resolve(strict=True), "cuda": a.cuda.resolve(strict=True)}
    root.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ECM_", "NTT_"))}
    versions = {backend: subprocess.check_output([str(v), "--version"], cwd=root, env=env, text=True).strip()
                for backend, v in exes.items()}
    telemetry = TelemetryRecorder(a.device, root / "telemetry.jsonl", interval=0.25)
    summary = dict(program=versions, device=telemetry.identity, sigma=SIGMA,
                   setup_threads=a.setup_threads, conditions=[])
    try:
        for bits in a.widths:
            curves, b1, block = FIXTURES[bits]
            block = a.block or block
            n = probable_prime(bits)
            samples, reference = [], None
            for repeat in range(a.repeats + 1):
                for backend in (exes if repeat % 2 == 0 else reversed(exes)):
                    row, points = run(exes[backend], backend, root, a.device, bits, n, b1,
                                      curves, block, repeat, a.setup_threads, kernel_cache, env)
                    reference = points if reference is None else reference
                    if points != reference:
                        raise ValueError(f"CUDA and OpenCL points differ at {bits} bit, {backend}, r{repeat}")
                    samples.append(row)
            medians = {backend: {k: statistics.median(row[k] for row in samples
                                                      if row["backend"] == backend and row["repeat"])
                                 for k in ("device", "wall", "construction", "init", "dictionary")}
                       for backend in exes}
            summary["conditions"].append(dict(bits=bits, modulus=str(n), b1=b1, curves=curves,
                                               block=block, samples=samples, medians=medians))
            (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    finally:
        telemetry.close()


if __name__ == "__main__":
    main()
