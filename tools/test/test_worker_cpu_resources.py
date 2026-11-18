"""Validate CPU resource projection with real Stage1 and Stage2 workers."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess


def available_cpus():
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                                   ctypes.POINTER(ctypes.c_size_t)]
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        assert kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system))
        return [index for index in range(ctypes.sizeof(process) * 8) if process.value & (1 << index)]
    return sorted(os.sched_getaffinity(0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage1", "cpu-stage2", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-stage2", type=Path)
    parser.add_argument("--opencl-stage1", type=Path)
    parser.add_argument("--opencl-stage2", type=Path)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--opencl-device", type=int, default=0)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True)
    first, second, third = available_cpus()[:3]

    def run(name, exe, config, flags, *, data=None, error=None):
        folder = out / name
        folder.mkdir()
        path = folder / "ecm.toml"
        path.write_text(config, encoding="utf-8")
        result = subprocess.run([str(exe.resolve()), "--config", str(path), "--worker", "2", *flags],
                                input=data, text=True, encoding="utf-8", errors="replace", cwd=folder,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
        (folder / "run.log").write_text(result.stdout, encoding="utf-8")
        if error:
            assert result.returncode and error in result.stdout, result.stdout
        else:
            assert result.returncode == 0, result.stdout
        return result.stdout

    common = f'''config_version = 3
[shared]
log_file = ""
cpu_affinity = "{third}"
tune_file = ""
[stage1]
target = "cpu"
param = "e"
cpu_implementation = "scalar"
edwards_naf_window_bits = 6
cpu_threads = 4
gpu.init_threads = 1
[stage2]
cpu.threads = 1
cpu.backend = "gwpoly"
tune.runtime_enabled = false
[worker.2]
shared.cpu_affinity = "{second}, {first}-{second}"
stage1.cpu.threads = 1
stage1.gpu.init_threads = 3
stage2.cpu.threads = 2
'''
    stage1 = ["-gpucurves", "4", "-sigma", "26"]
    save = out / "ed.save"
    cpu = run("cpu", args.stage1, common,
              [*stage1, "-save", str(save), "128", "128"], data="2^127-1\n")
    pool = f"logical_cpus={first},{second}"
    assert pool in cpu and re.search(r"Stage1 execution: .*backend=cpu .*\bthreads=1\b", cpu), cpu
    gpu = run("gpu_init", args.stage1, common,
              [*stage1, "--target", "gpu", "--gpu-algorithm", "ntt", "-d", str(args.device), "128", "128"],
              data="2^127-1\n")
    assert pool in gpu and "stage1_ntt_edwards_setup: threads=2 " in gpu, gpu
    thread = run("cuda_thread_init", args.stage1, common,
                 [*stage1, "--target", "gpu", "--gpu-algorithm", "thread", "-d", str(args.device), "128", "128"],
                 data="2^127-1\n")
    assert pool in thread and "construction_threads=2" in thread, thread
    if args.opencl_stage1:
        thread = run("opencl_init", args.opencl_stage1, common,
                     [*stage1, "--target", "gpu", "--gpu-backend", "opencl", "-d", str(args.opencl_device),
                      "128", "128"], data="2^127-1\n")
        assert pool in thread and "construction_threads=2" in thread, thread
    gpu = run("legacy_cli_override", args.stage1, common,
              [*stage1, "--target", "gpu", "--gpu-algorithm", "ntt", "-d", str(args.device),
               "--gpu-init-threads", "1", "--stage1-threads", "3", "128", "128"], data="2^127-1\n")
    assert "stage1_ntt_edwards_setup: threads=1 " in gpu, gpu
    legacy = common.replace('stage1.cpu.threads = 1\nstage1.gpu.init_threads = 3', 'stage1.cpu_threads = 2')
    cpu = run("legacy_worker_cpu", args.stage1, legacy, [*stage1, "128", "128"], data="2^127-1\n")
    assert re.search(r"Stage1 execution: .*backend=cpu .*\bthreads=2\b", cpu), cpu
    legacy = legacy.replace(f'shared.cpu_affinity = "{second}, {first}-{second}"', 'shared.cpu_affinity = "none"')
    gpu = run("legacy_worker_gpu", args.stage1, legacy,
              [*stage1, "--target", "gpu", "--gpu-algorithm", "ntt", "-d", str(args.device), "128", "128"],
              data="2^127-1\n")
    assert "worker_cpu_affinity:" not in gpu and "stage1_ntt_edwards_setup: threads=2 " in gpu, gpu

    cpu2 = ["--save", str(save), "--b2", "1000", "--exponent", "127", "--curves", "1"]
    log = run("cpu_stage2", args.cpu_stage2, common, cpu2)
    assert pool in log and "backend=gwpoly threads=2" in log and "gwpoly: D=" in log, log
    bad = common.replace(f'{second}, {first}-{second}', '1023')
    run("invalid_cpu", args.cpu_stage2, bad, cpu2, error="logical CPU 1023 is unavailable")
    bad = common.replace(f'{second}, {first}-{second}', '1-0')
    run("invalid_syntax", args.stage1, bad, [*stage1, "128", "128"], data="2^127-1\n",
        error="expected comma-separated CPU indices/ranges")
    if args.gpu_stage2:
        log = run("gpu_stage2", args.gpu_stage2, common,
                  ["--save", str(save), "--b2", "1000", "--device", str(args.device), "--d", "210", "--curves", "1"])
        assert pool in log and "curve_done:" in log, log
    if args.opencl_stage2:
        log = run("opencl_stage2", args.opencl_stage2, common,
                  ["--save", str(save), "--b2", "1000", "--device", str(args.opencl_device), "--d", "210", "--curves", "1"])
        assert pool in log and "curve_done:" in log, log
    (out / "summary.json").write_text(json.dumps({"passed": True, "pool": [first, second],
                                                  "device": args.device, "gpu_stage2": bool(args.gpu_stage2),
                                                  "opencl_stage1": bool(args.opencl_stage1),
                                                  "opencl_stage2": bool(args.opencl_stage2)}))
    print("PASS worker CPU resources, legacy fallback, CLI precedence and affinity")


if __name__ == "__main__":
    main()
