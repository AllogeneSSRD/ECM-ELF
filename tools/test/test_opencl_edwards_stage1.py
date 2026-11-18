"""Verify the OpenCL or CUDA one-thread-per-curve Edwards worker."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time

from stage1_checkpoint_decode import batch
from test_cuda_queue_stop import SecurityAttributes
from test_edwards_gpu_pipeline import records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("opencl", "cuda-thread"), default="opencl")
    parser.add_argument("--mul", choices=("auto", "cios", "fips", "fips_ptx"), default="auto")
    parser.add_argument("--sqr", choices=("auto", "cios", "fips", "fips_ptx"), default="auto")
    parser.add_argument("--resume", action="store_true", help="also interrupt and resume a live worker")
    args = parser.parse_args()
    cuda = args.backend == "cuda-thread"
    if cuda and (args.mul != "auto" or args.sqr != "auto"):
        parser.error("CUDA thread uses its built-in field arithmetic")
    exe = args.exe.resolve(strict=True)
    root = args.out.resolve()
    root.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ECM_", "NTT_"))}
    summary = {"device": args.device, "checks": []}
    version = subprocess.check_output([str(exe), "--version"], env=env, cwd=root, text=True).strip()
    summary["program"] = version

    def cfg(folder, target, doubling, cap=0, period=0, start_cios=False, threads=0):
        conf = folder / "ecm.toml"
        mul = "cios" if start_cios else args.mul
        sqr = "cios" if start_cios else args.sqr
        thread_key = "stage1_cpu_threads" if target == "cpu" else "gpu_init_threads"
        conf.write_text(
            f'config_version = 4\n[shared]\nlog_file = ""\n[stage1]\nparam = "e"\n'
            f'target = "{target}"\nstage1_gpu_backend = "{"cuda" if cuda else "opencl"}"\nstage1_gpu_device = {args.device}\n'
            f'gpu_algorithm = "{"thread" if cuda and target == "gpu" else "cgbn"}"\n'
            f'cpu_implementation = "scalar"\nedwards_doubling = {doubling}\n'
            f'opencl_kernel_mul = "{mul}"\nopencl_kernel_sqr = "{sqr}"\n'
            f'naf_memory_mb = {cap}\ncheckpoint_interval_seconds = {period}\n{thread_key} = {threads}\n'
            f'handoff_mode = "off"\nstage1_save_dir = {json.dumps(folder.as_posix())}\n',
            encoding="ascii",
        )
        return conf

    def command(folder, config, bound, curves, window=6, sigma=20260922):
        return [str(exe), "--config", str(config), "--naf-w", str(window),
                "-sigma", str(sigma), "-gpucurves", str(curves),
                "-savea", str(folder / "point.save"), str(bound), "0"]

    def invoke(folder, config, value, bound, curves, window=6, sigma=20260922, expect=True):
        run = subprocess.run(command(folder, config, bound, curves, window, sigma),
                             input=f"{value}\n", capture_output=True, text=True,
                             encoding="utf-8", errors="replace", env=env, cwd=folder, timeout=180)
        log = run.stdout + run.stderr
        (folder / "run.log").write_text(log, encoding="utf-8")
        assert (run.returncode == 0) == expect, (folder, log[-3000:])
        return log

    # Top limbs are full width, including the exact limit for this arithmetic.
    fips = args.mul.startswith("fips") or args.sqr.startswith("fips")
    for width, gap in [(128, 159), (192, 237), (256, 189), (384, 317), (512, 569),
                       (768, 825), (1024, 105)]:
        if fips and width > 512:
            break
        n = (1 << width) - gap
        cpu = root / f"cpu_{width}"
        gpu = root / f"{args.backend}_{width}"
        cpu.mkdir()
        gpu.mkdir()
        bound, curves = 300, 3
        doubling = int(width != 384)
        window = {128: 2, 192: 8, 768: 10, 1024: 12}.get(width, 6)
        invoke(cpu, cfg(cpu, "cpu", doubling), n, bound, curves, max(window, 3))
        log = invoke(gpu, cfg(gpu, "gpu", doubling, threads=3), n, bound, curves, window)
        assert f"kernel_weight={width} tpi=1 naf={window}" in log
        assert f"doubling={5 if doubling else 4}" in log
        assert f"backend={args.backend} " in log
        if args.mul != "auto":
            assert f"field_mul={args.mul} " in log
        if args.sqr != "auto":
            sqr = f"{args.sqr}_square" if args.sqr.startswith("fips") else args.sqr
            assert f"field_sqr={sqr} " in log
        oracle = records(cpu / "point.save")
        assert len(oracle) == curves and records(gpu / "point.save") == oracle
        summary["checks"].append({"bits": width, "window": window, "points": len(oracle)})
        print(f"PASS {width} bits, w={window}", flush=True)

    factor_case = root / "factor"
    factor_case.mkdir()
    n = 1009 * ((1 << 127) - 1)
    invoke(factor_case, cfg(factor_case, "gpu", 1, threads=4), n, 1000, 4, sigma=17)
    hits = [row[1] for row in records(factor_case / "point.save") if row[-1] == "FACTOR"]
    assert hits and set(hits) == {1009}
    summary["checks"].append({"factor": 1009, "hits": len(hits)})

    memory = root / "memory"
    memory.mkdir()
    log = invoke(memory, cfg(memory, "gpu", 1, cap=1), (1 << 128) - 159, 300, 32, window=12)
    assert "NAF window 12 -> 10" in log and len(records(memory / "point.save")) == 32
    summary["checks"].append({"budget_mb": 1, "requested_w": 12, "selected_w": 10})

    oversized = root / "oversized"
    oversized.mkdir()
    log = invoke(oversized, cfg(oversized, "gpu", 1), (1 << 1024) + 1, 300, 1, expect=False)
    assert "up to 1024-bit" in log
    if fips:
        fips_limit = root / "fips_limit"
        fips_limit.mkdir()
        log = invoke(fips_limit, cfg(fips_limit, "gpu", 1), (1 << 512) + 1, 300, 1, expect=False)
        assert "FIPS supports up to 512-bit" in log

    if args.resume:
        folder = root / "resume"
        folder.mkdir()
        config = cfg(folder, "gpu", 1, period=0.01, start_cios=fips)
        bound, curves, n = (300000 if cuda else 100000), 4, (1 << 127) - 1
        launch = command(folder, config, bound, curves)
        event = None
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateEventW.argtypes = [ctypes.POINTER(SecurityAttributes), wintypes.BOOL,
                                            wintypes.BOOL, wintypes.LPCWSTR]
            kernel.CreateEventW.restype = wintypes.HANDLE
            kernel.SetEvent.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), None, True)
            event = kernel.CreateEventW(ctypes.byref(attributes), True, False, None)
            if not event:
                raise OSError(ctypes.get_last_error(), "CreateEventW")
            launch.extend(["--stage1-stop-handle", str(event)])
        with (folder / "input.txt").open("w+", encoding="ascii") as source, (folder / "stopped.log").open("wb") as output:
            source.write(f"{n}\n")
            source.seek(0)
            proc = subprocess.Popen(
                launch, stdin=source, stdout=output, stderr=subprocess.STDOUT, env=env, cwd=folder,
                close_fds=event is None, creationflags=subprocess.CREATE_NO_WINDOW if event else 0,
            )
            try:
                deadline = time.monotonic() + 120
                done = 0
                while not done and time.monotonic() < deadline and proc.poll() is None:
                    for path in folder.glob("*.ckpt"):
                        if ".bak" in path.name:
                            continue
                        try:
                            payload = batch(path)[1][0][1]
                            prefix = b"CUDA_ED_THREAD_V1" if cuda else b"OCL_ED_NAF_V1"
                            if payload.startswith(prefix):
                                done = struct.unpack_from("<I", payload, len(prefix))[0]
                        except (OSError, AssertionError, ValueError):
                            continue
                    if not done:
                        time.sleep(0.01)
                assert done and proc.poll() is None, "finished before the progress checkpoint"
                if event:
                    assert kernel.SetEvent(event)
                else:
                    proc.send_signal(signal.SIGINT)
                assert proc.wait(timeout=60) != 0
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if event:
                    kernel.CloseHandle(event)
        assert not (folder / "point.save").stat().st_size
        config = cfg(folder, "gpu", 1, period=0.01, threads=4)
        log = invoke(folder, config, n, bound, curves)
        assert re.search(r"resume=[1-9]\d*", log) and not list(folder.glob("*.ckpt"))
        full = root / "full"
        full.mkdir()
        invoke(full, cfg(full, "gpu", 1), n, bound, curves)
        assert (full / "point.save").read_bytes() == (folder / "point.save").read_bytes()
        summary["checks"].append({"resume": True, "b1": bound, "curves": curves})

    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {args.backend} Edwards save points, factors and memory admission")


if __name__ == "__main__":
    main()
