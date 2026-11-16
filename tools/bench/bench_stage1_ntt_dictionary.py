"""Compare Edwards NTT dictionary builders on the same Stage1 inputs."""

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))
from bench_stage1_naf_windows import monitor_gpu
from test_stage1_ntt_production import saved_points

CASES = ((4423, 512), (16381, 512), (65521, 256), (262139, 256))
BUILDERS = ("default", "host", "gpu")

def run(exe, folder, device, p, b1, curves, window, backend):
    folder.mkdir()
    config = folder / "ecm.toml"
    save = folder / "out.save"
    config.write_text(
        'config_version = 3\n[shared]\nlog_file = ""\n'
        f'[stage1]\nparam = "e"\ntarget = "gpu"\ngpu_algorithm = "ntt"\ngpu_device = {device}\n'
        f'edwards_naf_window_bits = {window}\nntt.convolution = "weighted"\nntt.window_bits = 16\n'
        f'save_dir = "{folder.as_posix()}"\nexponent_cache_dir = "off"\nhandoff.mode = "off"\n')
    env = {key: value for key, value in os.environ.items() if not key.startswith(("ECM_", "NTT_"))}
    if backend != "default":
        env["ECM_EDWARDS_NTT_DICTIONARY"] = backend
    start = time.perf_counter()
    path = folder / "run.log"
    with path.open("wb") as log:
        subprocess.run([str(exe), "--config", str(config), "--ckpt", "0", "-gpucurves", str(curves),
                        "-sigma", "26", "-save", str(save), str(b1), str(b1)],
                       cwd=folder, input=f"2^{p}-1\n".encode(), stdout=log, stderr=subprocess.STDOUT,
                       env=env, timeout=180, check=True)
    seconds = time.perf_counter() - start
    log = path.read_text()
    def find(pattern):
        match = re.search(pattern, log)
        assert match, (path, pattern)
        return match.groups()
    columns, device_ms, wall_ms, setup_ms = map(float, find(
        r"stage1_ntt_timing: (?:bits|columns)=(\d+) device_ms=([\d.]+) wall_ms=([\d.]+) "
        r"setup_ms=([\d.]+) readback_ms=[\d.]+ stopped=0"))
    owned_bytes, observed_bytes = map(int, find(r"stage1_ntt_memory: owned_bytes=(\d+) observed_bytes=(\d+)"))
    length, bpw = map(int, find(r"curves=\d+ length=(\d+) bpw=(\d+)"))
    version, = find(r"ECM-ELF (\d+\.\d+\.\d+[^ ]*) \(Stage1\)")
    naf_init = re.search(
        r"stage1_ntt_naf_init: ms=(?P<naf_init_ms>[\d.]+) encoding_bytes=(?P<encoding_bytes>\d+) "
        r"host_dictionary_bytes=(?P<host_dictionary_bytes>\d+) "
        r"dictionary_spectra_bytes=(?P<dictionary_spectra_bytes>\d+) "
        r"dictionary_workspace_bytes=(?P<dictionary_workspace_bytes>\d+)", log)
    prep = dict(naf_init_ms=None, encoding_bytes=None, host_dictionary_bytes=None,
                dictionary_spectra_bytes=None, dictionary_workspace_bytes=None)
    if naf_init:
        for key, value in naf_init.groupdict().items():
            prep[key] = float(value) if key == "naf_init_ms" else int(value)
    parts = re.search(r"stage1_ntt_naf_init_parts: ([^\r\n]+)", log)
    if parts:
        for item in parts[1].split():
            key, value = item.split("=", 1)
            prep[key] = float(value) if key.endswith("_ms") else int(value)
    if window:
        host_ms, = map(float, find(r"host_dictionary_ms=([\d.]+)"))
        gpu_ms, actual, inverse_ms = find(r"stage1_ntt_edwards_dictionary: gpu_ms=([\d.]+) backend=(\w+) root_inverse_ms=([\d.]+)")
        assert actual in ("host", "gpu") and (backend == "default" or actual == backend), path
    else:
        host_ms, gpu_ms, inverse_ms, actual = 0, 0, 0, "none"
    return dict(version=version, columns=int(columns), device_ms=device_ms, wall_ms=wall_ms, setup_ms=setup_ms,
                process_s=seconds, length=length, bpw=bpw, owned_bytes=owned_bytes, observed_bytes=observed_bytes,
                peak_bytes=max(owned_bytes, observed_bytes),
                host_dictionary_ms=host_ms, gpu_dictionary_ms=float(gpu_ms), root_inverse_ms=float(inverse_ms),
                backend=actual, **prep), saved_points(save, p, b1, True)


def collect(exe, device, output):
    dest = output / f"gpu{device}"
    dest.mkdir()
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(dest/"telemetry.csv", device, stop))
    monitor.start()
    rows = []
    try:
        for p, b1 in CASES:
            for curves in (1, 8, 30):
                cases = [(0, "default")] + [(w, builder) for w in (6, 8) for builder in BUILDERS]
                measurements = {case: [] for case in cases}
                expected = None
                for repeat in range(3):
                    for window, backend in cases[repeat:] + cases[:repeat]:
                        row, points = run(exe, dest/f"m{p}-c{curves}-w{window}-{backend}-r{repeat}",
                                          device, p, b1, curves, window, backend)
                        if expected is None:
                            expected = points
                        assert points == expected and len(points) == curves
                        measurements[window, backend].append(row)
                for (window, backend), samples in measurements.items():
                    rows.append(dict(p=p, b1=b1, curves=curves, window=window, requested=backend, samples=samples))
                (dest/"summary.json").write_text(json.dumps(rows, indent=2) + "\n")
                init_s = {f"w{w}-{b}": round(statistics.median(s["naf_init_ms"] for s in samples) / 1000, 4)
                          for (w, b), samples in measurements.items()
                          if all(s["naf_init_ms"] is not None for s in samples)}
                process_s = {f"w{w}-{b}": round(statistics.median(s["process_s"] for s in samples), 4)
                             for (w, b), samples in measurements.items()}
                print(device, p, curves, f"naf_init_s={init_s}", f"process_s={process_s}", flush=True)
    finally:
        stop.set()
        monitor.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=(0, 1))
    args = parser.parse_args()
    exe, output = args.exe.resolve(strict=True), args.output.resolve()
    output.mkdir(parents=True)
    with ThreadPoolExecutor(len(args.devices)) as pool:
        tasks = [pool.submit(collect, exe, device, output) for device in args.devices]
        for task in tasks:
            task.result()
