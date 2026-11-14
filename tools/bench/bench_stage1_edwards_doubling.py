"""Pair HWCD and 2XY runs on one CUDA GPU using the production Stage1 executable."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from bench_edwards_gpu_stage1 import validate
from bench_stage1_naf_windows import GPU_FIELDS, gpu_info, marker, monitor_gpu, write_report


# (Mersenne exponent, B1, compiled Edwards CGBN container, TPI).
CASES = [
    (347, 1_000_000, 384, 4),
    (607, 500_000, 768, 8),
    (1279, 250_000, 1280, 8),
    (2203, 120_000, 2560, 16),
    (4253, 80_000, 4608, 16),
    (8191, 50_000, 8192, 16),
    (10267, 50_000, 11264, 32),
    (12511, 50_000, 13312, 32),
    (16319, 50_000, 16384, 32),
]
SIGMA = 20260922
TPB = 128
WINDOW = 6
ALGORITHMS = {"hwcd": 4, "2xy": 5}


def run(exe, folder, device, sms, case, algorithm, repeat):
    bits, b1, container, tpi = case
    blocks = 2 * sms
    curves = blocks * (TPB // tpi)
    folder.mkdir()
    config = folder / "ecm.toml"
    config.write_text('config_version = 3\n[stage1]\nhandoff.mode = "off"\n')
    command = [str(exe), "--config", str(config), "--param", "e", "-sigma", str(SIGMA),
               "--target", "gpu", "--gpu-backend", "cuda", "--gpu-algorithm", "cgbn",
               "-d", str(device), "-gpucurves", str(curves), "--naf-w", str(WINDOW),
               "--ckpt", "0", "--exp-cache", "off", "--tmp-dir", str(folder),
               "-save", "completed.save", "-v", str(b1), "0"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ECM_", "NTT_"))}
    env["ECM_EDWARDS_STAGE1_ALGO"] = "naf"
    env["ECM_EDWARDS_DBL_ALGO"] = str(ALGORITHMS[algorithm])
    (folder / "command.json").write_text(json.dumps(
        {"argv": command, "doubling": ALGORITHMS[algorithm], "modulus": f"2^{bits}-1"},
        indent=2) + "\n")
    started = time.perf_counter()
    with (folder / "run.log").open("wb") as log:
        subprocess.run(command, cwd=folder, env=env, input=f"(2^{bits}-1)\n".encode(),
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    wall = time.perf_counter() - started
    lines = [(0, line) for line in (folder / "run.log").read_text(errors="replace").splitlines()]
    _, banner = marker(lines, r"ECM-ELF (\d+\.\d+\.\d+) \(Stage1\)")
    marker(lines, rf"GPU Edwards Stage1: device={device} curves={curves} N_bits={bits} "
                  rf"scalar_bits=\d+ window={WINDOW} doubling={ALGORITHMS[algorithm]}\b")
    marker(lines, rf"GPU Edwards geometry: container_bits={container} tpi={tpi} "
                  rf"tpb={TPB} blocks={blocks}\b")
    _, metrics = marker(lines, r"GPU Edwards NAF: columns=(\d+) additions=(\d+) "
                        r"container_bits=(\d+) tpi=(\d+) dictionary_bytes=(\d+) "
                        r"dictionary_gpu_ms=([\d.]+) loop_gpu_ms=([\d.]+) launches=(\d+)")
    columns, additions, actual_container, actual_tpi, dictionary_bytes = map(int, metrics.groups()[:5])
    expected_bytes = curves * (container // 8) * 4 * 2**(WINDOW - 2)
    if (actual_container, actual_tpi, dictionary_bytes) != (container, tpi, expected_bytes):
        raise ValueError(f"{folder}: unexpected NAF geometry or dictionary allocation")
    save = folder / "completed.save"
    points = validate(save, (1 << bits) - 1, b1, curves, SIGMA, "ed")
    return dict(device=device, bits=bits, b1=b1, curves=curves, container_bits=container, tpi=tpi,
                blocks=blocks, algorithm=algorithm, repeat=repeat, version=banner[1],
                columns=columns, additions=additions, launches=int(metrics[8]), wall_s=wall,
                dictionary_gpu_s=float(metrics[6]) / 1000, loop_gpu_s=float(metrics[7]) / 1000,
                log=folder.name), points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, choices=(0, 1), required=True)
    parser.add_argument("--bits", type=int, nargs="+", choices=[case[0] for case in CASES])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3 or (args.bits and len(args.bits) != len(set(args.bits))):
        parser.error("use at least three pairs and unique exponents")
    exe = args.exe.absolute()
    if not exe.is_file():
        parser.error(f"executable not found: {exe}")
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    device = args.device
    sms = 60 if device == 0 else 24
    (out / "gpu_inventory.csv").write_text(GPU_FIELDS + "\n" + gpu_info(device))
    report = dict(complete=False, device=device, sigma=SIGMA, window=WINDOW, samples=[])
    report_path = out / "report.json"
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(out / "telemetry.csv", device, stop))
    monitor.start()
    try:
        cases = [case for case in CASES if not args.bits or case[0] in args.bits]
        for case_index, case in enumerate(cases):
            points = None
            for repeat in range(args.repeats):
                order = list(ALGORITHMS)
                if (repeat + case_index + device) % 2:
                    order.reverse()
                for algorithm in order:
                    name = f"m{case[0]}_{algorithm}_r{repeat}"
                    print("START", name, flush=True)
                    row, saved = run(exe, out / name, device, sms, case, algorithm, repeat)
                    if report.setdefault("version", row["version"]) != row["version"]:
                        raise ValueError(f"{name}: executable version changed during collection")
                    if points is not None and points != saved:
                        raise ValueError(f"{name}: saved points differ")
                    points = saved
                    report["samples"].append(row)
                    write_report(report_path, report)
                    print(f'PASS {name} loop={row["loop_gpu_s"]:.3f}s wall={row["wall_s"]:.3f}s',
                          flush=True)
        report["complete"] = True
        write_report(report_path, report)
    finally:
        stop.set()
        monitor.join()


if __name__ == "__main__":
    main()
