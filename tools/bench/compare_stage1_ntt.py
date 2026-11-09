#!/usr/bin/env python3
"""Compare two Stage1 NTT implementations in alternating order and verify their saves."""
import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path

from bench_stage1_ntt_production import GPU_FIELDS, records


def measure(exe, directory, args, p, curves, convolution, length):
    directory.mkdir()
    save = directory / "out.save"
    config = directory / "ecm.toml"
    lines = ["[stage1]", 'curve = "mont"', "gpu_param = 0", 'lcm = "1"',
             f"device = {args.device}", 'exp_cache = "off"', 'log_file = ""',
             "ckpt_seconds = 0", "gpu_algorithm = 2"]
    if convolution:
        lines += [f'ntt_convolution = "{convolution}"']
    if length:
        lines += [f"ntt_length = {length}"]
    config.write_text("\n".join(lines) + "\n", encoding="ascii")
    cmd = [str(exe), "--config", str(config), "--gpu-backend", "cuda",
           "-sigma", str(args.sigma), "-gpucurves", str(curves),
           "-save", str(save), str(args.b1), str(args.b1)]
    process = subprocess.run(cmd, cwd=directory, input=f"2^{p}-1\n", text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             encoding="utf-8", errors="replace", timeout=180)
    log = directory / "driver.log"
    log.write_text(process.stdout, encoding="utf-8")
    plan = re.search(r"stage1_ntt_plan: (.+)", process.stdout)
    timing = re.search(r"stage1_ntt_timing: bits=(\d+) device_ms=([.\d]+) wall_ms=([.\d]+)"
                       r" setup_ms=([.\d]+) readback_ms=([.\d]+) stopped=0", process.stdout)
    if process.returncode or not plan or not timing:
        raise RuntimeError(f"missing successful Stage1 timing: {log}")
    saved = records(save)
    if set(saved) != set(range(args.sigma, args.sigma + curves)):
        raise RuntimeError(f"missing curve results: {log}")
    factors = sorted(re.findall(r"factor\[\d+\]=(\d+) curve=\d+ sigma=(\d+)", process.stdout))
    row = dict(zip(("bits", "device_ms", "wall_ms", "setup_ms", "readback_ms"), map(float, timing.groups())))
    row["plan"] = dict(item.split("=", 1) for item in plan[1].split())
    return row, (saved, factors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-convolution", choices=("linear", "weighted"))
    parser.add_argument("--candidate-convolution", choices=("linear", "weighted"))
    parser.add_argument("--baseline-length", type=int, default=0)
    parser.add_argument("--candidate-length", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--b1", type=int, default=256)
    parser.add_argument("--sigma", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--exponents", type=int, nargs="+", default=(4003, 8191, 16001, 32749, 65521, 131071))
    parser.add_argument("--curves", type=int, nargs="+", default=(1, 4, 8))
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    workers = [args.baseline.resolve(strict=True), args.candidate.resolve(strict=True)]
    convolution = [args.baseline_convolution, args.candidate_convolution]
    lengths = [args.baseline_length, args.candidate_length]
    versions = [subprocess.run([str(exe), "--version"], cwd=args.output, check=True,
                              capture_output=True, text=True, timeout=30).stdout.strip() for exe in workers]
    report = dict(versions=versions, convolution=convolution, device=args.device, b1=args.b1,
                  sigma=args.sigma, rounds=args.rounds, groups=[])
    with (args.output / "telemetry.csv").open("w", encoding="utf-8") as log:
        telemetry = subprocess.Popen(
            ["nvidia-smi", "-i", str(args.device), "-lms", "200", f"--query-gpu={GPU_FIELDS}",
             "--format=csv,nounits"], stdout=log, stderr=subprocess.STDOUT)
        try:
            for p in args.exponents:
                for curves in args.curves:
                    samples = [[], []]
                    reference = None
                    for sample in range(args.rounds):
                        for i in (sample % 2, 1 - sample % 2):
                            folder = args.output / f"p{p}_b{curves}_side{i}_sample{sample + 1}"
                            row, values = measure(workers[i], folder, args, p, curves, convolution[i], lengths[i])
                            if reference is None:
                                reference = values
                            if values != reference:
                                raise RuntimeError(f"Stage1 save or factor differs: {folder}")
                            samples[i].append(row)
                    medians = [statistics.median(r["device_ms"] for r in side) for side in samples]
                    report["groups"].append(dict(p=p, curves=curves, samples=samples, median_ms=medians))
                    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    print(f"p={p} curves={curves}: {medians[0]:.3f} -> {medians[1]:.3f} ms", flush=True)
        finally:
            telemetry.terminate()
            telemetry.wait()


if __name__ == "__main__":
    main()
