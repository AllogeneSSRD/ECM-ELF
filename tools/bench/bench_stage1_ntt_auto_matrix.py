#!/usr/bin/env python3
"""Compare the Stage1 production auto selector against the best tuned binary plan."""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import tomllib

from bench_stage1_ntt_best import HISTORICAL_EXPONENTS, run
from bench_stage1_ntt_production import EXPONENTS, GPU_FIELDS, records

MIXED_RANGES = ((53249, 64000, 2560), (102401, 128000, 5120), (204801, 245760, 10240))
REPRESENTATIVES = (58631, 115201, 225287)
STEPS = 1478
PLAN_FIELDS = {"length": "length", "bpw": "bpw", "tile_layers": "tile",
               "outer_max_layers": "max_radix_bits", "graph_bits": "graph_bits", "warp_tail": "warp_tail",
               "forward_passes": "passes_forward", "forward_layers": "forward", "inverse_layers": "inverse"}


def best_binary(tune, batch):
    columns, *values = tomllib.loads(tune.read_text(encoding="utf-8"))["ntt"]["matrix"]
    rows = [dict(zip(columns, row)) for row in values]
    candidates = [row for row in rows if row["batch"] == batch and row["length"] > 0
                  and not (row["length"] & (row["length"] - 1)) and row["ntt_type"].startswith("stage1_")]
    return max(candidates, key=lambda row: row["iter_per_s"])


def fixed_config(row):
    mode = row["ntt_type"].removeprefix("stage1_")
    config = f'ntt_convolution = "{mode}"\nntt_warp_tail = "{str(bool(row["warp_tail"])).lower()}"\n'
    if mode == "linear":
        config += f'ntt_bpw = {row["bpw"]}\n'
    keys = {"length": "length", "tile": "tile", "outer_max": "max_radix_bits", "window": "graph_bits"}
    return config + "".join(f"ntt_{k} = {row[v]}\n" for k, v in keys.items())


def read_sample(log, batch):
    line = re.search(r"stage1_ntt_plan: ([^\r\n]+)", log)[1]
    plan = dict(re.findall(r"(\w+)=([^ ]+)", line))
    if int(plan["curves"]) != batch:
        raise RuntimeError("Unexpected active batch size")
    steps, ms = re.search(r"stage1_ntt_timing: bits=(\d+) device_ms=([.\d]+)", log).groups()
    return plan, dict(steps=int(steps), device_ms=float(ms), us_per_iter=1000 * float(ms) / (int(steps) * batch))


def measure_case(exe, probe, out, device, p, boundary, rounds):
    work = out / f"p{p}"
    work.mkdir()
    tune = work / "tune.toml"
    batch = 32 if p <= 32749 else 24
    b1 = 1024 if p > 20000 or p < 128 else 10000
    sigma = 20260922
    config = (f'[stage1]\ndevice = {device}\ngpu_algorithm = "ntt"\ncurve = "mont"\ngpu_param = 0\n'
              f'ntt_tune_file = "{tune.as_posix()}"\n'
              'lcm = "1"\nexp_cache = "off"\nlog_file = ""\nckpt_seconds = 0\n')
    flags = ["-sigma", str(sigma), "-gpucurves", str(batch), "-save", "point.save", str(b1), str(b1)]

    def trial(name, fixed=None):
        folder = work / name
        if boundary:
            folder.mkdir()
            command = [str(probe), str(device), str(p), str(batch), str(STEPS), str(tune)]
            if fixed:
                command += [str(int(fixed["ntt_type"] == "stage1_weighted"))]
                command += [str(fixed[k]) for k in ("length", "bpw", "tile", "max_radix_bits", "graph_bits", "warp_tail")]
            result = subprocess.run(command, cwd=folder, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=600)
            log = result.stdout + result.stderr
            (folder / "driver.log").write_text(log, encoding="utf-8")
            result.check_returncode()
            endpoint = re.search(r"stage1_ntt_state: hash=(\w+)", log)[1]
        else:
            log = run(exe, folder, config + (fixed_config(fixed) if fixed else ""), flags, f"2^{p}-1\n")
            endpoint = records(folder / "point.save")
            if set(endpoint) != set(range(sigma, sigma + batch)):
                raise RuntimeError(f"Incomplete production save at {folder}")
        plan, timing = read_sample(log, batch)
        if fixed and (any(plan[k] != str(fixed[v]) for k, v in PLAN_FIELDS.items()) or
                      fixed["ntt_type"] != f'stage1_{plan["convolution"]}'):
            raise RuntimeError(f"Wrong binary plan in {folder}")
        if not fixed and name != "seed" and "stage1_ntt_selection: source=tune" not in log:
            raise RuntimeError(f"Auto timing must use cached tuning in {folder}")
        return plan, timing, endpoint

    auto, _, reference = trial("seed")
    binary = best_binary(tune, batch)
    samples = {"auto": [], "binary": []}
    for repeat in range(rounds):
        order = ("auto", "binary") if repeat % 2 == 0 else ("binary", "auto")
        for mode in order:
            plan, timing, endpoint = trial(f"{mode}{repeat+1}", binary if mode == "binary" else None)
            if endpoint != reference or (mode == "auto" and plan != auto):
                raise RuntimeError(f"Endpoint or cached plan changed at p={p} ({mode})")
            samples[mode].append(timing)
    return dict(p=p, batch=batch, b1=None if boundary else b1, synthetic=boundary, auto_plan=auto,
                binary_plan=binary, samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=1, type=int)
    parser.add_argument("--rounds", default=3, type=int)
    args = parser.parse_args()
    exe, probe = (p.resolve(strict=True) for p in (args.exe, args.probe))
    versions = [subprocess.check_output([str(p), "--version"], text=True).strip() for p in (exe, probe)]
    if versions != ["ECM-ELF 0.60.2"] * 2 or args.rounds < 1:
        parser.error("Use 0.60.2 worker and probe with at least one repeat")
    device = args.device
    gpu_model = subprocess.check_output(["nvidia-smi", "-i", str(device), "--query-gpu=name",
                                         "--format=csv,noheader"], text=True, timeout=10).strip()
    edges = [p for low, high, _ in MIXED_RANGES for p in (low - 1, low, high, high + 1)]
    matrix = sorted(set(EXPONENTS + HISTORICAL_EXPONENTS + REPRESENTATIVES) - set(edges))
    out = args.output.resolve()
    out.mkdir(parents=True)
    report = dict(version=versions[0], device=device, gpu_model=gpu_model, rounds=args.rounds,
                  mixed_ranges=MIXED_RANGES, exponents=sorted(edges + matrix), groups=[])
    with (out / "telemetry.csv").open("w", encoding="utf-8") as log:
        telemetry = subprocess.Popen(["nvidia-smi", "-i", str(device), "-lms", "200", f"--query-gpu={GPU_FIELDS}",
                                      "--format=csv,nounits"], stdout=log, stderr=subprocess.STDOUT)
        try:
            for p in edges + matrix:
                row = measure_case(exe, probe, out, device, p, p in edges, args.rounds)
                report["groups"].append(row)
                (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                a, b = (statistics.median(s["us_per_iter"] for s in row["samples"][name]) for name in ("auto", "binary"))
                print(f'p={p} {"edge" if p in edges else "production"} '
                      f'{row["auto_plan"]["length"]}/{row["binary_plan"]["length"]} '
                      f'{a:.4f}/{b:.4f} us/iter/curve', flush=True)
        finally:
            telemetry.terminate()
            telemetry.wait()


if __name__ == "__main__":
    main()
