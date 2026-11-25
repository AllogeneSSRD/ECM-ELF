#!/usr/bin/env python3
"""Revalidate an Edwards Stage1 tune profile with timed production batches."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import sys
import time
import tomllib

from bench_stage1_ntt_best import stepped_b1
from bench_stage1_ntt_production import GPU_FIELDS
from plot_stage1_tune import candidate_lengths, size_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))
from test_stage1_ntt_production import saved_points


def prime_at_most(p):
    for n in range(p if p % 2 else p - 1, 2, -2):
        if all(n % d for d in range(3, math.isqrt(n) + 1, 2)):
            return n
    raise ValueError(f"Exponent must be at least 3: {p}")


def write_report(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def choose_rows(fields, rows, p, budget):
    limits = size_limits()
    shapes = candidate_lengths(p, True, limits)
    contenders = [r for r in rows if r["curve"] == "edwards" and r["benchmark_kind"] == "stage1_naf_stratified"
                  and r["max_exp"] >= p and 0 < r["ntt_memory_bytes"] <= budget and r["iter_per_s"] > 0]
    matching = [r for r in contenders if (r["ntt_type"], r["length"]) in shapes]
    if not matching and contenders:
        shortest = min(r["length"] for r in contenders)
        matching = [r for r in contenders if r["length"] == shortest]
    # Several measurements may name the same configuration. Try each shape once.
    seen = set()
    keys = ("ntt_type", "length", "bpw", "batch", "naf_window", "tile", "max_radix_bits", "graph_bits", "warp_tail", "dataflow")
    for row in sorted(matching, key=lambda r: r["iter_per_s"], reverse=True):
        identity = tuple(row[k] for k in keys)
        if identity not in seen:
            seen.add(identity)
            yield row, [row[k] for k in fields]


def measure(args, folder, database, p, b1, sigma, row):
    folder.mkdir()
    save = folder / "point.save"
    cfg = folder / "ecm.toml"
    lines = ["config_version = 4", "[shared]", 'log_file = ""', f'tune_file = "{database.as_posix()}"',
             "[stage1]", 'param = "e"', 'target = "gpu"', 'gpu_algorithm = "ntt"',
             f"stage1_gpu_device = {args.device}", f"ntt_memory_mib = {args.memory_mib}",
             f"edwards_naf_window_bits = {row['naf_window']}", f"ntt_length = {row['length']}",
             f'ntt_convolution = "{row["ntt_type"].removeprefix("stage1_")}"',
             f"ntt_tile = {row['tile']}", f"ntt_outer_radix_limit = {row['max_radix_bits']}",
             f"ntt_window_bits = {row['graph_bits']}", f'ntt_warp_tail = "{str(bool(row["warp_tail"])).lower()}"',
             'exponent_cache_dir = "off"',
             'handoff_mode = "off"']
    if args.naf_gpu_min_bits is not None:
        lines.append(f"ntt_naf_gpu_min_bits = {args.naf_gpu_min_bits}")
    if row["ntt_type"] == "stage1_linear":
        lines.append(f"ntt_bits_per_word = {row['bpw']}")
    lines += ["[stage2]", "tune_runtime_enabled = false"]
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    batch = row["batch"]
    command = [str(args.exe), "--config", str(cfg), "--ckpt", "0", "-sigma", str(sigma),
               "-gpucurves", str(batch), "-save", str(save), str(b1), str(b1)]
    path = folder / "driver.log"
    start = time.perf_counter()
    with path.open("wb") as handle:
        process = subprocess.run(command, cwd=folder, input=f"2^{p}-1\n".encode(), stdout=handle,
                                 stderr=subprocess.STDOUT, timeout=300, env=args.env)
    wall = time.perf_counter() - start
    log = path.read_text(encoding="utf-8", errors="replace")
    plans = re.findall(r"stage1_ntt_plan: ([^\r\n]+)", log)
    windows = re.findall(r"stage1_ntt_edwards_naf: window=(\d+)", log)
    timing = re.findall(r"stage1_ntt_timing: columns=(\d+) device_ms=([\d.]+) wall_ms=([\d.]+) "
                        r"setup_ms=([\d.]+) readback_ms=([\d.]+) stopped=0", log)
    if (process.returncode or "Stage1 NTT tune read:" in log or len(plans) != 1
            or len(timing) != 1 or windows != [str(row["naf_window"])]):
        raise RuntimeError(f"Incomplete native Edwards batch: {path}")
    plan = dict(re.findall(r"(\w+)=([^ ]+)", plans[0]))
    layout_bits = (1 << 25) | (1 << 24) | 384
    if (int(plan["device"]) != args.device or int(plan["p"]) != p or int(plan["length"]) != row["length"]
            or int(plan["tile_layers"]) != row["tile"] or int(plan["graph_bits"]) != row["graph_bits"]
            or int(plan["outer_max_layers"]) != row["max_radix_bits"] or int(plan["warp_tail"]) != row["warp_tail"]
            or plan["convolution"] != row["ntt_type"].removeprefix("stage1_")
            or int(plan["dataflow"]) & layout_bits != row["dataflow"] & layout_bits
            or not 0 < int(plan["curves"]) <= batch):
        raise RuntimeError(f"Tune plan did not replay: {path}")
    points = saved_points(save, p, b1, edwards=True) if save.exists() else {}
    identities = set(map(int, re.findall(r"Edwards identity: sigma=(\d+)", log)))
    if set(points) | identities != set(range(sigma, sigma + batch)):
        raise RuntimeError(f"Missing curve outcomes: {path}")
    statuses = {}
    if save.exists():
        for line in save.read_text(encoding="utf-8").splitlines():
            fields = dict(item.strip().split("=", 1) for item in line.split(";") if "=" in item)
            key = int(fields["SIGMA"])
            statuses[key] = fields["STATUS"]
            if statuses[key] == "FACTOR":
                modulus = (1 << p) - 1
                factor = points[key]
                if not (1 < factor < modulus and modulus % factor == 0):
                    raise ValueError(f"Invalid factor: {path}")
    outcomes = hashlib.sha256()
    for key, value in sorted(points.items()):
        if key in identities and (value != 0 or statuses[key] != "IDENTITY"):
            raise RuntimeError(f"Identity disagrees with save record: {path}")
        outcomes.update(f"{key} {statuses[key]} {value:x}\n".encode("ascii"))
    for key in sorted(identities - points.keys()):
        outcomes.update(f"{key} identity\n".encode("ascii"))
    columns, device_ms, loop_ms, setup_ms, readback_ms = timing[0]
    dictionary = re.search(r"stage1_ntt_edwards_dictionary: .*?backend=(gpu|host)", log)
    return dict(columns=int(columns), device_ms=float(device_ms), loop_wall_ms=float(loop_ms),
                setup_ms=float(setup_ms), readback_ms=float(readback_ms), process_seconds=wall,
                active_curves=int(plan["curves"]), plan=plan, outcomes_sha256=outcomes.hexdigest(), identities=len(identities),
                dictionary_backend=dictionary[1] if dictionary else None,
                log=path.relative_to(args.output).as_posix())


def collect(args, report, tune):
    table = tune.read_text(encoding="utf-8-sig")
    tune_doc = tomllib.loads(table)
    fields, *values = tune_doc["ntt"]["matrix"]
    uuid = report["gpu_uuid"].lower().removeprefix("gpu-").replace("-", "")
    conditions = tune_doc["condition"]
    rows = [dict(zip(fields, row)) for row in values if conditions[str(row[fields.index("condition")])]["uuid_hex"] == uuid]
    header = (f'[profile]\nformat = {tune_doc["profile"]["format"]}\nunit = "{tune_doc["profile"]["unit"]}"\n'
              f'\n[condition.0]\nuuid_hex = "{uuid}"\n\n[ntt]\n')
    for p in report["exponents"]:
        if any(group["p"] == p for group in report["groups"]):
            continue
        group = args.output / f"p{p}"
        group.mkdir(exist_ok=True)
        for index, (row, values) in enumerate(choose_rows(fields, rows, p, args.memory_mib * 2**20)):
            attempt = group / f"plan{index}"
            generation = 0
            while attempt.exists():
                generation += 1
                attempt = group / f"plan{index}-run{generation}"
            attempt.mkdir()
            database = attempt / "plan.toml"
            values[fields.index("condition")] = 0
            database.write_text(header + "matrix = [\n" + json.dumps(fields) + ",\n" + json.dumps(values) + ",\n]\n",
                                encoding="utf-8")
            sigma = args.sigma
            # Start with roughly one tenth of the target time, using 1.442 NAF columns per B1.
            b1 = stepped_b1(1, 1442 * row["batch"] / row["iter_per_s"], args.seconds / 10)
            try:
                pilot = measure(args, attempt / "pilot", database, p, b1, sigma, row)
                b1 = stepped_b1(b1, pilot["device_ms"], args.seconds)
                if not b1:
                    raise RuntimeError(f"Invalid B1: {attempt}")
                for trial in range(4):
                    sample = measure(args, attempt / f"calibrate{trial}", database, p, b1, sigma, row)
                    revised = stepped_b1(b1, sample["device_ms"], args.seconds)
                    if abs(sample["device_ms"] / (1000 * args.seconds) - 1) <= 0.17 or revised == b1 or trial == 3:
                        break
                    b1 = revised
                samples = [sample]
                for trial in range(1, args.rounds):
                    samples.append(measure(args, attempt / f"formal{trial}", database, p, b1, sigma, row))
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                print(f"M{p} retry: {exc}", flush=True)
                if index >= 4:
                    raise
                continue
            fingerprints = {(s["outcomes_sha256"], s["active_curves"], s["columns"], s["plan"]["dataflow"]) for s in samples}
            if len(fingerprints) != 1:
                raise RuntimeError(f"Repeated results differ: {attempt}")
            item = dict(p=p, b1=b1, sigma=sigma, batch=row["batch"], active_batch=sample["active_curves"],
                        tune=row, pilot=pilot, samples=samples, raw_source=attempt.relative_to(args.output).as_posix(),
                        projected_seconds_per_curve_1e6=[s["device_ms"] * 1000 / (s["active_curves"] * b1) for s in samples])
            report["groups"].append(item)
            write_report(args.output / "summary.json", report)
            t = median(s["device_ms"] for s in samples) / 1000
            print(f"{len(report['groups']):02}/{len(report['exponents'])} M{p} L={row['length']} w{row['naf_window']} "
                  f"{sample['active_curves']}/{row['batch']}c B1={b1} main={t:.3f}s dataflow={sample['plan']['dataflow']}",
                  flush=True)
            break
        else:
            raise RuntimeError(f"No compatible tune plan accepted for M{p}: {group}")


def lock_output(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def run(args):
    args.env = {k: v for k, v in os.environ.items() if not k.startswith(("ECM_", "NTT_"))}
    gpu = ["nvidia-smi", "-i", str(args.device)]
    name, uuid, driver = subprocess.check_output(
        gpu + ["--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip().split(", ")
    info = dict(version=subprocess.check_output([str(args.exe), "--version"], text=True).strip(),
                executable_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),
                tune_sha256=hashlib.sha256(args.tune.read_bytes()).hexdigest(),
                curve="ed", device=args.device, gpu_model=name, gpu_uuid=uuid, driver=driver, rounds=args.rounds,
                reference_b1=1_000_000, target_main_seconds=args.seconds, memory_mib=args.memory_mib,
                exponents=list(dict.fromkeys(prime_at_most(p) for p in args.exponents)))
    summary = args.output / "summary.json"
    report = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else dict(info, groups=[])
    if any(report[k] != value for k, value in info.items()):
        raise ValueError("Measurement identity changed. Choose a new output directory.")
    write_report(summary, report)
    trace = args.output / f"telemetry-{time.time_ns()}.csv"
    with trace.open("wb") as log:
        monitor = subprocess.Popen(gpu + [f"--query-gpu={GPU_FIELDS}", "--format=csv", "-lms", "500"],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            collect(args, report, args.tune)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("exe", "tune", "output"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    parser.add_argument("--exponents", required=True, nargs="+", type=int,
                        help="Use the prime exponent at or below each input width")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--memory-mib", type=int, default=8000)
    parser.add_argument("--naf-gpu-min-bits", type=int, help="Override the production automatic dictionary threshold")
    parser.add_argument("--sigma", type=int, default=20260922)
    args = parser.parse_args()
    if args.rounds < 1 or args.seconds <= 0 or args.memory_mib < 1:
        parser.error("Rounds, target seconds, and memory must be positive")
    args.exe = args.exe.resolve(strict=True)
    args.tune = args.tune.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "collector.lock").open("a+b") as handle:
        try:
            lock_output(handle)
        except OSError as exc:
            parser.error(f"Cannot lock the benchmark output {args.output}: {exc}")
        run(args)


if __name__ == "__main__":
    main()
