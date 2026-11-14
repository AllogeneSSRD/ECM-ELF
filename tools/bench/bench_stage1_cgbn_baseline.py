"""Measure CUDA Edwards NAF and PARAM0 PRAC on the historical cofactor matrix."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading

from analyze_prime95_stage1 import ROOT, input_scope
from analyze_stage1_gpu import exact_modulus
from bench_edwards_gpu_stage1 import integer_expression
from bench_stage1_naf_windows import GPU_FIELDS, gpu_info, monitor_gpu, write_report
from test_edwards_gpu_pipeline import checksum


SIGMA = 20260922
REFERENCE_B1 = 1_000_000
WINDOW = 12
SEARCH = 50
DEPTH = 3
MODELS = ("ed", "mont0")


def one(log, pattern):
    matches = re.findall(pattern, log, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern!r}, found {len(matches)}")
    return matches[0]


def verify_save(path, n, b1, curves, model):
    # Montgomery save rows are Stage2 points only. A factor or identity can
    # be represented only in the result log, so count completed outcomes there.
    rows = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        keys = dict(part.strip().split("=", 1) for part in line.split(";") if "=" in part)
        sigma, x, saved_n = int(keys["SIGMA"]), int(keys["X"], 0), integer_expression(keys["N"])
        if (sigma in rows or not SIGMA <= sigma < SIGMA + curves or int(keys["B1"]) != b1
                or saved_n != n or not 0 <= x < n):
            raise ValueError(f"invalid save row in {path}: sigma {sigma}")
        if model == "ed":
            status = keys["STATUS"]
            expected = checksum(n, x, sigma, b1, status)
            if (keys["METHOD"], keys["VERSION"], keys["SCALAR"]) != ("ECM-EDWARDS-AM", "1", "48LCM"):
                raise ValueError(f"unexpected Edwards save type: {path}")
            if status not in ("POINT", "FACTOR", "IDENTITY"):
                raise ValueError(f"unexpected Edwards save status: {path}")
            if status == "FACTOR" and (not 1 < x < n or n % x):
                raise ValueError(f"invalid Edwards save factor: {path}")
        else:
            status = "POINT"
            expected = b1 * sigma * n * x % 4294967291
            if keys["METHOD"] != "ECM" or int(keys.get("PARAM", 0)) != 0:
                raise ValueError(f"unexpected Montgomery save type: {path}")
        if int(keys["CHECKSUM"]) != expected:
            raise ValueError(f"invalid save checksum: {path}")
        rows[sigma] = (status, x)
    return rows


def run_case(exe, folder, device, model, case, prior, env, version):
    folder.mkdir()
    p = case["exponent"]
    factors = case["intended_known_factors"]
    n = exact_modulus(p, factors)
    bits, b1, curves, container, tpi, tpb, blocks = (
        prior[key] for key in ("bits", "b1", "curves", "container_bits", "tpi", "tpb", "blocks"))
    if n.bit_length() != bits or tpb != 128:
        raise ValueError(f"M{p}: historical geometry or modulus differs")
    name = f'{"e" if model == "ed" else "m0"}_m{p}_{b1}.save'
    save = folder / name
    work = folder / "worktodo.txt"
    work.write_text(f'ECMSTAGE1=1,2,{p},-1,"{save.as_posix()}",{b1},0,{curves},"{",".join(factors)}"\n')
    (folder / "worktodo.initial.txt").write_bytes(work.read_bytes())
    config = folder / "ecm.toml"
    config.write_text(
        'config_version = 3\n[shared]\nlog_file = ""\nworktodo_file = "worktodo.txt"\n'
        '[stage1]\ntarget = "gpu"\ngpu_backend = "cuda"\ngpu_algorithm = "cgbn"\n'
        f'param = "{"e" if model == "ed" else "m0"}"\ngpu_device = {device}\nsigma = {SIGMA}\n'
        'verbose = true\ncheckpoint_interval_seconds = 0\nhandoff.mode = "off"\n'
        'exponent_cache_dir = "off"\ncuda_low_cpu = false\n'
        f'cuda_slice_depth = {DEPTH}\nsave_dir = {json.dumps(folder.as_posix())}\n' +
        (f'edwards_naf_window_bits = {WINDOW}\nedwards_doubling = 1\n' if model == "ed" else
         'montgomery_lcm_multiplier = "1"\n'), encoding="utf-8")
    if model == "mont0":
        env = env | {"ECM_GPU_STAGE1_ALGO": "prac", "ECM_PRAC_SEARCH": str(SEARCH),
                     "ECM_PRAC_VARIANT": "baseline"}
    with (folder / "run.log").open("wb") as log:
        subprocess.run([str(exe), "--config", str(config)], env=env, cwd=folder, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=900)
    log = (folder / "run.log").read_text(encoding="utf-8", errors="replace")
    if one(log, r"ECM-ELF (\d+\.\d+\.\d+) \(Stage1\)") != version or work.read_text().strip():
        raise ValueError(f"{folder}: version mismatch or unfinished work")
    if "checkpoint resumed" in log or "GPU Edwards resume:" in log:
        raise ValueError(f"{folder}: unexpected checkpoint resume")
    if model == "ed":
        one(log, rf"GPU Edwards Stage1: device={device} curves={curves} N_bits={bits} "
                 rf"scalar_bits=\d+ window={WINDOW} doubling=5\b")
        one(log, rf"GPU Edwards geometry: container_bits={container} tpi={tpi} tpb={tpb} blocks={blocks}\b")
        columns, adds, dictionary_bytes, dictionary_ms, loop_ms, launches = one(
            log, rf"GPU Edwards NAF: columns=(\d+) additions=(\d+) container_bits={container} "
                 rf"tpi={tpi} dictionary_bytes=(\d+) dictionary_gpu_ms=([\d.]+) "
                 r"loop_gpu_ms=([\d.]+) launches=(\d+)")
        loop = float(loop_ms) / 1000
        if int(dictionary_bytes) != curves * (container // 8) * 4 * 2**(WINDOW - 2):
            raise ValueError(f"{folder}: NAF allocation differs from the requested window")
        metrics = dict(columns=int(columns), additions=int(adds), launches=int(launches),
                       dictionary_gpu_seconds=float(dictionary_ms) / 1000,
                       dictionary_bytes=int(dictionary_bytes), window=WINDOW, doubling=5)
    else:
        one(log, rf"preparing Prime95 PRAC plan \(B1={b1}, t=1, search={SEARCH}\)")
        one(log, r"algorithm=prac; resident Montgomery domain")
        one(log, r"GPU: PRAC register policy=per-tier")
        one(log, rf"GPU: sigma={SIGMA}, CGBN<{tpi},{container}>, curves={curves}, blocks={blocks},")
        work_count = int(one(log, r"GPU: PRAC plan [^\n]+; work=(\d+)"))
        event_seconds, execution_wall = one(
            log, r"GPU: Stage1 PRAC completed; kernel=([\d.]+) s, execution-wall=([\d.]+) s")
        loop = float(event_seconds)
        metrics = dict(prac_search=SEARCH, prac_work=work_count, execution_wall_seconds=float(execution_wall))
    outcomes = {}
    for line in (folder / "stage1_results.jsonl").read_text(encoding="utf-8-sig").splitlines():
        row = json.loads(line)
        sigma = int(row["Edwards"]["sigma"] if model == "ed" else row["sigma"])
        if (sigma in outcomes or int(row["b1"]) != b1 or row["status"] not in ("F", "NF")
                or not SIGMA <= sigma < SIGMA + curves):
            raise ValueError(f"{folder}: invalid outcome at sigma {sigma}")
        if row["status"] == "F":
            for value in row["factors"]:
                factor = int(value)
                if not 1 < factor < n or n % factor:
                    raise ValueError(f"{folder}: invalid factor at sigma {sigma}")
        outcomes[sigma] = row["status"]
    if len(outcomes) != curves or not math.isfinite(loop) or loop <= 0:
        raise ValueError(f"{folder}: incomplete outcomes or invalid main-loop timing")
    saved = verify_save(save, n, b1, curves, model)
    if any(outcomes[sigma] != ("F" if status == "FACTOR" else "NF")
           for sigma, (status, _) in saved.items()):
        raise ValueError(f"{folder}: saved curve status differs from internal results")
    return dict(exponent=p, bits=bits, b1=b1, curves=curves, container_bits=container, tpi=tpi,
                tpb=tpb, blocks=blocks, samples=1, gpu_loop_batch_seconds=loop,
                projected_gpu_seconds_per_curve_1e6=loop * REFERENCE_B1 / (curves * b1),
                saved_rows=len(saved), factor_rows=sum(value == "F" for value in outcomes.values()),
                **metrics)


def collect(args, version, output):
    cases = input_scope(args.worktodo)
    history = {}
    for model in MODELS:
        path = args.historical / f"gpu{args.device}_{model}_summary.json"
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        history[model] = {row["exponent"]: row for row in report["rows"]}
        if set(history[model]) != set(cases) or len(report["rows"]) != len(cases):
            raise ValueError(f"{path}: historical matrix differs from the worktodo")
    inventory = gpu_info(args.device)
    (output / "gpu_inventory.csv").write_text(GPU_FIELDS + "\n" + inventory)
    gpu_row = next(csv.reader(inventory.splitlines(), skipinitialspace=True))
    if int(gpu_row[1]) != args.device:
        raise ValueError("GPU telemetry device index differs from the requested device")
    gpu_name = gpu_row[2]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ECM_", "NTT_"))}
    reports = {model: dict(program_version=version, gpu_model=gpu_name, device=args.device,
                           model=model, target_b1=REFERENCE_B1, sigma=SIGMA, slice_depth=DEPTH,
                           algorithm="naf" if model == "ed" else "prac",
                           gpu_timing="sum of main-loop CUDA event intervals", complete=False, rows=[])
               for model in MODELS}
    stop = threading.Event()
    thread = threading.Thread(target=monitor_gpu, args=(output / "telemetry.csv", args.device, stop))
    thread.start()
    try:
        for p in sorted(cases):
            for model in MODELS:
                name = f"{model}_m{p}"
                print("START", name, flush=True)
                row = run_case(args.exe, output / name, args.device, model, cases[p],
                               history[model][p], env, version)
                report = reports[model]
                report["rows"].append(row)
                write_report(output / f"{model}.json", report)
                print(f'PASS {name} {row["gpu_loop_batch_seconds"]:.6f} s', flush=True)
    finally:
        stop.set()
        thread.join()
    for model, report in reports.items():
        report["complete"] = True
        write_report(output / f"{model}.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--device", type=int, choices=(0, 1), required=True)
    parser.add_argument("--worktodo", type=Path,
                        default=ROOT / "tools/benchmarks/prime95/worktodo_ecm_stage1.txt")
    parser.add_argument("--historical", type=Path,
                        default=ROOT / "data/benchmarks/stage1_cgbn_curve_cost_0623")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.exe = args.exe.absolute()
    version = subprocess.check_output([str(args.exe), "--version"], text=True).strip()
    if version != f"ECM-ELF {args.version}":
        parser.error(f"expected ECM-ELF {args.version}, found {version}")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    collect(args, args.version, output)


if __name__ == "__main__":
    main()
