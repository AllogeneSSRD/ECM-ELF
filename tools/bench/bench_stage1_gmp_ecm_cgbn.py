#!/usr/bin/env python3
"""Measure local GMP-ECM CGBN Stage1 curves under WSL and write plot input."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from math import prod
from pathlib import Path
import re
import subprocess


TIMING = "GMP-ECM native CUDA Stage1 event"
GPU_TOTAL = re.compile(r"Computing (\d+) Step 1 took \d+ms of CPU time / (\d+)ms of GPU time")
KERNEL = re.compile(r"CGBN<(\d+), (\d+)> running kernel<(\d+) block x (\d+) threads> input number is (\d+) bits")
VERSION = re.compile(r"^GMP-ECM ([\d.]+)\b.*$", re.MULTILINE)
ARCH = re.compile(r"GPU: Using device code targeted for architecture (\S+)")
DEVICE = re.compile(r"^GPU: will use device (\d+): (.+), compute capability", re.MULTILINE)
BOUNDS = re.compile(r"^Using B1=(\d+), B2=0, sigma=3:(\d+)-(\d+) \((\d+) curves\)$", re.MULTILINE)
STAGE2 = re.compile(r"^(?:Curve \d+ Step 2 took|Computing \d+ Step 2 on CPU)", re.MULTILINE)
SAVE = re.compile(r"^METHOD=ECM;\s*PARAM=3;\s*SIGMA=(\d+);\s*B1=(\d+);")


def load_cases(worktodo, exponents):
    cases = []
    for line in worktodo.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("ECMSTAGE1="):
            continue
        row = next(csv.reader([line.split("=", 1)[1]]))
        if row[:2] != ["1", "2"] or row[3] != "-1":
            raise ValueError(f"Expected a Mersenne ECMSTAGE1 line: {line}")
        exponent = int(row[2])
        if exponents and exponent not in exponents:
            continue
        factors = row[8].split(",") if len(row) > 8 and row[8] else []
        modulus, remainder = divmod((1 << exponent) - 1, prod(map(int, factors)))
        if remainder or modulus <= 1:
            raise ValueError(f"Invalid cofactor for M{exponent}")
        expression = f"(2^{exponent}-1)"
        if factors:
            expression += f"/({'*'.join(factors)})"
        b1 = int(row[5])
        if b1 < 2:
            raise ValueError(f"B1 below 2 for M{exponent}")
        cases.append({"exponent": exponent, "bits": modulus.bit_length(), "b1": b1, "expression": expression})
    if not cases or len({case["exponent"] for case in cases}) != len(cases):
        raise ValueError("Expected a nonempty worktodo with one line per exponent")
    if exponents and exponents != {case["exponent"] for case in cases}:
        raise ValueError("Requested exponent missing from worktodo")
    return cases


def read_devices():
    fields = "index,name,pci.bus_id,driver_version"
    report = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"], text=True)
    return {int(index): {"device": int(index), "gpu_model": name, "pci_bus_id": bus, "driver": driver}
            for index, name, bus, driver in csv.reader(io.StringIO(report), skipinitialspace=True)}


def read_batch(log, save, identity, case, sigma, exit_code):
    content = log.read_text(encoding="utf-8", errors="replace")
    totals = GPU_TOTAL.findall(content)
    kernels = KERNEL.findall(content)
    version, arch = VERSION.search(content), ARCH.search(content)
    if not version or not arch or len(totals) != 1 or len(kernels) != 1 or exit_code not in (0, 2, 6, 8, 10, 14):
        raise RuntimeError(f"Incomplete Stage1 or unexpected exit {exit_code}: {log}")
    if DEVICE.findall(content) != [(str(identity["device"]), identity["gpu_model"])]:
        raise RuntimeError(f"CUDA device differs from nvidia-smi: {log}")
    curves, milliseconds = map(int, totals[0])
    b1 = case["b1"]
    expected = (b1, sigma, sigma + curves - 1, curves)
    if [tuple(map(int, bounds)) for bounds in BOUNDS.findall(content)] != [expected] or STAGE2.search(content):
        raise RuntimeError(f"Expected B1={b1}, B2=0 and a complete PARAM3 sigma range: {log}")
    (container, tpi, blocks, threads, bits), = kernels
    if int(bits) != case["bits"] or curves != int(blocks) * int(threads) // int(tpi):
        raise RuntimeError(f"Unexpected CGBN width or launch geometry: {log}")
    saved = set()
    if save.exists():
        with save.open(encoding="utf-8", errors="replace") as data:
            for line in data:
                match = SAVE.match(line)
                if not match or int(match[2]) != b1:
                    raise RuntimeError(f"Unexpected save record: {save}")
                saved_sigma = int(match[1])
                if not sigma <= saved_sigma < sigma + curves or saved_sigma in saved:
                    raise RuntimeError(f"Out-of-range or duplicate sigma in {save}")
                saved.add(saved_sigma)
    if curves <= 0 or milliseconds <= 0 or (exit_code == 0 and len(saved) != curves):
        raise RuntimeError(f"Invalid timing or incomplete save ({len(saved)}/{curves}): {log}")
    return {"seconds": milliseconds / 1000, "curves": curves, "saved_curves": len(saved), "exit_code": exit_code,
            "version": version.group(1), "banner": version.group(), "compiled_arch": arch.group(1),
            "container": int(container), "tpi": int(tpi), "blocks": int(blocks), "threads": int(threads)}


def run_batch(args, identity, case, repeat, place):
    save = place / f"r{repeat}.save"
    log = place / f"r{repeat}.log"
    command = [str(args.exe), "-v", "-savea", str(save), "-gpu", "-gpudevice", str(identity["device"]),
               "-gpucurves", str(args.curves), "-gpuckpt", "0", "-sigma", f"3:{args.sigma}", str(case["b1"]), "0"]
    with log.open("w", encoding="utf-8") as output:
        output.write(f"Command: {command!r}\nInput: {case['expression']}\n")
        output.flush()
        process = subprocess.run(command, cwd=place, input=case["expression"] + "\n", text=True,
                                 stdout=output, stderr=subprocess.STDOUT)
    return read_batch(log, save, identity, case, args.sigma, process.returncode)


def run_device(args, identity, cases):
    device = identity["device"]
    out = args.output / f"gpu{device}"
    out.mkdir()
    rows = []
    for case in cases:
        place = out / f"m{case['exponent']}"
        place.mkdir()
        runs = []
        for repeat in range(1, args.repeats + 1):
            run = run_batch(args, identity, case, repeat, place)
            runs.append(run)
            print(f"GPU {device} M{case['exponent']} {repeat}/{args.repeats}: "
                  f"{run['curves']} curves, {run['seconds']:.3f} s", flush=True)
        curves, = set(run["curves"] for run in runs)
        rows.append({**case, "curves": curves, "stage1_seconds": [run["seconds"] for run in runs], "runs": runs})
        (out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    return {**identity, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--worktodo", type=Path, required=True, help="ECMSTAGE1 matrix with B1 and known factors")
    parser.add_argument("--output", type=Path, required=True, help="New experiment directory")
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--exponents", nargs="+", type=int)
    parser.add_argument("--curves", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sigma", type=int, default=20260922)
    args = parser.parse_args()
    if min(args.curves, args.repeats, args.sigma) <= 0 or len(set(args.devices)) != len(args.devices):
        parser.error("Positive curves, repeats and sigma, and distinct devices, are required")
    args.exe, args.output = args.exe.resolve(strict=True), args.output.resolve()
    cases = load_cases(args.worktodo, set(args.exponents or []))
    devices = read_devices()
    args.output.mkdir(parents=True)
    config = subprocess.check_output([str(args.exe), "-printconfig"], text=True)
    (args.output / "printconfig.txt").write_text(config, encoding="utf-8")
    with ThreadPoolExecutor(max_workers=len(args.devices)) as workers:
        tasks = [workers.submit(run_device, args, devices[device], cases) for device in args.devices]
        results = [task.result() for task in tasks]
    versions = {run["version"] for device in results for row in device["rows"] for run in row["runs"]}
    version, = versions
    summary = {"version": version, "timing": TIMING, "binary": str(args.exe), "param": 3,
               "requested_curves": args.curves, "sigma_start": args.sigma, "b2": 0, "devices": results}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
