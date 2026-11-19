"""Replay the frozen 0.66.1 CPU Stage2 width matrices with a current build."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "data/experiments"
FULL = EXPERIMENTS / "stage2_cpu_full_0661"
WIDE = EXPERIMENTS / "stage2_wide_b1000"
SERIAL = ROOT / "data/benchmarks/stage2_parallel_0620/small_b2_comparison/serial.json"
SAMPLER = Path(__file__).with_name("bench_cpu_stage2_blocks.py")
AFFINITY = {1: (5,), 4: (1, 3, 5, 7)}
PHASES = {"baby", "ftree", "reciprocal", "giant", "gtree", "fold", "scale", "descent", "leaf_gcd"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_fields(path):
    lines = path.read_text(encoding="ascii").strip().splitlines()
    if not lines or any(line != lines[0] for line in lines):
        raise ValueError(f"Expected one repeated Edwards save point in {path}")
    return dict(field.strip().split("=", 1) for field in lines[0].strip(";").split(";"))


def frozen_inputs(matrix):
    if matrix == "full":
        rows = read_json(SERIAL)["rows"]
        if len(rows) != 13:
            raise ValueError("Expected the 13-width serial reference")
        sources = [(r["bits"], r["exponent"], r["B1"], r["B2"], FULL / f"{r['bits']}.save", None)
                   for r in rows]
    else:
        rows = read_json(WIDE / "summary.json")
        if len(rows) != 8:
            raise ValueError("Expected the eight wide cases")
        sources = [(r["bits"], r["e"], 1000, r["b2"], WIDE / r["directory"] / f"m{r['e']}.save", r["factors"])
                   for r in rows]

    result = []
    for bits, exponent, b1, b2, path, factors in sorted(sources):
        fields = save_fields(path)
        sigma = 27 if exponent == 2003 else 26
        modulus = int(fields["N"], 0)
        quotient, residue = divmod((1 << exponent) - 1, modulus)
        if factors is None:
            factors = [quotient] if quotient != 1 else []
        if (fields["METHOD"] != "ECM-EDWARDS-AM" or fields["STATUS"] != "POINT" or
                int(fields["B1"]) != b1 or int(fields["SIGMA"]) != sigma or
                modulus.bit_length() != bits or residue or math.prod(factors) != quotient):
            raise ValueError(f"Save identity mismatch at {path}")
        result.append(dict(bits=bits, e=exponent, b1=b1, requested=b2, sigma=sigma, factors=factors,
                           save=str(path), save_sha256=sha256(path)))
    if len({row["bits"] for row in result}) != len(result):
        raise ValueError("Duplicate cofactor width in the matrix")
    return result


def sample(folder, inputs, version):
    measurement = folder / "measurement.json"
    if measurement.is_file():
        if read_json(folder / "input.json") != inputs:
            raise ValueError(f"Sample input changed at {folder}")
    else:
        if folder.exists():
            # Keep an interrupted/failed attempt separate from the complete sample.
            folder.rename(folder.with_name(f"{folder.name}_incomplete_{time.time_ns()}"))
        while True:
            folder.mkdir()
            write_json(folder / "input.json", inputs)
            result = subprocess.run([sys.executable, SAMPLER, "--sample", folder])
            if result.returncode == 0:
                break
            accounting = folder / "accounting.json"
            if not accounting.is_file() or not read_json(accounting).get("compile_overlap"):
                result.check_returncode()
            folder.rename(folder.with_name(f"{folder.name}_overlap_{time.time_ns()}"))
    measured = read_json(measurement)
    if (not measured["dll"] or measured["dll"] not in version or measured["actual"] != inputs["requested"] or
            not math.isfinite(measured["seconds"]) or measured["seconds"] <= 0 or
            set(measured["phases"]) != PHASES or
            any(not math.isfinite(v) or v < 0 for v in measured["phases"].values())):
        raise ValueError(f"Unexpected DLL identity, coverage or phase timing at {folder}")
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix", choices=("full", "wide"), default="full")
    parser.add_argument("--repeats", type=int, help="Default 3 for full and 2 for wide")
    parser.add_argument("--widths", type=int, nargs="+", help="Only replay selected cofactor widths")
    args = parser.parse_args()
    repeats = args.repeats if args.repeats is not None else (3 if args.matrix == "full" else 2)
    if repeats < 1:
        parser.error("Choose at least one repeat")
    cases = frozen_inputs(args.matrix)
    if args.widths:
        widths = set(args.widths)
        if not widths.issubset({case["bits"] for case in cases}):
            parser.error("Width not in the reference matrix")
        cases = [case for case in cases if case["bits"] in widths]
    exe = args.exe.resolve(strict=True)
    files = [exe, exe.with_name("gwnum.dll"), exe.with_name("gmp-10.dll")]
    version = subprocess.check_output([exe, "--version"], text=True)
    identity = dict(matrix=args.matrix, version=version, repeats=repeats, memory_mib=4096,
                    binary={file.name: sha256(file) for file in files}, cases=cases)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    if summary_path.is_file():
        previous = read_json(summary_path)
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Cannot resume with different inputs or binaries at {out}")
    report = dict(**identity, runs=[], complete=False)
    for rep in range(1, repeats + 1):
        for case in cases:
            for threads in ((1, 4) if rep % 2 else (4, 1)):
                cpus = list(AFFINITY[threads])
                inputs = dict(case, exe=str(exe), cpus=cpus, memory=4096, d=0, arm="exact", check=False)
                folder = out / f"m{case['bits']}_t{threads}_r{rep}"
                measured = sample(folder, inputs, version)
                report["runs"].append(dict(measured, bits=case["bits"], exponent=case["e"], b1=case["b1"],
                                           b2=case["requested"], sigma=case["sigma"], repeat=rep,
                                           threads=threads, cpus=cpus, save_sha256=case["save_sha256"]))
                write_json(summary_path, report)
    report["complete"] = True
    write_json(summary_path, report)


if __name__ == "__main__":
    main()
