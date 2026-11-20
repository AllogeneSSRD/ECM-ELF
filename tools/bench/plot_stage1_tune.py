#!/usr/bin/env python3
"""Overlay a Stage1 tune TOML on the historical STAGE1 main figures.

Example: python tools/bench/plot_stage1_tune.py --tune D:/bench/tune-4070ti.toml
Writes log/linear PNGs and one audit CSV to data/figures/stage1_tune_preview.
Each matrix exponent is represented by the fastest sampled configuration across
its automatic transform lengths. This is a length-range projection, independent
of the narrower dispatch checks used to reuse a tune row in production.
"""
import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path
import re
import sys
import tomllib

from plot_stage1_ntt_best import plot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/test"))
from stage2_tune_tables import summaries

DEFAULT_BASELINE = ROOT / "data/figures/stage1_prime95_mont_wide_20261117"
SIZE_TABLE = ROOT / "docs/usage/TUNE.md"
# Devices used in the archived main figure. The compact tune table identifies
# hardware by UUID rather than by model or the current CUDA device index.
MATRIX_GPUS = {"d9139abeea58ffc8b97d2066e52c9702": "4070ti", "8a67b1f8ef1c3177a822813a7ac2224d": "4060lp"}
# Highest nonzero wNAF column + 1 of 48 * lcm(1..1_000_000),
# in window order 2..12 (EcmEdwardsNaf in src/ecm/stage1/scalar.cpp).
NAF_COLUMNS = (1442105, 1442106, 1442105, 1442106, 1442106, 1442100,
               1442100, 1442105, 1442106, 1442096, 1442096)
# The binary ladder skips the leading bit: bit_length(scalar) - 1.
BINARY_STEPS = {"edwards": 1442104, "mont0": 1442098}


def size_ranges():
    section = SIZE_TABLE.read_text(encoding="utf-8").split("#### NTT and transform sizes", 1)[1]
    rows = []
    for line in section.splitlines():
        if rows and not line.startswith("|"):
            break
        if not re.match(r"^\| [\d,]+ \|", line):
            continue
        cells = line.strip("|").replace(",", "").split("|")
        def endpoints(cell):
            if cell.strip() == "—":
                return range(0)
            lo, hi = map(int, cell.strip().split("–"))
            return range(lo, hi+1)
        rows.append((int(cells[0]), *map(endpoints, cells[2:5]), set(map(int, re.findall(r"\d+", cells[5])))))
    if not rows:
        raise ValueError(f"Missing transform size matrix in {SIZE_TABLE}")
    return rows


def candidate_lengths(p, native, ranges):
    linear = next(n for n, interval, _, _, _ in ranges if p in interval)
    weighted = next((n for n, _, _, _, exceptions in ranges if native and p in exceptions), None)
    if weighted is None:
        weighted = next(n for n, _, conservative, edwards, _ in ranges if p in (edwards if native else conservative))
    # Auto also tries the shortest safe pure power of two when mixed wins.
    binary = 1 << (weighted-1).bit_length()
    return {("stage1_linear", linear), ("stage1_weighted", weighted), ("stage1_weighted", binary)}


def reference_points(path):
    points = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            points[r["series"], r["version"]].append((
                int(r["mersenne_exponent"]), int(r["arithmetic_bits"]), float(r["median_s_per_curve_at_1e6"]),
                float(r["min_s_per_curve_at_1e6"]) if r["min_s_per_curve_at_1e6"] else None,
                float(r["max_s_per_curve_at_1e6"]) if r["max_s_per_curve_at_1e6"] else None))
    return points


def one_value(values, option):
    values = sorted(set(values))
    if len(values) != 1:
        raise ValueError(f"Select {option} from {values}" if values else f"No tune samples matching {option}")
    return values[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", required=True, type=Path)
    ap.add_argument("--curve", choices=BINARY_STEPS, default="edwards")
    ap.add_argument("--gpu", choices=("4070ti", "4060lp"),
                    help="Historical GPU series (inferred for the two reference devices)")
    ap.add_argument("--uuid", help="Select one GPU if the file contains multiple devices")
    ap.add_argument("--version", help="Select one program version if the file contains multiple builds")
    ap.add_argument("--baseline", type=Path, help="Main figure comparison.csv for the selected curve")
    ap.add_argument("--output", type=Path, default=ROOT / "data/figures/stage1_tune_preview")
    args = ap.parse_args()
    model = "ed" if args.curve == "edwards" else "mont0"
    baseline = args.baseline or DEFAULT_BASELINE / model / "comparison.csv"
    blob = args.tune.read_bytes()  # The writer atomically replaces the file. All outputs use this one snapshot.
    digest = hashlib.sha256(blob).hexdigest()
    doc = tomllib.loads(blob.decode("utf-8-sig"))
    rows = []
    for r in summaries(doc, "ntt"):
        kind = r.get("benchmark_kind")
        if r.get("curve") != args.curve or not math.isfinite(r.get("iter_per_s", 0)) or r.get("iter_per_s", 0) <= 0:
            continue
        if kind == "stage1_naf_stratified" and args.curve == "edwards" and 2 <= r["naf_window"] <= 12:
            r["reference_steps"] = NAF_COLUMNS[r["naf_window"]-2]
        elif kind == "stage1_graph_replay" and r.get("naf_window") == 0:
            r["reference_steps"] = BINARY_STEPS[args.curve]
        else:
            continue
        r["uuid"] = doc["condition"][str(r["condition"])]["uuid_hex"]
        if args.gpu and r["uuid"] in MATRIX_GPUS and MATRIX_GPUS[r["uuid"]] != args.gpu:
            continue
        if (not args.uuid or args.uuid.lower().removeprefix("gpu-").replace("-", "") == r["uuid"]) and (
                not args.version or r["program_version"] == args.version):
            rows.append(r)
    uuid = one_value((r["uuid"] for r in rows), "--uuid")
    version = one_value((r["program_version"] for r in rows), "--version")
    gpu = args.gpu or MATRIX_GPUS.get(uuid)
    if not gpu:
        ap.error(f"Unknown GPU {uuid}; select the comparison device with --gpu")
    points = reference_points(baseline)
    old_role = "ntt_overlay" if gpu == "4070ti" else "ntt"
    old_version = one_value((v for role, v in points if role == old_role), "historical NTT version")
    ranges = size_ranges()
    audit, projected = [], []
    for p, bits, old_cost, _, _ in sorted(points[old_role, old_version]):
        candidates = {native: candidate_lengths(p, native, ranges) for native in (False, True)}
        matching = [r for r in rows if (r["ntt_type"], r["length"]) in candidates[bool(r["naf_window"])]]
        result = dict(p=p, bits=bits, baseline_s_per_curve=old_cost, baseline_version=old_version,
                      sampled_transforms=" ".join(f"{kind}:{n}" for kind,n in sorted({(r["ntt_type"],r["length"]) for r in matching})),
                      candidates=len(matching), uuid=uuid, program_version=version, tune_sha256=digest)
        cost = math.nan  # Break the new line across missing matrix positions.
        if matching:
            best = max(matching, key=lambda r: r["iter_per_s"])
            cost = best["reference_steps"] / best["iter_per_s"]
            result.update(s_per_curve_at_1e6=cost, old_over_projected=old_cost/cost,
                          **{k: best.get(k, "") for k in ("iter_per_s", "reference_steps", "reference_bits", "ntt_type",
                             "length", "bpw", "naf_window", "batch", "graph_bits", "tile", "max_radix_bits", "dataflow",
                             "b1", "repeats", "sample_phase", "sample_target_ms", "updated_at")})
        audit.append(result)
        projected.append((p, bits, cost, None, None))
    count = sum(math.isfinite(r[2]) for r in projected)
    if not count:
        ap.error("No measured transform lengths cover the baseline exponent matrix")
    points["tune_overlay" if gpu == "4070ti" else "tune", version] = projected
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"stage1-{model}-{gpu}-tune"
    csv_path = args.output / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dict.fromkeys(k for row in audit for k in row))
        writer.writeheader()
        writer.writerows(audit)
    bounds = sorted({r["b1"] for r in rows})
    source_b1 = str(bounds[0]) if len(bounds) == 1 else f"{bounds[0]}–{bounds[-1]}"
    note = (f"Tune · B1={source_b1} sample → B1=1M projection · peak iter/s per transform range"
            f" · {count}/{len(audit)} points · power & clocks unrecorded")
    for scale in ("log", "linear"):
        plot(points, args.output / f"{stem}-{scale}.png", scale == "log", model, "RTX 4060 Laptop · historical 1500 MHz",
             "RTX 4070 Ti · historical 2600 MHz", f"5 runs for {old_version}", ("0.62.3",), True, note)
    print(f"{args.tune}: {len(rows)} samples, {count}/{len(audit)} matrix points; SHA256 {digest}\n{csv_path}")


if __name__ == "__main__":
    main()
