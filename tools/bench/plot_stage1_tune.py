#!/usr/bin/env python3
"""Overlay a Stage1 tune TOML on the historical STAGE1 main figures.

Example: python tools/bench/plot_stage1_tune.py --tune D:/bench/tune-4070ti.toml
Writes log/linear PNGs and one audit CSV to data/figures/stage1_tune_preview.
Each matrix exponent is represented by the fastest sampled configuration across
its automatic transform lengths. Pass --measured with a completed production
matrix to also redraw the main comparison and the throughput/batch matrix.
"""
import argparse
import csv
import hashlib
import json
import math
from statistics import median
from collections import defaultdict
from pathlib import Path
import re
import sys
import tomllib

from plot_stage1_ntt_best import ntt_points, ntt_version, plot
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/test"))
from stage2_tune_tables import summaries

DEFAULT_BASELINE = ROOT / "data/figures/stage1_prime95_mont_wide_20261117"
SIZE_TABLE = ROOT / "include/ecm/stage1/ntt_sizes.h"
# Devices used in the archived main figure. The compact tune table identifies
# hardware by UUID rather than by model or the current CUDA device index.
MATRIX_GPUS = {"d9139abeea58ffc8b97d2066e52c9702": "4070ti", "8a67b1f8ef1c3177a822813a7ac2224d": "4060lp"}
# Highest nonzero wNAF column + 1 of 48 * lcm(1..1_000_000),
# in window order 2..12 (EcmEdwardsNaf in src/ecm/stage1/scalar.cpp).
NAF_COLUMNS = (1442105, 1442106, 1442105, 1442106, 1442106, 1442100,
               1442100, 1442105, 1442106, 1442096, 1442096)
# The binary ladder skips the leading bit: bit_length(scalar) - 1.
BINARY_STEPS = {"edwards": 1442104, "mont0": 1442098}


def size_limits():
    # Use the same continuous capacities as the production packing selector.
    source = SIZE_TABLE.read_text(encoding="utf-8")
    table = source.split("ntt_sizes[] = {", 1)[1].split("};", 1)[0]
    entries = re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", table)
    if not entries:
        raise ValueError(f"Missing transform size matrix in {SIZE_TABLE}")
    return [tuple(map(int, entry)) for entry in entries]


def candidate_lengths(p, native, limits):
    linear = next(n for n, maximum, _, _ in limits if p <= maximum)
    weighted = next(n for n, _, conservative, edwards in limits if p <= (edwards if native else conservative))
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


def naf_rate(sample):
    return 1000 * sample["active_curves"] * sample["columns"] / sample["device_ms"]


def plot_matrix(report, path):
    groups = sorted(report["groups"], key=lambda group: group["p"])
    widths, batch, rates = [], [], []
    fused = []
    for group in groups:
        widths.append(group["p"])
        batch.append(group["active_batch"])
        rates.append([naf_rate(s) for s in group["samples"]])
        if int(group["samples"][0]["plan"]["dataflow"]) & (1 << 25):
            fused.append(len(widths) - 1)
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, (ax_rate, ax_batch) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
        fig.suptitle(f"Edwards Stage1 NTT · {report['gpu_model'].replace('NVIDIA GeForce ', '')}\n"
                     f"{ntt_version(report)} · {len(groups)} widths up to {widths[-1]:,} bits · "
                     f"{report['rounds']} runs per width",
                     fontsize=14)
        mid = [median(r) for r in rates]
        ax_rate.plot(widths, mid, "o-", color="#14804a", markersize=4)
        ax_rate.fill_between(widths, [min(r) for r in rates], [max(r) for r in rates], color="#14804a", alpha=.18)
        if fused:
            ax_rate.scatter([widths[i] for i in fused], [mid[i] for i in fused], marker="*",
                            color="#a41b74", s=84, zorder=4, label="PFA + first outer pass")
            ax_rate.legend(frameon=False)
        ax_rate.set_ylabel("Aggregate NAF columns/s")
        ax_batch.plot(widths, batch, "D-", color="#2866a2", markersize=4)
        ax_batch.set_ylabel("Active curves per batch")
        ax_batch.set_xlabel("Mersenne exponent (bits)")
        for ax in (ax_rate, ax_batch):
            ax.set_xscale("log")
            ax.set_yscale("log", base=2 if ax is ax_batch else 10)
            ax.set_xticks([64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 5242880])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/2**20:g}M" if x >= 2**20 else
                                                       f"{x/1024:g}K" if x >= 1024 else f"{x:g}"))
        ax_batch.set_yticks([2**i for i in range(10)], labels=[str(2**i) for i in range(10)])
        fig.savefig(path, dpi=180)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", required=True, type=Path)
    ap.add_argument("--curve", choices=BINARY_STEPS, default="edwards")
    ap.add_argument("--gpu", choices=("4070ti", "4060lp"),
                    help="Historical GPU series (inferred for the two reference devices)")
    ap.add_argument("--uuid", help="Select one GPU if the file contains multiple devices")
    ap.add_argument("--version", help="Restrict samples to one build (default: all builds in the snapshot)")
    ap.add_argument("--baseline", type=Path, help="Main figure comparison.csv for the selected curve")
    ap.add_argument("--measured", type=Path, help="Production remeasurement summary using the same tune snapshot")
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
    versions = {r["program_version"] for r in rows}
    version = next(iter(versions)) if len(versions) == 1 else "mixed builds"
    gpu = args.gpu or MATRIX_GPUS.get(uuid)
    if not gpu:
        ap.error(f"Unknown GPU {uuid}; select the comparison device with --gpu")
    points = reference_points(baseline)
    old_role = "ntt_overlay" if gpu == "4070ti" else "ntt"
    old_version = one_value((v for role, v in points if role == old_role), "historical NTT version")
    history = points[old_role, old_version]
    history_at = {p: cost for p, _, cost, _, _ in history}
    measured = {}
    memory_budget = None
    matrix = [(p, bits) for p, bits, _, _, _ in history]
    scope = f"5 runs for {old_version}"
    if args.measured:
        report = json.loads(args.measured.read_text(encoding="utf-8"))
        if report["curve"] != model or report["tune_sha256"] != digest or (
                report["gpu_uuid"].lower().removeprefix("gpu-").replace("-", "") != uuid):
            ap.error("Production summary must use the selected curve, GPU and tune snapshot")
        current = sorted(ntt_points(report, 1_000_000))
        measured = {g["p"]: g for g in report["groups"]}
        memory_budget = report["memory_mib"] * 2**20
        matrix = [(p, bits) for p, bits, _, _, _ in current]
        points[old_role + "_previous", old_version] = points.pop((old_role, old_version))
        points[old_role, ntt_version(report)] = current
        scope = f"{report['rounds']} runs for {ntt_version(report)}, 5 for {old_version}"
    limits = size_limits()
    audit, projected = [], []
    for p, bits in sorted(matrix):
        old_cost = history_at.get(p)
        candidates = {native: candidate_lengths(p, native, limits) for native in (False, True)}
        group = measured.get(p)
        if group:
            plan = group["tune"]
            candidates[True].add((plan["ntt_type"], plan["length"]))
        matching = [r for r in rows if r.get("max_exp", p) >= p
                    and (r["ntt_type"], r["length"]) in candidates[bool(r["naf_window"])]
                    and (memory_budget is None or 0 < r.get("ntt_memory_bytes", 0) <= memory_budget)]
        result = dict(p=p, bits=bits, baseline_s_per_curve=old_cost, baseline_version=old_version,
                      sampled_transforms=" ".join(f"{kind}:{n}" for kind,n in sorted({(r["ntt_type"],r["length"]) for r in matching})),
                      candidates=len(matching), uuid=uuid, program_version="", tune_sha256=digest)
        cost = math.nan  # Break the new line across missing matrix positions.
        if matching:
            best = max(matching, key=lambda r: r["iter_per_s"])
            cost = best["reference_steps"] / best["iter_per_s"]
            result.update(s_per_curve_at_1e6=cost, old_over_projected=old_cost/cost if old_cost else None,
                          **{k: best.get(k, "") for k in ("iter_per_s", "reference_steps", "reference_bits", "max_exp", "ntt_type",
                             "length", "bpw", "naf_window", "batch", "graph_bits", "tile", "max_radix_bits", "dataflow", "program_version",
                             "b1", "repeats", "sample_phase", "sample_target_ms", "updated_at")})
        if group:
            times = group["projected_seconds_per_curve_1e6"]
            samples = group["samples"]
            result.update(measured_b1=group["b1"], measured_batch=group["active_batch"],
                          measured_length=plan["length"], measured_ntt_type=plan["ntt_type"],
                          measured_naf_window=plan["naf_window"], measured_dataflow=samples[0]["plan"]["dataflow"],
                          measured_iter_per_s=median(naf_rate(s) for s in samples),
                          measured_main_seconds=median(s["device_ms"] for s in samples) / 1000,
                          measured_process_seconds=median(s["process_seconds"] for s in samples),
                          measured_s_per_curve_at_1e6=median(times), measured_min=min(times), measured_max=max(times))
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
    device = "RTX 4060 Laptop · historical 1500 MHz"
    overlay = "RTX 4070 Ti" if measured else "RTX 4070 Ti · historical 2600 MHz"
    for scale in ("log", "linear"):
        plot(points, args.output / f"{stem}-{scale}.png", scale == "log", model, device,
             overlay, scope, ("0.62.3",), True, note)
        if measured:
            measured_points = {key: series for key, series in points.items() if key[0] not in {"tune", "tune_overlay"}}
            plot(measured_points, args.output / f"stage1-{model}-{gpu}-measured-{scale}.png", scale == "log",
                 model, device, overlay, scope, ("0.62.3",), True)
    if measured:
        plot_matrix(report, args.output / f"stage1-{model}-{gpu}-measured-matrix.png")
    print(f"{args.tune}: {len(rows)} samples, {count}/{len(audit)} matrix points; SHA256 {digest}\n{csv_path}")


if __name__ == "__main__":
    main()
