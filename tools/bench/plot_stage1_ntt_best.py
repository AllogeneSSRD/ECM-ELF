#!/usr/bin/env python3
"""Compare Stage1 NTT measurements with CGBN and Prime95 baselines."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


def ntt_points(report, target_b1):
    if report["reference_b1"] != target_b1:
        raise ValueError("benchmark B1 projections differ")
    exponents = [g["p"] for g in report["groups"]]
    if not exponents or len(set(exponents)) != len(exponents) or sorted(exponents) != sorted(report["exponents"]):
        raise ValueError("NTT benchmark is incomplete or repeated")
    points = []
    for group in report["groups"]:
        p, samples = group["p"], group["projected_seconds_per_curve_1e6"]
        points.append((p, p, statistics.median(samples), min(samples), max(samples)))
    return points


def ntt_version(report):
    return report["version"].removeprefix("ECM-ELF ")


def version_order(report):
    version = ntt_version(report)
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", version)
    if match is None:
        raise ValueError(f"unrecognized NTT version {version!r}")
    return tuple(map(int, match.groups()))


def make_points(ntt, historical, previous_ntt=()):
    target_b1 = historical["target_b1"]
    current_version = version_order(ntt)
    seen_versions = {current_version}
    points = {("ntt", ntt_version(ntt)): ntt_points(ntt, target_b1)}
    for report in sorted(previous_ntt, key=version_order, reverse=True):
        version = version_order(report)
        if version in seen_versions or version > current_version:
            raise ValueError("each previous NTT version must be distinct and older than --ntt")
        if any(report[key] != ntt[key] for key in ("curve", "gpu_model")):
            raise ValueError("NTT series must use the same GPU model and curve construction")
        if sorted(report["exponents"]) != sorted(ntt["exponents"]):
            raise ValueError("NTT series must cover the same exponents")
        seen_versions.add(version)
        points["ntt_previous", ntt_version(report)] = ntt_points(report, target_b1)
    cgbn_exponents = [row["exponent"] for row in historical["rows"]]
    if not cgbn_exponents or len(set(cgbn_exponents)) != len(cgbn_exponents) or historical.get("missing_cpu_reference_exponents"):
        raise ValueError("CGBN and Prime95 reference conditions are incomplete or repeated")
    cgbn, prime95 = ("cgbn", historical.get("program_version", "")), ("prime95", "")
    points[cgbn], points[prime95] = [], []
    for row in historical["rows"]:
        p, bits = row["exponent"], row["bits"]
        points[cgbn].append((p, bits, row["projected_gpu_seconds_per_curve_1e6"], None, None))
        # Prime95 sets up GWNUM with k,b,n,c, even when the factor target is a cofactor.
        points[prime95].append((p, p, row["cpu_seconds_1e6"], row["cpu_min_seconds_1e6"], row["cpu_max_seconds_1e6"]))
    return {key: sorted(rows, key=lambda r: r[1]) for key, rows in points.items()}


def plot(points, output, logarithmic, curve, device):
    model = "Edwards" if curve == "ed" else "Montgomery"
    scheme = "Edwards" if curve == "ed" else "PARAM0"
    ntt_color = to_rgb("#176fa5")
    history = [version for backend, version in points if backend == "ntt_previous"]
    styles = {}
    for key in points:
        backend, version = key
        if backend in ("ntt", "ntt_previous"):
            fade = 0 if backend == "ntt" else .57 * (history.index(version) + 1) / (len(history) + 1)
            color = tuple(value + (1 - value) * fade for value in ntt_color)
            styles[key] = (f"NTT {scheme} {version} · {device} · full Mersenne", color, "o")
        elif backend == "cgbn":
            styles[key] = (f"CGBN {scheme}{' NAF' if curve == 'ed' else ''} {version} · {device} · cofactor", "#cb6719", "s")
        else:
            styles[key] = (f"Prime95 {model} · 1 core at 4 GHz · full-M FFT", "#72459a", "^")
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.25 + .28 * len(history)), layout="constrained")
        fig.suptitle(f"Stage1 {model} cost per curve by arithmetic width", fontweight="bold")
        for i, ax in enumerate(axes):
            ax.set_title(("Historical width range (≤ 16,500 bits)", "Full NTT measurement range")[i])
            for key, rows in points.items():
                backend, version = key
                visible = rows if i else [r for r in rows if r[1] <= 16500]
                if not visible:
                    continue
                _, x, y, lo, hi = zip(*visible)
                label, color, marker = styles[key]
                previous = backend == "ntt_previous"
                ax.plot(x, y, label=label, color=color, marker=marker, markersize=4, linewidth=1.7,
                        linestyle="--" if previous else "-", markerfacecolor="none" if previous else color,
                        zorder=2 - .1 * history.index(version) if previous else 3)
                if not previous and all(value is not None for value in lo + hi):
                    ax.fill_between(x, lo, hi, color=color, alpha=.14, linewidth=0)
            ax.set_xlabel("Actual arithmetic modulus width (bits)")
            if logarithmic:
                ax.set_yscale("log")
                if i:
                    ax.set_xscale("log")
            else:
                ax.set_ylim(bottom=0)
                ax.ticklabel_format(style="plain", axis="x")
        axes[0].set_ylabel("Seconds per curve at B1 = 1,000,000 (scaled)")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=2 if history else 3, fontsize=9, frameon=False)
        fig.savefig(output, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntt", required=True, type=Path)
    parser.add_argument("--previous-ntt", action="append", default=[], type=Path,
                        help="Overlay a historical NTT benchmark as a dashed line; may be repeated")
    parser.add_argument("--comparison", "--historical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ntt, comparison = (json.loads(p.read_text(encoding="utf-8-sig")) for p in (args.ntt, args.comparison))
    curve = ntt["curve"]
    if ntt["gpu_model"] != comparison["gpu_model"] or comparison["model"] != ("ed" if curve == "ed" else "mont0"):
        parser.error("CGBN and NTT must use the same GPU model and curve construction")
    device = ntt["gpu_model"].removeprefix("NVIDIA GeForce ").removesuffix(" GPU")
    previous = [json.loads(path.read_text(encoding="utf-8-sig")) for path in args.previous_ntt]
    try:
        points = make_points(ntt, comparison, previous)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = "stage1-edwards-ntt-cgbn-prime95" if curve == "ed" else "stage1-ntt-best-cgbn-prime95"
    for suffix in ("log", "linear"):
        plot(points, args.output / f"{prefix}-{suffix}.png", suffix == "log", curve, device)
    with (args.output / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("series", "mersenne_exponent", "arithmetic_bits", "median_s_per_curve_at_1e6",
                         "min_s_per_curve_at_1e6", "max_s_per_curve_at_1e6", "version"))
        for (backend, version), rows in points.items():
            writer.writerows((backend, *row, version) for row in rows)


if __name__ == "__main__":
    main()
