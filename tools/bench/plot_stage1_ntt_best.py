#!/usr/bin/env python3
"""Compare measured NTT Stage1 throughput with the historical Montgomery baselines."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_points(ntt, historical):
    points = {"ntt": [], "cgbn": [], "prime95": []}
    if ntt["reference_b1"] != historical["target_b1"]:
        raise ValueError("benchmark B1 projections differ")
    if sorted(g["p"] for g in ntt["groups"]) != sorted(ntt["exponents"]):
        raise ValueError("NTT benchmark is incomplete")
    for group in ntt["groups"]:
        samples = group["projected_seconds_per_curve_1e6"]
        points["ntt"].append((group["p"], group["p"], statistics.median(samples), min(samples), max(samples)))
    for row in historical["rows"]:
        p, bits = row["exponent"], row["bits"]
        points["cgbn"].append((p, bits, row["projected_gpu_seconds_per_curve_1e6"], None, None))
        # Prime95 sets up GWNUM with k,b,n,c, even when the factor target is a cofactor.
        points["prime95"].append((p, p, row["cpu_seconds_1e6"],
                                 row["cpu_min_seconds_1e6"], row["cpu_max_seconds_1e6"]))
    return {key: sorted(rows, key=lambda r: r[1]) for key, rows in points.items()}


def plot(points, output, logarithmic):
    styles = {
        "ntt": ("NTT PARAM0 · RTX 4060 Laptop · full Mersenne", "#176fa5", "o"),
        "cgbn": ("CGBN PARAM0 · RTX 4070 Ti · cofactor", "#cb6719", "s"),
        "prime95": ("Prime95 Montgomery · 1 core at 4 GHz · full-M FFT", "#72459a", "^"),
    }
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.25), layout="constrained")
        fig.suptitle("Stage1 Montgomery cost per curve by arithmetic width", fontweight="bold")
        for i, ax in enumerate(axes):
            ax.set_title(("Historical width range (≤ 16,500 bits)", "Full NTT measurement range")[i])
            for key, rows in points.items():
                visible = rows if i else [r for r in rows if r[1] <= 16500]
                if not visible:
                    continue
                _, x, y, lo, hi = zip(*visible)
                label, color, marker = styles[key]
                ax.plot(x, y, label=label, color=color, marker=marker, markersize=4, linewidth=1.7)
                if all(value is not None for value in lo + hi):
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
        fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=9, frameon=False)
        fig.savefig(output, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntt", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ntt, old = (json.loads(p.read_text(encoding="utf-8-sig")) for p in (args.ntt, args.historical))
    points = make_points(ntt, old)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("log", "linear"):
        plot(points, args.output / f"stage1-ntt-best-cgbn-prime95-{suffix}.png", suffix == "log")
    with (args.output / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("series", "mersenne_exponent", "arithmetic_bits", "median_s_per_curve_at_1e6",
                         "min_s_per_curve_at_1e6", "max_s_per_curve_at_1e6"))
        for key, rows in points.items():
            writer.writerows((key, *row) for row in rows)


if __name__ == "__main__":
    main()
