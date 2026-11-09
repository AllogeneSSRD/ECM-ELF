#!/usr/bin/env python3
"""Plot Stage1 auto versus binary-only ladder cost for a complete matrix."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUTO = "#176fa5"
BINARY = "#cb6719"


def summarize(report):
    groups = sorted(report["groups"], key=lambda row: row["p"])
    if [r["p"] for r in groups] != report["exponents"]:
        raise ValueError("The matrix is incomplete or duplicated")
    rows = []
    for group in groups:
        auto, binary = group["auto_plan"], group["binary_plan"]
        row = dict(p=group["p"], input="synthetic" if group["synthetic"] else "PARAM0",
                   batch=group["batch"], b1=group["b1"], auto_length=int(auto["length"]),
                   auto_convolution=auto["convolution"], auto_radix5="r5" in auto["forward_layers"],
                   auto_bpw=int(auto["bpw"]), auto_tile=int(auto["tile_layers"]),
                   auto_outer=int(auto["outer_max_layers"]), auto_warp_tail=int(auto["warp_tail"]),
                   auto_forward=auto["forward_layers"], auto_inverse=auto["inverse_layers"],
                   auto_graph_bits=int(auto["graph_bits"]), binary_length=binary["length"],
                   binary_convolution=binary["ntt_type"].removeprefix("stage1_"),
                   binary_bpw=binary["bpw"], binary_tile=binary["tile"],
                   binary_outer=binary["max_radix_bits"], binary_warp_tail=binary["warp_tail"],
                   binary_forward=binary["forward"],
                   binary_inverse=binary["inverse"], binary_graph_bits=binary["graph_bits"])
        steps = {s["steps"] for samples in group["samples"].values() for s in samples}
        if len(steps) != 1:
            raise ValueError(f"Mismatched ladder steps at p={row['p']}")
        row["steps"] = steps.pop()
        for mode in ("auto", "binary"):
            times = [s["us_per_iter"] for s in group["samples"][mode]]
            if len(times) != report["rounds"] or min(times) <= 0:
                raise ValueError(f"Invalid {mode} timings at p={row['p']}")
            row[f"{mode}_us_per_curve_iter"] = median(times)
            row[f"{mode}_us_per_batch_iter"] = row[f"{mode}_us_per_curve_iter"] * row["batch"]
            row[f"{mode}_min"] = min(times)
            row[f"{mode}_max"] = max(times)
        row["change_pct"] = 100 * (1 - row["auto_us_per_curve_iter"] / row["binary_us_per_curve_iter"])
        rows.append(row)
    return rows


def overview(rows, device, output):
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, (cost, change) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
        fig.suptitle(f"Stage1 NTT auto vs binary-only · {device} · 0.60.2", fontweight="bold")
        for input_kind, marker in (("PARAM0", "o"), ("synthetic", "D")):
            group = [r for r in rows if r["input"] == input_kind]
            p = [r["p"] for r in group]
            for mode, color in (("auto", AUTO), ("binary", BINARY)):
                y = [r[f"{mode}_us_per_curve_iter"] for r in group]
                cost.scatter(p, y, marker=marker, s=36 if mode == "auto" else 15,
                             facecolors="none", edgecolors=color,
                             label=f"{mode} · {input_kind}")
            change.scatter(p, [r["change_pct"] for r in group], marker=marker, s=24, color=AUTO,
                           label=input_kind)
        for ax in (cost, change):
            ax.set_xscale("log")
            ax.set_xlabel("Mersenne exponent p (bits)")
            ax.legend(fontsize=8)
        cost.set_yscale("log")
        cost.set_ylabel("µs per curve per ladder step")
        change.axhline(0, color="0.45", linewidth=1)
        change.set_ylabel("Time saved vs binary-only (%)")
        fig.savefig(output, dpi=180)
        plt.close(fig)


def boundaries(rows, ranges, output):
    by_p = {r["p"]: r for r in rows}
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(1, len(ranges), figsize=(13, 4.6), layout="constrained")
        fig.suptitle("Stage1 NTT · exact radix-5 boundaries · 24 synthetic states", fontweight="bold")
        for ax, (low, high, length) in zip(axes, ranges):
            points = (low - 1, low, high, high + 1)
            ax.axvspan(.5, 2.5, color=AUTO, alpha=.06)
            for mode, color, offset in (("auto", AUTO, -.13), ("binary", BINARY, .13)):
                group = [by_p[p] for p in points]
                x = [i + offset for i in range(len(points))]
                y = [r[f"{mode}_us_per_curve_iter"] for r in group]
                err = [[v - r[f"{mode}_min"] for v, r in zip(y, group)],
                       [r[f"{mode}_max"] - v for v, r in zip(y, group)]]
                ax.errorbar(x, y, yerr=err, marker="o", linestyle="none", color=color,
                            capsize=3, label=mode)
                for i, row in enumerate(group):
                    ax.annotate(f'{row[f"{mode}_length"]:,}', (x[i], y[i]), xytext=(0, 8 if mode == "auto" else -12),
                                textcoords="offset points", ha="center", fontsize=7, color=color)
            ax.margins(y=.22)
            ax.set_xlim(-.48, 3.48)
            ax.set_xticks(range(len(points)), [f"{p:,}" for p in points], rotation=25, ha="right")
            ax.set_xlabel("Mersenne exponent p (bits)")
            ax.set_title(f"{length:,}-point candidate")
        axes[0].set_ylabel("µs per curve per ladder step")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
        fig.savefig(output, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--figures", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    rows = summarize(report)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    device = report["gpu_model"].removeprefix("NVIDIA GeForce ").removesuffix(" GPU")
    overview(rows, device, args.figures / "stage1-ntt-auto-vs-binary.png")
    boundaries(rows, report["mixed_ranges"], args.figures / "stage1-ntt-auto-boundaries.png")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
