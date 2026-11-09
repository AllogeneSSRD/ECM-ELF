#!/usr/bin/env python3
"""Summarize production Stage1 ladder cost by the selected NTT length."""
import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(report):
    groups = report["groups"]
    if not groups or sorted(group["p"] for group in groups) != sorted(set(report["exponents"])):
        raise ValueError("Expected one measured group per requested exponent")
    by_length = defaultdict(list)
    for group in groups:
        length, batch = group["tune"]["length"], group["batch"]
        samples = group["samples"]
        if (length < 2 or batch < 1 or batch != group["tune"]["batch"]
                or not samples or len(samples) != report["rounds"]):
            raise ValueError(f"Invalid plan or sample count for M{group['p']}")
        times = []
        for sample in samples:
            if any(not math.isfinite(sample[k]) or sample[k] <= 0 for k in ("steps", "device_ms")):
                raise ValueError(f"Invalid ladder timing for M{group['p']}")
            times.append(1000 * sample["device_ms"] / sample["steps"])
        by_length[length].append((batch, times))

    rows = []
    for length, cells in sorted(by_length.items()):
        row = dict(curve=report["curve"], version=report["version"].split()[-1],
                   length=length, input_count=len(cells), repeat_count=sum(len(t) for _, t in cells),
                   batches=",".join(map(str, sorted({b for b, _ in cells}))))
        for name, per_curve in (("curve", True), ("batch", False)):
            series = [[t / (batch if per_curve else 1) for t in times] for batch, times in cells]
            row[f"median_us_per_{name}_iter"] = median(map(median, series))
            row[f"min_us_per_{name}_iter"] = min(map(min, series))
            row[f"max_us_per_{name}_iter"] = max(map(max, series))
        rows.append(row)
    return rows


def plot(rows, device, output):
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
        fig.suptitle(f"Stage1 NTT cost by transform length · {device}", fontweight="bold")
        for ax, metric, title, ylabel in zip(axes, ("curve", "batch"),
                                             ("Amortized across curves", "Whole batch"),
                                             ("µs per curve per ladder step", "µs per batch per ladder step")):
            lengths = [r["length"] for r in rows]
            values = [r[f"median_us_per_{metric}_iter"] for r in rows]
            low = [r[f"min_us_per_{metric}_iter"] for r in rows]
            high = [r[f"max_us_per_{metric}_iter"] for r in rows]
            ax.fill_between(lengths, low, high, color="#176fa5", alpha=.15, linewidth=0)
            ax.plot(lengths, values, linewidth=1.6, marker="o", markersize=5,
                    markerfacecolor="white", color="#176fa5")
            ax.set_title(title)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(lengths, [f"{v:,}" for v in lengths], rotation=35, ha="right")
            ax.set_xlabel("NTT length (points)")
            ax.set_ylabel(ylabel)
        batch_ranges = []
        for batches, run in groupby(rows, key=lambda row: row["batches"]):
            lengths = [row["length"] for row in run]
            span = f"{lengths[0]:,}–{lengths[-1]:,}" if len(lengths) > 1 else f"{lengths[0]:,}"
            batch_ranges.append(f"{span} points: {batches} curves/batch")
        axes[1].text(.03, .96, "\n".join(batch_ranges), transform=axes[1].transAxes,
                     ha="left", va="top", fontsize=9, color="#484848")
        fig.savefig(output, dpi=180)
        plt.close(fig)


def weighted_steps(lengths):
    modulus = (1 << 64) - (1 << 32) + 1
    next_bit = 32
    for length in sorted(lengths):
        max_bpw = max(b for b in range(1, 32) if 2 * length * ((1 << b) - 1)**2 < modulus)
        end = min(262144, length * max_bpw)
        start = max(next_bit, length)
        if start <= end:
            yield start, end, length
            next_bit = end + 1


def plot_steps(report, output):
    binary = {1 << k for k in range(1, 17)}
    mixed = {5 << k for k in range(9, 13)}
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(9.5, 5), layout="constrained")
        for lengths, label, color in ((binary, "Weighted · powers of two", "#cb6719"),
                                      (binary | mixed, "Weighted · default radix-5 range", "#176fa5")):
            steps = list(weighted_steps(lengths))
            edges = [p for p, _, _ in steps] + [steps[-1][1] + 1]
            ax.stairs([length for _, _, length in steps], edges, baseline=None,
                      label=label, color=color, linewidth=1.7)
        groups = report["groups"]
        ax.scatter([g["p"] for g in groups], [g["tune"]["length"] for g in groups],
                   label="Measured production choices (linear or weighted)", s=18, color="#45346e", zorder=3)
        ax.set(xscale="log", yscale="log", xlabel="Mersenne exponent p (bits)",
               ylabel="NTT length (points)", title="Stage1 NTT length thresholds")
        ax.set_yscale("log", base=2)
        ax.set_yticks([1 << k for k in range(1, 15)], [f"{1 << k:,}" for k in range(1, 15)])
        ax.legend(loc="upper left", frameon=True)
        fig.savefig(output, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One Stage1 production summary")
    parser.add_argument("--figure", type=Path, required=True, help="Output PNG")
    parser.add_argument("--csv", type=Path, required=True, help="Grouped benchmark statistics")
    parser.add_argument("--steps-figure", type=Path, required=True, help="Exponent-to-length staircase PNG")
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8-sig"))
    rows = summarize(report)
    device = report["gpu_model"].removeprefix("NVIDIA GeForce ").removesuffix(" GPU")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.steps_figure.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, f"{device} · {rows[0]['version']}", args.figure)
    plot_steps(report, args.steps_figure)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
