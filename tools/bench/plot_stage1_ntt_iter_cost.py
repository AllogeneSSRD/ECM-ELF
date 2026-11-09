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
    plan_fields = ("length", "batch", "ntt_type", "bpw", "tile", "max_radix_bits",
                   "graph_bits", "warp_tail", "forward", "inverse")
    by_length = defaultdict(list)
    plans = {}
    for group in groups:
        length, batch = group["tune"]["length"], group["batch"]
        samples = group["samples"]
        if (length < 2 or length & (length - 1) or batch < 1 or batch != group["tune"]["batch"]
                or not samples or len(samples) != report["rounds"]):
            raise ValueError(f"Invalid plan or sample count for M{group['p']}")
        times = []
        for sample in samples:
            if any(not math.isfinite(sample[k]) or sample[k] <= 0 for k in ("steps", "device_ms")):
                raise ValueError(f"Invalid ladder timing for M{group['p']}")
            times.append(1000 * sample["device_ms"] / sample["steps"])
        by_length[length].append((batch, times))
        plans[group["p"]] = tuple(group["tune"][key] for key in plan_fields)

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
    return rows, plans


def plot(series, device, output):
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
        fig.suptitle(f"Stage1 NTT cost by transform length · {device}", fontweight="bold")
        for ax, metric, title, ylabel in zip(axes, ("curve", "batch"),
                                             ("Amortized across curves", "Whole batch"),
                                             ("µs per curve per ladder step", "µs per batch per ladder step")):
            for rows, label, color, marker, line in series:
                lengths = [r["length"] for r in rows]
                values = [r[f"median_us_per_{metric}_iter"] for r in rows]
                low = [r[f"min_us_per_{metric}_iter"] for r in rows]
                high = [r[f"max_us_per_{metric}_iter"] for r in rows]
                ax.fill_between(lengths, low, high, color=color, alpha=.15, linewidth=0)
                ax.plot(lengths, values, linestyle=line, linewidth=1.6, marker=marker,
                        markersize=5, markerfacecolor="white", color=color, label=label)
            ax.set_title(title)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            lengths = sorted({r["length"] for rows, *_ in series for r in rows})
            ax.set_xticks(lengths, [f"{v:,}" for v in lengths], rotation=35, ha="right")
            ax.set_xlabel("NTT length (points)")
            ax.set_ylabel(ylabel)
        batch_ranges = []
        for batches, run in groupby(series[0][0], key=lambda row: row["batches"]):
            lengths = [row["length"] for row in run]
            span = f"{lengths[0]:,}–{lengths[-1]:,}" if len(lengths) > 1 else f"{lengths[0]:,}"
            batch_ranges.append(f"{span} points: {batches} curves/batch")
        axes[1].text(.03, .96, "\n".join(batch_ranges), transform=axes[1].transAxes,
                     ha="left", va="top", fontsize=9, color="#484848")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
        fig.savefig(output, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edwards", type=Path, required=True)
    parser.add_argument("--montgomery", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True, help="Output PNG")
    parser.add_argument("--csv", type=Path, required=True, help="Grouped benchmark statistics")
    args = parser.parse_args()
    ed, mont = (json.loads(path.read_text(encoding="utf-8-sig")) for path in (args.edwards, args.montgomery))
    if (ed["curve"], mont["curve"]) != ("ed", "mont") or any(ed[k] != mont[k] for k in ("device", "gpu_model")):
        parser.error("Expected Edwards and Montgomery reports from the same GPU")
    ed_rows, ed_plans = summarize(ed)
    mont_rows, mont_plans = summarize(mont)
    if ed_plans != mont_plans:
        parser.error("Expected the same measured plans for the two curve models")

    series = [(mont_rows, f"Montgomery PARAM0 · {mont_rows[0]['version']}", "#c3631c", "s", "-"),
              (ed_rows, f"Edwards · {ed_rows[0]['version']}", "#176fa5", "o", "--")]
    device = ed["gpu_model"].removeprefix("NVIDIA GeForce ").removesuffix(" GPU")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    plot(series, device, args.figure)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ed_rows[0]))
        writer.writeheader()
        writer.writerows(ed_rows + mont_rows)


if __name__ == "__main__":
    main()
