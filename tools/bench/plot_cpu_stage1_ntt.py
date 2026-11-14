#!/usr/bin/env python3
"""Plot CPU weighted NTT round-one timings and phase breakdowns."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


def load_samples(path):
    timings, phases = defaultdict(list), defaultdict(list)
    with path.open(newline="") as source:
        for kind, bits, operation, batch, sample, *data in csv.reader(source):
            if kind == "time":
                key = (int(bits), operation, int(batch), data[0])
                timings[key].append(float(data[3]))
            elif kind == "phase":
                key = (int(bits), operation, int(batch), data[0])
                phases[key].append(tuple(map(float, data[1:])))
    if any(len(group) != 3 for group in (*timings.values(), *phases.values())):
        raise ValueError("expected three samples for every benchmark condition")
    return timings, phases


def plot(timings, phases, output):
    widths = sorted({key[0] for key in timings})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    series = [
        ("zmm-canonical", 1, "Original ZMM, one chain", "#7c8492", "--"),
        ("best_ntt", 1, "Best NTT, one chain", "#c05425", "-"),
        ("best_ntt", 8, "Best NTT, eight chains", "#c05425", ":"),
        ("ifma", 1, "IFMA, one active lane", "#216397", "--"),
        ("ifma", 8, "IFMA, eight active lanes", "#216397", "-"),
        ("gwnum", 1, "GWNUM, one chain", "#1b1d21", "-"),
    ]
    for ax, operation in zip(axes, ("multiply", "square")):
        for backend, batch, label, color, style in series:
            values = []
            for width in widths:
                if backend == "best_ntt":
                    candidates = [group for (p, op, n, kind), group in timings.items()
                                  if (p, op, n) == (width, operation, batch)
                                  and kind.startswith(("ymm", "zmm"))]
                    values.append(min(candidates, key=median))
                else:
                    values.append(timings[width, operation, batch, backend])
            center = [median(group) for group in values]
            err = [[value - min(group) for value, group in zip(center, values)],
                   [max(group) - value for value, group in zip(center, values)]]
            ax.errorbar(widths, center, yerr=err, label=label, color=color, linestyle=style,
                        marker="o", markersize=4, capsize=2)
        ax.set(xscale="log", yscale="log", xlabel="Mersenne exponent (bits)")
        ax.set_title("Modular multiplication" if operation == "multiply" else "Modular squaring",
                     loc="left", fontweight="bold")
        ax.set_xticks([4000, 8000, 16000, 32000, 64000], labels=["4K", "8K", "16K", "32K", "64K"])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("µs per modular operation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, fontsize=9)
    fig.suptitle("CPU Stage1 modular arithmetic · one logical core", x=0.1, ha="left", fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.25, wspace=0.16)
    fig.savefig(output / "stage1-cpu-ntt-round1.png", dpi=180, facecolor="white")
    plt.close(fig)

    phase_names = ["Pack + weight", "Forward", "Pointwise", "Inverse", "Unweight + unpack", "Carry + writeback"]
    colors = ["#a46538", "#286599", "#6c7580", "#4792c1", "#aa78a6", "#527a55"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for ax, operation in zip(axes, ("multiply", "square")):
        data = [tuple(map(median, zip(*phases[p, operation, 1, "zmm-canonical"]))) for p in widths]
        left = [0.0] * len(widths)
        for label, color, phase in zip(phase_names, colors, zip(*data)):
            ax.barh([f"M{p}" for p in widths], phase, left=left, color=color, label=label)
            left = [a + b for a, b in zip(left, phase)]
        ax.invert_yaxis()
        ax.set_xlabel("Instrumented µs per operation")
        ax.set_title("Multiplication · 2 forward" if operation == "multiply" else "Square · 1 forward",
                     loc="left", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, fontsize=9)
    fig.suptitle("Original ZMM NTT · one dependent chain", x=0.1, ha="left", fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.24, wspace=0.22)
    fig.savefig(output / "stage1-cpu-ntt-phases.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, operation in zip(axes, ("multiply", "square")):
        for width_name, color in (("zmm", "#225d94"), ("ymm", "#b75226")):
            for mode, style, marker in (("canonical", ":", "o"), ("lazy", "-", "s"),
                                        ("canonical-fused8", "--", "D"), ("lazy-fused8", "-.", "^")):
                ratios = [median(timings[p, operation, 1, f"{width_name}-{mode}"]) /
                          median(timings[p, operation, 1, "zmm-canonical"]) for p in widths]
                ax.plot(widths, ratios, color=color, linestyle=style, marker=marker, markersize=4,
                        label=f"{width_name.upper()} {mode.replace('-', ' ')}")
        ax.axhline(1, color="#777d85", linewidth=0.8)
        ax.set(xscale="log", xlabel="Mersenne exponent (bits)")
        ax.set_xticks([4000, 8000, 16000, 32000, 64000], labels=["4K", "8K", "16K", "32K", "64K"])
        ax.set_title("Modular multiplication" if operation == "multiply" else "Modular squaring",
                     loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Time / original ZMM time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, frameon=False, fontsize=8)
    fig.suptitle("CPU weighted NTT · short layers and lazy reduction", x=0.1, ha="left", fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.25, wspace=0.16)
    fig.savefig(output / "stage1-cpu-ntt-variants.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--phases", type=Path, help="Separate baseline-only phase capture")
    parser.add_argument("--output", type=Path, default=Path("data/figures/cpu_stage1_ntt_round1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timings, phases = load_samples(args.samples)
    if args.phases:
        _, phases = load_samples(args.phases)
    plot(timings, phases, args.output)
