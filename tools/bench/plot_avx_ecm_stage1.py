#!/usr/bin/env python3
"""Plot the avx-ecm and ECM-ELF Stage1 wall-time matrix."""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


SERIES = (
    ("avx52_ifma", "avx-ecm 52-bit IFMA", "#1c735c", "o", "-"),
    ("avx52_fma", "avx-ecm 52-bit FMA", "#6b5099", "s", "-"),
    ("avx52_ifma_fold", "avx-ecm 52-bit IFMA, extended fold", "#1c735c", "^", "--"),
    ("avx32", "avx-ecm 32-bit REDC", "#8b661e", "v", "-"),
    ("avx32_maskfix", "avx-ecm 32-bit fold, mask fix", "#8b661e", "v", "--"),
    ("ecm_mont", "ECM-ELF Montgomery IFMA", "#225e9c", "D", "-"),
    ("ecm_ed", "ECM-ELF Edwards IFMA", "#ba5227", "P", "-"),
)


def series_points(rows, checks, variant, timer):
    for row in rows:
        width = row["exponent"]
        group = row["samples"].get(variant)
        if group and timer in group[0]:
            if variant.startswith("avx") and not checks[str(width)][variant]["valid"]:
                continue
            values = [sample[timer] / sample["lanes"] for sample in group]
            yield width, min(values), median(values), max(values)


def draw(rows, checks, output):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, timer, title in zip(axes, ("native", "wall"),
                                ("avx-ecm Stage1 loop", "Process start to exit")):
        for variant, label, color, marker, line_style in SERIES:
            points = list(series_points(rows, checks, variant, timer))
            if not points:
                continue
            width, low, middle, high = zip(*points)
            ax.errorbar(width, middle, yerr=([m - lo for m, lo in zip(middle, low)],
                                           [hi - m for hi, m in zip(high, middle)]),
                        label=label, color=color, marker=marker, linestyle=line_style,
                        linewidth=1.6, markersize=4, capsize=2)
        ax.axvline(2048, color="#6c7278", linewidth=0.9, linestyle=":")
        ax.annotate("2,048-bit REDC switch", xy=(2048, 0.96), xycoords=ax.get_xaxis_transform(),
                    xytext=(6, 0), textcoords="offset points", fontsize=8, va="top",
                    color="#5b6167", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.set_xlabel("Mersenne exponent (bits)")
        ax.set_yscale("log")
        ax.set_xticks((0, 1000, 2000, 3000, 4000))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Seconds per curve (full SIMD batch throughput)")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.03),
                  ncols=3, frameon=False, fontsize=9)
    figure.suptitle("CPU Stage1 on Ryzen AI 9 HX 370", x=0.09, ha="left", fontweight="bold")
    figure.text(0.09, 0.91, "One logical CPU · B1 = B2 = 100,000 · median and range of three runs",
                fontsize=9, color="#4f5962")
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.28, top=0.82, wspace=0.13)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("verification", type=Path, help="Per-width and per-variant correctness gate")
    parser.add_argument("--supplement", type=Path, help="Optional additional timings for the same widths")
    parser.add_argument("--output", type=Path,
                        default=Path("data/figures/avx_ecm_4k/stage1-avx-ecm-4k.png"))
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if args.supplement:
        by_width = {row["exponent"]: row for row in rows}
        for extra in json.loads(args.supplement.read_text(encoding="utf-8")):
            by_width[extra["exponent"]]["samples"].update(extra["samples"])
    checks = json.loads(args.verification.read_text(encoding="utf-8"))
    draw(sorted(rows, key=lambda row: row["exponent"]), checks, args.output)
