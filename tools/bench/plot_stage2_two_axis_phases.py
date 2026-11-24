#!/usr/bin/env python3
"""Plot paired CUDA Stage2 phase times at the original and tenfold B2 bounds."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


PHASES = (
    ("baby", "Baby points", "#71859a"),
    ("f_tree", "F tree", "#1664aa"),
    ("inv", "Newton inverse", "#8651a5"),
    ("giant", "Giant points", "#a9b9c9"),
    ("g_tree", "G tree", "#0a887c"),
    ("fold", "H fold", "#d27d15"),
    ("descent", "F descent", "#b94066"),
    ("accum", "Accum. / GCD", "#546374"),
)
WIDTHS = ("2k", "4k", "8k", "16k")
DUAL_PHASES = (("f_tree", "F tree"), ("inv", "Newton"), ("g_tree", "G tree"), ("fold", "H fold"), ("descent", "F descent"))
INK = "#162435"
GRID = "#d5dee8"
HATCH = "///"


def read_curves(runs):
    curves = {}
    case_keys = ("n_bits", "carrier", "sigma", "b1", "b2", "d", "p", "giant", "loops", "identity", "fixture", "known_factor")
    for width in WIDTHS:
        samples = [run for run in runs if run["width"] == width]
        if not samples:
            raise ValueError(f"missing {width} samples")
        reference = samples[0]
        for key in case_keys:
            if any(run[key] != reference[key] for run in samples):
                raise ValueError(f"unpaired {key} for {width}")
        if reference["identity"]["status"] != "NF":
            raise ValueError(f"factor search would change the phase breakdown for {width}")
        modes = {}
        for mode in ("off", "on"):
            batch = [run for run in samples if run["mode"] == mode]
            if not batch or len({run["rep"] for run in batch}) != len(batch):
                raise ValueError(f"invalid repeat counts for {width} {mode}")
            counts = batch[0]["counts"]
            if any(run["counts"] != counts for run in batch) or any(counts.values()) != (mode == "on"):
                raise ValueError(f"inconsistent two-axis counts for {width} {mode}")
            if any(min(run["seconds"].values()) < 0 or not math.isclose(sum(run["seconds"].values()), run["total"])
                   for run in batch):
                raise ValueError(f"phase times do not cover the curve for {width} {mode}")
            times = {name: mean(run["seconds"][name] for run in batch) for name, _, _ in PHASES}
            modes[mode] = (times, counts, {run["rep"]: run["seconds"] for run in batch})
        if modes["off"][2].keys() != modes["on"][2].keys():
            raise ValueError(f"unbalanced pairs for {width}")
        curves[width] = (reference, modes)
    return curves


def load_tiers(baseline, base_2k, tenfold):
    source = [json.loads(path.read_text(encoding="utf-8")) for path in (baseline, base_2k, tenfold)]
    if len({report["sha256"] for report in source}) != 1:
        raise ValueError("the two B2 tiers require one worker binary")
    base, high = read_curves(source[0]["runs"] + source[1]["runs"]), read_curves(source[2]["runs"])
    keys = ("n_bits", "carrier", "sigma", "b1", "d", "p", "fixture", "known_factor")
    for width in WIDTHS:
        low_case, high_case = base[width][0], high[width][0]
        if high_case["b2"] != 10 * low_case["b2"] or any(low_case[key] != high_case[key] for key in keys):
            raise ValueError(f"the {width} tiers must differ only in B2")
    return base, high


def style_axis(axis, units):
    axis.set_xlabel(units, color=INK)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color=GRID, linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8495a8")
    axis.tick_params(axis="both", colors=INK, length=3, width=0.65)
    axis.tick_params(axis="y", length=0, pad=6)


def curve_center(index):
    return (len(WIDTHS) - 1 - index) * 2.28


def draw_curves(axis, curves, title):
    mode_ticks = []
    labels = []
    longest = 0.0
    for index, width in enumerate(WIDTHS):
        run, modes = curves[width]
        center = curve_center(index)
        axis.text(-0.11, center + 0.21, width, transform=axis.get_yaxis_transform(),
                  va="center", ha="right", fontweight="bold", color=INK)
        axis.text(-0.11, center - 0.34, f"B₂ {run['b2'] / 1e9:g}B", transform=axis.get_yaxis_transform(),
                  va="center", ha="right", fontsize=8, color="#4b5d71")
        for mode, shift in (("off", 0.40), ("on", -0.40)):
            y = center + shift
            mode_ticks.append(y)
            labels.append(mode.capitalize())
            times, counts, _ = modes[mode]
            left = 0.0
            for key, _, color in PHASES:
                duration = times[key]
                if duration > 0:
                    axis.barh(y, duration, left=left, height=0.62, color=color, edgecolor="white",
                              linewidth=0.45, hatch=HATCH if counts.get(key, 0) else None)
                left += duration
            longest = max(longest, left)
            axis.annotate(f"{left:.2f}", (left, y), xytext=(5, 0), textcoords="offset points",
                          va="center", color=INK, fontsize=8)
    axis.set_yticks(mode_ticks, labels)
    axis.set_ylim(-0.88, curve_center(0) + 0.88)
    axis.set_xlim(0.0, longest * 1.15)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=7))
    axis.set_title(title, loc="left", color=INK, weight="bold", pad=10)
    style_axis(axis, "Measured phase time (s / curve)")


def draw_changes(axis, curves):
    for row, width in enumerate(WIDTHS):
        y = curve_center(row)
        _, modes = curves[width]
        _, _, off_runs = modes["off"]
        _, counts, on_runs = modes["on"]
        for column, (key, _) in enumerate(DUAL_PHASES):
            if not counts.get(key) or any(run[key] <= 0 for run in off_runs.values()):
                axis.text(column, y, "—", ha="center", va="center", color=INK, fontsize=8.5)
                continue
            changes = [(on_runs[rep][key] - off[key]) / off[key] for rep, off in off_runs.items()]
            change = median(changes)
            consistent = all(delta < 0 for delta in changes) or all(delta > 0 for delta in changes)
            visible = abs(change) >= 0.0005
            if not visible:
                label, background, color = "0.0%", "#eef2f6", INK
            elif change < 0:
                label, background, color = f"−{abs(change):.1%}", "#e1edf8", "#174d81"
            else:
                label, background, color = f"+{change:.1%}", "#f9e9e5", "#8b3c2f"
            axis.barh(y, 0.89, left=column - 0.445, height=0.88, color=background)
            marker = "*" if consistent and visible else ""
            axis.text(column, y, label + marker, ha="center", va="center", color=color, fontsize=8)
    axis.set_xlim(-0.5, len(DUAL_PHASES) - 0.5)
    axis.set_ylim(-0.88, curve_center(0) + 0.88)
    axis.set_xticks(range(len(DUAL_PHASES)), [label for _, label in DUAL_PHASES], fontsize=8)
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=4, colors=INK)
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title("Median paired change (%)", fontsize=9, weight="bold", pad=20, color=INK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="original 4k, 8k and 16k summary JSON")
    parser.add_argument("base_2k", type=Path, help="2k baseline summary JSON")
    parser.add_argument("tenfold", type=Path, help="all widths at tenfold B2 summary JSON")
    parser.add_argument("--output", type=Path, default=Path("data/figures/stage2-two-axis-phases.png"))
    args = parser.parse_args()

    with plt.rc_context({"font.size": 10, "hatch.linewidth": 0.7, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), width_ratios=(1.7, 1.0))
        fig.subplots_adjust(left=0.16, right=0.975, top=0.90, bottom=0.19, hspace=0.48, wspace=0.18)
        for (stack, changes), curves, title in zip(
            axes, load_tiers(args.baseline, args.base_2k, args.tenfold), ("(a)  Baseline B₂", "(b)  Tenfold B₂"), strict=True
        ):
            draw_curves(stack, curves, title)
            draw_changes(changes, curves)
        handles = [Patch(facecolor=color, edgecolor="none", label=label) for _, label, color in PHASES]
        handles.append(Patch(facecolor="#546374", edgecolor="white", hatch=HATCH, label="Two-axis used"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.045), ncol=5,
                   frameon=False, fontsize=8, labelcolor=INK, columnspacing=1.55, handlelength=1.65)
        fig.text(0.50, 0.023, "* All measured pairs changed in the same direction", ha="center", color=INK, fontsize=8)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=225, facecolor="white", bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)


if __name__ == "__main__":
    main()
