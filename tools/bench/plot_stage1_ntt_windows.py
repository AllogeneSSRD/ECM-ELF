#!/usr/bin/env python3
"""Plot Stage1 Edwards NTT window timings and NAF allocation sizes from a measured matrix."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator, StrMethodFormatter


WINDOWS = (8, 10, 12)
MARKERS = ("s", "o", "^")
DEVICES = (
    ("RTX 4070 Ti", ("#606570", "#168353", "#2866a2")),
    ("RTX 4060 Laptop", ("#606570", "#d4771d", "#b73339")),
)
MEMORY_FIELDS = (
    ("dictionary_spectra_bytes", "Per-curve dictionary spectra (VRAM)", "#377ca4"),
    ("dictionary_workspace_bytes", "Dictionary-build scratch (VRAM)", "#7abdb7"),
    ("naf_host_dictionary_bytes", "Dictionary or seeds (RAM)", "#d58a47"),
    ("naf_encoding_bytes", "Shared packed and decoded NAF (RAM)", "#8b5ca5"),
)
# The root inverse is also measured separately inside device_dictionary.
INIT_TIMINGS = ("encoding", "curve_setup", "host_points", "device_dictionary", "other", "root_inverse")
DICTIONARY_COUNTS = ("entries_per_curve", "host_dictionary_curves", "device_dictionary_curves")
WIDTH_BOUNDS = (0, 1024, 4096, 16384, 65536, math.inf)


def load_matrix(path):
    groups = sorted(json.loads(path.read_text(encoding="utf-8-sig")), key=lambda row: row["p"])
    if not groups or len({row["p"] for row in groups}) != len(groups):
        raise ValueError(f"Empty or duplicated exponent matrix in {path}")
    rows = []
    for group in groups:
        for window in WINDOWS:
            samples = group["samples"][str(window)]
            if not samples:
                raise ValueError(f"No w{window} samples at p={group['p']} in {path}")
            for trial, sample in enumerate(samples, start=1):
                if (sample.get("naf_window", window) != window
                        or not 0 <= sample["active_curves"] <= group["batch"]
                        or not math.isfinite(sample["naf_init_ms"]) or sample["naf_init_ms"] <= 0
                        or not math.isfinite(sample["device_ms"]) or sample["device_ms"] < 0
                        or any(sample[field] < 0 for field, _, _ in MEMORY_FIELDS)):
                    raise ValueError(f"Invalid w{window} sample {trial} at p={group['p']} in {path}")
                timings_ms = {name: sample.get(f"{name}_ms") for name in INIT_TIMINGS}
                rows.append(dict(
                    p=group["p"], b1=group["b1"], sigma=group["sigma"], length=group["length"],
                    requested_curves=group["batch"], active_curves=sample["active_curves"],
                    window=window, trial=trial, dictionary_backend=sample["dictionary_backend"],
                    raw_source=group.get("raw_source", ""),
                    naf_init_seconds=sample["naf_init_ms"] / 1000,
                    **{f"{name}_seconds": elapsed / 1000 if elapsed is not None else None
                       for name, elapsed in timings_ms.items()},
                    main_loop_seconds=sample["device_ms"] / 1000,
                    naf_columns=sample.get("steps"),
                    naf_nonzero_columns=sample.get("additions"),
                    **{field: sample.get(field) for field in DICTIONARY_COUNTS},
                    **{field: sample[field] for field, _, _ in MEMORY_FIELDS},
                ))
    return rows


def summarize(rows):
    points = {}
    for row in rows:
        points.setdefault(row["p"], {}).setdefault(row["window"], []).append(row)
    summary = {}
    for p, windows in points.items():
        summary[p] = {}
        for window, samples in windows.items():
            active = {sample["active_curves"] for sample in samples}
            main = [sample["main_loop_seconds"] / sample["active_curves"] for sample in samples
                    if sample["active_curves"] and sample["main_loop_seconds"]]
            summary[p][window] = dict(
                batch=samples[0]["requested_curves"], active=active,
                main=median(main) if main else math.nan,
                init=median(sample["naf_init_seconds"] for sample in samples),
                memory=tuple(median(sample[field] for sample in samples) for field, _, _ in MEMORY_FIELDS),
            )
    return summary


def bit_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks((64, 256, 1024, 4096, 16384, 65536, 262144))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda value, _: f"{int(value) // 1024}K" if value >= 1024 else str(int(value))))


def loop_gain(baseline, candidate):
    if baseline["active"] != candidate["active"] or len(baseline["active"]) != 1:
        return math.nan
    return 100 * (1 - candidate["main"] / baseline["main"])


def timings(devices, output, version):
    fig, axes = plt.subplots(2, len(devices), figsize=(11, 6.6), sharex="col", sharey="row",
                             layout="constrained", squeeze=False)
    fig.suptitle(f"Stage1 Edwards NTT · Native NAF window size · {version}", weight="semibold")
    fig.supxlabel("Mersenne exponent p (bits, log scale)")
    for col, (name, colors, rows) in enumerate(devices):
        main, init = axes[:, col]
        bit_axis(init)
        main.set_title(name)
        main.axhline(0, color="#606570", lw=0.8, linestyle="--")
        init.set_yscale("log")
        init.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        init.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        init.yaxis.set_minor_locator(NullLocator())
        for window, color, marker in zip(WINDOWS, colors, MARKERS):
            style = dict(color=color, marker=marker, markersize=3.8, lw=1.55, label=f"w{window}")
            if window != 8:
                main.plot(list(rows), [loop_gain(r[8], r[window]) for r in rows.values()], **style)
            init.plot(list(rows), [r[window]["init"] for r in rows.values()], **style)
        init.legend(loc="best", ncol=3, fontsize=9)
    axes[0, 0].set_ylabel("Main loop time saved vs w8 (%)")
    axes[1, 0].set_ylabel("NAF initialization time (s / batch)")
    fig.savefig(output, dpi=185)
    plt.close(fig)


def memory_panel(ax, rows, exponents, y_max):
    for offset, window in zip((-0.255, 0, 0.255), WINDOWS):
        positions = [index + offset for index in range(len(exponents))]
        bases = [0.] * len(exponents)
        for component, (_, _, color) in enumerate(MEMORY_FIELDS):
            sizes = [rows[p][window]["memory"][component] / 2**20 if p in rows else 0 for p in exponents]
            ax.bar(positions, sizes, width=0.235, bottom=bases, color=color, zorder=2)
            bases = [base + size for base, size in zip(bases, sizes)]
    ax.set_xticks(range(len(exponents)), [f"{p:,}\n{rows[p][8]['batch']}c" if p in rows else f"{p:,}"
                                         for p in exponents], fontsize=8)
    ax.set_ylim(0, 1.08 * y_max)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", visible=False)


def encoding_panel(ax, rows):
    ax.set_title("Shared encoding detail", fontsize=10)
    encoding = [[row[window]["memory"][-1] / 1024 for row in rows.values()] for window in WINDOWS]
    colors = ("#606570", "#168353", "#8b5ca5")
    if all(min(values) == max(values) for values in encoding):
        bars = ax.barh([f"w{window}" for window in WINDOWS], [values[0] for values in encoding], color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
        ax.set_xlim(0, 1.18 * max(values[0] for values in encoding))
        ax.invert_yaxis()
        ax.set_xlabel("RAM (KiB / batch)", fontsize=8, labelpad=0)
        ax.grid(axis="y", visible=False)
        return
    for window, marker, color, values in zip(WINDOWS, MARKERS, colors, encoding):
        ax.plot(list(rows), values, color=color, marker=marker, markerfacecolor="none",
                markersize=3.3, lw=1.1, label=f"w{window}")
    bit_axis(ax)
    ax.set_ylabel("RAM (KiB / batch)")
    ax.legend(loc="best", ncol=3, fontsize=8)


def memory_stack(rows):
    return {p: [(row[w]["batch"], row[w]["memory"]) for w in WINDOWS] for p, row in rows.items()}


def memory(devices, output, version):
    widths = sorted({p for _, _, rows in devices for p in rows})
    ranges = [[p for p in widths if lo <= p < hi] for lo, hi in zip(WIDTH_BOUNDS, WIDTH_BOUNDS[1:])]
    ranges = [exponents for exponents in ranges if exponents]
    same = all(memory_stack(rows) == memory_stack(devices[0][2]) for _, _, rows in devices)
    nrows = (len(ranges) + 2) // 2 if same else len(ranges) + 1
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 2.17 * nrows + 0.75), squeeze=False,
                             gridspec_kw={"hspace": 0.58, "wspace": 0.2})
    fig.suptitle(f"Stage1 Edwards NTT · NAF memory by window size · {version}", weight="semibold", y=0.98)
    subtitle = "Same allocation sizes on RTX 4070 Ti and RTX 4060 Laptop · " if same else ""
    fig.text(0.5, 0.938, subtitle + "Each group shows w8, w10, then w12 · c denotes requested curves",
             ha="center", fontsize=9)
    for index, exponents in enumerate(ranges):
        y_max = max(sum(value["memory"]) / 2**20 for _, _, rows in devices for p in exponents
                    if p in rows for value in rows[p].values())
        for col, (name, _, rows) in enumerate(devices[:1] if same else devices):
            ax = axes.flat[index] if same else axes[index, col]
            memory_panel(ax, rows, exponents, y_max)
            label = f"{exponents[0]:,}–{exponents[-1]:,} bits" if len(exponents) > 1 else f"{exponents[0]:,} bits"
            ax.set_title(label if same else f"{name} · {label}", fontsize=10)
            if (same and index % 2 == 0) or (not same and col == 0):
                ax.set_ylabel("Logical allocation (MiB)")
    for col, (_, _, rows) in enumerate(devices[:1] if same else devices):
        ax = axes.flat[len(ranges)] if same else axes[-1, col]
        encoding_panel(ax, rows)
    if same and len(ranges) % 2 == 0:
        axes.flat[-1].axis("off")
    legend = [Patch(facecolor=color, label=label) for _, label, color in MEMORY_FIELDS]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.043), ncol=4,
               fontsize=8.5, frameon=False)
    fig.text(0.5, 0.025, "Stacks add host and device allocation sizes across setup, including transient scratch.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86 if same else 0.89, bottom=0.112 if same else 0.092)
    fig.savefig(output, dpi=185)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory with gpu0/summary.json and gpu1/summary.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True, help="Version of the executable used to measure the inputs")
    args = parser.parse_args()
    raw = []
    devices = []
    for index, (name, colors) in enumerate(DEVICES):
        rows = load_matrix(args.input / f"gpu{index}" / "summary.json")
        raw.extend(dict(version=args.version, device=name, **row) for row in rows)
        devices.append((name, colors, summarize(rows)))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    version = args.version.replace(".", "")
    with plt.style.context("seaborn-v0_8-whitegrid"):
        timings(devices, args.output / f"stage1-edwards-ntt-naf-init-{version}.png", args.version)
        memory(devices, args.output / f"stage1-edwards-ntt-naf-memory-{version}.png", args.version)


if __name__ == "__main__":
    main()
