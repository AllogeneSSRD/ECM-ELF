#!/usr/bin/env python3
"""Plot the best measured ECM-ELF RTX 4070 Ti throughput against Prime95."""

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import shutil

import matplotlib

from analyze_stage1_gpu import core_value_labels

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke


MODELS = {"ed": ("Edwards", "#187e8b", "o"), "mont0": ("Montgomery PARAM0", "#7954a5", "s")}
GPU_SERIES = {
    "cgbn_overlay": "CGBN", "cgbn_overlay_previous": "CGBN", "cgbn_overlay_paired_ladder": "CGBN",
    "ntt_overlay": "NTT", "ntt_overlay_previous": "NTT",
}
PREFIX = "stage1-edwards-mont0-rtx4070ti-cpu-cores"


def load_report(path, model):
    cpu, gpu = {}, defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            exponent = int(row["mersenne_exponent"])
            series = row["series"]
            if series == "prime95":
                if exponent in cpu:
                    raise ValueError(f"Duplicate Prime95 reference for M{exponent}: {path}")
                cpu[exponent] = row
            elif series in GPU_SERIES:
                gpu[exponent].append(row)

    rows, ntt_only = [], []
    for exponent, candidates in gpu.items():
        cofactor_widths = {int(row["arithmetic_bits"]) for row in candidates if GPU_SERIES[row["series"]] == "CGBN"}
        if not cofactor_widths and exponent not in cpu:
            continue
        if (len(cofactor_widths) > 1 or exponent not in cpu
                or not any(GPU_SERIES[row["series"]] == "NTT" for row in candidates)):
            raise ValueError(f"Incomplete matched input M{exponent}: {path}")
        cpu_row = cpu[exponent]
        low, median, high = (float(cpu_row[f"{stat}_s_per_curve_at_1e6"]) for stat in ("min", "median", "max"))
        costs = [(float(row["median_s_per_curve_at_1e6"]), row) for row in candidates]
        if not (0 < low <= median <= high < math.inf) or any(not 0 < cost < math.inf for cost, _ in costs):
            raise ValueError(f"Invalid Stage1 timing for M{exponent}: {path}")
        gpu_seconds, fastest = min(costs, key=lambda item: item[0])
        arithmetic_bits = int(fastest["arithmetic_bits"])
        target = rows if cofactor_widths else ntt_only
        target.append({"model": model, "exponent": exponent, "bits": next(iter(cofactor_widths), arithmetic_bits),
                       "backend": GPU_SERIES[fastest["series"]], "gpu_series": fastest["series"],
                       "gpu_version": fastest["version"], "gpu_arithmetic_bits": arithmetic_bits,
                       "gpu_seconds_1e6": gpu_seconds, "cpu_seconds_1e6": median,
                       "equivalent_cpu_cores": median / gpu_seconds, "equivalent_cpu_min": low / gpu_seconds,
                       "equivalent_cpu_max": high / gpu_seconds})
    return {"rows": sorted(rows, key=lambda row: row["bits"]), "ntt_only": ntt_only}


def plot_series(ax, rows, x):
    name, color, marker = MODELS[rows[0]["model"]]
    y = [row["equivalent_cpu_cores"] for row in rows]
    errors = [[value - row["equivalent_cpu_min"] for value, row in zip(y, rows)],
              [row["equivalent_cpu_max"] - value for value, row in zip(y, rows)]]
    ax.errorbar(x, y, yerr=errors, color=color, marker=marker, markersize=4, capsize=2, label=name, zorder=3)


def plot(reports, mixes, wide, path, scale):
    fig, (mix_ax, ax) = plt.subplots(1, 2, figsize=(18.5 if wide else 15, 6),
                                     gridspec_kw={"width_ratios": [1, 5 if wide else 3.6]})
    rows = reports[0]["rows"]
    inputs = [(row["exponent"], row["bits"]) for row in rows]
    inputs += sorted({(row["exponent"], row["bits"]) for row in wide})
    positions = {key: i for i, key in enumerate(inputs)}
    offsets = {"ed": -.055, "mont0": .055}
    styles = {model: (name, "", color) for model, (name, color, _) in MODELS.items()}
    for i, (report, mix) in enumerate(zip(reports, mixes)):
        model = report["rows"][0]["model"]
        data = report["rows"] + [row for row in wide if row["model"] == model]
        _, color, _ = MODELS[model]
        pos = len(reports) - i - 1
        mix_ax.barh(pos, mix, color=color, height=.34)
        mix_ax.text(mix * 1.025, pos, f"{mix:.2f}", va="center", color=color, fontweight="bold")
        plot_series(ax, data, [positions[row["exponent"], row["bits"]] + offsets[model] for row in data])

    for j in range(1, len(rows)):
        switched = [MODELS[report["rows"][j]["model"]][1] for report in reports
                    if report["rows"][j]["backend"] != report["rows"][j - 1]["backend"]]
        if switched:
            ax.axvline(j - .5, color=switched[0] if len(switched) == 1 else "#53606d",
                       linestyle="--", linewidth=1.2, alpha=.75, zorder=2)

    mix_ax.set(xlim=(0, max(mixes) * 1.28), ylim=(-.65, 1.65),
               title=f"Equal {len(rows)}-input mix", xlabel="Equivalent CPU cores")
    mix_ax.set_yticks([1, 0], ["Edwards", "Montgomery\nPARAM0"])
    mix_ax.grid(axis="x", alpha=.2)
    ax.set_yscale(scale)
    if scale == "linear":
        all_rows = [row for report in reports for row in report["rows"]] + wide
        ax.set_ylim(0, max(row["equivalent_cpu_max"] for row in all_rows) * 1.16)
    else:
        ax.margins(y=.08)
    ax.set_xticks(range(len(inputs)), [f"M{exponent}\n{bits:,}" for exponent, bits in inputs], fontsize=8)
    ax.set(xlabel="Mersenne exponent and modulus bits (cofactor where CGBN was measured)",
           ylabel=f"Equivalent CPU cores ({scale})",
           title="Best measured CGBN or NTT per input")
    ax.grid(axis="y", alpha=.25)
    for panel in (mix_ax, ax):
        panel.spines[["right", "top"]].set_visible(False)
        panel.set_axisbelow(True)
    ax.legend(loc="upper right" if wide else "upper center", ncols=2, frameon=False)
    fig.suptitle("Stage1 · RTX 4070 Ti vs Prime95 on one 4.0 GHz core", fontweight="bold")
    fig.text(.5, .045, "B1 normalized to 1,000,000. Whiskers use the three Prime95 samples. "
             "Vertical dashes mark a backend switch between samples.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.10, right=.985, bottom=.18, top=.85, wspace=.32)
    core_value_labels(fig, ax, reports, styles)
    for row in wide:
        model = row["model"]
        x = positions[row["exponent"], row["bits"]] + offsets[model]
        y = row["equivalent_cpu_max" if model == "ed" else "equivalent_cpu_min"]
        ax.annotate(f'{row["equivalent_cpu_cores"]:.2f}', (x, y),
                    xytext=(0, 7 if model == "ed" else -17), textcoords="offset points", ha="center",
                    color=MODELS[model][1], fontsize=8, path_effects=[withStroke(linewidth=2.5, foreground="white")])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", required=True, type=Path, help="Main plot output with ed/ and mont0/ CSVs")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attachments", type=Path)
    args = parser.parse_args()
    reports = [load_report(args.comparison_root / model / "comparison.csv", model) for model in MODELS]
    keys = [[(row["exponent"], row["bits"]) for row in report["rows"]] for report in reports]
    if not keys[0] or keys[0] != keys[1]:
        raise ValueError("Both curves need the same matched input matrix")
    last_common = max(exponent for exponent, _ in keys[0])
    wide = sorted((row for report in reports for row in report["ntt_only"] if row["exponent"] > last_common),
                  key=lambda row: (row["model"], row["exponent"]))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.attachments:
        args.attachments.mkdir(parents=True, exist_ok=True)
    mixes = []
    for report in reports:
        rows = report["rows"]
        mix = sum(row["cpu_seconds_1e6"] for row in rows) / sum(row["gpu_seconds_1e6"] for row in rows)
        mixes.append(mix)
        print(f'{rows[0]["model"]}: {mix:.4f} equivalent cores, '
              f'{sum(row["backend"] == "NTT" for row in rows)}/{len(rows)} inputs use NTT')
    selected = [row for report in reports for row in report["rows"]] + wide
    with (args.output / "selected.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    for scale in ("linear", "log"):
        path = args.output / f"{PREFIX}-{scale}.png"
        plot(reports, mixes, wide, path, scale)
        if args.attachments:
            shutil.copyfile(path, args.attachments / path.name)


if __name__ == "__main__":
    main()
