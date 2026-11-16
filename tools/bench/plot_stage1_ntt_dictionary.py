"""Plot matched host/GPU Edwards NTT dictionary runs."""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def metric(row, measure):
    samples = row["samples"]
    if measure == "dictionary":
        return median((s["host_dictionary_ms"] + s["gpu_dictionary_ms"]) / 1000 for s in samples)
    return median(row["curves"] / s["process_s"] for s in samples)


def matched_points(rows, curves, window, builder):
    by_width = {}
    for row in rows:
        if row["curves"] != curves or row["window"] != window:
            continue
        if not all(sample["backend"] == builder for sample in row["samples"]):
            continue
        width = row["p"]
        if width not in by_width or row.get("requested") == builder:
            by_width[width] = row
    return [by_width[width] for width in sorted(by_width)]


def plot(devices, output, measure, log):
    batches = sorted({row["curves"] for _, rows in devices for row in rows})
    fig, axes = plt.subplots(len(batches), len(devices), figsize=(11, 8.1),
                             sharex=True, sharey="row", squeeze=False)
    for col, (device, rows) in enumerate(devices):
        widths = sorted({row["p"] for row in rows})
        colors = ("#007d68", "#2855a0") if col == 0 else ("#c55413", "#963e83")
        for index, curves in enumerate(batches):
            ax = axes[index][col]
            if measure == "throughput":
                baseline = sorted((r for r in rows if r["curves"] == curves and r["window"] == 0),
                                  key=lambda r: r["p"])
                ax.plot([r["p"] for r in baseline], [metric(r, measure) for r in baseline],
                        label="x-only", color="#353535", marker=".", linewidth=1.4)
            for window, color, marker in zip((6, 8), colors, ("o", "s")):
                for builder, linestyle in (("gpu", "-"), ("host", "--")):
                    points = matched_points(rows, curves, window, builder)
                    ax.plot([r["p"] for r in points], [metric(r, measure) for r in points],
                            label=f"w{window} {builder.upper()}", color=color, linestyle=linestyle,
                            marker=marker, markerfacecolor=color if builder == "gpu" else "white",
                            markersize=4.7, linewidth=1.4)
            ax.grid(visible=True, alpha=0.24)
            ax.set_xscale("log", base=2)
            ax.set_xticks(widths, [f"{p:,}" for p in widths])
            if log:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            if index == 0:
                ax.set_title(device)
            label_x, label_side = (0.025, "left") if measure == "dictionary" else (0.975, "right")
            ax.text(label_x, 0.94, f"{curves} curve{'s' if curves != 1 else ''}",
                    transform=ax.transAxes, ha=label_side, va="top", fontsize=9)
            if index == len(batches) - 1:
                ax.set_xlabel("Mersenne exponent (bits)")
    if not log:
        for index, curves in enumerate(batches):
            peak = max(metric(r, measure) for _, rows in devices for r in rows if r["curves"] == curves)
            axes[index][0].set_ylim(0, 1.09 * peak)
    title = "NTT odd-multiple dictionary setup" if measure == "dictionary" else "Cold-process Stage1 throughput"
    fig.suptitle(f"Edwards 2XY NAF • {title}", y=0.982, fontweight="semibold")
    fig.supylabel("Batch seconds" if measure == "dictionary" else "Curves per second", x=0.017)
    for col in range(len(devices)):
        handles, labels = axes[0][col].get_legend_handles_labels()
        axes[-1][col].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.34),
                            ncol=len(labels), fontsize=8, frameon=False, handlelength=1.9)
    fig.subplots_adjust(left=0.096, right=0.98, top=0.9, bottom=0.15, hspace=0.25, wspace=0.14)
    fig.savefig(output, dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    devices = [(name, json.loads((args.input / f"gpu{index}" / "summary.json").read_text()))
               for index, name in enumerate(("RTX 4070 Ti", "RTX 4060 Laptop"))]
    args.output.mkdir(parents=True, exist_ok=True)
    for metric_name, log in (("dictionary", True), ("throughput", False), ("throughput", True)):
        filename = f"stage1-edwards-ntt-{metric_name}-{'log' if log else 'linear'}.png"
        plot(devices, args.output / filename, metric_name, log)


if __name__ == "__main__":
    main()
