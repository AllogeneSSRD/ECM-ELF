"""Plot paired Stage1 elapsed-time savings from the 2XY doubling formula."""

import argparse
from collections import defaultdict
import csv
import json
from math import isfinite
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, PercentFormatter


DEVICES = {
    0: ("RTX 4070 Ti", "#248255", "o"),
    1: ("RTX 4060 Laptop", "#be601f", "s"),
}
METRICS = {
    "loop_gpu_s": "GPU NAF loop",
    "wall_s": "Process wall time",
}
CONDITIONS = ("b1", "curves", "container_bits", "tpi", "columns", "additions")


def summarize(paths):
    data = {}
    builds = set()
    input_matrix = None
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        device = report["device"]
        if not report["complete"] or device in data:
            raise ValueError(f"incomplete or duplicate device report: {path}")
        builds.add((report["version"], report["window"], report["sigma"]))
        cells = defaultdict(lambda: defaultdict(dict))
        for sample in report["samples"]:
            if sample["device"] != device or sample["version"] != report["version"]:
                raise ValueError(f"sample device or version differs from report: {sample['log']}")
            if any(not isfinite(sample[metric]) or sample[metric] <= 0 for metric in METRICS):
                raise ValueError(f"invalid timing: {sample['log']}")
            variants = cells[sample["bits"]][sample["repeat"]]
            algorithm = sample["algorithm"]
            if algorithm in variants:
                raise ValueError(f"duplicate sample: {sample['log']}")
            variants[algorithm] = sample
        rows = []
        for bits, repeats in sorted(cells.items()):
            conditions = {tuple(s[k] for k in CONDITIONS) for pair in repeats.values()
                          for s in pair.values()}
            if (len(conditions) != 1 or len(repeats) < 3
                    or any(pair.keys() != {"hwcd", "2xy"} for pair in repeats.values())):
                raise ValueError(f"inconsistent or incomplete tests at M{bits}: {path}")
            row = dict(device=device, bits=bits, pairs=len(repeats), sigma=report["sigma"],
                       report_dir=path.parent.name,
                       **dict(zip(CONDITIONS, conditions.pop())))
            for metric in METRICS:
                samples = [(pair["hwcd"][metric], pair["2xy"][metric])
                           for pair in repeats.values()]
                gains = [100 * (1 - alternative / baseline) for baseline, alternative in samples]
                row.update({f"{metric}_hwcd": median(t[0] for t in samples),
                            f"{metric}_2xy": median(t[1] for t in samples),
                            f"{metric}_gain": median(gains),
                            f"{metric}_gain_min": min(gains),
                            f"{metric}_gain_max": max(gains)})
            rows.append(row)
        matrix = [(row["bits"], row["b1"], row["container_bits"], row["tpi"]) for row in rows]
        if not matrix or (input_matrix is not None and matrix != input_matrix):
            raise ValueError(f"device input matrices differ: {path}")
        input_matrix = matrix
        data[device] = rows
    if len(builds) != 1:
        raise ValueError("version, window or sigma differs between reports")
    version, window, _ = builds.pop()
    return data, version, window


def plot(path, data, version, window):
    fig, axs = plt.subplots(1, len(METRICS), figsize=(12, 4.6), layout="constrained", sharey=True)
    fig.suptitle(f"Edwards Stage 1 · 2XY vs HWCD | ECM-ELF {version}, NAF w{window}",
                 fontsize=15, fontweight="bold")
    widths = [row["bits"] for rows in data.values() for row in rows]
    for ax, (metric, title) in zip(axs, METRICS.items()):
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_xlim(min(widths) / 1.16, max(widths) * 1.16)
        ticks = (256, 512, 1024, 2048, 4096, 8192, 16384)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{n/1024:g}K" if n >= 1024 else str(n)
                                                    for n in ticks]))
        ax.axhline(0, color="#555b61", linewidth=0.9)
        ax.grid(axis="y", color="#e5e8ea")
        ax.set_axisbelow(True)
        ax.set_xlabel("Modulus width (bits)")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        for device, (label, color, marker) in DEVICES.items():
            if device not in data:
                continue
            rows = data[device]
            gains = [row[f"{metric}_gain"] for row in rows]
            errors = [[abs(row[f"{metric}_gain_{bound}"] - gain) for row, gain in zip(rows, gains)]
                      for bound in ("min", "max")]
            ax.errorbar([row["bits"] for row in rows], gains, yerr=errors,
                        color=color, label=label, marker=marker, linewidth=1.7,
                        linestyle="-" if device == 0 else "--",
                        markerfacecolor=color if device == 0 else "white",
                        elinewidth=1, capsize=3, markersize=5)
    axs[0].set_ylabel("Elapsed time saved by 2XY")
    axs[-1].legend(loc="best")
    fig.supxlabel("Points: median of paired runs · bars: measured min–max", fontsize=9)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def replace_cell(data, replacement, version, window, expected_version, expected_window):
    if (version, window) != (expected_version, expected_window):
        raise ValueError("replacement report uses a different version or NAF window")
    for device, rows in replacement.items():
        original = {row["bits"]: row for row in data[device]}
        for row in rows:
            previous = original[row["bits"]]
            if any(row[key] != previous[key] for key in (*CONDITIONS, "sigma")):
                raise ValueError(f"replacement changed conditions at M{row['bits']}")
            original[row["bits"]] = row
        data[device] = [original[bits] for bits in sorted(original)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--replace", type=Path, action="append", default=[],
                        help="Replace complete cells with a verified rerun report.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    data, version, window = summarize(args.input)
    for path in args.replace:
        replacement, rerun_version, rerun_window = summarize([path])
        replace_cell(data, replacement, rerun_version, rerun_window, version, window)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for device in DEVICES if device in data for row in data[device]]
    with args.summary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(args.output_dir / "stage1-edwards-hwcd-2xy.png", data, version, window)


if __name__ == "__main__":
    main()
