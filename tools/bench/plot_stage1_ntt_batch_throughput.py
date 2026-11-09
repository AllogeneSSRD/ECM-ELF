"""Plot Stage1 NTT batch throughput normalized to one curve per shape."""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def read_confirmed(path):
    cells = defaultdict(list)
    conditions = set()
    for row in json.loads(path.read_text(encoding="utf-8")):
        if row["kind"] != "confirm":
            continue
        if int(row["bad"]) or int(row["curves"]) != int(row["requested_curves"]):
            raise ValueError("confirmation row failed arithmetic or changed the batch size")
        conditions.add(tuple(row[k] for k in ("device", "b1", "bits", "version", "window")))
        samples = row["event"]
        if not samples or any(not math.isfinite(t) or t <= 0 for t in samples):
            raise ValueError("confirmation row has no valid device timings")
        shape = (int(row["length"]), int(row["p"]))
        cells[shape, int(row["curves"])].append(statistics.median(samples))
    if len(conditions) != 1:
        raise ValueError("expected confirmed batches from one device, B1, version and window")
    rates = defaultdict(dict)
    for (shape, count), times_ms in cells.items():
        rates[shape][count] = 1000 * count / statistics.median(times_ms)
    if any(1 not in batch for batch in rates.values()):
        raise ValueError("each NTT shape needs a one-curve baseline")
    return conditions.pop(), rates


def plot(input_path, output_path, device_name):
    (device, b1, _, version, window), rates = read_confirmed(input_path)
    fig, ax = plt.subplots(figsize=(8.2, 4.9), layout="constrained")
    colors = ["#1c58a0", "#248a8a", "#31915b", "#ba801c", "#bb533f", "#744189"]
    markers = ["o", "s", "^", "D", "v", "P"]
    counts = sorted({b for batch in rates.values() for b in batch})
    for i, ((length, p), batch) in enumerate(sorted(rates.items())):
        x = sorted(batch)
        y = [batch[b] / batch[1] for b in x]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        label = f"L = {length:,}"
        if sum(l == length for l, _ in rates) > 1:
            label += f" (p = {p:,})"
        ax.plot(x, y, marker=marker, ms=5, lw=1.8, color=color, label=label)
    ax.set_xticks(counts)
    ax.set_xlim(counts[0] - 0.2, counts[-1] + 0.2)
    ax.set_xlabel("Curves per batch")
    ax.set_ylabel("Normalized throughput (one curve = 1)")
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(1, color="#666666", ls="--", lw=1, zorder=0)
    ax.text(0.98, 1.05, "Single-curve baseline = 1", ha="right", va="bottom",
            fontsize=9, color="#555555", transform=ax.get_yaxis_transform())
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center",
               ncols=min(3, len(rates)), fontsize=9, frameon=False)
    fig.suptitle("Stage1 NTT batch throughput\n"
                 f"{device_name} GPU{device} · PARAM0 · B1 = {b1} · Graph window {window} · v{version}",
                 fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Batch benchmark summary.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-name", default="RTX 4060 Laptop")
    args = parser.parse_args()
    plot(args.input, args.output, args.device_name)
