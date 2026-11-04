"""Plot fixed-D ordinary Stage2 samples from an audited production summary CSV."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8-sig", newline="") as file:
        samples = sorted(csv.DictReader(file), key=lambda r: (int(r["bits"]), int(r["B2"])))
    if not samples:
        parser.error("Empty production summary")

    fig, (timing, phases) = plt.subplots(1, 2, sharey=True, figsize=(12, 3.4 + 0.34 * len(samples)))
    labels = []
    fields = ("baby", "ftree_inverse", "giant", "gtree_fold", "leaf_oracle", "descent")
    split_tree = any(r.get("g_build") for r in samples)
    colors = {"Baby": "#5ca496", "F tree + inverse": "#83c5be", "Giant": "#e39742"}
    if split_tree:
        colors.update({"G + fold (unsplit)": "#9ca9b9", "G build": "#486c9d",
                       "H x G": "#649fcb", "Remainder": "#243d69"})
    else:
        colors["G trees + fold"] = "#486c9d"
    colors.update({"Leaf oracle": "#9b8ac4", "Descent": "#c97478", "Other": "#d3d7da"})
    for i, r in enumerate(samples):
        cu, cl = float(r["cuda"]), float(r["opencl"])
        values = [float(r[f]) for f in fields]
        if split_tree:
            nested = [float(r[f]) for f in ("g_build", "fold_product", "remainder")] if r.get("g_build") else []
            if sum(nested) > values[3] + 1e-5:
                raise ValueError("G build and fold intervals exceed their enclosing phase")
            # Packing and transfer enter Other when the three inner intervals exist.
            values[3:4] = [0., *nested] if nested else [values[3], 0., 0., 0.]
        if not (cu > 0 and cl > 0 and all(math.isfinite(v) and v >= 0 for v in values)
                and sum(values) <= cl + 0.00001):
            raise ValueError(f"Invalid component or engine time at {r['bits']} bits")
        policy = " (GMP)" if r["policy"] == "forced_cpu" else ""
        labels.append(f"{int(r['bits']):,} bits{policy}   |   {int(r['B2'])/1e12:g}T")
        timing.plot([cu, cl], [i, i], color="#a7b3bf", zorder=1)
        timing.scatter(cu, i, color="#205bac", label="CUDA" if i == 0 else None)
        timing.scatter(cl, i, color="#c2571e", marker="D", label="OpenCL" if i == 0 else None)
        timing.annotate(f"{cl/cu:.1f}×", (max(cu, cl), i), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9)
        left = 0.
        for (name, color), seconds in zip(colors.items(), [*values, max(0., cl-sum(values))]):
            percent = 100 * seconds / cl
            phases.barh(i, percent, height=.55, left=left, color=color, label=name if i == 0 else None)
            left += percent
    timing.set_xscale("log")
    timing.margins(x=.17)
    timing.set(xlabel="Full engine seconds (log scale)", title="Same save, D and B2",
               yticks=range(len(labels)), yticklabels=labels)
    timing.invert_yaxis()
    timing.grid(axis="x", alpha=.25)
    timing.legend(loc="upper left", bbox_to_anchor=(0, -.22), ncol=2, frameon=False)
    phases.set(xlim=(0, 100), xlabel="Share of full engine time (%)", title="OpenCL phases")
    phases.legend(loc="upper left", bbox_to_anchor=(0, -.22), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Production Stage2  |  RTX 4060 Laptop  |  Ordinary cofactor")
    fig.subplots_adjust(left=.22, right=.98, bottom=.32 if split_tree else .26, top=.84, wspace=.15)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
