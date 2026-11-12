#!/usr/bin/env python3
"""Report fixed-B2 Stage2 geometry and optional fixed-shape two-slot replays."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def route(item):
    samples = item["samples"]
    if not samples or len({(s["leaf_hash"], s["ntt_counts"]) for s in samples}) != 1:
        raise ValueError("route lacks reproducible leaf and workload evidence")
    fields = ("D", "P", "carrier", "layout", "chain", "baby", "giant", "fold_length", "tree_length", "I", "G")
    seconds = [sample["engine_seconds"] for sample in samples]
    if min(seconds) <= 0:
        raise ValueError("invalid engine timing")
    return dict(**{key: item[key] for key in fields}, samples=len(seconds), median_engine_s=median(seconds),
                min_engine_s=min(seconds), max_engine_s=max(seconds),
                max_log2_ntt=max(map(int, re.findall(r"\b(\d+):\d+", samples[0]["ntt_counts"]))),
                peak_estimate_mib=item["peak_estimate_bytes"] / 1048576,
                peak_device_used_mib=max(s["used_mib"] for s in samples), source="+".join(item["sources"]))


def summarize(data):
    winners, candidates, parallel = [], [], []
    seen = set()
    for cell in sorted(data["results"], key=lambda row: (row["bound"] != "small", row["bits"])):
        selected, reference = (route(cell[key]) for key in ("selected", "baseline"))
        keys = {k: cell[k] for k in ("bound", "bits", "exponent", "B1", "B2")}
        if (keys["bound"], keys["bits"]) in seen or min(selected["samples"], reference["samples"]) < 2:
            raise ValueError(f"repeated width or insufficient replays: {keys}")
        seen.add((keys["bound"], keys["bits"]))
        gain = 100 * (1 - selected["median_engine_s"] / reference["median_engine_s"])
        winners.append(dict(**keys, saved_pct=gain, candidates=len(cell["candidates"]),
                            **{"selected_"+k: v for k,v in selected.items()},
                            **{"baseline_"+k: v for k,v in reference.items()}))
        candidates.extend(dict(**keys, **route(candidate)) for candidate in cell["candidates"])
    for pair in data.get("pairs", []):
        keys = ("bound", "bits", "status", "D", "P", "carrier", "layout", "serial_seconds", "parallel_seconds", "gain_pct")
        parallel.append({key: pair.get(key, "") for key in keys})
    return winners, candidates, parallel


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path, data, rows):
    title = f"{data['device']} · ECM-ELF {data['version']} · B1={data['B1']:,}"
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.0), layout="constrained")
        fig.suptitle(f"Stage2 D/P and G-tree layout  |  {title}", fontweight="bold")
        for i, bound in enumerate(("small", "large")):
            group = [row for row in rows if row["bound"] == bound]
            label = "Low B2" if bound == "small" else "High B2"
            for j, ax in enumerate(axes[i]):
                if not group:
                    ax.axis("off")
                    continue
                bits = [row["bits"] for row in group]
                for role, desc, color, style, marker in (
                        ("baseline", "Best native default", "#66717d", "--", "s"),
                        ("selected", "Best confirmed candidate", "#165e94", "-", "o")):
                    ax.plot(bits, [row[role + "_median_engine_s"] for row in group], label=desc, color=color,
                            linestyle=style, marker=marker, markersize=3.5, linewidth=1.5)
                    ax.fill_between(bits, [row[role + "_min_engine_s"] for row in group],
                                    [row[role + "_max_engine_s"] for row in group], color=color, alpha=.13)
                ax.set(title=f"{label} · {'logarithmic' if j else 'linear'}",
                       xlabel="Target cofactor width (bits)")
                if j:
                    ax.set(xscale="log", yscale="log")
                else:
                    ax.set(ylabel="Engine seconds per curve", ylim=(0, None))
                if i == 0:
                    ax.legend(loc="upper left", fontsize=9, frameon=True)
        fig.savefig(path, dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figures", type=Path, default=Path("data/figures/stage2_geometry_matrix"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows, candidates, parallel = summarize(data)
    if not rows:
        parser.error("the input has no completed width/B2 cases")
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "by-width.csv", rows)
    write_csv(args.output / "candidates.csv", candidates)
    write_csv(args.output / "parallel.csv", parallel)
    plot(args.figures / "stage2-geometry-by-width.png", data, rows)
    for bound in ("small", "large"):
        group = [r for r in rows if r["bound"] == bound]
        if not group:
            continue
        samples = sum(r["samples"] for r in candidates if r["bound"] == bound)
        routes = Counter((bool(row["selected_carrier"]), row["selected_layout"]) for row in group)
        old, new = (sum(row[k + "_median_engine_s"] for row in group) for k in ("baseline", "selected"))
        print(f"{bound}: {len(group)} widths, {samples} runs, carrier/layout={dict(routes)}, "
              f"{old:.3f} -> {new:.3f}s ({100*(1-new/old):.2f}% saved)")


if __name__ == "__main__":
    main()
