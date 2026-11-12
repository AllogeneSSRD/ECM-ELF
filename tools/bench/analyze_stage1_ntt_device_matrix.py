#!/usr/bin/env python3
"""Summarize one GPU's Stage1 NTT batch search and actual curve costs."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def load_reports(pairs):
    reports, plans = {}, {}
    identity = None
    for mont, ed in pairs:
        for path, curve in ((mont, "mont"), (ed, "ed")):
            report = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            condition = tuple(report[key] for key in ("version", "device", "gpu_model", "reference_b1"))
            if identity is None:
                identity = condition
            if condition != identity or report["curve"] != curve:
                raise ValueError("matrix GPU, program, projection bound or curve differs")
            groups = {group["p"]: group for group in report["groups"]}
            if len(groups) != len(report["groups"]) or sorted(groups) != sorted(report["exponents"]):
                raise ValueError(f"incomplete width matrix: {path}")
            for p, group in groups.items():
                if (curve, p) in reports or len(group["samples"]) != report["rounds"]:
                    raise ValueError(f"repeated input or incomplete timing samples: {path} M{p}")
                reports[curve, p] = group
                tune = path / f"p{p}" / "tune" / "tune.toml"
                if tune.is_file():
                    columns, *rows = tomllib.loads(tune.read_text(encoding="utf-8"))["ntt"]["matrix"]
                    measured = [dict(zip(columns, row)) for row in rows]
                    if p in plans and plans[p] != measured:
                        raise ValueError(f"conflicting tune measurements for M{p}")
                    plans[p] = measured
    if not plans or set(reports) != {(curve, p) for p in plans for curve in ("mont", "ed")}:
        raise ValueError("each tuned exponent needs both production curve reports")
    return identity, reports, plans


def row_fields(prefix, row):
    keys = ("batch", "ntt_type", "length", "bpw", "tile", "max_radix_bits", "forward", "inverse", "graph_bits",
            "warp_tail", "iter_per_s", "owned_bytes", "peak_bytes", "budget_bytes")
    return {prefix + key: row[key] for key in keys}


def summaries(reports, plans):
    result, batch_table = [], []
    for p, tune in sorted(plans.items()):
        mont, ed = (reports[curve, p] for curve in ("mont", "ed"))
        best = {}
        for row in tune:
            n = row["batch"]
            if n not in best or row["iter_per_s"] > best[n]["iter_per_s"]:
                best[n] = row
        if not best or 1 not in best or any(n < 1 or n > 128 for n in best):
            raise ValueError(f"M{p} requires a legal single-curve measurement")
        if mont["batch_search"] == "exhaustive" and sorted(best) != list(range(1, max(best) + 1)):
            raise ValueError(f"M{p} exhaustive search has missing batch sizes")
        peak = max(best.values(), key=lambda row: row["iter_per_s"])
        if peak != mont["tune"] or peak != ed["tune"] or mont["b1"] != ed["b1"]:
            raise ValueError(f"M{p} actual and tuned plans differ")
        near = next(best[n] for n in sorted(best) if best[n]["iter_per_s"] >= .95 * peak["iter_per_s"])
        batch_table.extend(dict(p=p, fraction_of_peak=best[n]["iter_per_s"] / peak["iter_per_s"], **row_fields("", best[n]))
                           for n in sorted(best))
        row = dict(p=p, b1=mont["b1"], measured_batches=len(best), max_batch=max(best),
                   **row_fields("single_", best[1]), **row_fields("peak_", peak),
                   **row_fields("near_", near), peak_speedup=peak["iter_per_s"] / best[1]["iter_per_s"])
        for curve, group in (("mont", mont), ("ed", ed)):
            projected = group["projected_seconds_per_curve_1e6"]
            active = group.get("active_batch", group["batch"])
            row.update({curve + "_seconds_per_curve": group["measured_seconds_per_curve"],
                        curve + "_active_batch": active,
                        curve + "_s_per_curve_at_1e6": median(projected),
                        curve + "_min_at_1e6": min(projected), curve + "_max_at_1e6": max(projected),
                        curve + "_us_per_curve_step": median(sample["device_ms"] * 1000 / (sample["steps"] * active)
                                                               for sample in group["samples"])})
        result.append(row)
    return result, batch_table


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(output, identity, rows, batches):
    version, _, gpu, b1 = identity
    if b1 != 1000000:
        raise ValueError("plot expects projection to B1=1,000,000")
    title = f"{gpu.removeprefix('NVIDIA GeForce ')} · {version}"
    p = [row["p"] for row in rows]
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.65), layout="constrained")
        fig.suptitle(f"Stage1 NTT cost per curve  |  {title}", fontweight="bold")
        for index, ax in enumerate(axes):
            for kind, name, color, style, marker in (("ed", "Edwards", "#14804a", "-", "D"),
                                                     ("mont", "Montgomery PARAM0", "#2866a2", "--", "o")):
                ax.plot(p, [row[kind + "_s_per_curve_at_1e6"] for row in rows], label=name,
                        color=color, linestyle=style, marker=marker, markersize=3, linewidth=1.5)
                ax.fill_between(p, [row[kind + "_min_at_1e6"] for row in rows],
                                [row[kind + "_max_at_1e6"] for row in rows], color=color, alpha=.15)
            ax.set_xlabel("Mersenne exponent p (bits)")
            if index:
                ax.set(xscale="log", yscale="log", title="Logarithmic axes")
            else:
                ax.set(ylim=(0, None), title="Linear axes", ylabel="Seconds per curve at B1 = 1,000,000 (scaled)")
            ax.legend(loc="upper left", fontsize=9)
        fig.savefig(output / "stage1-ntt-cost-by-width.png", dpi=170)
        plt.close(fig)

        fig, (ax, fit) = plt.subplots(2, 1, height_ratios=(1.2, 4), sharex=True, figsize=(12.0, 6.5), layout="constrained")
        fig.suptitle(f"Stage1 NTT batch tuning  |  {title}", fontweight="bold")
        order = range(len(rows))
        ax.plot(order, [row["peak_speedup"] for row in rows], color="#14804a", marker="o", markersize=3)
        ax.set_ylabel("Peak rate /\n1-curve rate")
        rates = {(row["p"], row["batch"]): row["fraction_of_peak"] for row in batches}
        cap = max(row["batch"] for row in batches)
        values = [[rates.get((exponent, batch), float("nan")) for exponent in p] for batch in range(1, cap + 1)]
        image = fit.imshow(values, origin="lower", aspect="auto", interpolation="nearest", cmap="viridis",
                           norm=Normalize(0, 1), extent=(-.5, len(rows)-.5, .5, cap+.5))
        fit.scatter(order, [row["peak_batch"] for row in rows], s=22, marker="x", color="#000000", label="Measured peak")
        fit.scatter(order, [row["near_batch"] for row in rows], s=21, marker="o", facecolors="none",
                    edgecolors="#d11938", linewidths=1.15, label="Smallest batch at ≥95% of peak")
        fit.set(xlabel="Mersenne exponent p (ordered)", ylabel="Concurrent curves",
                xticks=list(order), xticklabels=p, yticks=sorted({1, cap, *range(max(4, cap//8), cap, max(4, cap//8))}))
        fit.tick_params(axis="x", labelrotation=65, labelsize=8)
        fit.grid(False)
        handles, labels = fit.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right", fontsize=8.5)
        fig.colorbar(image, ax=fit, label="Fraction of peak curves × steps/s")
        fig.savefig(output / "stage1-ntt-batch-by-width.png", dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, nargs=2, metavar=("MONT", "ED"), required=True, action="append",
                        help="Matching bench_stage1_ntt_best.py output directories. May be repeated.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=Path("data/figures/stage1_ntt_device_matrix"))
    args = parser.parse_args()
    identity, reports, plans = load_reports(args.pair)
    rows, batches = summaries(reports, plans)
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "by-width.csv", rows)
    write_csv(args.output / "by-batch.csv", batches)
    plots(args.figures, identity, rows, batches)


if __name__ == "__main__":
    main()
