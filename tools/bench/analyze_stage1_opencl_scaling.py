"""Plot Stage1 batch saturation and compare confirmed OpenCL / CGBN plans."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


MODELS = {"e": "Edwards 2XY NAF", "m3": "Montgomery PARAM3"}
STYLES = {
    ("e", "opencl"): ("#d56819", "-", "o"),
    ("e", "cuda"): ("#704eb6", "-", "o"),
    ("m3", "opencl"): ("#d56819", "--", "s"),
    ("m3", "cuda"): ("#704eb6", "--", "s"),
}
BACKENDS = {"opencl": "OpenCL", "cuda": "CUDA CGBN"}


def plan_rates(condition, metric):
    samples = {}
    for row in condition["samples"]:
        if row["repeat"]:
            key = row["backend"], row["curves"], row["wg"]
            samples.setdefault(key, []).append(row["curves"] / row[metric])
    return {key: statistics.median(values) for key, values in samples.items()}


def rows_for_run(run):
    for condition in run["conditions"]:
        for metric in ("inner", "wall"):
            rates = plan_rates(condition, metric)
            for backend in BACKENDS:
                candidates = [(count, wg, rate) for (name, count, wg), rate in rates.items() if name == backend]
                peak_count, peak_wg, peak = max(candidates, key=lambda row: row[2])
                min_count, min_wg, _ = min(row for row in candidates if row[2] >= .95 * peak)
                rate768 = rates.get((backend, 768, 0))
                peak_samples = [r["curves"] / r[metric] for r in condition["samples"]
                                if r["repeat"] and (r["backend"], r["curves"], r["wg"]) ==
                                (backend, peak_count, peak_wg)]
                yield {
                    "gpu": run["device"]["name"], "model": condition["model"], "bits": condition["bits"],
                    "b1": condition["b1"], "backend": backend, "metric": metric,
                    "peak_curves": peak_count, "peak_wg": peak_wg, "peak_curves_s": peak,
                    "peak_sample_min_curves_s": min(peak_samples), "peak_sample_max_curves_s": max(peak_samples),
                    "minimum_95pct_curves": min_count, "minimum_95pct_wg": min_wg,
                    "relative_768": rate768 / peak if rate768 else "",
                }


def all_plans(run):
    for c in run["conditions"]:
        inner = plan_rates(c, "inner")
        wall = plan_rates(c, "wall")
        for (backend, curves, wg), inner_rate in sorted(inner.items()):
            yield {
                "phase": run["phase"], "gpu": run["device"]["name"], "model": c["model"], "bits": c["bits"],
                "b1": c["b1"], "backend": backend, "curves": curves, "wg": wg,
                "loop_curves_s": inner_rate, "wall_curves_s": wall[backend, curves, wg],
            }


def gpu_name(value):
    return value.replace("NVIDIA GeForce ", "").replace(" Laptop GPU", " Laptop")


def sweep_plot(runs, widths, out):
    fig, axes = plt.subplots(len(runs), len(widths), figsize=(4.3 * len(widths), 3.25 * len(runs)),
                             squeeze=False, sharex=True, sharey=True, layout="constrained")
    ticks = sorted({r["curves"] for run in runs for c in run["conditions"] for r in c["samples"]
                    if r["repeat"] and not r["wg"]})
    labels = [f"{n:,}" for n in ticks]
    for run, ax_row in zip(runs, axes):
        for bits, ax in zip(widths, ax_row):
            for c in run["conditions"]:
                if c["bits"] != bits:
                    continue
                for backend in BACKENDS:
                    rates = sorted((count, rate) for (name, count, wg), rate in plan_rates(c, "inner").items()
                                   if name == backend and wg == 0)
                    if not rates:
                        continue
                    counts, values = zip(*rates)
                    color, ls, mark = STYLES[c["model"], backend]
                    ax.plot(counts, [v / max(values) for v in values], color=color, ls=ls, marker=mark,
                            label=f"{MODELS[c['model']]} · {BACKENDS[backend]}", markersize=4)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks, labels, fontsize=8)
            ax.axhline(.95, color="#7b8491", lw=1, ls=":")
            ax.set_title(f"{gpu_name(run['device']['name'])} · {bits} bit")
            ax.grid(alpha=.18)
        ax_row[0].set_ylabel("Throughput / measured peak")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="outside lower center", frameon=False)
    axes[-1][len(widths) // 2].set_xlabel("Curves per batch (log₂)")
    fig.suptitle("Stage1 batch scaling · main-loop timing")
    fig.savefig(out / "stage1-opencl-batch-scaling.png", dpi=190)
    plt.close(fig)


def crossover_plot(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), layout="constrained")
    device_colors = ["#24834d", "#d56819"]
    for ax, (model, title) in zip(axes, MODELS.items()):
        metrics = [("wall", "o", "-", "full wall")]
        edge_widths = (383, 384, 385)
        edge_series = []
        if model == "e":
            metrics.append(("inner", "s", "--", "device loop"))
        for run, color in zip(runs, device_colors):
            grouped = list(rows_for_run(run))
            for metric, marker, ls, label in metrics:
                by_backend = {
                    backend: {r["bits"]: r["peak_curves_s"] for r in grouped
                              if r["backend"] == backend and r["metric"] == metric and r["model"] == model}
                    for backend in BACKENDS
                }
                bits = sorted(by_backend["opencl"].keys() & by_backend["cuda"].keys())
                if bits:
                    ratios = {w: by_backend["opencl"][w] / by_backend["cuda"][w] for w in bits}
                    ax.plot(bits, list(ratios.values()),
                            color=color, label=f"{gpu_name(run['device']['name'])} · {label}",
                            marker=marker, ls=ls)
                    if model == "e" and metric == "inner" and all(w in ratios for w in edge_widths):
                        edge_series.append((color, [ratios[w] for w in edge_widths]))
        ax.axhline(1, color="#555e6d", lw=1, ls=":")
        ax.set_xlabel("Modulus width (bit)")
        ax.set_title(title)
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
        ax.grid(alpha=.18)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8, frameon=False)
        if model == "m3":
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, .48), max(hi, 2.05))
        if edge_series:
            inset = ax.inset_axes([.27, .54, .30, .40])
            for color, ratios in edge_series:
                inset.plot(range(len(edge_widths)), ratios, color=color, marker="s", markersize=3)
            inset.set_xticks(range(len(edge_widths)), edge_widths)
            inset.axhline(1, color="#555e6d", lw=1, ls=":")
            inset.grid(alpha=.16)
            inset.set_title("Device loop near 384 bit", fontsize=7)
            inset.tick_params(labelsize=7)
    fig.supylabel("OpenCL throughput / CUDA CGBN throughput")
    fig.suptitle("Stage1 crossover · confirmed batch plans")
    fig.savefig(out / "stage1-opencl-crossover.png", dpi=190)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", nargs="+", required=True, type=Path)
    p.add_argument("--confirm", nargs="+", default=[], type=Path)
    p.add_argument("--plot-widths", nargs="+", type=int, default=[127, 421, 1021])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--csv-out", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_out = args.csv_out or args.out
    csv_out.mkdir(parents=True, exist_ok=True)
    sweep, confirm = ([json.loads(path.read_text(encoding="utf-8")) for path in paths]
                      for paths in (args.sweep, args.confirm))
    tables = {
        "sweep": [row for run in sweep for row in rows_for_run(run)],
        "confirm": [row for run in confirm for row in rows_for_run(run)],
        "plans": [row for run in sweep + confirm for row in all_plans(run)],
    }
    for name, rows in tables.items():
        if not rows:
            continue
        with (csv_out / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    sweep_plot(sweep, args.plot_widths, args.out)
    if confirm:
        crossover_plot(confirm, args.out)


if __name__ == "__main__":
    main()
