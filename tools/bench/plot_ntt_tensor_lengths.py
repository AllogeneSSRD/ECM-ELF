"""Plot exact NTT length scans; retain timing boundaries and individual samples."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"CUDA production": "#28677b", "Tensor64": "#b06c36", "Tensor40": "#7457a4"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="Directories containing measurements.json")
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--analysis-output", type=Path, help="Default: repository data/benchmarks/<prefix>_analysis.json")
    p.add_argument("--clock-note", default="Clock/power settings unchanged; actual clocks are recorded in telemetry")
    a = p.parse_args()
    records, sources, telemetry = [], [], []
    for directory in a.inputs:
        doc = json.loads((directory / "measurements.json").read_text(encoding="utf-8"))
        sources.append({"path": str(directory.resolve()), "sha256": doc["sha256"], "arguments": doc["arguments"], "gate": doc["gate"]})
        for run in doc["runs"]:
            for row in run["summary"]:
                records.append(dict(row, source=str(directory.resolve())))
        trace = directory / "telemetry.csv"
        if trace.exists():
            with trace.open(encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    try:
                        normalized = {k.strip(): v.strip() for k, v in row.items() if k and v}
                        def number(prefix):
                            value = next(v for k, v in normalized.items() if k.startswith(prefix))
                            return float(value.split()[0])
                        telemetry.append({"sm_mhz": number("clocks.current.sm"), "watts": number("power.draw"), "temperature": number("temperature.gpu"), "utilization": number("utilization.gpu"), "memory_mb": number("memory.used")})
                    except (ValueError, StopIteration):
                        pass
    first = json.loads((a.inputs[0] / "measurements.json").read_text(encoding="utf-8"))
    with (a.inputs[0] / "devices_before.csv").open(encoding="utf-8-sig") as handle:
        devices = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(handle)]
    device_name = next(row["name"] for row in devices if int(row["index"]) == first["device"])
    unique = {}
    for r in records:
        if r["batch"] == 1 and r["tensor_threads"] == 256 and r["tile_bits"] == min(12, r["log2_length"]):
            unique.setdefault((r["execution"], r["operation"], r["log2_length"], r["mode"]), r)
    main_rows = list(unique.values())
    native = [r for r in main_rows if r["execution"] == "native" and r["operation"] == "convolution"]
    graph = [r for r in main_rows if r["execution"] == "graph" and r["operation"] == "convolution"]
    a.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2, "figure.dpi": 130, "savefig.dpi": 170})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for ax, rows, title, speedup in [(axes[0, 0], native, "Full convolution latency: native submission", False),
                                     (axes[0, 1], native, "Tensor throughput / production CUDA: native", True),
                                     (axes[1, 0], graph, "Full convolution latency: CUDA Graph diagnostic", False),
                                     (axes[1, 1], graph, "Tensor throughput / production CUDA: Graph", True)]:
        for mode, color in COLORS.items():
            if speedup and mode == "CUDA production":
                continue
            group = sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["log2_length"])
            if not group:
                continue
            values = [r["speedup_vs_cuda"] if speedup else r["median_seconds"] * 1000 for r in group]
            ax.plot([r["log2_length"] for r in group], values, marker="o", markersize=3.5, color=color, label=mode)
            if not speedup:
                ax.errorbar([r["log2_length"] for r in group], values, yerr=[r["mad_seconds"] * 1000 for r in group], fmt="none", ecolor=color, capsize=2)
        ax.set_title(title)
        ax.set_xlabel("NTT length L = 2^k (k)")
        ax.set_xticks(sorted({r["log2_length"] for r in rows}))
        ax.set_ylabel("Speed ratio (CUDA = 1)" if speedup else "Median time per convolution (ms, log scale)")
        if speedup:
            ax.axhline(1, color="0.35", linestyle="--", linewidth=1)
        else:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle(f"Exact Goldilocks NTT: production CUDA vs Tensor64 vs Tensor40\n{device_name} / batch 1 / Tensor CTA 256\n{a.clock_note}", fontsize=12)
    repeats_note = ",".join(str(n) for n in sorted({r["repeats"] for r in native + graph}))
    fig.supxlabel(f"{first['date']} | three transforms + pointwise/scale; pack/carry/S4 excluded; median/MAD, n={repeats_note}; single-operation Graph; every final A/B word checked", fontsize=8)
    path = a.output_prefix.with_suffix(".png")
    fig.savefig(path)
    plt.close(fig)

    batch_rows = [r for r in records if r["batch"] > 1 and r["operation"] == "convolution" and r["execution"] == "native"]
    if batch_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
        for mode, color in COLORS.items():
            for batch in sorted({r["batch"] for r in batch_rows}):
                group = sorted((r for r in batch_rows if r["mode"] == mode and r["batch"] == batch), key=lambda r: r["log2_length"])
                axes[0].plot([r["log2_length"] for r in group], [r["operations_per_s"] for r in group], color=color, marker="o", linestyle="-" if batch == 8 else "--", label=f"{mode}, batch {batch}")
                if mode != "CUDA production":
                    axes[1].plot([r["log2_length"] for r in group], [r["speedup_vs_cuda"] for r in group], color=color, marker="o", linestyle="-" if batch == 8 else "--", label=f"{mode}, batch {batch}")
        axes[0].set_ylabel("Independent convolutions per second (log scale)")
        axes[0].set_yscale("log")
        axes[0].set_title("Short NTT batched throughput")
        axes[1].set_ylabel("Speed ratio vs CUDA at the same batch")
        axes[1].axhline(1, color="0.35", linestyle="--")
        axes[1].set_title("Tensor speed ratio at fixed batch")
        for ax in axes:
            ax.set_xlabel("NTT length L = 2^k (k)")
            ax.legend(fontsize=7)
        fig.suptitle(f"{device_name} / native / Tensor CTA 256\n{a.clock_note}")
        fig.supxlabel("The isolated k=8, batch=8 positive point did not reproduce in the chained-Graph diagnostic; see the report.", fontsize=8)
        fig.savefig(a.output_prefix.parent / (a.output_prefix.name + "_batch.png"))
        plt.close(fig)

    ratios = []
    for row in graph:
        if row["mode"] != "CUDA production":
            continue
        original = next((n for n in native if n["log2_length"] == row["log2_length"] and n["mode"] == row["mode"]), None)
        if original:
            ratios.append({"log2_length": row["log2_length"], "native_seconds": original["median_seconds"], "graph_seconds": row["median_seconds"], "gain_percent": 100 * (1 - row["median_seconds"] / original["median_seconds"])})
    telemetry_summary = {}
    for subset_name, subset in [("all_samples", telemetry), ("gpu_utilization_ge_50", [t for t in telemetry if t["utilization"] >= 50])]:
        telemetry_summary[subset_name] = {"count": len(subset)}
        if subset:
            for metric in subset[0]:
                values = [t[metric] for t in subset]
                telemetry_summary[subset_name][metric] = {"min": min(values), "median": statistics.median(values), "max": max(values)}
    report = {"sources": sources, "records": records, "native_vs_graph_cuda": ratios, "telemetry": telemetry_summary,
              "note": "NTT-only experiment, not complete Stage2. Graph chains multiple operations; its benefit is not a production Stage2 claim."}
    analysis_path = a.analysis_output or Path(__file__).resolve().parents[2] / "data" / "benchmarks" / (a.output_prefix.name + "_analysis.json")
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps({"rows": len(records), "checked_words": sum(r["checked_words"] for r in records), "native_vs_graph_cuda": ratios, "telemetry": telemetry_summary}, indent=2))


if __name__ == "__main__":
    main()
