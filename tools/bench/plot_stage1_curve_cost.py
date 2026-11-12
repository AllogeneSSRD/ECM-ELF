"""Compare Edwards and Montgomery PARAM0 on matching Stage1 inputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from analyze_stage1_gpu import ROOT, exact_modulus, parse_batches, parse_mont_batches


GPU_COLORS = {
    "CGBN": {"RTX 4070 Ti": "#2866a2", "RTX 4060 Laptop": "#d4771d"},
    "NTT": {"RTX 4070 Ti": "#14804a", "RTX 4060 Laptop": "#b73339"},
}
NTT_PLAN_FIELDS = ("ntt_type", "length", "bpw", "tile", "max_radix_bits",
                   "forward", "inverse", "graph_bits", "warp_tail", "batch")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def percent(edwards, montgomery):
    return 100 * (edwards / montgomery - 1)


def cgbn_reports(directory, ed_cpu, mont_cpu):
    plan = read_json(directory / "prepared.json")
    log = (directory / "screen_stage1_gpu.log").read_text(encoding="utf-8-sig")
    sms, device = plan["sm_count"], plan["device"]
    reports = (parse_batches(log, ed_cpu, sms), parse_mont_batches(log, mont_cpu, sms, loop_timing=True))
    expected = {(row["model"], row["exponent"]): row for row in plan["cases"]}
    parsed = {(row["model"], row["exponent"]): row for report in reports for row in report["rows"]}
    if expected.keys() != parsed.keys() or any(report["queue_interrupted"] for report in reports):
        raise ValueError(f"{directory}: incomplete CGBN matrix")
    geometry = dict(cofactor_bits="bits", container_bits="container_bits", tpi="tpi", tpb="tpb",
                    launched_blocks="blocks", curves_per_batch="curves", b1="b1")
    for key, row in parsed.items():
        if row["device"] != device or any(row[v] != expected[key][k] for k, v in geometry.items()):
            raise ValueError(f"{directory}: measured CGBN geometry differs from plan for {key}")
    for report, model in zip(reports, ("ed", "mont0"), strict=True):
        if f'ECM-ELF {report["program_version"]}' != plan["program_version"]:
            raise ValueError(f"{directory}: CGBN version differs from plan")
        report.update(gpu_model=plan["gpu_model"], model=model, target_b1=1_000_000,
                      gpu_timing="sum of main-loop CUDA event intervals", source=str(directory))
    if any(row["native_gpu_model"] != plan["gpu_model"] for row in reports[1]["rows"]):
        raise ValueError(f"{directory}: unexpected Montgomery GPU")
    return reports


def cgbn_pair(ed, mont, ed_cpu, mont_cpu):
    ed_rows, mont_rows = ({r["exponent"]: r for r in report["rows"]} for report in (ed, mont))
    if ed_rows.keys() != mont_rows.keys():
        raise ValueError("CGBN curve matrices cover different inputs")
    rows = []
    for p, e in ed_rows.items():
        m = mont_rows[p]
        n_ed, n_mont = (exact_modulus(p, cpu[p]["intended_known_factors"]) for cpu in (ed_cpu, mont_cpu))
        if n_ed != n_mont or any(e[k] != m[k] for k in ("bits", "b1", "curves", "tpi", "tpb", "blocks")):
            raise ValueError(f"M{p}: incompatible CGBN curve pair")
        te, tm = (row["gpu_loop_batch_seconds"] for row in (e, m))
        rows.append(dict(backend="CGBN", gpu=ed["gpu_model"], device=e["device"], version=e["program_version"],
                         p=p, bits=e["bits"], b1=e["b1"], batch=e["curves"], samples=1,
                         ed_seconds=te, mont_seconds=tm, delta_pct=percent(te, tm),
                         tpi=e["tpi"], ed_container_bits=e["container_bits"], mont_container_bits=m["container_bits"]))
    return rows


def ntt_pair(mont, ed):
    if (mont["curve"], ed["curve"]) != ("mont", "ed") or any(
        mont[k] != ed[k] for k in ("gpu_model", "device", "version", "exponents", "rounds")
    ):
        raise ValueError("Expected matching NTT curve reports on one GPU")
    groups = [{g["p"]: g for g in report["groups"]} for report in (ed, mont)]
    if any(len(group) != len(report["groups"]) or group.keys() != set(report["exponents"])
           for group, report in zip(groups, (ed, mont), strict=True)):
        raise ValueError("Incomplete or repeated NTT exponents")
    rows = []
    for p, e in groups[0].items():
        m = groups[1][p]
        if (any(e[k] != m[k] for k in ("b1", "sigma", "batch"))
                or any(e["tune"][k] != m["tune"][k] for k in NTT_PLAN_FIELDS)):
            raise ValueError(f"M{p}: incompatible NTT curve pair")
        times, steps = [], []
        dataflow = {s["dataflow"] for g in (e, m) for s in g["samples"]}
        if len(dataflow) != 1:
            raise ValueError(f"M{p}: NTT curves used different device dataflow")
        for g in (e, m):
            samples = [s["device_ms"] / 1000 for s in g["samples"]]
            counts = {s["steps"] for s in g["samples"]}
            if len(samples) != ed["rounds"] or any(not math.isfinite(t) or t <= 0 for t in samples) or len(counts) != 1:
                raise ValueError(f"M{p}: invalid NTT measurements")
            count = counts.pop()
            if not math.isfinite(count) or count <= 0 or count != int(count):
                raise ValueError(f"M{p}: invalid NTT step count")
            times.append(samples)
            steps.append(int(count))
        et, mt = times
        te, tm = median(et), median(mt)
        rows.append(dict(backend="NTT", gpu=ed["gpu_model"], device=ed["device"],
                         version=ed["version"].removeprefix("ECM-ELF "), p=p, bits=p, b1=e["b1"], batch=e["batch"],
                         samples=ed["rounds"], ed_seconds=te, mont_seconds=tm, delta_pct=percent(te, tm),
                         delta_min_pct=percent(min(et), max(mt)), delta_max_pct=percent(max(et), min(mt)),
                         ed_steps=steps[0], mont_steps=steps[1], delta_per_step_pct=percent(te / steps[0], tm / steps[1]),
                         ntt_length=e["tune"]["length"], convolution=e["tune"]["ntt_type"]))
    return rows


def plot(rows, output, backend):
    series = defaultdict(list)
    for row in rows:
        series[row["device"], row["gpu"], row["version"]].append(row)
    if len(series) != 2 or len({key[0] for key in series}) != 2:
        raise ValueError(f"Expected two GPUs with one version each for {backend}")
    is_ntt = backend == "NTT"
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(10.5, 5), layout="constrained")
        fig.suptitle(f"Stage1 {backend} · Edwards vs Montgomery PARAM0", fontweight="bold")
        ax.axhline(0, color="#6b7280", linewidth=1, linestyle="--", zorder=1)
        for i, ((_, gpu, version), group) in enumerate(sorted(series.items())):
            name = gpu.removeprefix("NVIDIA GeForce ").removesuffix(" GPU")
            color = GPU_COLORS[backend].get(name, tuple(GPU_COLORS[backend].values())[i])
            group.sort(key=lambda r: r["bits"])
            x = [r["bits"] for r in group]
            delta = [r["delta_pct"] for r in group]
            marker = ("D", "o")[i]
            ax.plot(x, delta, color=color, marker=marker, markersize=4, linewidth=1.7, label=f"{name} · {version}")
            if is_ntt:
                ax.fill_between(x, [r["delta_min_pct"] for r in group], [r["delta_max_pct"] for r in group],
                                color=color, alpha=.13, linewidth=0)
            else:
                distinct = [r for r in group if r["ed_container_bits"] != r["mont_container_bits"]]
                ax.scatter([r["bits"] for r in distinct], [r["delta_pct"] for r in distinct],
                           color="white", edgecolor=color, marker=marker, s=26, linewidth=1.25, zorder=4)
        if is_ntt:
            ax.set_xscale("log")
            ax.set_xlabel("Full Mersenne modulus width (bits)")
            ax.set_title("Matched NTT plans · three runs per curve family · band shows observed ratio range", fontsize=10)
        else:
            ax.set_xlabel("Mersenne cofactor width (bits)")
            ax.set_title("Matched inputs and launch geometry · two blocks per SM · one batch per curve family", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        if not is_ntt and any(r["ed_container_bits"] != r["mont_container_bits"] for r in rows):
            handles.append(Line2D([], [], color="#6b7280", marker="o", markerfacecolor="white", linestyle="none"))
            labels.append("Different container widths")
        ax.legend(handles, labels, loc="best", frameon=True, fontsize=9)
        ax.set_ylabel("Edwards time / Montgomery time − 1\n(negative means Edwards is faster)")
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        fig.savefig(output, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cgbn", type=Path, nargs=2, required=True, help="Two completed CGBN run directories")
    parser.add_argument("--ntt-pair", type=Path, nargs=2, action="append", required=True, metavar=("MONT", "ED"))
    parser.add_argument("--cpu-ed", type=Path, default=ROOT / "data/benchmarks/stage1_prime95_20261024/summary.json")
    parser.add_argument("--cpu-mont", type=Path, default=ROOT / "data/benchmarks/stage1_prime95_mont_20261024/summary.json")
    parser.add_argument("--output", type=Path, required=True, help="Directory for CSV and CGBN summaries")
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    ed_cpu, mont_cpu = ({r["exponent"]: r for r in read_json(path)["rows"]} for path in (args.cpu_ed, args.cpu_mont))
    cgbn, reports = [], {}
    for directory in args.cgbn:
        ed, mont = cgbn_reports(directory, ed_cpu, mont_cpu)
        cgbn.extend(cgbn_pair(ed, mont, ed_cpu, mont_cpu))
        for report in (ed, mont):
            device = report["rows"][0]["device"]
            reports[f'gpu{device}_{report["model"]}_summary.json'] = report
    ntt = [row for pair in args.ntt_pair for row in ntt_pair(*(read_json(path) for path in pair))]
    for rows in (cgbn, ntt):
        if len({(r["device"], r["p"]) for r in rows}) != len(rows):
            raise ValueError("Repeated GPU and exponent in comparison")
    if {(r["device"], r["gpu"]) for r in cgbn} != {(r["device"], r["gpu"]) for r in ntt}:
        raise ValueError("CGBN and NTT must cover the same GPUs")
    conditions = defaultdict(set)
    for row in cgbn:
        conditions[row["device"]].add(tuple(row[k] for k in (
            "p", "bits", "b1", "tpi", "ed_container_bits", "mont_container_bits")))
    if len(conditions) != 2 or len({frozenset(cases) for cases in conditions.values()}) != 1:
        raise ValueError("CGBN GPUs must cover the same inputs and arithmetic geometry")
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        (args.output / name).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for rows, backend in ((cgbn, "CGBN"), (ntt, "NTT")):
        with (args.output / f"{backend.lower()}_curve_delta.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["device"], r["bits"])))
        plot(rows, args.figures / f"stage1-{backend.lower()}-ed-mont-delta.png", backend)
    print(f"CGBN: {len(cgbn)} pairs; NTT: {len(ntt)} pairs")


if __name__ == "__main__":
    main()
