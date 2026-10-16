"""Visualize the audited Edwards Stage1 study without running GPU work.

Input: analyze_edwards_gpu_stage1.py's summary.json. Write PNG/SVG figures and
a provenance manifest. Preserve all three formal samples; whiskers are observed
min/max, not confidence intervals. Normalization always stays within one suite,
width and timer. Initial baseline builds never enter NAF speedup calculations.
Requires Python 3.10+ and matplotlib.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
STUDY = "edwards_naf_20261016"
WIDTHS = [347, 677, 991, 2203, 4003, 8191]
SMALL_WIDTHS = [347, 2203, 8191]
WINDOWS = [6, 8, 10, 12]
COLORS = ["#727a84", "#28677b", "#b06c36"]
METRICS = ("process_seconds", "arithmetic_api_seconds", "cuda_event_span_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(report):
    """Check plot scope and cached statistics; raw evidence audit remains upstream."""
    require(report.get("complete") is True, "Need a complete audited summary")
    require(report["device"] == 1 and report["uuid"] ==
            "GPU-8a67b1f8-ef1c-3177-a822-813a7ac2224d", "Different study device")
    require(report["b1"] == 10000, "Different B1; captions require the qualified study")
    suites = {s["evidence"]: s for s in report["suites"]}
    expected = {
        "bench_initial_01": (WIDTHS[:-1], ["ed-binary", "mont0", "mont3"], 768),
        "bench_initial_wide_01": ([8191], ["ed-binary", "mont0", "mont3"], 768),
        "bench_paired_01": (WIDTHS, ["ed-binary", "ed"], 768),
        "bench_doubling_01": (SMALL_WIDTHS, ["ed", "ed-5"], 128),
        **{f"bench_window{w}_01": (SMALL_WIDTHS, ["ed"], 128) for w in WINDOWS},
    }
    require(len(suites) == len(report["suites"]) and set(suites) == set(expected),
            "Missing, duplicate or unexpected suite")
    formal = 0
    for name, (bits, models, curves) in expected.items():
        suite = suites[name]
        require(sorted(suite["qualified_bits"]) == bits, f"Different qualified scope: {name}")
        require(suite["collector_complete"] or name == "bench_initial_01",
                f"Incomplete collector outside qualified initial subset: {name}")
        rows = suite["rows"]
        keys = [(r["bits"], r["model"]) for r in rows]
        require(len(keys) == len(set(keys)) and set(keys) ==
                {(b, m) for b in bits for m in models}, f"Different matrix: {name}")
        for row in rows:
            require(row["evidence"] == name and row["b1"] == report["b1"] and
                    row["curves"] == curves and row["sigma"] == 20260922 and
                    row["binary_sha256"] == suite["binary_sha256"], f"Scope mismatch: {name}")
            require(len(row["binary_sha256"]) == 64, "Missing executable SHA256")
            for metric in METRICS:
                if metric not in row:
                    require(metric == "cuda_event_span_seconds" and row["model"] == "ed-binary"
                            or row["model"].startswith("mont")
                            or name.startswith("bench_initial"), f"Missing timer: {name}/{metric}")
                    continue
                stat = row[metric]
                values = stat["samples"]
                require(len(values) == 3 and all(math.isfinite(v) and v > 0 for v in values),
                        f"Need three positive formal samples: {name}/{metric}")
                median = statistics.median(values)
                actual = dict(median=median, min=min(values), max=max(values),
                              mad=statistics.median(abs(v - median) for v in values))
                require(all(math.isclose(stat[k], v, rel_tol=1e-10, abs_tol=1e-10)
                            for k, v in actual.items()), f"Stale statistics: {name}/{metric}")
            formal += len(row["process_seconds"]["samples"])
            if "arithmetic_api_seconds" in row:
                require(all(a <= p for a, p in zip(row["arithmetic_api_seconds"]["samples"],
                                                   row["process_seconds"]["samples"])), "API exceeds process")
            if row["model"] in ("ed", "ed-5"):
                window = 8 if name == "bench_doubling_01" else (
                    6 if name == "bench_paired_01" else int(name.split("window")[1].split("_")[0]))
                require(row["window"] == window and row["doubling"] ==
                        (5 if row["model"] == "ed-5" else 4), f"Wrong NAF path: {name}")
                require("cuda_event_span_seconds" in row, "Missing CUDA event spans")
    require(formal == report["formal_samples"] == 144 and
            report["warmup_samples"] == 48, "Different sample accounting")
    require(report["initial"] == suites["bench_initial_01"]["rows"] +
            suites["bench_initial_wide_01"]["rows"], "Initial rows disagree with qualified suites")
    modern = [s for n, s in suites.items() if not n.startswith("bench_initial")]
    require(len({s["binary_sha256"] for s in modern}) == 1, "Modern suites use different executables")
    paired = {(r["bits"], r["model"]): r for r in suites["bench_paired_01"]["rows"]}
    for b in WIDTHS:
        require(all(paired[b, "ed-binary"]["geometry"][k] == paired[b, "ed"]["geometry"][k]
                    for k in ("container_bits", "tpi", "tpb", "blocks")),
                "Paired binary/NAF geometry differs")
    return suites


def lookup(suite, bits, model):
    return next(r for r in suite["rows"] if r["bits"] == bits and r["model"] == model)


def observations(ax, x, stat, color, denominator=1):
    """All raw observations, their median, and observed range (no inferred CI)."""
    samples = [v / denominator for v in stat["samples"]]
    median = stat["median"] / denominator
    ax.scatter([x - .035, x, x + .035], samples, s=18, facecolors="none",
               edgecolors=color, linewidths=.8, alpha=.8, zorder=3)
    ax.errorbar(x, median, yerr=[[median - min(samples)], [max(samples) - median]],
                fmt="o", markersize=5, color=color, capsize=3, linewidth=1.2, zorder=4)
    return median


def style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d7dce0", linewidth=.6)
    ax.set_axisbelow(True)


def categories(ax, labels, xlabel):
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel(xlabel)
    ax.set_xlim(-.55, len(labels) - .45)


def legend(fig, labels, colors):
    handles = [Line2D([], [], marker="o", linestyle="none", color=c, label=label, markersize=5)
               for label, c in zip(labels, colors)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .89),
               ncol=len(labels), frameon=False, fontsize=9)


def frame(fig, title, batch, notes, source, conditions):
    fig.suptitle(title, y=.985, fontsize=15, fontweight="bold")
    fig.text(.5, .932, f"2026-10-16 | GPU1 RTX 4060 Laptop | B1=10,000 | sigma=20260922 | batch={batch}",
             ha="center", fontsize=9.5, color="#3c4650")
    footer = ["Timing panels: filled dots = medians; open dots = all 3 runs; whiskers = observed min/max (not confidence intervals).",
              conditions, *notes, source]
    fig.text(.055, .025, "\n".join(footer), fontsize=7.8, linespacing=1.55, color="#444d56", va="bottom")
    fig.subplots_adjust(left=.08, right=.98, top=.79, bottom=.27, wspace=.30)


def plot_initial(report, suites, source, conditions):
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    models = ["ed-binary", "mont0", "mont3"]
    index = {(r["bits"], r["model"]): r for r in report["initial"]}
    for j, (model, color) in enumerate(zip(models, COLORS)):
        for i, bits in enumerate(WIDTHS):
            observations(ax, i + (j - 1) * .23, index[bits, model]["process_seconds"], color)
    ax.axvspan(4.5, 5.5, color="#edf1f3", zorder=0)
    ax.text(5, .98, "Separate build", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8, color="#59636d")
    ax.set_yscale("log")
    ax.set_ylim(.5, 45)
    ax.yaxis.set_major_locator(FixedLocator([.5, 1, 2, 5, 10, 20, 40]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    style(ax, "Complete process time (s / batch; log scale)")
    categories(ax, [f"M{b}" for b in WIDTHS], "Intact Mersenne input (categorical spacing)")
    legend(fig, ["Edwards binary", "Montgomery PARAM0", "Montgomery PARAM3"], COLORS)
    first = suites["bench_initial_01"]["binary_sha256"][:12]
    wide = suites["bench_initial_wide_01"]["binary_sha256"][:12]
    frame(fig, "Initial Stage1 cost across three curve families", 768,
          ["Fresh process: startup, preparation, GPU, GCD/normalization and save; independent CPU oracle excluded.",
           f"Exe SHA256: M347..M4003 {first}; M8191 {wide}. Different scalar families and container/TPI geometry.",
           "The shaded M8191 group uses the independently rebuilt frozen baseline. No cross-build optimization ratio."],
          source, conditions)
    return fig


def plot_paired(suites, source, conditions):
    suite = suites["bench_paired_01"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.8), sharey=True)
    for ax, metric, title in zip(axes, METRICS[:2], ["Complete process", "Arithmetic API (inside process time)"]):
        for i, bits in enumerate(WIDTHS):
            binary, naf = lookup(suite, bits, "ed-binary"), lookup(suite, bits, "ed")
            denominator = binary[metric]["median"]
            observations(ax, i - .13, binary[metric], COLORS[0], denominator)
            ratio = observations(ax, i + .13, naf[metric], COLORS[1], denominator)
            ax.annotate(f"{100 * (ratio - 1):.1f}%", (i + .13, naf[metric]["min"] / denominator), xytext=(0, -15),
                        textcoords="offset points", ha="center", fontsize=8, color=COLORS[1])
        ax.axhline(1, color="#7c858d", linestyle="--", linewidth=.8)
        ax.set_title(title, fontsize=11)
        style(ax, "Time / binary median (ratio; lower is faster)")
        categories(ax, [str(b) for b in WIDTHS], "Mersenne exponent p (categorical spacing)")
        ax.set_ylim(min(.48, ax.dataLim.y0 - .07), max(1.20, ax.dataLim.y1 + .05))
    legend(fig, ["Edwards binary", "NAF w=6 / HWCD"], COLORS[:2])
    frame(fig, "Same executable: binary versus window NAF", 768,
          ["Each sample is divided by its own width's binary median for the same timer; percentages compare medians.",
           "Two nested timing boundaries, shown separately. M991 ranges overlap; small differences remain uncertain.",
           f"Source suite: bench_paired_01; exe SHA256 {suite['binary_sha256'][:12]}. No initial-build samples used."],
          source, conditions)
    return fig


def dictionary_mib(row):
    """Logical allocation, matching ecm_cuda_edwards_naf.inl; not process peak."""
    return row["curves"] * (row["geometry"]["container_bits"] // 8) * 4 * (1 << (row["window"] - 2)) / 2**20


def plot_windows(suites, source, conditions):
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.8))
    for j, (metric, title) in enumerate([
            ("process_seconds", "Complete process / w=6 median"),
            ("cuda_event_span_seconds", "CUDA event spans / w=6 (zoom)")]):
        ax = axes[j]
        for b, color, offset in zip(SMALL_WIDTHS, COLORS, [-.10, 0, .10]):
            denominator = lookup(suites["bench_window6_01"], b, "ed")[metric]["median"]
            medians = []
            for i, window in enumerate(WINDOWS):
                row = lookup(suites[f"bench_window{window}_01"], b, "ed")
                medians.append(observations(ax, i + offset, row[metric], color, denominator))
            ax.plot([i + offset for i in range(4)], medians, color=color, linewidth=.8, alpha=.7)
        ax.axhline(1, color="#7c858d", linestyle="--", linewidth=.8)
        ax.set_title(title, fontsize=10.5)
        style(ax, "Time ratio (lower is faster)")
        categories(ax, [str(w) for w in WINDOWS], "NAF window w")
    # Independent scales deliberately expose process jitter and small event changes.
    axes[1].text(.02, .02, "Independent y scales", transform=axes[1].transAxes, fontsize=8, color="#59636d")
    memory = [dictionary_mib(lookup(suites[f"bench_window{w}_01"], 8191, "ed")) for w in WINDOWS]
    axes[2].plot(range(4), memory, "s-", color=COLORS[2], markersize=5, linewidth=1)
    for i, value in enumerate(memory):
        axes[2].annotate(f"{value:g} MiB", (i, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    axes[2].set_yscale("log", base=2)
    axes[2].set_ylim(4, 1200)
    axes[2].yaxis.set_major_locator(FixedLocator(memory))
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    axes[2].set_title("M8191 dictionary allocation", fontsize=10.5)
    style(axes[2], "Logical dictionary (MiB; log2 scale)")
    categories(axes[2], [str(w) for w in WINDOWS], "NAF window w")
    legend(fig, [f"M{b}" for b in SMALL_WIDTHS], COLORS)
    frame(fig, "Window tradeoff: event spans, complete time and dictionary size", 128,
          ["HWCD; window groups collected separately. Each width/timer is normalized to its w=6 median.",
           "Event spans sum dictionary + loop CUDA events, including possible first-load/launch gaps; not profiler kernel time.",
           "Dictionary = curves x container bytes x 4 coordinates x 2^(w-2); derived allocation, not a measured process peak.",
           f"Suites: bench_window{{6,8,10,12}}_01; exe SHA256 {suites['bench_window6_01']['binary_sha256'][:12]}."],
          source, conditions)
    return fig


def plot_doubling(suites, source, conditions):
    suite = suites["bench_doubling_01"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8))
    for ax, metric, title in zip(axes, [METRICS[0], METRICS[2]],
                               ["Complete process", "CUDA event spans (zoomed scale)"]):
        for i, bits in enumerate(SMALL_WIDTHS):
            hwcd, xy = lookup(suite, bits, "ed"), lookup(suite, bits, "ed-5")
            denominator = hwcd[metric]["median"]
            observations(ax, i - .10, hwcd[metric], COLORS[0], denominator)
            ratio = observations(ax, i + .10, xy[metric], COLORS[1], denominator)
            if metric == METRICS[2]:
                ax.annotate(f"{100 * (ratio - 1):.2f}%", (i + .10, xy[metric]["min"] / denominator), xytext=(0, -15),
                            textcoords="offset points", ha="center", fontsize=9, color=COLORS[1])
        ax.axhline(1, color="#7c858d", linestyle="--", linewidth=.8)
        ax.set_title(title, fontsize=11)
        style(ax, "Time / HWCD median (ratio; lower is faster)")
        categories(ax, [f"M{b}" for b in SMALL_WIDTHS], "Intact Mersenne input (categorical spacing)")
    axes[0].set_ylim(.45, 1.3)
    axes[1].set_ylim(.90, 1.025)
    legend(fig, ["HWCD", "2XY"], COLORS[:2])
    frame(fig, "Doubling forms: small event differences at large widths", 128,
          ["NAF w=8; each width/timer is normalized to its HWCD median. Percentages compare medians; axes differ.",
           "Event spans sum dictionary + loop CUDA events; not profiler kernel time. Complete-process ranges overlap.",
           f"Source suite: bench_doubling_01; exe SHA256 {suite['binary_sha256'][:12]}. Default remains HWCD."],
          source, conditions)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/benchmarks" / STUDY / "final_02/summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/figures" / STUDY)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/benchmarks" / STUDY / "plots_01.json")
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    suites = validate(report)
    # This script plots the qualified GPU1 study, not arbitrary hardware/B1 inputs.
    telemetry = [s["loaded_telemetry"] for s in suites.values()]
    observed = {k: [min(s[k]["min"] for s in telemetry), max(s[k]["max"] for s in telemetry)]
                for k in ("sm_mhz", "power_w", "temperature_c")}
    conditions = (f"Observed under load: SM {observed['sm_mhz'][0]:g}..{observed['sm_mhz'][1]:g} MHz; "
                  f"power {observed['power_w'][0]:.2f}..{observed['power_w'][1]:.2f} W; power limit N/A. "
                  "Hardware settings unchanged; one excluded device warm-up/case.")
    source = f"Source: {args.input.parent.name}/{args.input.name}; SHA256 {sha(args.input)[:12]}. Full identities and plot hashes: {args.manifest.name}."
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlepad": 10,
                         "svg.hashsalt": STUDY, "savefig.dpi": 180})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, figure in [
            ("initial", lambda: plot_initial(report, suites, source, conditions)),
            ("paired_naf", lambda: plot_paired(suites, source, conditions)),
            ("windows", lambda: plot_windows(suites, source, conditions)),
            ("doubling", lambda: plot_doubling(suites, source, conditions))]:
        fig = figure()
        for ext in ("png", "svg"):
            path = args.output_dir / f"{name}.{ext}"
            fig.savefig(path, facecolor="white", metadata={"Date": None} if ext == "svg" else None)
            outputs.append(dict(path=str(path.resolve()), sha256=sha(path)))
        plt.close(fig)
    manifest = dict(input=str(args.input.resolve()), input_sha256=sha(args.input),
                    plotter_sha256=sha(Path(__file__)), analyzer_sha256=report["analyzer_sha256"],
                    python=platform.python_version(), matplotlib=matplotlib.__version__,
                    device=report["device"], uuid=report["uuid"], observed=observed,
                    formal_samples=report["formal_samples"], warmup_samples=report["warmup_samples"],
                    binaries={n: s["binary_sha256"] for n, s in suites.items()},
                    statistics="All three formal runs, median and observed min/max; no confidence intervals.",
                    normalization="Paired: per-width binary median; windows: per-width w6 median; doubling: per-width HWCD median. Timer boundaries stay separate.",
                    dictionary="Derived curves * (container_bits/8) * 4 * 2^(window-2); logical allocation only.",
                    outputs=outputs)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(outputs)} PNG/SVG files; 144 formal samples retained. Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
