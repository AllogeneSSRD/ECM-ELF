"""Summarize completed Prime95 Stage1 timers and linearly project B1.

Reuses the worker/curve/FFT parser. Task completion messages are not Stage1
timers: in particular, a resumed Stage2 contributes no Stage1 sample.
No benchmark processes are launched and no input files are changed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
LOCAL_INPUTS = TOOLS / "benchmarks/prime95"
sys.path.insert(0, str(TOOLS / "log_parser"))
from parser import parse_log  # noqa: E402


def measure(values):
    if not values:
        return dict(n=0, samples=[], median=None, mean=None, min=None, max=None)
    return dict(n=len(values), samples=values, median=statistics.median(values),
                mean=statistics.mean(values), min=min(values), max=max(values))


def input_scope(path):
    """Worktodo identifies the intended integers, not proof of logged factors."""
    scope = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith(("ECM=", "ECM2=")):
            continue
        fields = next(csv.reader([line.split("=", 1)[1]], strict=True))
        k, base, exponent, c = map(int, fields[:4])
        if (k, base, c) != (1, 2, -1):
            raise ValueError("Only Mersenne work lines are supported")
        factors = [int(f) for f in fields[-1].split(",")]
        if not factors or any(f <= 1 for f in factors):
            raise ValueError("Invalid known factors")
        cofactor, remainder = divmod((1 << exponent) - 1, math.prod(factors))
        if remainder:
            raise ValueError(f"Known factors do not divide M{exponent}")
        item = dict(exponent=exponent, intended_cofactor_bits=cofactor.bit_length(),
                    intended_known_factors=[str(f) for f in factors])
        if exponent in scope and scope[exponent] != item:
            raise ValueError(f"Inconsistent inputs for M{exponent}")
        scope[exponent] = item
    if not scope:
        raise ValueError("No ECM inputs in worktodo")
    return scope


def analyze(sources, scope, cpu_ghz, program_version, target_bounds, curve_type="Edwards"):
    parsed = []
    for source in sources:
        for run in parse_log(source["text"]):
            run["source_log"] = source["name"]
            run["warmup"] = run["curve"] in source["exclude_curves"]
            parsed.append(run)
    grouped = defaultdict(list)
    samples, excluded = [], []
    for run in parsed:
        if run["exponent"] not in scope:
            raise ValueError(f"Unexpected exponent: {run['exponent']}")
        if run["curve_type"] != curve_type or run["b1"] != 1_000_000:
            raise ValueError(f"Expected {curve_type} B1=1,000,000")
        if run["s1_time"] is None:
            excluded.append(dict(exponent=run["exponent"], curve=run["curve"],
                                 source_log=run["source_log"], start_line=run["start_line"],
                                 reason="no_stage1_completion_timer"))
            continue
        if run["warmup"]:
            excluded.append(dict(exponent=run["exponent"], curve=run["curve"],
                                 source_log=run["source_log"], start_line=run["start_line"],
                                 seconds=run["s1_time"], reason="user_designated_warmup"))
            continue
        if not math.isfinite(run["s1_time"]) or run["s1_time"] <= 0:
            raise ValueError("Invalid Stage1 duration")
        sample = {k: run[k] for k in ("worker", "exponent", "curve", "curve_type", "s", "b1",
                                      "s1_time", "s1_fft", "s1_fft_type", "start_line", "source_log",
                                      "start_ts", "s1_end_ts")}
        samples.append(sample)
        grouped[run["exponent"]].append(sample)
    timers = sum(len(re.findall(r"Stage 1 complete\..*?Total time:", s["text"])) for s in sources)
    warmups = sum(e["reason"] == "user_designated_warmup" for e in excluded)
    if timers != len(samples) + warmups:
        raise ValueError("Not all Stage1 completion lines were attributed to curves")
    if len({r["worker"] for r in parsed}) != 1:
        raise ValueError("Expected one worker for this single-core study")
    rows = []
    for exponent, case in sorted(scope.items()):
        runs = grouped[exponent]
        if len({(r["source_log"], r["curve"], r["s"]) for r in runs}) != len(runs):
            raise ValueError(f"Duplicate Stage1 observation for M{exponent}")
        if len(runs) not in (0, 3):
            raise ValueError("Expected either a missing cell or three observations")
        fft = {(r["s1_fft"], r["s1_fft_type"]) for r in runs}
        if len(fft) > 1:
            raise ValueError("Stage1 FFT changed within a cell")
        stats = measure([r["s1_time"] for r in runs])
        length, kind = next(iter(fft)) if fft else (None, None)
        rows.append(dict(**case, fft_length=length, fft_type=kind, measured=stats,
                         projections=[dict(b1=b1, scale=b1 / 1_000_000,
                                           median_seconds=stats["median"] * b1 / 1_000_000
                                           if stats["n"] else None)
                                      for b1 in target_bounds]))
    return dict(schema=1, cpu_ghz=cpu_ghz, cpu_cores=1,
                cpu_frequency_evidence="user-confirmed fixed frequency",
                program="Prime95", program_version=program_version, curve_type=curve_type,
                cpu_model="not recorded; log reports AMD Zen architecture",
                cpu_power="not recorded", measured_b1=1_000_000,
                input_identity="intended worktodo scope; known factors not printed in screen log",
                timing="native Stage 1 complete timer; excludes Stage2 and full process startup",
                projection="T(B1)=T(1,000,000)*B1/1,000,000; assumed linear, not a multi-B1 fit",
                aggregation="median; only explicitly user-designated warmups excluded",
                parsed_curves=len(parsed), completed_stage1_samples=len(samples),
                observation_span=dict(start=min(s["start_ts"] for s in samples),
                                      end=max(s["s1_end_ts"] for s in samples), timezone="not recorded"),
                excluded_warmups=warmups,
                missing_exponents=[r["exponent"] for r in rows if r["measured"]["n"] == 0],
                rows=rows, samples=samples, excluded=excluded)


def plots(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    rows = report["rows"]
    valid = [(i, r) for i, r in enumerate(rows) if r["measured"]["n"]]
    outputs = []
    for scale in ("linear", "log"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
        for ax in axes:
            ax.set_yscale(scale)
            ax.set_xticks(range(len(rows)), [str(r["exponent"]) for r in rows], rotation=45)
            ax.set_xlabel("Original Mersenne exponent p (categorical spacing)")
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            for i, r in enumerate(rows):
                if not r["measured"]["n"]:
                    ax.axvspan(i - .4, i + .4, color="#f3e4df")
                    ax.text(i, .97, "No S1 timer", transform=ax.get_xaxis_transform(),
                            rotation=90, ha="center", va="top", fontsize=8, color="#994c39")
        ax = axes[0]
        for i, r in valid:
            s = r["measured"]
            ax.errorbar(i, s["median"], yerr=[[s["median"] - s["min"]],
                                           [s["max"] - s["median"]]], fmt="o", color="#177c86",
                        capsize=3, markersize=5)
            ax.scatter([i - .13, i, i + .13], s["samples"], s=22,
                       facecolors="none", edgecolors="#177c86", zorder=3)
        ax.set_title("Measured Stage1: B1 = 1,000,000")
        ax.set_ylabel("Native Stage1 time (seconds / curve)")
        for j, (color, bound) in enumerate(zip(("#bf6b2e", "#5968a4"),
                                              rows[0]["projections"])):
            xs = [i for i, _ in valid]
            ys = [r["projections"][j]["median_seconds"] for _, r in valid]
            axes[1].scatter(xs, ys, marker=("s", "^")[j], color=color,
                            label=f'B1 = {bound["b1"] / 1e6:g} million')
            for i, r in valid:
                s = r["measured"]
                axes[1].vlines(i, s["min"] * bound["scale"], s["max"] * bound["scale"],
                               color=color, alpha=.5)
        axes[1].set_title("Linear B1 extrapolation (not measured)")
        axes[1].set_ylabel("Projected Stage1 time (seconds / curve)")
        axes[1].legend(loc="upper left", frameon=False)
        for ax in axes:
            if scale == "linear":
                ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            if scale == "log":
                ax.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        fig.suptitle(f'Prime95 {report["curve_type"]} Stage1 | one CPU core at {report["cpu_ghz"]:.1f} GHz',
                     fontsize=15, fontweight="bold")
        fig.text(.07, .025,
                 f'{report["completed_stage1_samples"]} completed observations; n=3 per measured cell; filled = median, open = all samples, whiskers = observed min/max.\n'
                 f'{report["excluded_warmups"]} designated warmup(s) excluded. No other complete timer discarded.\n'
                 "Source logs and sample provenance are recorded in summary.json; exact Prime95 version, CPU model and power not recorded.",
                 fontsize=8, color="#49535d", linespacing=1.5)
        fig.subplots_adjust(left=.07, right=.985, top=.86, bottom=.29, wspace=.21)
        for ext in ("png", "svg"):
            model = "" if report["curve_type"] == "Edwards" else "-montgomery"
            path = output / f"stage1-prime95{model}-4ghz-{scale}.{ext}"
            fig.savefig(path, dpi=170)
            outputs.append(str(path))
        plt.close(fig)
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, default=LOCAL_INPUTS / "screen_stage1_1e6.log")
    p.add_argument("--worktodo", type=Path, default=LOCAL_INPUTS / "worktodo_ecm_stage1.txt")
    p.add_argument("--supplement-log", type=Path)
    p.add_argument("--exclude-supplement-curves", type=int, nargs="*", default=[],
                   help="Curve numbers explicitly designated as warmups, supplement only")
    p.add_argument("--cpu-ghz", type=float, required=True)
    p.add_argument("--curve-type", choices=("Edwards", "Montgomery"), default="Edwards")
    p.add_argument("--program-version", default=None, help="Use only a version verified for this log")
    p.add_argument("--output", type=Path, default=Path("data/benchmarks/stage1_prime95_20261024"))
    p.add_argument("--evidence-dir", type=Path, default=Path("data/experiments/stage1_prime95_20261024"))
    p.add_argument("--figures", type=Path, default=Path("data/figures/stage1_prime95_20261024"))
    args = p.parse_args()
    if not math.isfinite(args.cpu_ghz) or args.cpu_ghz <= 0:
        p.error("CPU frequency must be positive")
    if args.exclude_supplement_curves and args.supplement_log is None:
        p.error("Supplement warmup exclusions require --supplement-log")
    paths = [args.log] + ([args.supplement_log] if args.supplement_log else [])
    sources = [dict(name=str(path.resolve()), raw=path.read_bytes(),
                    exclude_curves=set(args.exclude_supplement_curves) if i else set())
               for i, path in enumerate(paths)]
    for s in sources:
        s["text"] = s["raw"].decode("utf-8-sig")
    report = analyze(sources, input_scope(args.worktodo),
                     args.cpu_ghz, args.program_version, (11_000_000, 44_000_000), args.curve_type)
    report["source_logs"] = [s["name"] for s in sources]
    report["intended_worktodo"] = str(args.worktodo.resolve())
    for path in (args.output, args.evidence_dir, args.figures):
        path.mkdir(parents=True, exist_ok=True)
    report["frozen_logs"] = []
    for path, source in zip(paths, sources):
        frozen = args.evidence_dir / path.name
        if frozen.exists() and frozen.read_bytes() != source["raw"]:
            raise ValueError("Evidence snapshot already exists with different contents; use a new directory")
        frozen.write_bytes(source["raw"])
        report["frozen_logs"].append(str(frozen.resolve()))
    worktodo_snapshot = args.evidence_dir / "worktodo.snapshot.txt"
    worktodo_raw = args.worktodo.read_bytes()
    if worktodo_snapshot.exists() and worktodo_snapshot.read_bytes() != worktodo_raw:
        raise ValueError("Worktodo snapshot differs; use a new evidence directory")
    worktodo_snapshot.write_bytes(worktodo_raw)
    report["frozen_worktodo"] = str(worktodo_snapshot.resolve())
    report["figures"] = plots(report, args.figures)
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    columns = ("exponent", "intended_cofactor_bits", "fft_type", "fft_length", "samples",
               "median_1e6_s", "min_1e6_s", "max_1e6_s", "projected_11e6_s", "projected_44e6_s")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for r in report["rows"]:
            values = dict(exponent=r["exponent"], intended_cofactor_bits=r["intended_cofactor_bits"],
                                 fft_type=r["fft_type"], fft_length=r["fft_length"], samples=r["measured"]["n"],
                                 median_1e6_s=r["measured"]["median"], min_1e6_s=r["measured"]["min"],
                                 max_1e6_s=r["measured"]["max"],
                                 projected_11e6_s=r["projections"][0]["median_seconds"],
                                 projected_44e6_s=r["projections"][1]["median_seconds"])
            writer.writerow({key: format(value, ".9g") if isinstance(value, float) else value
                             for key, value in values.items()})
    print(json.dumps(dict(completed_stage1_samples=report["completed_stage1_samples"],
                          missing_exponents=report["missing_exponents"], output=str(args.output))))


if __name__ == "__main__":
    main()
