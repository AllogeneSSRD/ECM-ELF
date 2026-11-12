#!/usr/bin/env python3
"""Compare Stage1 NTT measurements with CGBN and Prime95 baselines."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from analyze_prime95_stage1 import stage1_plot_points


def ntt_points(report, target_b1):
    if report["reference_b1"] != target_b1:
        raise ValueError("benchmark B1 projections differ")
    exponents = [g["p"] for g in report["groups"]]
    if not exponents or len(set(exponents)) != len(exponents) or sorted(exponents) != sorted(report["exponents"]):
        raise ValueError("NTT benchmark is incomplete or repeated")
    points = []
    for group in report["groups"]:
        p, samples = group["p"], group["projected_seconds_per_curve_1e6"]
        points.append((p, p, statistics.median(samples), min(samples), max(samples)))
    return points


def ntt_version(report):
    return report["version"].removeprefix("ECM-ELF ")


def version_order(report):
    version = ntt_version(report)
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", version)
    if match is None:
        raise ValueError(f"unrecognized NTT version {version!r}")
    return tuple(map(int, match.groups()))


def cgbn_points(report):
    return [(row["exponent"], row["bits"], row["projected_gpu_seconds_per_curve_1e6"], None, None)
            for row in report["rows"]]


def reference_points(report):
    return {row["exponent"]: (row["bits"], row["b1"], row["cpu_seconds_1e6"],
                             row["cpu_min_seconds_1e6"], row["cpu_max_seconds_1e6"]) for row in report["rows"]}


def make_points(ntt, historical, previous_ntt=(), overlay_ntt=(), overlay_cgbn=None, prime95_extra=()):
    target_b1 = historical["target_b1"]
    current_version = version_order(ntt)
    seen_versions = {current_version}
    points = {("ntt", ntt_version(ntt)): ntt_points(ntt, target_b1)}
    for report in sorted(previous_ntt, key=version_order, reverse=True):
        version = version_order(report)
        if version in seen_versions or version > current_version:
            raise ValueError("each previous NTT version must be distinct and older than --ntt")
        if any(report[key] != ntt[key] for key in ("curve", "gpu_model")):
            raise ValueError("NTT series must use the same GPU model and curve construction")
        if sorted(report["exponents"]) != sorted(ntt["exponents"]):
            raise ValueError("NTT series must cover the same exponents")
        seen_versions.add(version)
        points["ntt_previous", ntt_version(report)] = ntt_points(report, target_b1)
    if overlay_ntt:
        first = overlay_ntt[0]
        identity = ("gpu_model", "device", "curve")
        if first["gpu_model"] == ntt["gpu_model"] or first["curve"] != ntt["curve"] or any(
            any(report[key] != first[key] for key in identity) for report in overlay_ntt
        ):
            raise ValueError("overlay requires one other GPU and the same curve construction")
        versions = {}
        for report in sorted(overlay_ntt, key=version_order, reverse=True):
            versions.setdefault(ntt_version(report), []).extend(ntt_points(report, target_b1))
        for i, (version, rows) in enumerate(versions.items()):
            if len({row[0] for row in rows}) != len(rows):
                raise ValueError("overlay NTT reports must cover distinct exponents within each version")
            points["ntt_overlay" if i == 0 else "ntt_overlay_previous", version] = rows
    cgbn_exponents = [row["exponent"] for row in historical["rows"]]
    if not cgbn_exponents or len(set(cgbn_exponents)) != len(cgbn_exponents) or historical.get("missing_cpu_reference_exponents"):
        raise ValueError("CGBN and Prime95 reference conditions are incomplete or repeated")
    cgbn, prime95 = ("cgbn", historical.get("program_version", "")), ("prime95", "")
    points[cgbn], points[prime95] = cgbn_points(historical), []
    if overlay_cgbn is not None:
        if (overlay_cgbn["model"] != historical["model"] or overlay_cgbn["gpu_model"] == historical["gpu_model"]
                or overlay_cgbn["target_b1"] != target_b1 or overlay_cgbn.get("gpu_timing") != historical.get("gpu_timing")
                or len(overlay_cgbn["rows"]) != len(cgbn_exponents)
                or reference_points(overlay_cgbn) != reference_points(historical)):
            raise ValueError("CGBN overlay must use another GPU with matching inputs, timers and CPU references")
        points["cgbn_overlay", overlay_cgbn.get("program_version", "")] = cgbn_points(overlay_cgbn)
    for row in historical["rows"]:
        p = row["exponent"]
        # Prime95 sets up GWNUM with k,b,n,c, even when the factor target is a cofactor.
        points[prime95].append((p, p, row["cpu_seconds_1e6"], row["cpu_min_seconds_1e6"], row["cpu_max_seconds_1e6"]))
    if set(cgbn_exponents) & {row[0] for row in prime95_extra}:
        raise ValueError("Supplemental Prime95 points overlap the historical baseline")
    points[prime95].extend(prime95_extra)
    return {key: sorted(rows, key=lambda r: r[1]) for key, rows in points.items()}


def plot(points, output, logarithmic, curve, device, overlay_device=None):
    model = "Edwards" if curve == "ed" else "Montgomery PARAM0"
    primary_history = [version for backend, version in points if backend == "ntt_previous"]
    overlay_history = [version for backend, version in points if backend == "ntt_overlay_previous"]
    colors = {
        "ntt": ("#b73339", "o"),
        "ntt_overlay": ("#14804a", "D"),
        "cgbn": ("#d4771d", "s"),
        "cgbn_overlay": ("#2866a2", "P"),
        "prime95": ("#000000", "^"),
    }
    styles = {}
    for key in points:
        backend, version = key
        previous = backend.endswith("_previous")
        if previous:
            overlay = backend == "ntt_overlay_previous"
            history = overlay_history if overlay else primary_history
            rgb = to_rgb("#237c55" if overlay else "#784788")
            fade = .56 * (history.index(version) + 1) / (len(history) + 1)
            color = tuple(value + (1 - value) * fade for value in rgb)
            marker = "D" if overlay else "o"
        else:
            color, marker = colors[backend]
        label = ("GWNUM · 1 core at 4 GHz" if backend == "prime95"
                 else f"{'NTT' if backend.startswith('ntt') else 'CGBN'} {version}")
        styles[key] = (label, color, marker, previous)
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.15 + .18 * len(primary_history)), layout="constrained")
        fig.suptitle(f"Stage1 {model} · Cost per curve\n"
                     "NTT · full Mersenne, 3 runs   |   CGBN · cofactor, 2 blocks/SM, 1 batch   |   "
                     "Prime95 · native Stage1, 3 runs",
                     fontsize=13, fontweight="medium", linespacing=1.45)
        lines = {}
        for i, ax in enumerate(axes):
            ax.set_title(("Detail through 16,500 bits", "Full NTT measurement range")[i])
            for key, rows in points.items():
                visible = rows if i else [r for r in rows if r[1] <= 16500]
                if not visible:
                    continue
                _, x, y, lo, hi = zip(*visible)
                label, color, marker, previous = styles[key]
                line, = ax.plot(x, y, label=label, color=color, marker=marker, markersize=4, linewidth=1.7,
                                linestyle="--" if previous else "-", markerfacecolor="none" if previous else color,
                                zorder=2 if previous else 3)
                if i:
                    lines[key] = line
                if not previous and all(value is not None for value in lo + hi):
                    ax.fill_between(x, lo, hi, color=color, alpha=.14, linewidth=0)
            ax.set_xlabel("Actual arithmetic modulus width (bits)")
            if logarithmic:
                ax.set_yscale("log")
                if i:
                    ax.set_xscale("log")
            else:
                ax.set_ylim(bottom=0)
                ax.ticklabel_format(style="plain", axis="x")
        axes[0].set_ylabel("Seconds per curve at B1 = 1,000,000 (scaled)")
        groups = (
            (overlay_device, ("ntt_overlay", "ntt_overlay_previous", "cgbn_overlay")),
            (device, ("ntt", "ntt_previous", "cgbn")),
            ("Prime95", ("prime95",)),
        )
        columns = [(name, [line for role in roles for key, line in lines.items() if key[0] == role])
                   for name, roles in groups if name]
        height = 1 + max(len(group) for _, group in columns)
        handles, labels = [], []
        for name, group in columns:
            handles += [Line2D([], [], linestyle="none")] + group
            labels += [name] + [line.get_label() for line in group]
            handles += [Line2D([], [], linestyle="none") for _ in range(height - len(group) - 1)]
            labels += [""] * (height - len(group) - 1)
        legend = fig.legend(handles, labels, loc="outside lower center", ncols=len(columns), fontsize=9, frameon=False)
        for heading in legend.get_texts()[::height]:
            heading.set_fontweight("bold")
        fig.savefig(output, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntt", required=True, type=Path)
    parser.add_argument("--previous-ntt", action="append", default=[], type=Path,
                        help="Overlay a historical NTT benchmark as a dashed line; may be repeated")
    parser.add_argument("--overlay-ntt", action="append", default=[], type=Path,
                        help="Add another GPU's NTT data; repeat for versions or disjoint exponents")
    parser.add_argument("--overlay-cgbn", type=Path, help="Add the other GPU's CGBN results with the same timing scope")
    parser.add_argument("--prime95-stage1", type=Path, nargs=2, metavar=("WORKTODO", "SCREEN_LOG"),
                        help="Extend the 4 GHz single-core Prime95 baseline with Stage1-only results")
    parser.add_argument("--comparison", "--historical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ntt, comparison = (json.loads(p.read_text(encoding="utf-8-sig")) for p in (args.ntt, args.comparison))
    curve = ntt["curve"]
    if ntt["gpu_model"] != comparison["gpu_model"] or comparison["model"] != ("ed" if curve == "ed" else "mont0"):
        parser.error("CGBN and NTT must use the same GPU model and curve construction")
    device = ntt["gpu_model"].removeprefix("NVIDIA GeForce ").removesuffix(" GPU")
    previous = [json.loads(path.read_text(encoding="utf-8-sig")) for path in args.previous_ntt]
    overlay = [json.loads(path.read_text(encoding="utf-8-sig")) for path in args.overlay_ntt]
    cgbn_overlay = json.loads(args.overlay_cgbn.read_text(encoding="utf-8-sig")) if args.overlay_cgbn else None
    if overlay and cgbn_overlay and overlay[0]["gpu_model"] != cgbn_overlay["gpu_model"]:
        parser.error("NTT and CGBN overlays must use the same GPU model")
    extra = overlay[0] if overlay else cgbn_overlay
    overlay_device = extra["gpu_model"].removeprefix("NVIDIA GeForce ").removesuffix(" GPU") if extra else None
    try:
        prime95_extra = ()
        if args.prime95_stage1:
            curve_type = "Edwards" if curve == "ed" else "Montgomery"
            prime95_extra = stage1_plot_points(*args.prime95_stage1, curve_type, comparison["target_b1"])
        points = make_points(ntt, comparison, previous, overlay, cgbn_overlay, prime95_extra)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = "stage1-edwards-ntt-cgbn-prime95" if curve == "ed" else "stage1-ntt-best-cgbn-prime95"
    for suffix in ("log", "linear"):
        plot(points, args.output / f"{prefix}-{suffix}.png", suffix == "log", curve, device, overlay_device)
    with (args.output / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("series", "mersenne_exponent", "arithmetic_bits", "median_s_per_curve_at_1e6",
                         "min_s_per_curve_at_1e6", "max_s_per_curve_at_1e6", "version"))
        for (backend, version), rows in points.items():
            writer.writerows((backend, *row, version) for row in rows)


if __name__ == "__main__":
    main()
