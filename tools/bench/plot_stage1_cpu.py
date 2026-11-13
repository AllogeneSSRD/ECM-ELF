"""Plot the matched Edwards Stage1 CPU matrix on linear and log time axes."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


ROOT = Path(__file__).resolve().parents[2]
GMP_FIXED_WIDTH_LIMIT = 10240
KARATSUBA_THRESHOLDS = ((4213, "Karatsuba mul"), (8165, "Karatsuba square"))
SERIES = (
    ("prime95", "stage1", 1, "#000000", "^"),
    ("scalar", "wall", 1, "#2363a2", "s"),
    ("simd", "wall", 8, "#b74421", "o"),
)
SCHOOLBOOK = ("simd_schoolbook", "wall", 8, "#d48050", "D")


def load_matrix(path):
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty array of measured exponent rows")
    bounds, repeats, widths = set(), set(), set()
    has_schoolbook = ["simd_schoolbook" in row["samples"] for row in rows]
    if any(has_schoolbook) and not all(has_schoolbook):
        raise ValueError("Expected IFMA schoolbook samples for every exponent")
    series = SERIES + (SCHOOLBOOK,) if all(has_schoolbook) else SERIES
    points = {key: [] for key, *_ in series}
    for row in rows:
        p, b1 = row["exponent"], row["b1"]
        if type(p) is not int or p < 2 or p in widths or type(b1) is not int or b1 < 1:
            raise ValueError(f"Invalid or repeated matrix condition p={p}, B1={b1}")
        widths.add(p)
        bounds.add(b1)
        for key, timer, curves, *_ in series:
            samples = row["samples"][key]
            durations = [sample[timer] / curves for sample in samples]
            if not durations or any(not math.isfinite(t) or t <= 0 for t in durations):
                raise ValueError(f"M{p}: invalid {key} samples")
            repeats.add(len(durations))
            points[key].append((p, median(durations), min(durations), max(durations)))
    if len(bounds) != 1 or len(repeats) != 1:
        raise ValueError("Expected the same B1 and sample count for each exponent and implementation")
    return {key: sorted(group) for key, group in points.items()}, bounds.pop(), repeats.pop()


def draw_series(ax, points, label, color, marker, dashed=False):
    x, middle, lo, hi = zip(*points)
    ax.errorbar(x, middle,
                yerr=([y - low for y, low in zip(middle, lo)],
                      [high - y for y, high in zip(middle, hi)]),
                label=label, color=color, marker=marker, markersize=5, linewidth=2,
                linestyle="--" if dashed else "-", markerfacecolor="white" if dashed else color,
                elinewidth=1, capsize=3, zorder=4)


def mark_threshold(ax, width, description, color, linestyle):
    ax.axvline(width, color=color, linestyle=linestyle, linewidth=1.1, zorder=2)
    ax.text(width, 1.02, f"{width:,}\n{description}", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", color=color, fontsize=8.5, linespacing=1.05)


def plot(points, b1, repeats, output, scale, cpu, ecm_version, prime95_version):
    has_schoolbook = SCHOOLBOOK[0] in points
    fallback = [row for row in points["scalar"] if row[0] > GMP_FIXED_WIDTH_LIMIT]
    min_width, max_width = points["prime95"][0][0], points["prime95"][-1][0]
    thresholds = [(width, label) for width, label in KARATSUBA_THRESHOLDS
                  if has_schoolbook and min_width <= width <= max_width]
    fixed_width_boundary = min_width <= GMP_FIXED_WIDTH_LIMIT < max_width
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(10.6, 5.8))
        fig.subplots_adjust(left=.105, right=.97, bottom=.214 if fallback else .18,
                            top=.78 if thresholds or fixed_width_boundary else .82)
        fig.suptitle("Edwards Stage1 on one CPU core", x=.105, y=.96, ha="left",
                     fontweight="bold", fontsize=18)
        fig.text(.105, .875, f"{cpu}  •  Full Mersenne moduli  •  "
                             f"B1 = B2 = {b1:,}  •  {repeats} runs per point",
                 fontsize=10, color="#434854")

        for width, description in thresholds:
            mark_threshold(ax, width, description, "#a15a41", "--")
        if fixed_width_boundary:
            mark_threshold(ax, GMP_FIXED_WIDTH_LIMIT, "GMP fixed-width limit", "#3f6d9c", ":")
        for key, _, _, color, marker in SERIES:
            group = points[key]
            if key == "scalar":
                fixed = [row for row in group if row[0] <= GMP_FIXED_WIDTH_LIMIT]
                if fixed:
                    draw_series(ax, fixed, "ECM-ELF GMP · fixed-width NAF · 1 curve", color, marker)
                if fallback:
                    draw_series(ax, fallback, "ECM-ELF GMP · mpz fallback · 1 curve", color, marker, dashed=True)
            else:
                label = f"Prime95 {prime95_version} · GWNUM" if key == "prime95" else "ECM-ELF IFMA · 8 curves"
                if key == "simd" and has_schoolbook:
                    label = "ECM-ELF IFMA + Karatsuba · 8 curves"
                draw_series(ax, group, label, color, marker)
        key, _, _, color, marker = SCHOOLBOOK
        if key in points:
            draw_series(ax, points[key], "ECM-ELF IFMA schoolbook · 8 curves", color, marker, dashed=True)

        ax.set_yscale(scale)
        if scale == "linear":
            ax.set_ylim(bottom=0)
        ax.set_xlabel("Full Mersenne exponent p (bits)")
        ax.set_ylabel("Wall time (seconds per curve)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", facecolor="white", edgecolor="#e2e5e9",
                  frameon=True, framealpha=1, fontsize=9)
        fig.text(.105, .097 if fallback else .063,
                 f"ECM-ELF {ecm_version}: full process wall time, IFMA batch divided by 8. "
                 "Prime95: native Stage1 timer.",
                 fontsize=9, color="#434854")
        if fallback:
            fig.text(.105, .063, "GMP above 10,240 bits uses the mpz binary fallback "
                                 "when scalar is explicitly selected.",
                     fontsize=9, color="#434854")
        fig.text(.105, .029, "Markers show medians. Whiskers show the observed minimum and maximum.",
                 fontsize=9, color="#434854")
        fig.savefig(output, dpi=180, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, help="Matched raw benchmark JSON")
    parser.add_argument("--output", type=Path, default=ROOT / "data/figures/stage1_cpu")
    parser.add_argument("--ecm-version", required=True)
    parser.add_argument("--prime95-version", required=True)
    parser.add_argument("--cpu", default="Ryzen AI 9 HX 370, 4.0 GHz setting")
    args = parser.parse_args()
    points, b1, repeats = load_matrix(args.measurements)
    args.output.mkdir(parents=True, exist_ok=True)
    for scale in ("linear", "log"):
        output = args.output / f"stage1-edwards-cpu-{scale}.png"
        plot(points, b1, repeats, output, scale, args.cpu, args.ecm_version, args.prime95_version)
        print(output)


if __name__ == "__main__":
    main()
