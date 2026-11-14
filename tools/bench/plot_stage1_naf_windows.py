#!/usr/bin/env python3
"""Plot Edwards Stage1 NAF-window throughput and elapsed-time breakdowns."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


BACKENDS = {
    "cpu-ifma": ("CPU · IFMA", 12, ("#375780", "#77729b", "#4a7e76")),
    "cpu-scalar": ("CPU · scalar", 12, ("#375780", "#77729b", "#4a7e76")),
    "rtx4060": ("RTX 4060 Laptop · CGBN", 6, ("#bd651d", "#ad4046", "#78568f")),
    "rtx4070ti": ("RTX 4070 Ti · CGBN", 6, ("#317952", "#267b87", "#325d9a")),
}
PHASES = {
    "startup_s": ("Process startup", "#9da6ad"),
    "init_s": ("Stage1 preparation", "#de974a"),
    "loop_s": ("Arithmetic loop", "#3379a3"),
    "tail_s": ("Finalization and exit", "#83a381"),
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def load_samples(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    cells = defaultdict(lambda: defaultdict(list))
    conditions = {}
    repeats = set()
    for row in report["samples"]:
        backend, bits, window, repeat = (row[k] for k in ("backend", "bits", "w", "repeat"))
        if repeat < 0:
            continue
        if backend not in BACKENDS:
            raise ValueError(f"unknown Stage1 backend {backend!r}")
        key = (backend, bits)
        condition = (row["b1"], row["curves"])
        if conditions.setdefault(key, condition) != condition:
            raise ValueError(f"B1 or curve count changed for {key}")
        repeat_key = (*key, window, repeat)
        if repeat_key in repeats:
            raise ValueError(f"duplicate sample {repeat_key}")
        repeats.add(repeat_key)
        times = [row[phase] for phase in PHASES]
        wall = row["wall_s"]
        if (any(not math.isfinite(t) or t < 0 for t in (*times, wall))
                or any(not math.isfinite(row[k]) or row[k] <= 0 for k in ("loop_s", loop_metric(backend)))):
            raise ValueError(f"invalid timings for {repeat_key}")
        if not math.isclose(sum(times), wall, rel_tol=0.002, abs_tol=0.002):
            raise ValueError(f"elapsed-time phases do not sum to the wall time for {repeat_key}")
        if backend.startswith("rtx"):
            dictionary = row["dictionary_gpu_s"]
            if not math.isfinite(dictionary) or dictionary < 0:
                raise ValueError(f"invalid GPU dictionary timing for {repeat_key}")
            row["prepared_loop_s"] = dictionary + row["loop_gpu_s"]
        else:
            row["prepared_loop_s"] = row["init_s"] + row["loop_s"]
        cells[key][window].append(row)
    if not cells:
        raise ValueError("benchmark report contains no measured samples")
    for (backend, bits), windows in cells.items():
        default = BACKENDS[backend][1]
        if default not in windows:
            raise ValueError(f"{backend} M{bits} has no default w{default} baseline")
    return report.get("version"), cells, conditions


def backend_order(cells):
    present = {backend for backend, _ in cells}
    return [backend for backend in BACKENDS if backend in present]


def phase_median(samples, key):
    return median(row[key] for row in samples)


def loop_metric(backend):
    return "loop_gpu_s" if backend.startswith("rtx") else "loop_s"


def fastest(windows, key):
    return min(sorted(windows), key=lambda w: phase_median(windows[w], key))


def save_figure(fig, output):
    fig.savefig(output, dpi=175, facecolor="white")
    plt.close(fig)


def plot_throughput(cells, output, version):
    backends = backend_order(cells)
    widths = sorted({bits for _, bits in cells})
    fig, axes = plt.subplots(3, len(backends), figsize=(3.65 * len(backends) + 0.55, 8.4),
                             squeeze=False, layout="constrained")

    for row in range(3):
        first_axes = {}
        for column, backend in enumerate(backends):
            label, default, colors = BACKENDS[backend]
            ax = axes[row, column]
            gpu = backend.startswith("rtx")
            metrics = ((loop_metric(backend), "Loop throughput"),
                       ("prepared_loop_s", "GPU dictionary + loop" if gpu else "CPU prep + loop"),
                       ("wall_s", "End-to-end throughput"))
            metric, ylabel = metrics[row]
            # CPU and GPU ranges differ enough to need separate scales.
            if gpu in first_axes:
                ax.sharey(first_axes[gpu])
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(ylabel)
                first_axes[gpu] = ax
            measured_windows = set()
            for index, bits in enumerate(widths):
                if (backend, bits) not in cells:
                    continue
                windows = cells[backend, bits]
                x = sorted(windows)
                measured_windows.update(x)
                reference = phase_median(windows[default], metric)
                rates = [[reference / sample[metric] for sample in windows[w]] for w in x]
                center = [reference / phase_median(windows[w], metric) for w in x]
                lower = [mid - min(values) for mid, values in zip(center, rates)]
                upper = [max(values) - mid for mid, values in zip(center, rates)]
                color = colors[index % len(colors)]
                ax.errorbar(x, center, yerr=(lower, upper), lw=1.7, elinewidth=0.9, capsize=2,
                            color=color, marker=MARKERS[index % len(MARKERS)], markersize=4)
                winner = fastest(windows, metric)
                ax.plot(winner, center[x.index(winner)], marker=MARKERS[index % len(MARKERS)],
                        color=color, markerfacecolor="white", markeredgewidth=1.5, markersize=6.5,
                        zorder=4)
            ax.axhline(1, color="#6c7479", ls="--", lw=0.9, zorder=0)
            ax.axvline(default, color="#9ba2a9", ls=":", lw=0.9, zorder=0)
            ax.set_xticks(sorted(measured_windows))
            ax.set_xlim(min(measured_windows) - 0.6, max(measured_windows) + 0.6)
            ax.yaxis.set_major_locator(MaxNLocator(6))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f×"))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.16)
            if row == 0:
                ax.set_title(f"{label}\nDefault w{default}", fontsize=10)
            elif row == 2:
                ax.set_xlabel("NAF window w")

    shapes = [Line2D([], [], marker=MARKERS[i % len(MARKERS)], color="#515962",
                     lw=0, markersize=6, label=f"M{bits:,}") for i, bits in enumerate(widths)]
    fig.legend(handles=shapes, loc="outside lower center", ncols=len(shapes), frameon=False)
    suffix = f" · ECM-ELF {version}" if version else ""
    fig.suptitle(f"Edwards Stage1 NAF-window throughput{suffix}\n"
                 "CPU loop uses an in-process timer · GPU loop uses CUDA events\n"
                 "Default = 1 · separate CPU/GPU scales · open markers show best median · whiskers show range",
                 fontsize=12)
    save_figure(fig, output)


def plot_phases(cells, conditions, output, version):
    backends = backend_order(cells)
    widths = sorted({bits for _, bits in cells})
    fig, axes = plt.subplots(len(widths), len(backends), squeeze=False, layout="constrained",
                             figsize=(3.65 * len(backends) + 0.55, 2.5 * len(widths) + 0.75))

    for column, backend in enumerate(backends):
        label, default, _ = BACKENDS[backend]
        first_width = min(bits for name, bits in cells if name == backend)
        for row, bits in enumerate(widths):
            ax = axes[row, column]
            if (backend, bits) not in cells:
                ax.set_axis_off()
                continue
            windows = cells[backend, bits]
            b1, curves = conditions[backend, bits]
            x = sorted(windows)
            bottom = [0.0] * len(x)
            for metric, (phase, color) in PHASES.items():
                millis = [1000 * phase_median(windows[w], metric) for w in x]
                ax.bar(x, millis, bottom=bottom, label=phase, color=color)
                bottom = [a + b for a, b in zip(bottom, millis)]
            heading = f"{label}\n" if bits == first_width else ""
            ax.set_title(f"{heading}M{bits:,} · B1={b1:,} · {curves} curve(s)", fontsize=9)
            ax.set_xticks(x, [f"{w}{'*' if w == default else ''}" for w in x])
            ax.set_ylabel("ms per batch")
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis="y", alpha=0.16)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

    handles, labels = next(ax.get_legend_handles_labels() for ax in axes.flat if ax.has_data())
    fig.legend(handles, labels, loc="outside lower center", ncols=len(PHASES), frameon=False)
    suffix = f" · ECM-ELF {version}" if version else ""
    fig.suptitle(f"Edwards Stage1 elapsed time by phase{suffix}\n"
                 "Host elapsed-time phases · independent medians · * default NAF window", fontsize=12)
    save_figure(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Benchmark report JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    version, cells, conditions = load_samples(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_throughput(cells, args.output_dir / "stage1-edwards-naf-throughput.png", version)
    plot_phases(cells, conditions, args.output_dir / "stage1-edwards-naf-phases.png", version)
    for backend in backend_order(cells):
        default = BACKENDS[backend][1]
        for _, bits in sorted(key for key in cells if key[0] == backend):
            windows = cells[backend, bits]
            best = []
            prep = "dict+loop" if backend.startswith("rtx") else "prep+loop"
            metrics = [("loop", loop_metric(backend)), (prep, "prepared_loop_s"), ("wall", "wall_s")]
            for title, metric in metrics:
                w = fastest(windows, metric)
                ratio = phase_median(windows[default], metric) / phase_median(windows[w], metric)
                best.append(f"{title} w{w} ({ratio:.3f}x default)")
            wall_best = windows[fastest(windows, "wall_s")]
            breakdown = " / ".join(f"{1000 * phase_median(wall_best, phase):.1f}" for phase in PHASES)
            print(f"{backend} M{bits}: {', '.join(best)}; wall winner phase medians {breakdown} ms")


if __name__ == "__main__":
    main()
