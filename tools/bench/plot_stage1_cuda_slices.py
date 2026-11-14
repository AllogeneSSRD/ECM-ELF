#!/usr/bin/env python3
"""Plot the CUDA Stage1 slice-depth matrix and blocking-wait CPU measurements."""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


GPUS = ("RTX 4070 Ti", "RTX 4060 Laptop GPU")
ALGORITHMS = {"edwards": "Edwards NAF", "ladder": "Mont. ladder", "prac": "Mont. PRAC 7"}
GROUPS = [(gpu, name) for gpu in GPUS for name in ALGORITHMS]
DEPTHS = {8: ("8 片组尾", "#53667c"), 2: ("2 片滚动", "#087c76"), 3: ("3 片滚动", "#ab5907")}
WAIT_MODES = {
    "standard": ("关闭 · 同一构建", "#53667c"),
    "blocking": ("开启 · 同一构建", "#087c76"),
    "final": ("开启 · 0.63.5 复测", "#ab5907"),
}
SHORT_RUNS = {
    "default": ("3 片 · 默认值", "#ab5907", 3),
    "legacy": ("2 片 · 环境变量", "#087c76", 2),
}
# These pairs had long host-side pauses in the same interval. Use their full
# reruns in the stable-run comparison and retain all original reports.
HOST_STALL_PAIRS = {("RTX 4070 Ti", "prac", 1),
                    ("RTX 4060 Laptop GPU", "ladder", 1),
                    ("RTX 4060 Laptop GPU", "prac", 1)}
INK = "#152a3d"
GRID = "#e0e7ed"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_depths(base):
    rows = read_json(base / "rolling-summary.json")
    by_key = {(r["gpu"], r["algorithm"], r["depth"]): r for r in rows}
    expected = {(*group, depth) for group in GROUPS for depth in DEPTHS}
    if len(rows) != len(expected) or by_key.keys() != expected:
        raise ValueError("expected one rolling summary per GPU, algorithm and depth")
    for row in rows:
        if (row["wall_count"], row["trace_count"], row["stop_count"]) != (2, 1, 3):
            raise ValueError("unexpected rolling sample counts")
    return by_key


def read_wait_modes(base):
    modes = {}
    hashes = {}
    endpoints = {}
    for mode in WAIT_MODES:
        modes[mode] = {}
        hashes[mode] = set()
        for index, gpu in enumerate(GPUS):
            report = read_json(base / f"configured-{mode}-wall-gpu{index}" / "results.json")
            if (report["gpu"], report["bits"], report["b1"], report["curves"]) != (gpu, 4423, 100000, 128):
                raise ValueError("unexpected blocking-wait benchmark conditions")
            samples = report["samples"]
            if len(samples) != len(ALGORITHMS) or {r["key"][1] for r in samples} != ALGORITHMS.keys():
                raise ValueError("expected one blocking-wait sample per GPU and algorithm")
            for row in samples:
                group = (gpu, row["key"][1])
                expected_mode = {"low_cpu": int(mode != "standard"),
                                 "projection": "default" if mode == "final" else "explicit"}
                if row["key"][2] != 3 or row["config"] != expected_mode:
                    raise ValueError(f"unexpected wait settings for {mode}, {group}")
                if endpoints.setdefault(group, row["endpoint"]) != row["endpoint"]:
                    raise ValueError(f"save mismatch for {mode}, {group}")
                modes[mode][group] = row["cpu_cores"]
                hashes[mode].add(row["binary_sha256"])
    if (len(hashes["standard"]) != 1 or hashes["standard"] != hashes["blocking"]
            or len(hashes["final"]) != 1 or hashes["final"] == hashes["standard"]):
        raise ValueError("expected one shared binary for the paired test and a separate final build")
    return modes


def read_short_runs(base):
    runs = {}
    for mode, (_, _, depth) in SHORT_RUNS.items():
        report = read_json(base / f"configured-{mode}-wall-gpu0" / "results.json")
        if (report["gpu"], report["bits"], report["b1"], report["curves"]) != (GPUS[0], 4423, 5000, 8):
            raise ValueError("unexpected short-run benchmark conditions")
        samples = report["samples"]
        if len(samples) != len(ALGORITHMS) or {r["key"][1] for r in samples} != ALGORITHMS.keys():
            raise ValueError("expected one short-run sample per algorithm")
        if any(r["key"][2] != depth or r["config"] != {"low_cpu": 1, "projection": mode} for r in samples):
            raise ValueError(f"unexpected short-run settings for {mode}")
        runs[mode] = {r["key"][1]: r["cpu_cores"] for r in samples}
    return runs


def read_throughput_pairs(base):
    pairs = {}
    binaries = set()
    for index, gpu in enumerate(GPUS):
        report = read_json(base / f"configured-throughput-paired-gpu{index}" / "results.json")
        if (report["gpu"], report["bits"], report["b1"], report["curves"]) != (gpu, 4423, 100000, 128):
            raise ValueError("unexpected throughput benchmark conditions")
        samples = report["samples"]
        runs = {(row["algorithm"], row["pair"], row["low_cpu"]): row for row in samples}
        expected = {(name, pair, mode) for name in ALGORITHMS for pair in range(1, 4) for mode in (0, 1)}
        if len(samples) != len(expected) or runs.keys() != expected:
            raise ValueError("expected three complete throughput pairs per GPU and algorithm")
        supplements = read_json(base / f"configured-throughput-supplement-gpu{index}" / "results.json")
        if any(supplements[key] != report[key] for key in ("gpu", "bits", "b1", "curves")):
            raise ValueError("supplement conditions differ from the original run")
        extra = supplements["samples"]
        reruns = {(row["algorithm"], row["low_cpu"]): row for row in extra if row["pair"] == 4}
        expected_reruns = {(name, mode) for device, name, _ in HOST_STALL_PAIRS if device == gpu
                           for mode in (0, 1)}
        if len(extra) != len(expected_reruns) or reruns.keys() != expected_reruns:
            raise ValueError("expected a complete repeat for each affected pair")
        all_runs = samples + extra
        if any(row["key"][2] != 3 or row["config"] != {"low_cpu": row["low_cpu"], "projection": "default"}
               for row in all_runs):
            raise ValueError("throughput pairs must use default three-slice scheduling")
        binaries.update(row["binary_sha256"] for row in all_runs)
        for name in ALGORITHMS:
            endpoints = {row["endpoint"] for row in all_runs if row["algorithm"] == name}
            if len(endpoints) != 1:
                raise ValueError(f"throughput save mismatch for {gpu}, {name}")
            pairs[gpu, name] = [
                tuple((reruns[name, mode] if (gpu, name, p) in HOST_STALL_PAIRS
                       else runs[name, p, mode])["wall_s"] for mode in (0, 1))
                for p in range(1, 4)
            ]
    if len(binaries) != 1:
        raise ValueError("throughput pairs must use one binary")
    return pairs


def setup_axis(ax, grid_axis):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)


def heading(fig, title, condition):
    fig.text(0.065, 0.966, title, fontsize=21, fontweight="bold", va="top")
    fig.text(0.065, 0.919, condition, fontsize=11, color="#42566a", va="top")


def plot_depths(rows):
    fig, axs = plt.subplots(2, 2, figsize=(13.6, 9.2))
    fig.subplots_adjust(left=0.067, right=0.975, top=0.786, bottom=0.146, wspace=0.17, hspace=0.50)
    heading(fig, "CUDA Stage1 切片深度对照", "M4423  /  B1 = 100,000  /  128 条曲线  /  标准等待")
    panels = (
        ("整批进程墙钟", "秒", "wall_s", (8, 23)),
        ("片间隔 P95 至 P99", "毫秒", "gap_p95_ms", (0, 4)),
        ("整进程 CPU 占用", "逻辑核当量", "cpu_cores", (0.8, 1.04)),
        ("停止信号至进程退出", "秒", "stop_s", (0, 1.65)),
    )
    positions = (0, 1, 2, 3.45, 4.45, 5.45)
    for ax, (title, unit, metric, limits) in zip(axs.flat, panels):
        setup_axis(ax, "y")
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.set_ylabel(unit)
        ax.set_ylim(limits)
        ax.set_xlim(-0.42, 5.87)
        ax.axvline(2.725, color=GRID)
        ax.set_xticks(positions, [ALGORITHMS[algorithm] for _, algorithm in GROUPS], fontsize=9)
        for position, label in ((1, "RTX 4070 Ti"), (4.45, "RTX 4060 Laptop")):
            ax.text(position, -0.14, label, transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontweight="bold", fontsize=10, color="#42566a")
        for (depth, (_, color)), offset in zip(DEPTHS.items(), (-0.24, 0.0, 0.24)):
            for group, position in zip(GROUPS, positions):
                row = rows[(*group, depth)]
                value = row[metric]
                x = position + offset
                if metric == "gap_p95_ms":
                    p99 = row["gap_p99_ms"]
                    ax.vlines(x, value, p99, color=color, linewidth=1.5)
                    ax.plot(x, p99, "o", color=color, markerfacecolor="white", markersize=7, zorder=3)
                else:
                    lo, hi = row[f"{metric}_range"]
                    ax.errorbar(x, value, yerr=[[value - lo], [hi - value]], fmt="none", color=color,
                                capsize=2.5, linewidth=1.4)
                ax.plot(x, value, "o", color=color, markersize=4.6, zorder=4)
    legend = [Line2D([], [], marker="o", linestyle="none", color=color, label=label)
              for label, color in DEPTHS.values()]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.057, 0.887), ncol=3, frameon=False)
    fig.text(0.065, 0.041, "墙钟与 CPU 每格 2 次，停止每格 3 次。实心点为中位数，误差线为实测最小值至最大值。"
             "片间隔每格 1 次跟踪：● P95   ○ P99。", fontsize=10, color="#42566a")
    return fig


def plot_wait_modes(modes, short_runs):
    fig, (ax, short_ax) = plt.subplots(2, 1, figsize=(13.6, 9.6), height_ratios=(3.6, 1.15))
    fig.subplots_adjust(left=0.181, right=0.965, top=0.799, bottom=0.105, hspace=0.55)
    heading(fig, "低 CPU 模式的实测占用", "M4423  /  B1 = 100,000  /  128 条曲线  /  三片滚动")
    setup_axis(ax, "x")
    positions = (5.6, 4.6, 3.6, 1.9, 0.9, -0.1)
    offsets = (0.23, 0, -0.23)
    for (mode, (_, color)), offset in zip(WAIT_MODES.items(), offsets):
        for group, position in zip(GROUPS, positions):
            value = modes[mode][group]
            ax.barh(position + offset, value, height=0.185, color=color)
            label_x = value + 0.014 if mode == "standard" else 0.115
            ax.text(label_x, position + offset, f"{value:.3f}", va="center", fontsize=9, color=INK)
    ax.axvline(0.1, color="#8c4767", linestyle="--", linewidth=1.1)
    ax.axhline(2.74, color=GRID)
    ax.set(xlim=(0, 1.085), ylim=(-0.61, 6.45))
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_yticks(positions, [ALGORITHMS[algorithm] for _, algorithm in GROUPS], fontsize=10)
    for y, name in ((6.21, "RTX 4070 Ti"), (2.54, "RTX 4060 Laptop")):
        ax.text(-0.015, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontweight="bold", fontsize=11)
    legend = [Patch(color=color, label=label) for label, color in WAIT_MODES.values()]
    legend.append(Line2D([], [], color="#8c4767", linestyle="--", label="0.1 核目标"))
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.057, 0.871), ncol=4, frameon=False)
    setup_axis(short_ax, "x")
    short_ax.set_title("短批次 · RTX 4070 Ti / B1 = 5,000 / 8 条曲线 / 低 CPU 模式开启",
                       loc="left", fontweight="bold", pad=13)
    short_ax.axvline(0.1, color="#8c4767", linestyle="--", linewidth=1.1)
    short_legend = []
    for (mode, (label, color, _)), offset in zip(SHORT_RUNS.items(), (0.14, -0.14)):
        short_legend.append(Patch(color=color, label=label))
        for algorithm, position in zip(ALGORITHMS, (2, 1, 0)):
            value = short_runs[mode][algorithm]
            short_ax.barh(position + offset, value, height=0.23, color=color)
            short_ax.text(value + 0.014, position + offset, f"{value:.3f}", va="center", fontsize=9)
    short_ax.legend(handles=short_legend, loc="upper right", ncol=2, frameon=False)
    short_ax.set(xlim=(0, 1.085), ylim=(-0.5, 2.5), xlabel="进程 CPU 时间 / 进程墙钟（逻辑核当量）")
    short_ax.xaxis.set_major_locator(MultipleLocator(0.1))
    short_ax.set_yticks((2, 1, 0), ALGORITHMS.values(), fontsize=10)
    fig.text(0.065, 0.032, "每组每种设置各 1 次。上图最终版复测只设置 cuda_low_cpu = 1，沿用默认三片。",
             fontsize=10, color="#42566a")
    return fig


def plot_throughput(pairs):
    fig, (wall_ax, rate_ax) = plt.subplots(1, 2, figsize=(13.6, 6.8), width_ratios=(1.05, 1))
    fig.subplots_adjust(left=0.155, right=0.96, top=0.738, bottom=0.195, wspace=0.28)
    heading(fig, "低 CPU 模式的整批耗时与吞吐", "M4423  /  B1 = 100,000  /  128 条曲线  /  三片滚动")
    positions = (5, 4, 3, 1.4, 0.4, -0.6)
    for ax in (wall_ax, rate_ax):
        setup_axis(ax, "x")
        ax.set_ylim(-1.13, 5.95)
        ax.axhline(2.19, color=GRID)
    wall_ax.set_title("完整进程墙钟", loc="left", fontweight="bold", pad=10)
    wall_ax.set_xlabel("秒 / 批")
    wall_ax.set_yticks(positions, [ALGORITHMS[name] for _, name in GROUPS], fontsize=10)
    for y, name in ((5.75, "RTX 4070 Ti"), (2.08, "RTX 4060 Laptop")):
        wall_ax.text(-0.02, y, name, transform=wall_ax.get_yaxis_transform(),
                     ha="right", va="center", fontweight="bold")
    wall_ax.set_xlim(0, 1.17 * max(t for samples in pairs.values() for pair in samples for t in pair))
    rate_ax.set_title("开启后的配对吞吐变化", loc="left", fontweight="bold", pad=10)
    rate_ax.set_xlabel("吞吐变化（%）")
    rate_ax.set_yticks([])
    rate_ax.axvline(0, color="#7f8d9a", linestyle="--", linewidth=1)
    colors = (WAIT_MODES["standard"][1], WAIT_MODES["blocking"][1])
    deltas = []
    for group, y in zip(GROUPS, positions):
        samples = pairs[group]
        for times, color, offset in zip(zip(*samples), colors, (0.15, -0.15)):
            center = median(times)
            wall_ax.hlines(y + offset, min(times), max(times), color=color, linewidth=2)
            wall_ax.plot(center, y + offset, "o", color=color, markersize=5)
            wall_ax.annotate(f"{center:.3f}", (max(times), y + offset), xytext=(6, 0),
                             textcoords="offset points", va="center", fontsize=9)
        changes = [100 * (off / on - 1) for off, on in samples]
        deltas.extend(changes)
        rate_ax.hlines(y, min(changes), max(changes), color=colors[1], linewidth=2)
        rate_ax.plot(changes, [y] * len(changes), "o", color=colors[1], markersize=5, markerfacecolor="white")
        center = median(changes)
        rate_ax.plot(center, y, "o", color=colors[1], markersize=5)
        rate_ax.annotate(f"{center:+.2f}%", (max(changes), y), xytext=(7, 0),
                         textcoords="offset points", va="center", fontsize=9)
    span = max(0.5, max(deltas) - min(deltas))
    rate_ax.set_xlim(min(0, min(deltas)) - 0.11 * span, max(0, max(deltas)) + 0.28 * span)
    legend = [Line2D([], [], marker="o", linestyle="none", color=color, label=label)
              for color, label in zip(colors, ("低 CPU 关闭", "低 CPU 开启"))]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.057, 0.86), ncol=2, frameon=False)
    fig.text(0.065, 0.066, "每格 3 对，同一构建交错切换。实心点为中位数，横线为最小值至最大值。"
             "吞吐图空心点为各对实测。", fontsize=10, color="#42566a")
    fig.text(0.065, 0.034, "主机长停顿期间的三对完整重测，原始长尾见正文。",
             fontsize=10, color="#42566a")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="M4423 benchmark directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for PNG and SVG figures")
    args = parser.parse_args()
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((font for font in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei")
                         if font in available_fonts), None)
    if chinese_font is None:
        parser.error("install Microsoft YaHei, Noto Sans CJK SC or SimHei to render the figure labels")
    plt.rcParams.update({
        "font.family": [chinese_font, "DejaVu Sans"],
        "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": "#42566a", "ytick.color": "#42566a",
        "svg.fonttype": "path", "svg.hashsalt": "stage1-cuda-slices",
    })
    depths = read_depths(args.input_dir)
    modes = read_wait_modes(args.input_dir)
    short_runs = read_short_runs(args.input_dir)
    pairs = read_throughput_pairs(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = (("stage1-cuda-slice-depths", plot_depths(depths)),
               ("stage1-cuda-low-cpu", plot_wait_modes(modes, short_runs)),
               ("stage1-cuda-low-cpu-throughput", plot_throughput(pairs)))
    for name, fig in figures:
        fig.savefig(args.output_dir / f"{name}.png", dpi=160)
        fig.savefig(args.output_dir / f"{name}.svg", metadata={"Date": None})
        plt.close(fig)


if __name__ == "__main__":
    main()
