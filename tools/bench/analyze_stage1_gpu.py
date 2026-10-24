"""Analyze completed Stage1 batches without launching benchmark jobs.

Uses the separately reported loop_gpu_ms CUDA event sum, excluding the NAF
dictionary, setup, coordinate readback, result checks and publication. CPU
comparison is a throughput proxy against the native Prime95 Stage1 timer.
Montgomery logs provide a global event interval including setup, which is
reported separately and must not be relabelled as a main-loop measurement.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
B1_BY_TPI = {4: 1_000_000, 8: 200_000, 16: 50_000, 32: 10_000}
DEFAULT_STUDY = "stage1_gpu_0_52_0_20261024/ed_4070ti_01"


def fields(text):
    return dict(re.findall(r"(\w+)=([^\s]+)", text))


def exact_modulus(exponent, factors):
    n = (1 << exponent) - 1
    for factor in factors:
        f = int(factor)
        if f <= 1 or n % f:
            raise ValueError(f"Invalid known factor for M{exponent}")
        n //= f
    return n


def parse_batches(text, references, sm_count):
    lines = text.splitlines()
    versions = re.findall(r"ECM-ELF ([0-9]+\.[0-9]+\.[0-9]+) \(Stage1\)", text)
    if len(set(versions)) != 1:
        raise ValueError("Expected one recorded program version")
    starts = [i for i, line in enumerate(lines) if "START: ECMSTAGE1=" in line]
    if not starts:
        raise ValueError("No ECMSTAGE1 tasks")
    rows, excluded = [], []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(lines)
        segment = "\n".join(lines[start:end])
        f = next(csv.reader([lines[start].split("ECMSTAGE1=", 1)[1]], strict=True))
        if len(f) != 9 or tuple(map(int, (f[0], f[1], f[3], f[6]))) != (1, 2, -1, 0):
            raise ValueError("Unexpected ECMSTAGE1 fields")
        exponent, b2, curves = int(f[2]), int(f[5]), int(f[7])
        save = re.fullmatch(r"(e|m0)_m(\d+)_([\d.eE+]+)\.save", f[4])
        if not save or int(save[2]) != exponent:
            raise ValueError("Invalid typed save name")
        if save[1] != "e":
            excluded.append(dict(exponent=exponent, start_line=start + 1,
                                 reason="Montgomery requires a separately qualified loop timer"))
            continue
        bound = float(save[3])
        if not math.isfinite(bound) or not bound.is_integer() or bound <= 0 or int(bound) != b2:
            raise ValueError("B1/B2 mismatch")
        b1 = int(bound)
        if exponent not in references:
            raise ValueError(f"No CPU reference for M{exponent}")
        cpu = references[exponent]
        n = exact_modulus(exponent, f[8].split(","))
        if n != exact_modulus(exponent, cpu["intended_known_factors"]):
            raise ValueError("CPU/GPU target integers differ")
        stage = re.findall(r"GPU Edwards Stage1: (.*)", segment)
        geometry = re.findall(r"GPU Edwards geometry: (.*)", segment)
        metrics = re.findall(r"GPU Edwards NAF: (.*)", segment)
        if not (len(stage) == len(geometry) == len(metrics) == 1):
            excluded.append(dict(exponent=exponent, start_line=start + 1,
                                 reason="missing or duplicated Edwards loop/geometry record"))
            continue
        stage, geometry, metrics = fields(stage[0]), fields(geometry[0]), fields(metrics[0])
        if int(stage["N_bits"]) != n.bit_length() or int(stage["curves"]) != curves:
            raise ValueError("Logged target width or batch size mismatch")
        tpi, tpb, container, blocks = (int(geometry[k]) for k in ("tpi", "tpb", "container_bits", "blocks"))
        if b1 != B1_BY_TPI.get(tpi) or container <= n.bit_length() or tpb % tpi:
            raise ValueError("TPI/B1 or Edwards geometry mismatch")
        if blocks != 2 * sm_count or curves != blocks * (tpb // tpi):
            raise ValueError("Expected two launched blocks per SM")
        if any(int(metrics[k]) != value for k, value in (("tpi", tpi), ("container_bits", container))):
            raise ValueError("NAF/geometry records disagree")
        window, doubling = int(stage["window"]), int(stage["doubling"])
        if window != 6 or doubling != 4:
            raise ValueError("Expected NAF w6 / HWCD")
        if int(metrics["dictionary_bytes"]) != curves * (container // 8) * 4 * (1 << (window - 2)):
            raise ValueError("Dictionary allocation disagrees with geometry")
        saves = re.findall(r"Edwards Stage1 save: (\S+) \((\d+) curves\)", segment)
        if saves != [(f[4], str(curves))] or segment.count("Stage1 results: ") != 1:
            excluded.append(dict(exponent=exponent, start_line=start + 1,
                                 reason="batch save/publication completion not recorded"))
            continue
        if re.search(r"\bERROR\b|CGBN.*error|Stage1 stopped|checkpoint restored", segment, re.I):
            raise ValueError("Failed/stopped/restored batch cannot be a complete timing sample")
        loop = float(metrics["loop_gpu_ms"]) / 1000
        dictionary = float(metrics["dictionary_gpu_ms"]) / 1000
        api = re.findall(r"GPU Edwards arithmetic: algorithm=naf api_seconds=([\d.]+)", segment)
        initialization = re.findall(r"GPU Edwards initialization: ([\d.]+) s", segment)
        if len(api) != 1 or len(initialization) != 1:
            raise ValueError("Missing API/preparation timer")
        api = float(api[0])
        if not all(math.isfinite(v) and v >= 0 for v in (loop, dictionary, api)) or loop <= 0 or api < loop:
            raise ValueError("Invalid or inconsistent timer")
        scale = 1_000_000 / b1
        measured_curve = loop / curves
        projected_curve = measured_curve * scale
        cpu_stats = cpu["measured"]
        if cpu_stats["n"] != 3 or cpu_stats["median"] <= 0:
            raise ValueError("Expected three completed CPU reference observations")
        row = dict(exponent=exponent, bits=n.bit_length(), model="ed", device=int(stage["device"]),
                   program_version=versions[0], samples=1, b1=b1, curves=curves, tpi=tpi, tpb=tpb,
                   container_bits=container, blocks=blocks, window=window, doubling=doubling,
                   columns=int(metrics["columns"]), launches=int(metrics["launches"]),
                   gpu_loop_batch_seconds=loop, dictionary_gpu_seconds=dictionary,
                   api_seconds=api, initialization_seconds=float(initialization[0]),
                   measured_gpu_seconds_per_curve=measured_curve, projection_to_b1_1e6=scale,
                   projected_gpu_seconds_per_curve_1e6=projected_curve,
                   cpu_seconds_1e6=cpu_stats["median"], cpu_min_seconds_1e6=cpu_stats["min"],
                   cpu_max_seconds_1e6=cpu_stats["max"], cpu_samples=cpu_stats["n"],
                   equivalent_cpu_cores=cpu_stats["median"] / projected_curve,
                   equivalent_cpu_min=cpu_stats["min"] / projected_curve,
                   equivalent_cpu_max=cpu_stats["max"] / projected_curve,
                   start_line=start + 1, end_line=end)
        rows.append(row)
    if not rows or len({r["exponent"] for r in rows}) != len(rows):
        raise ValueError("Empty or repeated GPU conditions; this is a single-sample study")
    if len({r["device"] for r in rows}) != 1:
        raise ValueError("Mixed GPU devices")
    return dict(rows=sorted(rows, key=lambda r: r["bits"]), excluded=excluded,
                program_version=versions[0], started_tasks=len(starts),
                queue_interrupted="STOP: queue interrupted" in text)


def parse_mont_batches(text, references, sm_count):
    """Select PARAM0 tasks, binding each one to its own preceding session header."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if "START: ECMSTAGE1=" in line]
    headers = [(i, m[1]) for i, line in enumerate(lines)
               if (m := re.search(r"ECM-ELF (\d+\.\d+\.\d+) \(Stage1\)", line))]
    rows, excluded = [], []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(lines)
        end = min([end] + [h for h, _ in headers if start < h < end])
        segment = "\n".join(lines[start:end])
        f = next(csv.reader([lines[start].split("ECMSTAGE1=", 1)[1]], strict=True))
        if len(f) != 9 or tuple(map(int, (f[0], f[1], f[3], f[6]))) != (1, 2, -1, 0):
            raise ValueError("Unexpected ECMSTAGE1 fields")
        exponent, b1, curves = int(f[2]), int(f[5]), int(f[7])
        save = re.fullmatch(r"(e|m0)_m(\d+)_([\d.eE+]+)\.save", f[4])
        if not save or int(save[2]) != exponent:
            raise ValueError("Invalid typed save name")
        if save[1] != "m0":
            excluded.append(dict(exponent=exponent, start_line=start + 1,
                                 reason="other_model; use the separate Edwards reference once"))
            continue
        versions = [v for h, v in headers if h < start]
        if not versions or float(save[3]) != b1 or b1 <= 0:
            raise ValueError("Missing session version or B1/save mismatch")
        if exponent not in references:
            raise ValueError(f"No CPU reference for M{exponent}")
        cpu = references[exponent]
        n = exact_modulus(exponent, f[8].split(","))
        if n != exact_modulus(exponent, cpu["intended_known_factors"]):
            raise ValueError("CPU/GPU target integers differ")
        if len(re.findall(r"queue_save: curve=mont param=0 B1=" + str(b1) + r"\b", segment)) != 1:
            raise ValueError("Expected PARAM0 task")
        if segment.count("GPU: parametrization = Suyama param0") != 1:
            raise ValueError("Expected native Suyama PARAM0 confirmation")
        geometry = re.findall(r"GPU: CGBN<(\d+),\s*(\d+)> kernel, N is (\d+) bits \((\d+) blocks x (\d+) threads\)", segment)
        devices = re.findall(r"GPU: will use device (\d+): (.+), compute capability [\d.]+, (\d+) MPs\.", segment)
        times = re.findall(r"GPU stage1 returned: (\d+) gputime=([\d.]+) ms", segment)
        saves = re.findall(r"Saved (\d+) Montgomery curve line\(s\) to (\S+)", segment)
        if len(geometry) != 1 or len(devices) != 1:
            raise ValueError("Missing or duplicated Montgomery geometry/device record")
        if len(times) != 1 or len(saves) != 1 or segment.count("Stage1 results: ") != 1:
            excluded.append(dict(exponent=exponent, start_line=start + 1,
                                 reason="batch timing/save/publication completion not recorded"))
            continue
        if int(times[0][0]) != 0 or re.search(r"\bERROR\b|CGBN.*error|Stage1 stopped|checkpoint restored", segment, re.I):
            raise ValueError("Failed/stopped/restored batch cannot be a complete timing sample")
        if int(saves[0][0]) != curves or saves[0][1].replace("\\", "/").rsplit("/", 1)[-1] != f[4]:
            raise ValueError("Completed save does not match task")
        tpi, container, bits, blocks, tpb = map(int, geometry[0])
        device, native_model, native_sms = devices[0]
        if bits != n.bit_length() or container < bits + 6 or tpb != 128 or b1 != B1_BY_TPI.get(tpi):
            raise ValueError("Montgomery target width, carry margin or TPI/B1 mismatch")
        if int(native_sms) != sm_count or blocks != 2 * sm_count or curves != blocks * (tpb // tpi):
            raise ValueError("Expected two launched blocks per SM")
        seconds = float(times[0][1]) / 1000
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid global timer")
        s = cpu["measured"]
        if s["n"] != 3 or s["median"] <= 0:
            raise ValueError("Expected three completed CPU reference observations")
        projected = seconds / curves * 1_000_000 / b1
        rows.append(dict(exponent=exponent, bits=bits, model="mont0", device=int(device),
                         native_gpu_model=native_model, program_version=versions[-1],
                         samples=1, b1=b1, curves=curves, tpi=tpi, tpb=tpb,
                         container_bits=container, blocks=blocks,
                         gpu_global_batch_seconds=seconds,
                         measured_gpu_seconds_per_curve=seconds / curves,
                         projection_to_b1_1e6=1_000_000 / b1,
                         projected_gpu_seconds_per_curve_1e6=projected,
                         cpu_seconds_1e6=s["median"], cpu_min_seconds_1e6=s["min"],
                         cpu_max_seconds_1e6=s["max"], cpu_samples=s["n"],
                         equivalent_cpu_cores=s["median"] / projected,
                         equivalent_cpu_min=s["min"] / projected,
                         equivalent_cpu_max=s["max"] / projected,
                         task_start=re.search(r"\[(.*?)\]", lines[start])[1],
                         result_publication=re.search(r"\[(.*?)\] Stage1 results:", segment)[1],
                         start_line=start + 1, end_line=end))
    if not rows or len({r["exponent"] for r in rows}) != len(rows):
        raise ValueError("Empty or repeated GPU conditions; this is a single-sample study")
    if len({(r["device"], r["native_gpu_model"], r["program_version"]) for r in rows}) != 1:
        raise ValueError("Mixed Montgomery devices or versions; analyze separately")
    return dict(rows=sorted(rows, key=lambda r: r["bits"]), excluded=excluded,
                program_version=rows[0]["program_version"], started_tasks=len(starts),
                selected_tasks=len(rows),
                queue_interrupted="STOP: queue interrupted" in "\n".join(lines[headers[-1][0]:]))


def make_plots(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    rows, gpu = report["rows"], report["gpu_model"]
    short = gpu.replace("NVIDIA GeForce ", "").replace(" GPU", "")
    x = list(range(len(rows)))
    labels = [f'{r["bits"]}\nM{r["exponent"]}' for r in rows]
    cpu_color, gpu_color = "#b26129", "#187e8b"
    prefix = f'stage1-{report["program_version"]}-{re.sub(r"[^a-z0-9]", "", short.lower())}'
    outputs = []

    def frame(fig, title):
        fig.suptitle(title, fontsize=15, fontweight="bold", y=.97)
        fig.text(.5, .906,
                 f'ECM-ELF {report["program_version"]} | {short} | NAF w6 / HWCD | 2 launched blocks/SM | GPU n=1, CPU n=3',
                 ha="center", fontsize=9, color="#48545e")
        fig.text(.07, .025,
                 "GPU: sum of main-loop CUDA event intervals; setup, dictionary, readback, GCD and publication excluded.\n"
                 "CPU: native Prime95 Stage1 at 4.0 GHz / one core (includes dictionary). Unequal timer boundaries: throughput proxy, not end-to-end speedup.\n"
                 "B1 normalization to 1e6 is linear extrapolation. GPU runtime clocks/power not recorded. Ranges show CPU min/max only; GPU variation unmeasured.",
                 fontsize=7.8, color="#4a545f", linespacing=1.5)

    def style(ax, scale):
        ax.set_yscale(scale)
        ax.set_xticks(x, labels, rotation=40, ha="right", fontsize=8.5)
        ax.set_xlabel("Exact cofactor bits / original Mersenne exponent (categorical spacing)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        if scale == "linear":
            ax.set_ylim(bottom=0)
        else:
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        for i in range(1, len(rows)):
            if rows[i]["tpi"] != rows[i - 1]["tpi"]:
                ax.axvline(i - .5, color="#bbc4c9", lw=.8, ls=":")

    def save(fig, suffix):
        for ext in ("png", "svg"):
            path = output / f"{prefix}-{suffix}.{ext}"
            fig.savefig(path, dpi=170)
            outputs.append(str(path.resolve()))
        plt.close(fig)

    for scale in ("linear", "log"):
        fig, ax = plt.subplots(figsize=(13.7, 6.4))
        values = [r["cpu_seconds_1e6"] for r in rows]
        ax.errorbar(x, values, yerr=[[r["cpu_seconds_1e6"] - r["cpu_min_seconds_1e6"] for r in rows],
                                    [r["cpu_max_seconds_1e6"] - r["cpu_seconds_1e6"] for r in rows]],
                    fmt="o-", color=cpu_color, capsize=3, lw=1.4,
                    label="Prime95 CPU: native Stage1 median")
        ax.plot(x, [r["projected_gpu_seconds_per_curve_1e6"] for r in rows], "s-",
                color=gpu_color, lw=1.4, label=f"{short}: GPU loop, B1 normalized")
        style(ax, scale)
        ax.set_ylabel(f"Seconds / curve at B1=1,000,000 ({scale} scale)")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        frame(fig, "Edwards Stage1: CPU reference versus GPU main-loop cost")
        fig.subplots_adjust(left=.07, right=.985, bottom=.32, top=.825)
        save(fig, f"times-{scale}")

        fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4), gridspec_kw={"width_ratios": [1, 3.4]})
        mix = report["equal_input_mix_cpu_cores"]
        axes[0].barh([0], [mix], color=gpu_color, height=.24)
        axes[0].set_xlim(0, mix * 1.3)
        axes[0].set_ylim(-.55, .55)
        axes[0].set_yticks([0], [short.replace(" ", "\n", 1)], fontsize=10)
        axes[0].set_xlabel("Equivalent CPU cores\n(linear scale)")
        axes[0].set_title("Equal 13-input mix", fontsize=11)
        axes[0].text(mix * 1.03, 0, f"{mix:.2f}", va="center", fontsize=13, fontweight="bold", color=gpu_color)
        axes[0].grid(axis="x", alpha=.15)
        axes[0].set_axisbelow(True)
        axes[0].text(.02, -.25, "Equal curve counts.\nB1=1e6, full-batch\namortized costs.\nRatio of summed costs.",
                     transform=axes[0].transAxes, fontsize=8.5, color="#4a545f", va="top")
        ax = axes[1]
        values = [r["equivalent_cpu_cores"] for r in rows]
        ax.errorbar(x, values, yerr=[[r["equivalent_cpu_cores"] - r["equivalent_cpu_min"] for r in rows],
                                    [r["equivalent_cpu_max"] - r["equivalent_cpu_cores"] for r in rows]],
                    fmt="o-", color=gpu_color, capsize=3, lw=1.3, label=short)
        for i, val in enumerate(values):
            ax.annotate(f"{val:.1f}", (i, rows[i]["equivalent_cpu_max"]),
                        textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        style(ax, scale)
        ax.set_ylim(top=ax.get_ylim()[1] * (1.3 if scale == "log" else 1.13))
        ax.set_ylabel(f"Equivalent 4.0 GHz CPU cores ({scale} scale)")
        ax.set_title("Per exact input: CPU seconds / normalized GPU loop seconds", fontsize=10.5)
        frame(fig, "GPU model versus CPU-core-equivalent throughput")
        fig.subplots_adjust(left=.08, right=.985, bottom=.36, top=.80, wspace=.30)
        save(fig, f"cpu-cores-{scale}")
    return outputs


def preserve(path, raw):
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"Frozen evidence differs: {path}; choose a new evidence directory")
    path.write_bytes(raw)


def core_value_labels(fig, ax, reports, styles):
    # Position in display units so the low values stay legible on linear axes.
    import matplotlib.patheffects as effects
    fig.canvas.draw()
    pixels_per_point = fig.dpi / 72
    lower = ax.bbox.y0 + 3 * pixels_per_point
    for i in range(len(reports[0]["rows"])):
        occupied_top = lower
        for report in reversed(reports):
            model = report["rows"][0]["model"]
            row = report["rows"][i]
            color = styles[model][2]
            xpos = i + (-.055 if model == "ed" else .055)
            value = row["equivalent_cpu_cores"]
            anchor = row["equivalent_cpu_max" if model == "ed" else "equivalent_cpu_min"]
            anchor_y = ax.transData.transform((xpos, anchor))[1]
            label_y = anchor_y + (7 if model == "ed" else -17) * pixels_per_point
            label_y = max(label_y, occupied_top)
            occupied_top = label_y + 13 * pixels_per_point
            point_y = ax.transData.transform((xpos, value))[1]
            ax.annotate(f"{value:.2f}", (xpos, value),
                        xytext=(0, (label_y - point_y) / pixels_per_point),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=color, zorder=5,
                        path_effects=[effects.withStroke(linewidth=2.5, foreground="white")])


def make_comparison_plots(reports, output):
    """Show the recorded boundaries side by side without subtracting setup estimates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    reports = sorted(reports, key=lambda r: r["rows"][0]["model"])
    for report in reports:
        report["rows"] = sorted(report["rows"], key=lambda r: r["bits"])
    base = reports[0]
    rows = base["rows"]
    keys = [(r["exponent"], r["bits"], r["b1"], r["curves"], r["tpi"]) for r in rows]
    for report in reports:
        if (report["gpu_model"], report["gpu_uuid"]) != (base["gpu_model"], base["gpu_uuid"]) or keys != [
                (r["exponent"], r["bits"], r["b1"], r["curves"], r["tpi"]) for r in report["rows"]]:
            raise ValueError("Comparison requires matching device, inputs, B1 and batch geometry")
    if len({r["rows"][0]["model"] for r in reports}) != len(reports):
        raise ValueError("Repeated model report")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    styles = {"ed": ("Edwards", "loop", "#187e8b", "#b26129", "o-"),
              "mont0": ("Montgomery PARAM0", "global incl. setup", "#7954a5", "#d89245", "s--")}
    short = base["gpu_model"].replace("NVIDIA GeForce ", "").replace(" GPU", "")
    x = list(range(len(rows)))
    labels = [f'{r["bits"]}\nM{r["exponent"]}' for r in rows]
    versions = "; ".join(f'{styles[r["rows"][0]["model"]][0]} {r["program_version"]}' for r in reports)
    models = "-".join("edwards" if r["rows"][0]["model"] == "ed" else "mont0" for r in reports)
    prefix = "stage1-" + models + "-" + re.sub(r"[^a-z0-9]", "", short.lower())
    outputs = []

    def frame(fig, title):
        fig.suptitle(title, fontsize=15, fontweight="bold", y=.97)
        fig.text(.5, .906, f'{short} | ECM-ELF {versions} | 2 launched blocks/SM | GPU n=1, CPU n=3 per model/input',
                 ha="center", fontsize=9, color="#48545e")
        fig.text(.07, .025,
                 "Edwards GPU = main-loop CUDA event sum; dictionary, setup, readback, GCD and publication excluded.\n"
                 "Montgomery GPU = global CUDA event interval, INCLUDING setup; excludes final readback, GCD and publication. No setup subtraction assumed.\n"
                 "CPU = Prime95 native Stage1, one core at 4.0 GHz. Unequal timer scopes: throughput proxies, not end-to-end or isolated algorithm speedups.\n"
                 "B1 normalized linearly to 1e6 (Montgomery setup also scales in this reference). GPU clocks/power unrecorded; whiskers show CPU min/max only.\n"
                 + ("Montgomery CPU and GPU logged execution intervals overlap; the runs were not isolated." if any(
                     r.get("cpu_gpu_logged_intervals_overlap") for r in reports) else "Runtime concurrent workloads were not recorded."),
                 fontsize=7.6, color="#4a545f", linespacing=1.5)

    def style(ax, scale):
        ax.set_yscale(scale)
        ax.set_xticks(x, labels, rotation=40, ha="right", fontsize=8.5)
        ax.set_xlabel("Exact cofactor bits / original Mersenne exponent (categorical spacing)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        if scale == "linear":
            ax.set_ylim(bottom=0)
        else:
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        for i in range(1, len(rows)):
            if rows[i]["tpi"] != rows[i - 1]["tpi"]:
                ax.axvline(i - .5, color="#bbc4c9", lw=.8, ls=":")

    def interval(ax, data, value, low, high, color, fmt, label, xs):
        ax.errorbar(xs, [r[value] for r in data],
                    yerr=[[r[value] - r[low] for r in data], [r[high] - r[value] for r in data]],
                    fmt=fmt, color=color, capsize=3, lw=1.4, markersize=4, label=label)

    def save(fig, suffix):
        for ext in ("png", "svg"):
            path = output / f"{prefix}-{suffix}.{ext}"
            fig.savefig(path, dpi=170)
            outputs.append(str(path.resolve()))
        plt.close(fig)

    for scale in ("linear", "log"):
        fig, ax = plt.subplots(figsize=(14.5, 7.2))
        for report in reports:
            name, scope, gpu_color, cpu_color, fmt = styles[report["rows"][0]["model"]]
            data = report["rows"]
            interval(ax, data, "cpu_seconds_1e6", "cpu_min_seconds_1e6", "cpu_max_seconds_1e6",
                     cpu_color, fmt, f"CPU {name}: native Stage1", x)
            ax.plot(x, [r["projected_gpu_seconds_per_curve_1e6"] for r in data], fmt,
                    color=gpu_color, lw=1.4, markersize=4, label=f"GPU {name}: {scope}")
        style(ax, scale)
        ax.set_ylabel(f"Seconds / curve at B1=1,000,000 ({scale} scale)")
        ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
        frame(fig, "Stage1: native CPU costs and " + ("two GPU timer scopes" if len(reports) == 2 else "the recorded GPU interval"))
        fig.subplots_adjust(left=.07, right=.985, bottom=.33, top=.825)
        save(fig, f"times-{scale}")

        fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), gridspec_kw={"width_ratios": [1.15, 3.4]})
        for j, report in enumerate(reports):
            name, scope, gpu_color, _, fmt = styles[report["rows"][0]["model"]]
            mix = report["equal_input_mix_cpu_cores"]
            pos = len(reports) - j - 1
            axes[0].barh(pos, mix, color=gpu_color, height=.32)
            axes[0].text(mix * 1.02, pos, f"{mix:.2f}", va="center", fontsize=11,
                         fontweight="bold", color=gpu_color)
            offset = -.055 if report["rows"][0]["model"] == "ed" else .055
            interval(axes[1], report["rows"], "equivalent_cpu_cores", "equivalent_cpu_min",
                     "equivalent_cpu_max", gpu_color, fmt,
                     f"{name}: GPU {scope}", [v + offset for v in x])
        max_mix = max(r["equal_input_mix_cpu_cores"] for r in reports)
        axes[0].set_xlim(0, max_mix * 1.3)
        axes[0].set_yticks(list(reversed(range(len(reports)))),
                           ["Edwards\nloop" if r["rows"][0]["model"] == "ed" else "Mont0\nglobal incl. setup" for r in reports], fontsize=9)
        axes[0].set_xlabel("Equivalent CPU cores\n(linear scale)")
        axes[0].set_title(f"{short}\nEqual {len(rows)}-input mix", fontsize=11)
        axes[0].grid(axis="x", alpha=.15)
        axes[0].set_axisbelow(True)
        axes[0].text(0, -.33, "Equal curve counts, B1=1e6.\nFull-batch amortized costs.\nRatio of summed costs;\neach model uses its own\nCPU reference and GPU scope.",
                     transform=axes[0].transAxes, fontsize=8.5, color="#4a545f", va="top")
        ax = axes[1]
        style(ax, scale)
        ax.set_ylabel(f"Equivalent 4.0 GHz CPU cores ({scale} scale)")
        ax.set_title("Per input: CPU native Stage1 / normalized GPU recorded interval", fontsize=10)
        ax.legend(loc="upper right", fontsize=8.5, frameon=False)
        frame(fig, "GPU model versus CPU-core-equivalent throughput (scopes differ)")
        fig.subplots_adjust(left=.10, right=.985, bottom=.37, top=.80, wspace=.30)
        core_value_labels(fig, ax, reports, styles)
        save(fig, f"cpu-cores-{scale}")
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, default=ROOT / "tools/benchmarks/stage1/screen_e.log")
    p.add_argument("--model", choices=("ed", "mont0"), default="ed")
    p.add_argument("--cpu", type=Path, default=ROOT / "data/benchmarks/stage1_prime95_20261024/summary.json")
    p.add_argument("--device-map", type=Path, required=True, help="nvidia-smi index,name,uuid,driver_version CSV")
    p.add_argument("--sm-count", type=int, required=True, help="SM count verified using device metadata")
    p.add_argument("--evidence", type=Path, default=ROOT / "data/experiments" / DEFAULT_STUDY)
    p.add_argument("--output", type=Path, default=ROOT / "data/benchmarks" / DEFAULT_STUDY)
    p.add_argument("--figures", type=Path, default=ROOT / "data/figures" / DEFAULT_STUDY)
    p.add_argument("--attachments", type=Path, help="Copy document PNGs into this directory")
    p.add_argument("--compare", type=Path, help="Other-model summary.json for matching comparison plots")
    a = p.parse_args()
    if a.sm_count <= 0:
        p.error("SM count must be positive")
    before = a.log.stat()
    raw = a.log.read_bytes()
    after = a.log.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        p.error("Source log changed during read; retry after this batch")
    if not raw.endswith((b"\n", b"\r")):
        p.error("Source log has an incomplete final line")
    cpu_raw = a.cpu.read_bytes()
    cpu = json.loads(cpu_raw.decode("utf-8-sig"))
    if cpu["cpu_cores"] != 1 or cpu["cpu_ghz"] != 4.0 or cpu["measured_b1"] != 1_000_000:
        p.error("Expected the confirmed 4 GHz single-core CPU reference")
    expected_cpu_model = "Edwards" if a.model == "ed" else "Montgomery"
    if cpu.get("curve_type", "Edwards") != expected_cpu_model:
        p.error("CPU curve type must match the GPU model")
    parser = parse_batches if a.model == "ed" else parse_mont_batches
    report = parser(raw.decode("utf-8-sig"), {r["exponent"]: r for r in cpu["rows"]}, a.sm_count)
    rows = report["rows"]
    device = rows[0]["device"]
    mapping_raw = a.device_map.read_bytes()
    mapping = {int(r[0]): [v.strip() for v in r[1:]]
               for r in csv.reader(mapping_raw.decode("utf-8-sig").splitlines()) if len(r) == 4}
    gpu, uuid, driver = mapping[device]
    if a.model == "mont0" and any(r["native_gpu_model"] != gpu for r in rows):
        p.error("Runtime GPU identity differs from the supplied device mapping")
    timer = "gpu_loop_batch_seconds" if a.model == "ed" else "gpu_global_batch_seconds"
    timing = ("sum of main-loop CUDA event intervals (loop_gpu_ms)" if a.model == "ed" else
              "global_start to final stop CUDA event interval (gputime); includes setup and host submission gaps, excludes final readback/GCD/publication")
    report.update(schema=1, analyzed_utc=datetime.now(timezone.utc).isoformat(),
                  source_log=str(a.log.resolve()), cpu_reference=str(a.cpu.resolve()),
                  gpu_model=gpu, gpu_uuid=uuid, driver_version=driver, sm_count=a.sm_count,
                  device_mapping_evidence=str(a.device_map.resolve()),
                  device_mapping_scope="queried during analysis; not runtime telemetry",
                  gpu_frequency_and_power="not recorded during the measurement",
                  cpu_ghz=cpu["cpu_ghz"], cpu_model=cpu["cpu_model"], cpu_version=cpu["program_version"],
                  cpu_input_identity=cpu.get("input_identity"),
                  model=a.model, target_b1=1_000_000, gpu_timing=timing,
                  cpu_timing=cpu["timing"],
                  interpretation="unequal timing scopes; core-equivalent throughput proxy, not end-to-end speedup",
                  correctness_scope="known-factor divisibility, logged GPU N matches CPU intended worktodo, native geometry/timers and completed save/publication; no endpoint oracle",
                  equal_input_mix_cpu_cores=sum(r["cpu_seconds_1e6"] for r in rows) /
                    sum(r["projected_gpu_seconds_per_curve_1e6"] for r in rows),
                  equal_input_mix_definition="equal curve counts across included inputs at B1=1e6, using full-batch-amortized costs; sum(CPU medians)/sum(normalized GPU per-curve seconds)",
                  cpu_median_sum_seconds=sum(r["cpu_seconds_1e6"] for r in rows),
                  normalized_gpu_per_curve_sum_seconds=sum(r["projected_gpu_seconds_per_curve_1e6"] for r in rows),
                  observed_interval_sum_seconds=sum(r[timer] for r in rows),
                  missing_cpu_reference_exponents=sorted(set(r["exponent"] for r in cpu["rows"]) - {r["exponent"] for r in rows}))
    if a.model == "mont0" and cpu.get("observation_span"):
        span = dict(start=min(r["task_start"] for r in rows),
                    end=max(r["result_publication"] for r in rows), timezone="not recorded")
        report["logged_execution_span"] = span
        report["cpu_gpu_logged_intervals_overlap"] = (span["start"] <= cpu["observation_span"]["end"] and
                                                       cpu["observation_span"]["start"] <= span["end"])
    for directory in (a.output, a.evidence, a.figures):
        directory.mkdir(parents=True, exist_ok=True)
    preserve(a.evidence / "screen.snapshot.log", raw)
    preserve(a.evidence / "cpu-reference.snapshot.json", cpu_raw)
    if (a.evidence / "device-map.csv").resolve() != a.device_map.resolve():
        preserve(a.evidence / "device-map.csv", mapping_raw)
    if a.model == "ed":
        report["observed_loop_sum_seconds"] = report["observed_interval_sum_seconds"]
    comparison = []
    if a.compare:
        comparison_raw = a.compare.read_bytes()
        comparison = [json.loads(comparison_raw.decode("utf-8-sig"))]
        preserve(a.evidence / "comparison-reference.snapshot.json", comparison_raw)
        report["comparison_reference"] = str(a.compare.resolve())
    report["figures"] = (make_plots(report, a.figures) if a.model == "ed" and not comparison else
                         make_comparison_plots([report] + comparison, a.figures))
    (a.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (a.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if a.attachments:
        a.attachments.mkdir(parents=True, exist_ok=True)
        for name in report["figures"]:
            if name.endswith(".png"):
                shutil.copy2(name, a.attachments / Path(name).name)
    print(json.dumps(dict(complete_batches=len(rows), missing_inputs=report["missing_cpu_reference_exponents"],
                          equivalent_cores_min=min(r["equivalent_cpu_cores"] for r in rows),
                          equivalent_cores_max=max(r["equivalent_cpu_cores"] for r in rows),
                          equal_input_mix_cpu_cores=report["equal_input_mix_cpu_cores"],
                          output=str(a.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
