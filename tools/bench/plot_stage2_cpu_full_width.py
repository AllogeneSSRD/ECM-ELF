"""Plot the Edwards full-width matrix with frozen references and an optional current build."""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from bench_cpu_stage2_full_width import AFFINITY


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data/benchmarks"
CPU = ROOT / "data/experiments/stage2_cpu_full_0661"
QUEUE = BENCH / "stage2_parallel_0620/small_b2_comparison"
ATTACHMENTS = ROOT / "docs/performance/attachments"
FIGURES = ROOT / "data/figures/stage2_cpu_full_0661"
WIDTHS = (318, 383, 1939, 2726, 3600, 4667, 5872, 6797, 7995, 10232, 12285, 14328, 16272)
# Device, CSV column, legend label, colour, marker, line.
SERIES = (
    ("RTX 4070 Ti", "cuda4070", "0.62.3   Geometry selection", "#19814b", "D", "--"),
    ("RTX 4060 Laptop", "serial", "0.62.0   Serial queue", "#b02e42", "^", "--"),
    ("RTX 4060 Laptop", "dual", "0.62.0   Two-slot queue", "#d06026", "v", "--"),
    ("RTX 4060 Laptop", "auto", "0.57.9   Selected backend", "#693499", "o", "--"),
    ("RTX 4060 Laptop", "scalar", "0.57.9   Scalar backend", "#9166ac", "+", "--"),
    ("RTX 4060 Laptop", "layout", "0.54.7   Tuned layout", "#b59ac8", "x", "--"),
    ("Ryzen AI 9 HX 370", "cpu1", "0.66.1   GWpoly 1T", "#1856a4", "o", "-"),
    ("Ryzen AI 9 HX 370", "cpu4", "0.66.1   GWpoly 4T", "#008391", "s", "-"),
    ("Ryzen AI 9 HX 370", "prime", "Prime95 31.4b05   4T", "#212121", "^", ":"),
)


def read_json(path):
    return json.loads(path.read_bytes())


def index(rows):
    indexed = {int(row["bits"]): row for row in rows}
    if len(rows) != len(WIDTHS) or set(indexed) != set(WIDTHS):
        raise ValueError("Expected exactly one row per reference width")
    return indexed


def timing(row, key, center, low, high):
    if not all(math.isfinite(v) for v in (center, low, high)) or not 0 < low <= center <= high:
        raise ValueError(f"Invalid timing range at {row['bits']} bits: {key}")
    row.update({key: center, key + "_min": low, key + "_max": high})


def samples(row, key, seconds):
    timing(row, key, median(seconds), min(seconds), max(seconds))


def collect():
    serial = read_json(QUEUE / "serial.json")
    dual = read_json(QUEUE / "summary.json")
    previous = read_json(BENCH / "stage2_giant_auto_20261101/matrix01/summary.json")
    report = read_json(CPU / "summary.json")
    if not report.get("complete") or len(report["runs"]) != 78:
        raise ValueError("CPU matrix is incomplete")
    if "0.66.1 CPU Stage2" not in report["version"] or "build=7444a53678a1e166" not in report["version"]:
        raise ValueError("Unexpected CPU executable or GWNUM DLL")
    if (serial["version"], dual["version"], previous["version"]) != ("0.62.0", "0.62.0", "0.57.9"):
        raise ValueError("Unexpected historical benchmark version")
    if hashlib.sha256((QUEUE / "serial.json").read_bytes()).hexdigest() != report["reference"]:
        raise ValueError("CPU collection referenced a different serial study")
    if not previous["complete"] or any(s["failures"] or s["repeats"] != 2 for s in (serial, dual)):
        raise ValueError("Historical benchmarks are incomplete")
    single, paired, original = index(serial["rows"]), index(dual["rows"]), index(previous["best"])
    with (BENCH / "stage2_4070ti_0623/by-width.csv").open(encoding="utf-8-sig", newline="") as stream:
        ti = index([r for r in csv.DictReader(stream) if r["bound"] == "small"])

    rows = []
    for bits in WIDTHS:
        base = single[bits]
        row = {k: base[k] for k in ("bits", "exponent", "B1", "B2")}
        for source in (paired[bits], original[bits], ti[bits]):
            if any(int(source[k]) != row[k] for k in row):
                raise ValueError(f"Input mismatch at {bits} bits")
        for key, study in (("serial", base), ("dual", paired[bits])):
            batch = study["samples"]
            if len(batch) != 2 or any(s["curves"] != 2 for s in batch):
                raise ValueError(f"Unexpected queue batch at {bits} bits")
            samples(row, key, [s["wall_seconds_per_curve"] for s in batch])
        for key in ("auto", "scalar"):
            v = original[bits]["values"][key]
            timing(row, key, v["seconds"], v["minimum"], v["maximum"])
        for key, column in (("layout", "layout_reference"), ("prime", "prime95_seconds")):
            samples(row, key, [original[bits][column]])
        timing(row, "cuda4070", *(float(ti[bits][f"selected_{s}_engine_s"])
                                  for s in ("median", "min", "max")))
        saved = hashlib.sha256((CPU / f"{bits}.save").read_bytes()).hexdigest()
        for threads, mask in ((1, "0x20"), (4, "0xaa")):
            runs = [r for r in report["runs"] if (r["bits"], r["threads"]) == (bits, threads)]
            if sorted(r["repeat"] for r in runs) != [1, 2, 3]:
                raise ValueError(f"Missing CPU repeat at {bits} bits, {threads}T")
            for run in runs:
                if (run["affinity"], run["save_sha256"], run["exponent"], run["b1"], run["b2"]) != (
                    mask, saved, row["exponent"], row["B1"], row["B2"]
                ):
                    raise ValueError(f"CPU sample identity mismatch at {bits} bits")
            samples(row, f"cpu{threads}", [r["seconds"] for r in runs])
        rows.append(row)
    return rows


def add_current(rows, report):
    version = re.search(r"^ECM-ELF (\d+\.\d+\.\d+) CPU Stage2\b", report["version"])
    repeats = report["repeats"]
    if (not version or report["matrix"] != "full" or not report["complete"] or
            repeats < 1 or len(report["runs"]) != 2 * len(rows) * repeats):
        raise ValueError("Expected a complete, versioned full-width CPU matrix")
    for row in rows:
        saved = hashlib.sha256((CPU / f"{row['bits']}.save").read_bytes()).hexdigest()
        for threads, cpus in AFFINITY.items():
            runs = [r for r in report["runs"] if (r["bits"], r["threads"]) == (row["bits"], threads)]
            if sorted(r["repeat"] for r in runs) != list(range(1, repeats + 1)):
                raise ValueError(f"Missing current CPU repeat at {row['bits']} bits, {threads}T")
            for run in runs:
                if (tuple(run["cpus"]), run["save_sha256"], run["exponent"], run["b1"], run["b2"]) != (
                    cpus, saved, row["exponent"], row["B1"], row["B2"]
                ):
                    raise ValueError(f"Current CPU sample identity mismatch at {row['bits']} bits")
                if not run["dll"] or run["dll"] not in report["version"]:
                    raise ValueError(f"Current CPU DLL mismatch at {row['bits']} bits")
            samples(row, f"current{threads}", [r["seconds"] for r in runs])
    return version[1]


def current_series(version):
    if not version:
        return SERIES
    return SERIES[:6] + (
        ("Ryzen AI 9 HX 370", "cpu1", "0.66.1   GWpoly 1T", "#6489bb", "o", "--"),
        ("Ryzen AI 9 HX 370", "cpu4", "0.66.1   GWpoly 4T", "#63a4ac", "s", "--"),
        ("Ryzen AI 9 HX 370", "current1", f"{version}   GWpoly 1T", "#1856a4", "o", "-"),
        ("Ryzen AI 9 HX 370", "current4", f"{version}   GWpoly 4T", "#008391", "s", "-"),
        SERIES[-1],
    )


def plot(rows, scale, figures, version):
    fig, (zoom, full, legend) = plt.subplots(
        1, 3, figsize=(14.5, 5.7), gridspec_kw={"width_ratios": [1.5, 2.4, 1.05]},
        layout="constrained"
    )
    legend.axis("off")
    handles, labels = [], []
    last_device = None
    for device, key, label, color, marker, style in current_series(version):
        if device != last_device:
            handles.append(Line2D([], [], linestyle="none"))
            labels.append(device)
            last_device = device
        for ax, data in ((zoom, [r for r in rows if r["bits"] <= 4667]), (full, rows)):
            x = [r["bits"] for r in data]
            line, = ax.plot(x, [r[key] for r in data], color=color, marker=marker, ls=style, ms=4,
                            lw=2 if style == "-" else 1.45,
                            markerfacecolor="white" if key == "prime" else color)
            ax.fill_between(x, [r[key + "_min"] for r in data], [r[key + "_max"] for r in data],
                            color=color, alpha=.10, linewidth=0)
        handles.append(line)
        labels.append(label)
    box = legend.legend(handles, labels, loc="center left", frameon=False, fontsize=9,
                        handlelength=2.5, labelspacing=.8, borderaxespad=0)
    for handle, label in zip(handles, box.get_texts()):
        if handle.get_linestyle() == "None":
            label.set_weight("bold")
    for ax, limit, tick, title in ((zoom, 4850, 1000, "318–4,667 bits"),
                                   (full, 16600, 2000, "Full width range")):
        ax.set(xlabel="Cofactor N (bits)", yscale=scale, xlim=(0, limit), title=title)
        if scale == "linear":
            ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MultipleLocator(tick))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="both", which="major", alpha=.24)
        ax.set_axisbelow(True)
    zoom.set_ylabel("Seconds per curve")
    fig.suptitle("Edwards Stage 2   |   13 cofactors, small B2", fontweight="bold")
    suffix = f"-{version}" if version else ""
    output = figures / f"stage2-edwards-cpu-gpu-full{suffix}-{scale}.png"
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".svg"))
    shutil.copyfile(output, ATTACHMENTS / output.name)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, help="Complete summary.json from bench_cpu_stage2_full_width.py")
    args = parser.parse_args()
    rows = collect()
    version = add_current(rows, read_json(args.current)) if args.current else None
    study = f"stage2_cpu_full_{version.replace('.', '')}" if version else CPU.name
    out = BENCH / study
    out.mkdir(parents=True, exist_ok=True)
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    figures = FIGURES.parent / study
    figures.mkdir(parents=True, exist_ok=True)
    for scale in ("log", "linear"):
        plot(rows, scale, figures, version)


if __name__ == "__main__":
    main()
