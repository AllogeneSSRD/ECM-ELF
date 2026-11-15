#!/usr/bin/env python3
"""Plot Stage2 version throughput for four fixed inputs within an actual B2 range."""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from statistics import median

from plot_version_throughput import PRIME95, WIDTHS, plot_version_panels, version_key


B2_MIN, B2_MAX = 26_000_000_000, 27_015_978_990
# Bounds for studies whose reports omit a bound shared by all runs.
REFERENCE_B1, REFERENCE_B2 = 50_000, 27_015_978_990
EXPONENT_BY_BITS = {bits: p for p, (_, bits) in WIDTHS.items()}
GPU0, GPU1 = "4070 Ti", "4060 Laptop"
ENGINE, QUEUE = "complete engine", "two-curve queue wall"
COLORS = {GPU0: "#14804a", GPU1: "#943e78", PRIME95: "#181818"}


@dataclass(frozen=True)
class Measurement:
    bits: int
    b1: int
    b2: int


@dataclass(frozen=True)
class Sample:
    exponent: int
    bits: int
    b1: int
    b2: int
    version: str
    series: str
    seconds_per_curve: float
    repeats: int
    variant: str
    timing: str
    source: str

    @property
    def throughput(self):
        return 1 / self.seconds_per_curve


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def measurement(row, width="bits", *, b1="B1", b2="B2"):
    bits = int(row[width])
    actual_b2 = int(row[b2])
    if bits not in EXPONENT_BY_BITS or not B2_MIN <= actual_b2 <= B2_MAX:
        return None
    p = EXPONENT_BY_BITS[bits]
    if int(row.get("exponent", p)) != p:
        raise ValueError(f"Unexpected input at {bits} bits")
    return Measurement(bits, int(row[b1]), actual_b2)


def sample(run, version, seconds, repeats, variant, path, *, device=GPU1, timing=ENGINE):
    return Sample(EXPONENT_BY_BITS[run.bits], run.bits, run.b1, run.b2, version, device,
                  float(seconds), int(repeats), variant, timing, path.as_posix())


def early_samples(bench):
    path = bench / "edwards_stage2_opt_20261017/final_03/comparison.csv"
    for row in read_csv(path):
        if (run := measurement(row, b1="b1", b2="b2")) is not None:
            route = "carrier" if int(row["carrier"]) else "ordinary"
            yield sample(run, row["version"], row["median"], row["n"], f"{row['model']}, {route}", path)

    path = bench / "gpu_stage2_original_20261018/final01/timings.csv"
    versions = {"baseline": "0.36.0", "candidate": "0.36.1"}
    for row in read_csv(path):
        if row["kind"] == "NF" and (run := measurement(row, b2="actual_B2")) is not None:
            yield sample(run, versions[row["arm"]], row["total_median_s"], row["n"], row["cell"], path)

    for version, directory in (("0.39.0", "stage2_d_search_0_39_0/default01"),
                               ("0.39.2", "stage2_d_search_0_39_2/paired01")):
        path = bench / directory / "cells.csv"
        for row in read_csv(path):
            if (run := measurement(row)) is not None:
                yield sample(run, version, row["candidate_seconds"], 1, row["engine"], path)

    # STAGE2_HOST_WAIT.md records B1=50000 for the complete-engine wait A/B.
    path = bench / "stage2_host_wait_final_20261025/summary.csv"
    for row in read_csv(path):
        if (run := measurement(dict(row, B1=REFERENCE_B1))) is not None:
            yield sample(run, "0.52.4", row["full_seconds"], row["samples"], row["variant"], path)


def engine_samples(bench):
    path = bench / "stage2_v0_38_4_20261018/final/cells.csv"
    for row in read_csv(path):
        if (run := measurement(row)) is not None:
            yield sample(run, row["version"], row["seconds"], row["n"], row["engine"], path)
            if row["engine"] == "gpu" and (run.b1, run.b2) == (REFERENCE_B1, REFERENCE_B2):
                yield sample(run, "", row["prime95_seconds"], 1, "Prime95, 4 threads", path, device=PRIME95,
                             timing="Prime95 init + main + GCD")

    path = bench / "stage2_runtime_v0_39_3_20261019/final/runtime_cells.csv"
    for row in read_csv(path):
        if (run := measurement(row)) is not None:
            yield sample(run, row["version"], row["seconds"], row["n"], "best eligible tuning mean", path)

    path = bench / "stage2_layout_matrix_20261031/low_final/cells.csv"
    for row in read_csv(path):
        if (run := measurement(row)) is not None:
            tuned = json.loads(row["tuned"])
            yield sample(run, "0.54.7", tuned["seconds"], tuned["n"], "independent tuned mean", path)

    # Both fixed Stage1 bounds are documented in STAGE2_PREPARATION.md.
    path = bench / "stage2_point_policy_20261031_03/summary.json"
    for table, bits in (("tail_abba02", 7995), ("wide_abba01", 16272)):
        for row in read_json(path)["tables"][table]:
            run = measurement(dict(row, bits=bits, B1=REFERENCE_B1), b2="actual_B2")
            if run is not None:
                yield sample(run, "0.55.0", row["mean"], row["n"], row["mode"], path)

    path = bench / "stage2_point_matrix_20261031/final01/summary.json"
    for row in read_json(path)["rows"]:
        if (run := measurement(dict(row, B1=REFERENCE_B1), "target_bits")) is not None:
            for mode in ("off", "on"):
                variant = f"{'carrier' if row['carrier'] else 'ordinary'}, tail balance {mode}"
                yield sample(run, "0.56.0", row[f"{mode}_seconds"], 1, variant, path)

    path = bench / "stage2_giant_auto_20261101/matrix01/cells.csv"
    for row in read_csv(path):
        if (run := measurement(dict(row, B1=REFERENCE_B1))) is not None:
            for mode in ("scalar", "auto"):
                yield sample(run, "0.57.9", row[f"{mode}_seconds"], row["n"], f"{row['family']}, {mode}", path)

    path = bench / "stage2_seed_coop_20261102/final02/cells.csv"
    for row in read_csv(path):
        if (run := measurement(dict(row, B1=REFERENCE_B1), "target_bits")) is not None:
            for mode in ("scalar", "coop"):
                yield sample(run, "0.57.11", row[f"{mode}_seconds"], 1, f"seed {mode} ({row['carrier']})", path)


def point_samples(data):
    bench = data / "benchmarks"
    # Fixed 7995-bit control from STAGE2_PREPARATION.md.
    path = data / "experiments/stage2_policy_default_20261101/fixed_ab02/analysis.json"
    report = read_json(path)
    if not report["complete"]:
        raise ValueError(f"Incomplete Stage2 tail balance control: {path}")
    run = measurement({"bits": 7995, "exponent": 8011, "B1": REFERENCE_B1, "B2": REFERENCE_B2})
    if run is not None:
        for row in report["results"]:
            for mode in ("off", "on"):
                yield sample(run, "0.57.0", row[f"mean_{mode}"], report["samples_per_side"],
                             f"{row['geometry']}, balance {mode}", path)

    # STAGE2_BIT_SCALING.md records the bounds and versions of these qualified runs.
    studies = (
        ("stage2_point_square_20261101/qualified", "0.56.4", "0.57.1"),
        ("stage2_point_coop_20261101/curves_final", "0.57.2", "0.57.3"),
        ("stage2_point_register_20261101/curves_qualified", "0.57.3", "0.57.4"),
        ("stage2_baby_coop_20261101/curves_qualified", "0.57.4", "0.57.5"),
        ("stage2_baby_policy_20261101/curves_final", "0.57.5", "0.57.6"),
    )
    for directory, stock, candidate in studies:
        path = bench / directory / "summary.json"
        report = read_json(path)
        if not report["complete"]:
            raise ValueError(f"Incomplete Stage2 point matrix: {path}")
        for row in report["cells"]:
            bounds = row.get("input", {"B1": REFERENCE_B1, "B2": REFERENCE_B2})
            if (run := measurement(dict(bounds, bits=row["bits"]))) is not None:
                for mode, stats in row["variants"].items():
                    version = stock if mode == "stock" else candidate
                    yield sample(run, version, stats["median_seconds"], stats["n"], f"{row['family']}, {mode}", path)

    # STAGE2_S4_REDUCTION.md records the fixed B1 of the S4 complete-curve control.
    path = bench / "stage2_s4_20261102/curves_final/cells.csv"
    for row in read_csv(path):
        if (run := measurement(dict(row, B1=REFERENCE_B1), "target_bits")) is not None:
            for mode in ("scalar", "coop"):
                yield sample(run, "0.57.13", row[f"{mode}_seconds"], row["samples_per_backend"],
                             f"S4 {mode} ({row['carrier']})", path)

    path = data / "experiments/stage2_s4_20261102/default_guard01/summary.json"
    report = read_json(path)
    if not report["complete"]:
        raise ValueError(f"Incomplete Stage2 S4 control: {path}")
    versions = {"old": "0.57.12", "new": "0.57.13"}
    for row in report["runs"]:
        if row["cell"] == "m16319_n16272_s26_ordinary":
            run = measurement({"bits": 16272, "exponent": 16319, "B1": REFERENCE_B1, "B2": REFERENCE_B2})
            if run is not None:
                yield sample(run, versions[row["version"]], row["seconds"], 1, "fixed-D ordinary S4 control", path)

    path = data / "experiments/stage2_carry_prepare_20261102/matrix02/summary.json"
    report = read_json(path)
    if not report["complete"] or report["failures"]:
        raise ValueError(f"Incomplete Stage2 carry matrix: {path}")
    cases = {row["id"]: row for row in report["cases"]}
    cells = {row["id"]: row for row in report["cells"]}
    for timed in report["runs"]:
        cell = cells[timed["cell"]]
        row = dict(cases[cell["case"]], B2=cell["B2"])
        if (run := measurement(row, b1="b1")) is not None:
            shift = "carrier" if cell["carrier"] else "ordinary"
            yield sample(run, "0.57.15", timed["seconds"], 1, f"{shift}, {timed['mode']}", path)


def recent_samples(data):
    bench = data / "benchmarks"
    pair = bench / "stage2_parallel_0620/small_b2_comparison"
    for name, mode in (("serial.json", "serial"), ("summary.json", "two slots")):
        path = pair / name
        report = read_json(path)
        if report["failures"]:
            raise ValueError(f"Incomplete Stage2 queue matrix: {path}")
        for row in report["rows"]:
            if (run := measurement(row)) is not None:
                samples = row["samples"]
                if any(s["curves"] != 2 or s["device"]["index"] != 1 for s in samples):
                    raise ValueError(f"Unexpected queue sample at {run.bits} bits")
                seconds = median(s["wall_seconds"] / s["curves"] for s in samples)
                yield sample(run, report["version"], seconds, len(samples), mode, path, timing=QUEUE)

    # This published short-window A/B used the matching bound only at 14328 and 16272 bits.
    path = data / "experiments/stage2_queue_pipeline_20261112/workspace_5/summary.json"
    for row in read_json(path):
        bits, label = row["bits"], row["label"]
        if bits not in EXPONENT_BY_BITS:
            continue
        console = path.parent / f"{bits}_{label}/console.log"
        log = console.read_text(encoding="utf-8")
        bounds = re.findall(r"^real_shape: B1=(\d+) B2=(\d+)\b", log, re.MULTILINE)
        if row["curves"] != 2 or not log.startswith("ECM-ELF 0.62.1 ") or len(bounds) != 2 or len(set(bounds)) != 1:
            raise ValueError(f"Unexpected Stage2 workspace trial: {console}")
        b1, b2 = bounds[0]
        if log.count(f"B1={b1} checksum=verified") < 2:
            raise ValueError(f"Unverified Stage1 points in {console}")
        if (run := measurement(dict(row, B1=b1, B2=b2))) is not None:
            yield sample(run, "0.62.1", row["wall_seconds"] / row["curves"], 1, label, path, timing=QUEUE)

    geometry = bench / "stage2_4070ti_0623"
    path = geometry / "by-width.csv"
    matched = {}
    for row in read_csv(path):
        if row["bound"] == "small" and (run := measurement(row)) is not None:
            matched[run.bits] = run
            yield sample(run, "0.62.3", row["selected_median_engine_s"], row["selected_samples"],
                         "selected D/P and layout", path, device=GPU0)

    path = geometry / "parallel.csv"
    for row in read_csv(path):
        bits = int(row["bits"])
        if row["bound"] == "small" and row["status"] == "checked" and bits in matched:
            for mode in ("serial", "parallel"):
                yield sample(matched[bits], "0.62.3", row[f"{mode}_seconds"], 2, mode, path, device=GPU0, timing=QUEUE)


def select(samples):
    best = {}
    for row in samples:
        if not math.isfinite(row.seconds_per_curve) or row.seconds_per_curve <= 0 or row.repeats < 1:
            raise ValueError(f"Invalid Stage2 result: {row}")
        key = row.exponent, row.version, row.series
        if key not in best or row.seconds_per_curve < best[key].seconds_per_curve:
            best[key] = row
    return sorted(best.values(), key=lambda row: (row.exponent, version_key(row.version), row.series))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    data = args.data
    bench = data / "benchmarks"
    selected = select([*early_samples(bench), *engine_samples(bench), *point_samples(data), *recent_samples(data)])
    figure = data / "figures/stage2_version_throughput/stage2-version-throughput.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    plot_version_panels(
        selected, figure, title="Stage2 throughput by version", ylabel="Completed curves/s",
        colors=COLORS, hatch=QUEUE, hatch_label="Two-curve queue wall", reference_label="Prime95 (4T)",
        note=f"Actual B2 = {B2_MIN:,}–{B2_MAX:,}. B1 as measured. One best measured route per GPU and version.",
    )
    summary = data / "benchmarks/stage2_version_throughput/selected.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(asdict(row), curves_per_second=row.throughput) for row in selected]
    with summary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(selected)} plotted points; {summary}")


if __name__ == "__main__":
    main()
