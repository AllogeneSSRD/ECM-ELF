#!/usr/bin/env python3
"""Plot selected, input-matched GPU Stage1 benchmarks by ECM-ELF version."""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import median
import subprocess

from plot_stage1_ntt_best import ntt_points, ntt_version
from plot_version_throughput import PRIME95, WIDTHS, plot_version_panels, version_key


REPO = Path(__file__).resolve().parents[2]
REFERENCE_B1 = 1_000_000
COLORS = {
    "4070 Ti CGBN": "#2866a2",
    "4070 Ti NTT": "#14804a",
    "4060 Laptop CGBN": "#ce6b18",
    "4060 Laptop NTT": "#943e78",
    PRIME95: "#181818",
}
HISTORICAL_COMPARISON = "2d091243:docs/performance/STAGE1.md"
HISTORICAL_SQUARE = "fed434ef:docs/performance/CGBN_SQUARE.md"
MATRIX_VERSION = "0.63.17"
MAIN_LOOP = "main loop"
WITH_SETUP = "device interval including preparation"


@dataclass(frozen=True)
class Sample:
    curve: str
    exponent: int
    bits: int
    version: str
    series: str
    seconds_per_curve_at_1e6: float
    variant: str
    timing: str
    source: str

    @property
    def throughput(self):
        return 1 / self.seconds_per_curve_at_1e6


def projected(seconds, b1, curves):
    return seconds * REFERENCE_B1 / (b1 * curves)


def git_table(source, header):
    document = subprocess.check_output(["git", "show", source], cwd=REPO, encoding="utf-8")
    lines = iter(document.splitlines())
    next(line for line in lines if line.startswith(header))
    next(lines)  # Markdown separator
    for line in lines:
        if not line.startswith("|"):
            break
        yield [cell.strip() for cell in line.strip("|").split("|")]


def historical_samples():
    for p, bits, b1, curves, ed, mont, *_ in git_table(
        HISTORICAL_COMPARISON, "| M 指数 | N bits | 实测 B1 | 曲线/批 |"
    ):
        p, bits, b1, curves = (int(value.replace(",", "")) for value in (p, bits, b1, curves))
        if p in WIDTHS:
            yield Sample("ed", p, bits, "0.52.0", "4070 Ti CGBN", projected(float(ed), b1, curves),
                         "NAF w6 HWCD", MAIN_LOOP, HISTORICAL_COMPARISON)
            yield Sample("mont0", p, bits, "0.52.2", "4070 Ti CGBN", projected(float(mont), b1, curves),
                         "ladder", WITH_SETUP, HISTORICAL_COMPARISON)

    by_bits = {bits: p for p, (_, bits) in WIDTHS.items()}
    for label, pair, *_ in git_table(HISTORICAL_SQUARE, "| N bits／模型 | GPU区间：原→融合 |"):
        width, model = label.split("／")
        bits = int(width)
        p = by_bits[bits]
        curves = 192 if p == 16319 else 384  # Source matrix uses a 32-thread group at 16K.
        ed = model == "Edwards"
        original, fused = map(float, pair.split("→"))
        for seconds, variant in ((original, "original CGBN"), (fused, "fused square REDC")):
            yield Sample("ed" if ed else "mont0", p, bits, "0.57.16", "4060 Laptop CGBN",
                         projected(seconds, 5000, curves), variant,
                         MAIN_LOOP if ed else WITH_SETUP, HISTORICAL_SQUARE)


def matrix_samples(root, curve):
    model = "ed" if curve == "ed" else "mont"
    for device, name in enumerate(("4070 Ti", "4060 Laptop")):
        path = root / f"gpu{device}" / model / "summary.json"
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        if (ntt_version(report) != MATRIX_VERSION or report["curve"] != model
                or report["device"] != device or name not in report["gpu_model"]
                or report["rounds"] != 5 or not WIDTHS.keys() <= set(report["exponents"])):
            raise ValueError(f"Unexpected Stage1 matrix identity in {path}")
        points = {p: seconds for p, _, seconds, _, _ in ntt_points(report, REFERENCE_B1)}
        for group in report["groups"]:
            p = group["p"]
            if p not in WIDTHS:
                continue
            measurements = group["projected_seconds_per_curve_1e6"]
            samples = group["samples"]
            active_batch = group["active_batch"]
            b1 = group["b1"]
            shapes = {(sample["chain"], sample["naf_window"], sample["active_curves"]) for sample in samples}
            if (len(samples) != 5 or len(measurements) != 5 or len(shapes) != 1 or active_batch <= 0 or b1 <= 0
                    or any(not math.isfinite(seconds) or seconds <= 0 for seconds in measurements)
                    or any(not math.isclose(seconds, projected(sample["device_ms"] / 1000, b1, active_batch),
                                            rel_tol=1e-12) for seconds, sample in zip(measurements, samples))):
                raise ValueError(f"Invalid NTT samples for M{p} in {path}")
            chain, window, measured_batch = next(iter(shapes))
            if measured_batch != active_batch:
                raise ValueError(f"Mismatched active batch for M{p} in {path}")
            if chain == "naf":
                chain = f"2XY NAF w{window}"
            tune = group["tune"]
            convolution = tune["ntt_type"].removeprefix("stage1_")
            variant = f"{convolution} N={tune['length']}, {chain}, {active_batch}c, B1={b1}"
            yield Sample(curve, p, p, MATRIX_VERSION, f"{name} NTT", points[p], variant, MAIN_LOOP, path.as_posix())


def comparison_samples(path, curve, current):
    matrix = {(sample.exponent, sample.series): sample for sample in current}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            p = int(row["mersenne_exponent"])
            role = row["series"]
            backend = role.partition("_")[0]
            if p not in WIDTHS:
                continue
            if role == "prime95":
                yield Sample(curve, p, int(row["arithmetic_bits"]), "", PRIME95,
                             float(row["median_s_per_curve_at_1e6"]), "GWNUM, 1 thread",
                             "native Prime95 Stage1", path.as_posix())
                continue
            if backend not in ("cgbn", "ntt"):
                continue
            version = row["version"]
            device = "4070 Ti" if "overlay" in role else "4060 Laptop"
            if backend == "ntt" and version == MATRIX_VERSION:
                actual = matrix[p, f"{device} NTT"]
                if not math.isclose(float(row["median_s_per_curve_at_1e6"]), actual.seconds_per_curve_at_1e6,
                                    rel_tol=1e-12):
                    raise ValueError(f"M{p} {device}: {path} and {actual.source} disagree")
                continue
            if backend == "ntt" and version == "0.63.15":
                continue
            if backend == "ntt":
                variant = "tuned NTT plan"
            elif curve == "ed":
                variant = "NAF w12" if version == "0.63.7" else "NAF w6"
            else:
                variant = "PRAC 50" if version == "0.63.7" and not role.endswith("_ladder") else "ladder"
            yield Sample(curve, p, int(row["arithmetic_bits"]), version, f"{device} {backend.upper()}",
                         float(row["median_s_per_curve_at_1e6"]), variant, MAIN_LOOP,
                         f"{path.as_posix()} ({role})")


def mont_auto_samples(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["version"] == "ECM-ELF 0.60.2" and report["device"] == 1
    for group in report["groups"]:
        p = group["p"]
        if group["synthetic"] or p not in WIDTHS:
            continue
        assert group["auto_plan"]["curve"] == "mont0"
        for mode, rows in group["samples"].items():
            assert len(rows) == report["rounds"]
            seconds = median(row["device_ms"] for row in rows) / 1000
            yield Sample("mont0", p, p, "0.60.2", "4060 Laptop NTT",
                         projected(seconds, group["b1"], group["batch"]), mode, MAIN_LOOP, path.as_posix())


def select(samples):
    best = {}
    for sample in samples:
        expected_bits = WIDTHS[sample.exponent][1] if sample.series.endswith("CGBN") else sample.exponent
        seconds = sample.seconds_per_curve_at_1e6
        if sample.bits != expected_bits or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Invalid benchmark point: {sample}")
        key = (sample.curve, sample.exponent, sample.version, sample.series)
        previous = best.get(key)
        if previous and previous.timing != sample.timing:
            raise ValueError(f"Mismatched timing scopes: {previous} and {sample}")
        if previous is None or sample.throughput > previous.throughput:
            best[key] = sample
    return sorted(best.values(), key=lambda row: (row.curve, row.exponent, version_key(row.version), row.series))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    matrix_dir = "stage1_main_matrix_" + MATRIX_VERSION.replace(".", "")
    parser.add_argument("--comparison-root", type=Path, default=Path("data/figures") / matrix_dir)
    parser.add_argument("--ntt-matrix-root", type=Path, default=Path("data/experiments") / matrix_dir)
    parser.add_argument("--mont-auto", type=Path,
                        default=Path("data/experiments/stage1_ntt_auto_matrix_0602/verified/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/figures/stage1_version_throughput"))
    parser.add_argument("--summary", type=Path,
                        default=Path("data/benchmarks/stage1_version_throughput/selected.csv"))
    args = parser.parse_args()
    curves = ("ed", "mont0")
    samples = list(historical_samples())
    samples.extend(mont_auto_samples(args.mont_auto))
    for curve in curves:
        current = list(matrix_samples(args.ntt_matrix_root, curve))
        samples.extend(current)
        samples.extend(comparison_samples(args.comparison_root / curve / "comparison.csv", curve, current))
    selected = select(samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for curve in curves:
        model = "Edwards" if curve == "ed" else "Montgomery PARAM0"
        plot_version_panels(
            [row for row in selected if row.curve == curve],
            args.output_dir / f"stage1-{curve}-version-throughput.png",
            title=f"{model} Stage1 throughput by version", ylabel="Curves/s at B1 = 1,000,000 (scaled)",
            colors=COLORS, hatch=WITH_SETUP, hatch_label="GPU interval includes prep",
            reference_label="Prime95 (1T)",
            note="CGBN uses the cofactor. NTT and Prime95 use the full exponent for arithmetic.",
            stagger_higher_labels=True,
        )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(asdict(row), curves_per_second_at_1e6=row.throughput) for row in selected]
    with args.summary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(selected)} plotted points; {args.summary}")


if __name__ == "__main__":
    main()
