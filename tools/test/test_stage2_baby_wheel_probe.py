"""Run the CUDA baby wheel probe on frozen replay witnesses and real Stage1 saves."""

import argparse
from dataclasses import dataclass
from fnmatch import fnmatchcase
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import sys

from test_edwards_gpu_pipeline import checksum, curve, mont


ROOT = Path(__file__).resolve().parents[2]
SAVE_DIR = ROOT / "data/experiments/stage2_giant_auto_20261101/matrix01/inputs"
REAL_INPUTS = ((2003, 1939, 27), (8011, 7995, 26), (16319, 16272, 26))


@dataclass(frozen=True)
class Case:
    name: str
    target: int
    arithmetic: int
    qx: int
    a24: int
    shape: tuple[int, int, int]
    mersenne_bits: int = 0
    guard: str = ""
    source: str = ""


def synthetic_cases():
    # Frozen from the independent integer probe. Common failures and replay
    # with a unit common step are both needed to exercise the recovery paths.
    shape = (2310, 30, 16)
    for n, x, a24, guard in (
        (77, 12, 73, "common"),
        (221, 128, 36, "common"),
        (1147, 756, 34, ""),
        (3481, 617, 895, "segment"),  # [13]Q has Z=0 modulo 59**2.
        (11413, 1068, 355, "segment"),
        (38021, 34322, 27234, "segment"),
        (63001, 54714, 28126, "segment"),
        (265189, 44126, 111044, "segment"),
        (1050589, 348187, 897977, "segment"),
    ):
        yield Case(f"toy_{n}", n, n, x, a24, shape, guard=guard)
    # M29 = 233 * 1103 * 2089. This case has one proper denominator
    # GCD at the target 233*1103 and a valid common difference.
    yield Case("toy_m29_carrier", 233 * 1103, (1 << 29) - 1, 167829, 143075,
               shape, mersenne_bits=29, guard="segment")
    # [527]Q has a denominator divisible by 2089 only. A guard that
    # mistakes the full carrier for the target replays this clean case.
    yield Case("toy_m29_carrier_only_nonunit", 233 * 1103, (1 << 29) - 1, 87625, 89759,
               shape, mersenne_bits=29, guard="clean")


def frozen_case():
    target = (1 << 128) + 1
    state = mont.stage1(26, 1000, target)
    return Case("frozen_stage2_factor_input", target, target, state["x"], state["a24"],
                (210, 30, 32), source="Suyama sigma=26 B1=1000")


def matches(name, patterns):
    return not patterns or any(fnmatchcase(name, pattern) for pattern in patterns)


def real_cases(save_dir, shapes, patterns):
    for exponent, bits, sigma in REAL_INPUTS:
        name = f"m{exponent}_n{bits}_s{sigma}"
        plans = [
            (f"{name}_d{d}_w{wheel}_l{walk}_{mode}", (d, wheel, walk), mode)
            for d, wheel, walk in shapes
            for mode in ("ordinary", "carrier")
        ]
        selected = [plan for plan in plans if matches(plan[0], patterns)]
        if not selected:
            continue
        path = save_dir / f"{name}.save"
        [line] = path.read_text(encoding="utf-8-sig").splitlines()
        fields = dict(token.strip().split("=", 1) for token in line.split(";") if token.strip())
        n, qx, got_sigma, b1 = (int(fields[k], 0) for k in ("N", "X", "SIGMA", "B1"))
        assert (fields["METHOD"], fields["VERSION"], fields["SCALAR"], fields["STATUS"]) == (
            "ECM-EDWARDS-AM", "1", "48LCM", "POINT")
        assert (n.bit_length(), got_sigma, b1) == (bits, sigma, 50000)
        assert int(fields["CHECKSUM"]) == checksum(n, qx, sigma, b1, "POINT")
        a24, _ = curve(n, sigma)
        for label, shape, mode in selected:
            if mode == "ordinary":
                yield Case(label, n, n, qx, a24, shape, source=str(path))
            else:
                yield Case(label, n, (1 << exponent) - 1, qx, a24, shape, exponent, source=str(path))


def geometry(shape):
    d, wheel, walk = shape
    if not 4 <= d <= 1200000 or not 2 <= wheel <= 30030 or wheel % 2 or d % wheel or not 2 <= walk <= 64:
        raise ValueError(f"invalid D/wheel/max walk: {shape}")
    indices = [j for j in range(1, d // 2 + 1) if math.gcd(j, d) == 1]
    starts, ends = {}, {}
    spans = []
    for j in indices:
        r = j % wheel
        if r not in starts or j - starts[r] >= wheel * walk:
            if r in starts:
                spans.append((ends[r] - starts[r]) // wheel + 1)
            starts[r] = j
        ends[r] = j
    spans.extend((ends[r] - start) // wheel + 1 for r, start in starts.items())
    return indices, dict(points=len(indices), segments=len(spans),
                         seeds=1 + sum(min(span, 2) for span in spans), walked=sum(spans))


def reference_replay(case, indices, total_segments):
    _, wheel, length = case.shape
    target, arithmetic = case.target, case.arithmetic

    def ladder(j):
        return mont.ladder(j, case.qx, 1, case.a24, arithmetic)

    dx, dz = ladder(wheel)
    if any(math.gcd(x, target) != 1 for x in (case.qx, case.a24, case.a24 - 1, dx, dz)):
        return True, total_segments, len(indices), 0
    groups = {}
    spans = []
    for j in indices:
        group = groups.get(j % wheel)
        if group is None or j - group[0] >= wheel * length:
            group = []
            groups[j % wheel] = group
            spans.append(group)
        group.append(j)

    bad = replayed = hole_only = 0
    for span in spans:
        first = span[0]
        selected = set(span)
        cur = ladder(first)
        failed = output_failed = False
        for j in range(first, span[-1] + 1, wheel):
            if j == first + wheel:
                previous, cur = cur, ladder(j)
            elif j > first + wheel:
                x, z = cur
                px, pz = previous
                previous, cur = cur, (pz * (x * dx - z * dz)**2 % arithmetic,
                                      px * (x * dz - z * dx)**2 % arithmetic)
            if math.gcd(cur[1], target) != 1:
                failed = True
                output_failed |= j in selected
        if failed:
            bad += 1
            replayed += len(span)
            hole_only += not output_failed
    return False, bad, replayed, hole_only


def verify(case, result, repeats):
    d, wheel, max_walk = case.shape
    indices, expected = geometry(case.shape)
    expected.update(status="PASS", target_bits=case.target.bit_length(),
                    arithmetic_bits=case.arithmetic.bit_length(), d=d, wheel=wheel, max_walk=max_walk,
                    verified_points=len(indices), gmp_samples=min(len(indices), 48))
    for key, wanted in expected.items():
        assert result[key] == wanted, (case.name, key, result[key], wanted)
    assert result["tpi"] == (32 if case.arithmetic.bit_length() > 8192 else 0), case.name
    for key in ("plan_ms", "setup_ms"):
        assert math.isfinite(result[key]) and result[key] >= 0, (case.name, key)
    for key in ("base_ms", "wheel_ms"):
        samples = result[key]
        assert len(samples) == repeats and all(math.isfinite(ms) and ms > 0 for ms in samples), (case.name, key)
    bad, replayed = result["bad_segments"], result["replayed_points"]
    failed = result["common_guard_failed"]
    assert isinstance(failed, bool), case.name
    assert 0 <= bad <= expected["segments"] and bad <= replayed <= len(indices), case.name
    assert bool(bad) == bool(replayed), case.name
    if failed:
        assert (bad, replayed) == (expected["segments"], len(indices)), case.name
    if case.guard == "common":
        assert failed, case.name
    if case.guard == "segment":
        assert not failed and 0 < bad <= expected["segments"], case.name
    if case.guard == "clean":
        assert not failed and (bad, replayed) == (0, 0), case.name
    proper, zero = result["z_nonunit"], result["z_zero"]
    assert 0 <= proper and 0 <= zero and proper + zero <= len(indices), case.name
    if case.target.bit_length() <= 129:
        gcds = [math.gcd(mont.ladder(j, case.qx, 1, case.a24, case.arithmetic)[1], case.target)
                for j in indices]
        assert proper == sum(1 < g < case.target for g in gcds), case.name
        assert zero == gcds.count(case.target), case.name
        common, bad_ref, replayed_ref, hole_only = reference_replay(case, indices, expected["segments"])
        assert (failed, bad, replayed) == (common, bad_ref, replayed_ref), (case.name, hole_only)


def run_case(args, case, repeats, folder):
    assert case.target > 1 and case.target & 1 and case.arithmetic % case.target == 0
    assert 0 <= case.qx < case.target and 0 <= case.a24 < case.target
    if case.mersenne_bits:
        assert (1 << case.mersenne_bits) - 1 == case.arithmetic
    folder.mkdir()
    command = [str(args.probe), *map(str, (args.device, *case.shape, repeats, case.mersenne_bits))]
    stdin = "".join(f"{value:x}\n" for value in (case.target, case.arithmetic, case.qx, case.a24)).encode("ascii")
    (folder / "input.txt").write_bytes(stdin)
    metadata = dict(case=case.name, source=case.source, command=command)
    (folder / "command.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    stdout = folder / "stdout.json"
    with stdout.open("wb") as out, (folder / "stderr.log").open("wb") as err:
        completed = subprocess.run(command, cwd=folder, input=stdin, stdout=out, stderr=err, timeout=args.timeout)
    assert completed.returncode == 0, (case.name, completed.returncode, folder)
    result = json.loads(stdout.read_text(encoding="utf-8"))
    verify(case, result, repeats)
    paired = [100 * (1 - wheel / base) for base, wheel in zip(result["base_ms"], result["wheel_ms"], strict=True)]
    row = metadata | result | dict(base_median_ms=median(result["base_ms"]),
                                   wheel_median_ms=median(result["wheel_ms"]),
                                   paired_saved_percent=median(paired))
    print(f'PASS {case.name}: {result["verified_points"]} points, '
          f'{result["replayed_points"]} replayed, {row["base_median_ms"]:.3f} -> '
          f'{row["wheel_median_ms"]:.3f} ms ({row["paired_saved_percent"]:+.2f}%)', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--group", action="append", choices=("synthetic", "frozen", "real"),
                        help="repeatable; default: synthetic and frozen")
    parser.add_argument("--case", action="append", help="shell-style name filter, repeatable")
    parser.add_argument("--repeats", type=int, help="override the default 3 real / 1 synthetic and frozen repeats")
    parser.add_argument("--real-shape", type=int, nargs=3, action="append", metavar=("D", "WHEEL", "MAX_WALK"))
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--list", action="store_true", help="list selected inputs without running the probe")
    args = parser.parse_args()
    if args.device < 0 or args.timeout < 1 or (args.repeats is not None and not 1 <= args.repeats <= 11):
        parser.error("device must be nonnegative, timeout positive, and repeats in 1..11")
    groups = args.group or ["synthetic", "frozen"]
    shapes = tuple(map(tuple, args.real_shape or [(180180, 210, 32)]))
    if len(set(groups)) != len(groups) or len(set(shapes)) != len(shapes):
        parser.error("groups and real shapes must be distinct")
    cases = []
    if "synthetic" in groups:
        cases.extend((case, 1) for case in synthetic_cases())
    if "frozen" in groups:
        cases.append((frozen_case(), 1))
    cases = [(case, args.repeats or count) for case, count in cases
             if matches(case.name, args.case)]
    if "real" in groups:
        cases.extend((case, args.repeats or 3) for case in real_cases(args.save_dir.resolve(), shapes, args.case))
    if not cases:
        parser.error("no matching cases")
    if args.list:
        for case, repeats in cases:
            print(case.name, geometry(case.shape)[1], f"repeats={repeats}")
        return
    if not args.probe or not args.output:
        parser.error("--probe and --output are required")
    args.probe = args.probe.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    telemetry = None
    if args.telemetry:
        sys.path.insert(0, str(ROOT / "tools/bench"))
        from gpu_telemetry import TelemetryRecorder
        telemetry = TelemetryRecorder(args.device, output / "telemetry.jsonl")
    report = dict(device=args.device, cases=[], complete=False)
    if telemetry:
        report["gpu"] = telemetry.identity
    try:
        for case, repeats in cases:
            report["cases"].append(run_case(args, case, repeats, output / case.name))
        report["complete"] = True
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if telemetry:
            telemetry.close()


if __name__ == "__main__":
    main()
