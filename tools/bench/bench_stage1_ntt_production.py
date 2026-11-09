#!/usr/bin/env python3
"""Tune CUDA Stage1 through the production CLI and verify every save against a CPU scalar reference."""
import argparse
import itertools
import json
import random
import re
import statistics
import subprocess
from pathlib import Path

EXPONENTS = (61, 127, 251, 521, 1279, 2539, 4003, 8191, 16001, 32749, 65521, 131071, 262139)
GPU_FIELDS = ("timestamp,name,utilization.gpu,clocks.current.sm,power.draw,"
              "power.limit,temperature.gpu")


def minimum_length(p):
    q = (1 << 64) - (1 << 32) + 1
    for b in range(31, 15, -1):
        digits = (p + b - 1) // b
        if digits * ((1 << b) - 1) ** 2 < q:
            return 1 << (2 * digits).bit_length()
    raise ValueError(f"no exact packing for {p}")


def records(path):
    if not path.exists():
        return {}
    saved = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = dict(part.strip().split("=", 1) for part in line.split(";") if "=" in part)
        if fields.get("METHOD") in ("ECM", "ECM-EDWARDS-AM"):
            sigma = int(fields["SIGMA"])
            if sigma in saved:
                raise ValueError(f"duplicate save sigma {sigma}: {path}")
            saved[sigma] = (fields["B1"], fields["X"], fields["CHECKSUM"])
    return saved


def run(args, directory, p, curves, preset=None):
    directory.mkdir()
    save = directory / "out.save"
    config = directory / "ecm.toml"
    lines = ["[stage1]", 'curve = "mont"', "gpu_param = 0", 'lcm = "1"',
             f"device = {args.device}", 'exp_cache = "off"',
             'log_file = ""', "ckpt_seconds = 0"]
    command = [str(args.exe), "--config", str(config), "-sigma", str(args.sigma),
               "-gpucurves", str(curves), "-save", str(save), str(args.b1), str(args.b1)]
    if preset is None:
        command += ["--target", "cpu", "--impl", "scalar"]
    else:
        tile, outer, window, warp = preset
        lines += ["gpu_algorithm = 2", 'ntt_convolution = "linear"', "ntt_profile = 1", f"ntt_tile = {tile}",
                  f"ntt_outer_max = {outer}", f"ntt_window = {window}",
                  f"ntt_warp_tail = {str(warp).lower()}"]
        if args.length:
            lines.append(f"ntt_length = {args.length}")
        command += ["--gpu-backend", "cuda"]
    config.write_text("\n".join(lines) + "\n", encoding="ascii")
    process = subprocess.run(command, cwd=directory, input=f"2^{p}-1\n", text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             encoding="utf-8", errors="replace", timeout=180)
    (directory / "driver.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"worker failed ({process.returncode}): {directory / 'driver.log'}")
    factors = dict((int(s), int(f)) for f, s in re.findall(
        r"factor\[\d+\]=(\d+) curve=\d+ sigma=(\d+)", process.stdout))
    points = records(save)
    if set(points) != set(range(args.sigma, args.sigma + curves)) or any(
            sigma not in points or int(points[sigma][1], 16) != factor for sigma, factor in factors.items()):
        raise RuntimeError(f"incomplete curve results or factor differs from save: {directory / 'driver.log'}")
    if preset is None:
        return points, factors
    timing = re.search(r"stage1_ntt_timing: bits=(\d+) device_ms=([.\d]+) wall_ms=([.\d]+)"
                       r" setup_ms=([.\d]+) readback_ms=([.\d]+) stopped=0", process.stdout)
    plan = re.search(r"stage1_ntt_plan: (.+)", process.stdout)
    if not timing or not plan:
        raise RuntimeError(f"missing NTT timings or plan: {directory / 'driver.log'}")
    row = dict(zip(("bits", "device_ms", "wall_ms", "setup_ms", "readback_ms"), map(float, timing.groups())))
    row["plan"] = dict(item.split("=", 1) for item in plan[1].split())
    if int(row["plan"]["length"]) != (args.length or minimum_length(p)):
        raise RuntimeError(f"unexpected transform length: {directory / 'driver.log'}")
    return row, points, factors


def probe_reference(args, directory, p, curves):
    """The arithmetic probe checks all four projective outputs against GMP."""
    directory.mkdir()
    if args.reuse_oracle:
        old = args.reuse_oracle / f"l{minimum_length(p)}_p{p}_b{curves}" / "oracle" / "driver.log"
        output = old.read_text(encoding="utf-8")
    else:
        cmd = [str(args.probe), str(args.device), str(p), str(args.b1), str(args.sigma),
               str(curves), "1", "12", "16", "print-x"]
        process = subprocess.run(cmd, cwd=directory, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=180)
        output = process.stdout + process.stderr
        if process.returncode:
            (directory / "driver.log").write_text(output, encoding="utf-8")
            raise RuntimeError(f"CPU oracle in the arithmetic probe failed: {directory / 'driver.log'}")
    (directory / "driver.log").write_text(output, encoding="utf-8")
    stamp = f" p={p} b1={args.b1} requested_curves={curves} "
    if not any(line.startswith("stage1_ntt: ") and stamp in line and " bad=0 " in line
               for line in output.splitlines()):
        raise RuntimeError(f"incomplete or mismatched arithmetic probe log: {directory / 'driver.log'}")
    points, factors = {}, {}
    n = (1 << p) - 1
    for sigma_text, status, factor, x in re.findall(
            r"stage1_curve: sigma=(\d+) status=(\w+)(?: factor=(\d+)| x=(0x[\da-f]+))?", output):
        sigma = int(sigma_text)
        if status == "no_factor":
            value = int(x, 16)
        elif status in ("factor", "construction_factor"):
            value = factors[sigma] = int(factor)
        else:
            raise RuntimeError(f"degenerate reference curve: {directory / 'driver.log'}")
        checksum = args.b1 * sigma * n * value % 4294967291
        points[sigma] = (str(args.b1), hex(value), str(checksum))
    if set(points) != set(range(args.sigma, args.sigma + curves)):
        raise RuntimeError(f"CPU oracle in the arithmetic probe failed: {directory / 'driver.log'}")
    return points, factors


def tune(args):
    random_order = random.Random(1729)
    version = subprocess.run([str(args.exe), "--version"], cwd=args.output, check=True,
                             capture_output=True, text=True, timeout=30).stdout.strip()
    report = dict(executable=str(args.exe), version=version, device=args.device, b1=args.b1,
                  sigma=args.sigma, scalar="lcm", rounds=args.rounds, groups=[])
    for p, curves in itertools.product(args.exponents, args.curves):
        length = args.length or minimum_length(p)
        k = length.bit_length() - 1
        top = min(k, 12)
        baseline = (top, 4, 16, True)
        tiles = sorted({top, max(1, top - 2), min(k, 8), min(k, 10)})
        # Screen tile and outer width before spending runs on the graph size and warp tail.
        candidates = [(t, m, 16, True) for t in tiles
                      for m in (4, 3, 2, 1) if m == 4 or m < k - t]
        group_dir = args.output / f"l{length}_p{p}_b{curves}"
        group_dir.mkdir()
        expected = (probe_reference(args, group_dir / "oracle", p, curves) if args.probe or args.reuse_oracle
                    else run(args, group_dir / "scalar", p, curves))
        samples = {}

        def measure(presets, rounds, phase):
            for _ in range(rounds):
                ordered = list(dict.fromkeys(presets))
                random_order.shuffle(ordered)
                for preset in ordered:
                    sample = samples.setdefault(preset, [])
                    label = "_".join(str(int(x)) for x in preset)
                    directory = group_dir / f"{phase}_{label}_{len(sample)+1}"
                    row, points, factors = run(args, directory, p, curves, preset)
                    if (points, factors) != expected:
                        raise RuntimeError(f"point or factor differs from scalar: {directory}")
                    sample.append(row)

        def score(preset):
            return statistics.median(r["device_ms"] for r in samples[preset])

        measure(candidates, 1, "screen")
        shortlisted = sorted(candidates, key=score)[:2]
        followup = [baseline]
        for tile, outer, _, _ in shortlisted:
            followup.extend((tile, outer, window, warp)
                            for window, warp in itertools.product((16, 32), (True, False)))
        # Re-measure every final candidate on equal sample counts and save the screen separately.
        screening, samples = samples, {}
        measure(followup, args.rounds, "confirm")
        winner = min(samples, key=score)
        selected = winner if score(winner) <= 0.97 * score(baseline) else baseline
        group = dict(p=p, length=length, curves=curves, reference_points=len(expected[0]) - len(expected[1]),
                     reference_factors=len(expected[1]), baseline=list(baseline),
                     winner=list(winner), selected=list(selected), baseline_ms=score(baseline), best_ms=score(winner),
                     screening=[dict(preset=list(k), samples=v) for k, v in screening.items()],
                     finalists=[dict(preset=list(k), samples=v, median_ms=score(k)) for k, v in samples.items()])
        report["groups"].append(group)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"L={length} curves={curves} {score(baseline):.3f} -> {score(winner):.3f} ms {winner}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--probe", type=Path,
                        help="use the GMP oracle in the standalone probe for widths outside CPU Stage1")
    parser.add_argument("--reuse-oracle", type=Path,
                        help="reuse matching, verified probe logs from a prior matrix")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--b1", type=int, default=256)
    parser.add_argument("--length", type=int, default=0, help="override the derived length")
    parser.add_argument("--sigma", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--exponents", type=int, nargs="+", default=EXPONENTS)
    parser.add_argument("--curves", type=int, nargs="+", default=(1, 4, 8))
    args = parser.parse_args()
    if args.rounds < 1 or not all(1 <= n <= 32 for n in args.curves):
        parser.error("use at least one round and 1–32 curves")
    args.exe, args.output = args.exe.resolve(strict=True), args.output.resolve()
    if args.probe:
        args.probe = args.probe.resolve(strict=True)
    if args.reuse_oracle:
        args.reuse_oracle = args.reuse_oracle.resolve(strict=True)
    args.output.mkdir(parents=True)
    with (args.output / "telemetry.csv").open("w", encoding="utf-8") as log:
        telemetry = subprocess.Popen(
            ["nvidia-smi", "-i", str(args.device), "-lms", "200", f"--query-gpu={GPU_FIELDS}",
             "--format=csv,nounits"], stdout=log, stderr=subprocess.STDOUT)
        try:
            tune(args)
        finally:
            telemetry.terminate()
            telemetry.wait()


if __name__ == "__main__":
    main()
