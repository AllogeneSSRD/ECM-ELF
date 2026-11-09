"""Verify and time whole-curve CUDA NTT Stage1, optionally against CGBN."""
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path


def invoke(command, folder, input_text=None):
    start = time.perf_counter()
    p = subprocess.run([str(arg) for arg in command], cwd=folder, input=input_text,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    (folder / "stdout.log").write_text(p.stdout, encoding="utf-8")
    (folder / "stderr.log").write_text(p.stderr, encoding="utf-8")
    (folder / "command.json").write_text(json.dumps(list(map(str, command))), encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"{p.returncode}: {folder}\n{p.stderr[-800:]}")
    return p.stdout + "\n" + p.stderr, time.perf_counter() - start


def fields(line):
    return dict(s.split("=", 1) for s in line.split() if "=" in s)


def ntt(args, folder, p, b1, sigma, count, tile, window, samples, print_x=True):
    folder.mkdir()
    command = [args.probe, args.device, p, b1, sigma, count, samples, tile, window]
    log, seconds = invoke(command + (["print-x"] if print_x else []), folder)
    record = fields(next(s for s in log.splitlines() if s.startswith("stage1_ntt: version=")))
    curves = {int(row["sigma"]): row for row in map(fields, (s for s in log.splitlines()
                                                             if s.startswith("stage1_curve:")))}
    assert record["bad"] == "0" and len(curves) == count and int(record["curves"]) > 0
    for curve in curves.values():
        if "factor" in curve:
            f = int(curve["factor"])
            assert 1 < f < (1 << p)-1 and ((1 << p)-1) % f == 0
    record.update(event=[float(v) for v in record["event_ms"].split(",")],
                  elapsed_seconds=seconds, tile_requested=tile, directory=folder.name)
    return record, curves


def cgbn(args, folder, p, b1, count):
    folder.mkdir()
    for file in (args.cgbn, *(args.cgbn.parent / n for n in ("gmp-10.dll", "libpari.dll"))):
        if file.exists():
            try:
                os.link(file, folder / file.name)
            except OSError:
                shutil.copy2(file, folder / file.name)
    save = folder / "points.save"
    log, seconds = invoke([folder / args.cgbn.name, "--curve", "mont", "--target", "gpu", "--gpu-param", 0,
                           "--lcm", 1, "-sigma", 11, "-gpucurves", count, "-d", args.device, "--ckpt", 0,
                           "--exp-cache", "off", "-save", save, b1, 0], folder, f"(2^{p}-1)\n")
    match = re.search(r"GPU stage1 returned:\s*[01]\s+gputime=([\d.]+) ms", log)
    if not match or f"will use device {args.device}:" not in log:
        raise RuntimeError(f"unexpected CGBN execution: {folder}")
    points = {}
    for line in save.read_text(encoding="utf-8").splitlines():
        vals = fields(line.replace(";", " "))
        if "X" in vals and "SIGMA" in vals:
            points[int(vals["SIGMA"])] = int(vals["X"], 0)
    version = re.search(r"ECM-ELF ([\d.]+) \(Stage1\)", log)[1]
    return dict(p=p, b1=b1, curves=count, event=[float(match[1])], elapsed_seconds=seconds,
                version=version, directory=folder.name), points


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    args.probe = args.probe.resolve()
    if args.cgbn:
        args.cgbn = args.cgbn.resolve()
    rows = []

    def record(row, kind):
        row["kind"] = kind
        rows.append(row)
        (out / "summary.json").write_text(json.dumps(rows, indent=2)+"\n", encoding="utf-8")
        print(kind, row["directory"], statistics.median(row["event"]), flush=True)

    if args.mode == "qualify":
        for p in (32, 33, 64, 65, 8192, 8193, 262144):
            folder = out / f"carry_{p}"
            folder.mkdir()
            log, _ = invoke([args.probe, "--check-carries", args.device, p], folder)
            assert "bad=0" in log
        row, result = ntt(args, out / "known_factor", 65, 40, 11, 8, 6, 7, 2)
        assert result[11]["status"] == "factor" and int(result[11]["factor"]) == 31*8191
        assert result[12]["status"] == "construction_factor" and int(result[12]["factor"]) == 31
        record(row, "known_factor")
        row, _ = ntt(args, out / "wide_sigma", 32749, 256, 4294967311, 4, 8, 16, 2)
        record(row, "wide_sigma")
        return

    if args.mode == "long":
        if not args.cgbn:
            raise ValueError("--mode long requires --cgbn")
        for repeat in range(args.rounds):
            for p, tile in ((8191, 10), (16001, 8)):
                row, curves = ntt(args, out / f"ntt_{p}_1_{repeat}", p, 5000, 11, 1, tile, 16, 3)
                record(row, "ntt")
                row, points = cgbn(args, out / f"cgbn_{p}_1_{repeat}", p, 5000, 1)
                expected = curves[11]
                x = int(expected["x"], 16) if "x" in expected else int(expected["factor"])
                assert points == {11: x}, p
                record(row, "cgbn")
        return

    if args.mode == "batches":
        # Tune the tile separately for each curve count; otherwise a tile chosen
        # for latency can bias the throughput ranking as the y grid gets taller.
        shapes = ((4003, 9), (8191, 10), (16001, 11), (32749, 12), (65521, 13), (131071, 14))
        for p, k in shapes:
            tiles = sorted({min(k, t) for t in (6, 8, 10, 12)})
            chosen = {}
            for count in range(1, 9):
                scores = []
                for tile in tiles:
                    row, _ = ntt(args, out / f"screen_{p}_{count}_{tile}", p, 13, 11,
                                 count, tile, 16, 3, print_x=False)
                    assert int(row["curves"]) == count and int(row["length"]) == 1 << k
                    scores.append((statistics.median(row["event"]), tile))
                    record(row, "screen")
                chosen[count] = min(scores)[1]
            for repeat in range(args.rounds):
                counts = range(1, 9) if repeat % 2 == 0 else range(8, 0, -1)
                for count in counts:
                    tile = chosen[count]
                    row, _ = ntt(args, out / f"confirm_{p}_{count}_{repeat}", p, 41, 11,
                                 count, tile, 16, 3, print_x=False)
                    assert int(row["curves"]) == count and int(row["length"]) == 1 << k
                    record(row, "confirm")
        return

    widths = (8191, 16001, 32749, 65521, 131071)
    tiles = {}
    for p in widths:
        scores = []
        for tile in (6, 8, 10, 12):
            row, _ = ntt(args, out / f"tile_{p}_{tile}", p, 41, 11, 1, tile, 16, 3)
            scores.append((statistics.median(row["event"]), tile))
            record(row, "tile")
        tiles[p] = min(scores)[1]
    for repeat in range(args.rounds):
        for p in widths:
            for count in (1, 4):
                row, curves = ntt(args, out / f"ntt_{p}_{count}_{repeat}", p, 256, 11, count, tiles[p], 16, 3)
                record(row, "ntt")
                if args.cgbn and p in (8191, 16001):
                    row, points = cgbn(args, out / f"cgbn_{p}_{count}_{repeat}", p, 256, count)
                    expected = {seed: int(v["x"], 16) if "x" in v else int(v["factor"])
                                for seed,v in curves.items() if "x" in v or "factor" in v}
                    assert len(expected) == count and expected == points, (p, count)
                    record(row, "cgbn")
            if args.cgbn and p in (8191,16001):
                row, _ = cgbn(args, out / f"cgbn_{p}_192_{repeat}", p, 256, 192)
                record(row, "cgbn")
    for p in (8191, 32749):
        row, _ = ntt(args, out / f"direct_{p}", p, 256, 11, 1, tiles[p], 0, 3)
        record(row, "direct")
    row, _ = ntt(args, out / "longer_32749", 32749, 5000, 11, 1, tiles[32749], 16, 3)
    record(row, "longer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--mode", choices=("qualify", "bench", "long", "batches"), default="qualify")
    parser.add_argument("--cgbn", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    run(args)
