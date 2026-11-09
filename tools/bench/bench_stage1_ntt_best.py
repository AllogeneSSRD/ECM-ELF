#!/usr/bin/env python3
"""Tune full Mersenne Stage1 shapes, freeze winners, then time real PARAM0 batches."""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import tomllib

from bench_stage1_ntt_production import EXPONENTS, GPU_FIELDS, records

HISTORICAL_EXPONENTS = (503, 1009, 2003, 3001, 4001, 5003, 6011, 7001, 8011, 10267, 12511, 14401, 16319)


def run(exe, work, config, flags, expression=None):
    work.mkdir()
    cfg = work / "ecm.toml"
    cfg.write_text(config, encoding="ascii")
    result = subprocess.run([str(exe), "--config", str(cfg), *flags], cwd=work, input=expression,
                            text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    log = work / "driver.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Stage1 failed. See {log}")
    return result.stdout


def tune(exe, work, config, p):
    run(exe, work, config, ["--tune", "ntt", "--bits", str(p)])
    table = tomllib.loads((work / "tune.toml").read_text(encoding="utf-8"))
    columns, *values = table["ntt"]["matrix"]
    rows = [dict(zip(columns, row)) for row in values]
    return max(rows, key=lambda row: row["iter_per_s"])


def measure(exe, work, config, p, b1, sigma, winner):
    batch = winner["batch"]
    mode = winner["ntt_type"].removeprefix("stage1_")
    warp_tail = str(bool(winner["warp_tail"])).lower()
    fixed = config + f'ntt_convolution = "{mode}"\nntt_warp_tail = "{warp_tail}"\n'
    options = {"length": "length", "tile": "tile", "outer_max": "max_radix_bits", "window": "graph_bits"}
    # A fixed weighted length determines ceil(p / length), even below the config's explicit bpw range.
    if mode == "linear":
        options["bpw"] = "bpw"
    fixed += "".join(f"ntt_{key} = {winner[field]}\n" for key, field in options.items())
    flags = ["-sigma", str(sigma), "-gpucurves", str(batch), "-save", "point.save", str(b1), str(b1)]
    log = run(exe, work, fixed, flags, f"2^{p}-1\n")
    plan = dict(re.findall(r"(\w+)=([^ ]+)", re.search(r"stage1_ntt_plan: ([^\r\n]+)", log)[1]))
    if int(plan["curves"]) != batch or plan["convolution"] != mode:
        raise RuntimeError(f"Actual plan differs from the frozen batch. See {work / 'driver.log'}")
    expected = {"length": winner["length"], "bpw": winner["bpw"], "tile_layers": winner["tile"],
                "outer_max_layers": winner["max_radix_bits"], "graph_bits": winner["graph_bits"],
                "warp_tail": winner["warp_tail"], "forward_passes": winner["passes_forward"],
                "forward_layers": winner["forward"], "inverse_layers": winner["inverse"]}
    if any(plan[key] != str(value) for key, value in expected.items()):
        raise RuntimeError(f"Actual plan differs from the frozen shape. See {work / 'driver.log'}")
    fields = re.search(r"stage1_ntt_timing: bits=(\d+) device_ms=([.\d]+) wall_ms=([.\d]+)"
                       r" setup_ms=([.\d]+) readback_ms=([.\d]+) stopped=0", log)
    sample = dict(zip(("steps", "device_ms", "wall_ms", "setup_ms", "readback_ms"), map(float, fields.groups())))
    save = records(work / "point.save")
    if set(save) != set(range(sigma, sigma + batch)):
        raise RuntimeError(f"Missing saved curves. See {work / 'driver.log'}")
    return sample, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=1, type=int)
    parser.add_argument("--rounds", default=3, type=int)
    parser.add_argument("--sigma", default=20260922, type=int)
    parser.add_argument("--exponents", nargs="+", type=int, default=sorted(set(EXPONENTS + HISTORICAL_EXPONENTS)))
    args = parser.parse_args()
    if args.rounds < 1 or len(set(args.exponents)) != len(args.exponents):
        parser.error("use positive rounds and distinct exponents")
    exe, out = args.exe.resolve(strict=True), args.output.resolve()
    out.mkdir(parents=True)
    version = subprocess.check_output([str(exe), "--version"], cwd=out, text=True, timeout=30).strip()
    config = f'[stage1]\ndevice = {args.device}\ngpu_algorithm = "ntt"\ncurve = "mont"\ngpu_param = 0\n' \
             'lcm = "1"\nexp_cache = "off"\nlog_file = ""\nckpt_seconds = 0\n'
    report = dict(version=version, device=args.device, sigma=args.sigma, rounds=args.rounds,
                  exponents=args.exponents, reference_b1=1000000, groups=[])
    with (out / "telemetry.csv").open("w", encoding="utf-8") as log:
        telemetry = subprocess.Popen(
            ["nvidia-smi", "-i", str(args.device), "-lms", "200", f"--query-gpu={GPU_FIELDS}",
             "--format=csv,nounits"], stdout=log, stderr=subprocess.STDOUT)
        try:
            for p in args.exponents:
                group = out / f"p{p}"
                group.mkdir()
                winner = tune(exe, group / "tune", config, p)
                b1 = 1024 if p > 20000 or p < 128 else 10000
                samples, reference = [], None
                for index in range(args.rounds):
                    sample, save = measure(exe, group / f"repeat{index + 1}", config, p, b1, args.sigma, winner)
                    if reference is None:
                        reference = save
                    if reference != save:
                        raise RuntimeError(f"Stage1 save differs across repeats for M{p}")
                    samples.append(sample)
                batch = winner["batch"]
                seconds = [s["device_ms"] / (1000 * batch) for s in samples]
                report["groups"].append(dict(p=p, b1=b1, batch=batch, tune=winner, samples=samples,
                    measured_seconds_per_curve=statistics.median(seconds),
                    projected_seconds_per_curve_1e6=[s * report["reference_b1"] / b1 for s in seconds]))
                (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(f"M{p} B1={b1} {batch}c {winner['ntt_type']} "
                      f"L={winner['length']} {statistics.median(seconds):.6f} s/curve", flush=True)
        finally:
            telemetry.terminate()
            telemetry.wait()


if __name__ == "__main__":
    main()
