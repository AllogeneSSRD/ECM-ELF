"""Measure Stage1 batch scaling and confirm selected OpenCL / CGBN plans."""
import argparse
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

from gpu_telemetry import TelemetryRecorder


BATCHES = [768, 1536, 3072, 6144, 12288, 24576, 49152]
WIDTHS = [127, 241, 319, 421, 511, 1021]
SIGMA = 20260922
MODULUS_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
ED_TIME = re.compile(r"Edwards NAF: columns=(\d+).*?\bloop_gpu_(ms|s)=([\d.]+)")
MONT_TIME = re.compile(r"GPU stage1 returned: .*gputime=([\d.]+) ms")


def probable_prime(bits):
    # Edwards and PARAM3 run the same candidate with no known small factors.
    for n in range((1 << bits) - 1, 1 << (bits - 1), -2):
        d = n - 1
        s = (d & -d).bit_length() - 1
        d >>= s
        for a in MODULUS_BASES:
            x = pow(a, d, n)
            if x not in (1, n - 1):
                for _ in range(s - 1):
                    x = x * x % n
                    if x == n - 1:
                        break
                else:
                    break
        else:
            return n
    raise ValueError(f"no fixture for width {bits}")


def read_points(path, count):
    points = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = dict(part.strip().split("=", 1) for part in line.split(";") if part.strip())
        sigma = int(fields["SIGMA"].rsplit(":", 1)[-1], 0)
        points[sigma] = int(fields["X"], 0)
    if len(points) != count or min(points) != SIGMA or max(points) != SIGMA + count - 1:
        raise ValueError(f"incomplete batch in {path}")
    return points


class Collector:
    def __init__(self, args):
        self.args = args
        self.root = args.out.resolve()
        self.root.mkdir(parents=True)
        self.exe = {"opencl": args.opencl.resolve(), "cuda": args.cuda.resolve()}
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("ECM_", "NTT_"))}
        self.versions = {
            backend: subprocess.check_output([str(exe), "--version"], cwd=self.root, text=True).strip()
            for backend, exe in self.exe.items()
        }
        self.telemetry = TelemetryRecorder(args.device, self.root / "telemetry.jsonl", interval=0.25)
        self.summary = {"device": self.telemetry.identity, "program": self.versions,
                        "phase": args.phase, "sigma": SIGMA, "conditions": []}
        self.random = random.Random(20261116 + args.device)

    def run(self, bits, n, model, b1, backend, curves, wg, repeat):
        folder = self.root / f"{model}_{bits}_{backend}_{curves}_wg{wg}_r{repeat}"
        folder.mkdir()
        save = folder / "point.save"
        cfg = folder / "ecm.toml"
        cfg.write_text(
            'config_version = 3\n[shared]\nlog_file = ""\n[stage1]\n'
            f'param = "{model}"\ntarget = "gpu"\ngpu_backend = "{backend}"\ngpu_device = {self.args.device}\n'
            'gpu_algorithm = "cgbn"\nedwards_naf_window_bits = 6\nedwards_doubling = 1\n'
            'opencl.threads_per_integer = 1\nhandoff.mode = "off"\ncheckpoint_interval_seconds = 0\n'
            f'opencl.workgroup_size = {wg}\nsave_dir = {json.dumps(folder.as_posix())}\n'
            f'exponent_cache_dir = {json.dumps((self.root / "scalar_cache").as_posix())}\n', encoding="ascii")
        seed = str(SIGMA) if model == "e" else f"3:{SIGMA}"
        cmd = [str(self.exe[backend]), "--config", str(cfg), "-sigma", seed, "-gpucurves",
               str(curves), "-savea", str(save), str(b1), "0"]
        start = time.perf_counter()
        result = subprocess.run(cmd, cwd=folder, env=self.env, input=f"{n}\n", text=True,
                                encoding="utf-8", errors="replace", capture_output=True, timeout=300)
        wall = time.perf_counter() - start
        log = result.stdout + result.stderr
        (folder / "run.log").write_text(log, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{folder}:\n{log[-1800:]}")
        execution, = re.findall(r"Stage1 execution: (.*)", log)
        if model == "e":
            columns, unit, number = ED_TIME.search(log).groups()
            inner = float(number) / (1000 if unit == "ms" else 1)
        else:
            columns = None
            inner = float(MONT_TIME.search(log)[1]) / 1000
        row = dict(backend=backend, curves=curves, wg=wg, repeat=repeat, wall=wall, inner=inner,
                   columns=int(columns) if columns is not None else None, execution=execution)
        return row, read_points(save, curves)

    def condition(self, bits, model, b1, plans, repeats, warmup_plans):
        n = probable_prime(bits)
        condition = dict(bits=bits, modulus=str(n), model=model, b1=b1, samples=[])
        ref = {}
        plans = list(plans)
        for repeat in range(repeats + 1):
            jobs = list(warmup_plans if repeat == 0 else plans)
            self.random.shuffle(jobs)
            for backend, curves, wg in jobs:
                row, points = self.run(bits, n, model, b1, backend, curves, wg, repeat)
                if any(ref[sigma] != x for sigma, x in points.items() if sigma in ref):
                    raise ValueError(f"save mismatch: {bits} {model} {row}")
                if len(points) > len(ref):
                    ref = points
                condition["samples"].append(row)
                print(f"{model} {bits:4} {backend:6} {curves:6} wg={wg:3} r{repeat}: "
                      f"{wall_rate(row):9.1f} curves/s wall, {row['inner']:.4f} s inner", flush=True)
        condition["verified_points"] = sum(s["curves"] for s in condition["samples"])
        self.summary["conditions"].append(condition)
        pending = self.root / "summary.json.pending"
        pending.write_text(json.dumps(self.summary, indent=2), encoding="utf-8")
        pending.replace(self.root / "summary.json")


def wall_rate(row):
    return row["curves"] / row["wall"]


def paired_plans(samples, backend, metric):
    plans = {}
    for r in samples:
        if r["repeat"] and r["backend"] == backend:
            plans.setdefault((r["curves"], r["wg"]), []).append(r["curves"] / r[metric])
    return sorted(((statistics.median(rates), *plan) for plan, rates in plans.items()), reverse=True)


def run_sweep(collector, args):
    for model in args.models:
        for bits in args.widths:
            # Small cases need more columns to keep device events measurable.
            b1 = 5000 if bits <= 255 else 1000
            plans = [("cuda", size, 0) for size in args.batches]
            plans.extend(("opencl", size, wg) for wg in args.workgroups for size in args.batches)
            warm = [("cuda", args.batches[0], 0), ("opencl", args.batches[0], args.workgroups[0])]
            collector.condition(bits, model, b1, plans, args.repeats, warm)


def run_confirm(collector, args):
    conditions = {}
    for path in args.sweep:
        sweep = json.loads(path.read_text(encoding="utf-8"))
        if sweep["device"] != collector.telemetry.identity or sweep["program"] != collector.versions:
            raise ValueError(f"sweep device or programs do not match confirmation: {path}")
        for c in sweep["conditions"]:
            key = c["model"], c["bits"]
            if key in conditions:
                existing = conditions[key]
                if (existing["modulus"], existing["b1"]) != (c["modulus"], c["b1"]):
                    raise ValueError(f"sweep workload changed for {key}")
                existing["samples"].extend(c["samples"])
            else:
                conditions[key] = c
    for _, c in sorted(conditions.items()):
        if c["bits"] not in args.widths or c["model"] not in args.models:
            continue
        # Edwards inner time is a device event on both sides. PARAM3's legacy
        # OpenCL field includes host submission time. Use it to size the run.
        ranked = {k: paired_plans(c["samples"], k, "inner") for k in collector.exe}
        peak = {backend: rows[0] for backend, rows in ranked.items()}
        plans = [(backend, curves, wg) for backend, (_, curves, wg) in peak.items()]
        # Use the smallest measured plan within 95% as a second candidate.
        for backend, rows in ranked.items():
            rate, _, _ = rows[0]
            _, curves, wg = min((r for r in rows if r[0] >= rate * .95), key=lambda r: (r[1], r[2]))
            plan = backend, curves, wg
            if plan not in plans:
                plans.append(plan)
        if all(wg for backend, _, wg in plans if backend == "opencl"):
            auto = next((row for row in ranked["opencl"] if not row[2]), None)
            if auto:
                plans.append(("opencl", auto[1], 0))
        min_seconds = min(curves / rate for rate, curves, _ in peak.values())
        b1 = min(260000, max(c["b1"], round(c["b1"] * args.target / min_seconds / 500) * 500))
        print(f"Confirming {c['model']} {c['bits']} bit with B1={b1}: {plans}", flush=True)
        collector.condition(c["bits"], c["model"], b1, plans, args.repeats, plans[:2])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--opencl", type=Path, required=True)
    p.add_argument("--cuda", type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=["sweep", "confirm"], default="sweep")
    p.add_argument("--widths", nargs="+", type=int, default=WIDTHS)
    p.add_argument("--batches", nargs="+", type=int, default=BATCHES)
    p.add_argument("--workgroups", nargs="+", type=int, default=[0])
    p.add_argument("--models", nargs="+", choices=["e", "m3"], default=["e", "m3"])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--sweep", nargs="+", type=Path)
    p.add_argument("--target", type=float, default=1.8, help="fastest peak plan's intended inner seconds")
    args = p.parse_args()
    if not all(32 <= w <= 1024 for w in args.widths) or len(set(args.widths)) != len(args.widths):
        p.error("--widths requires distinct modulus widths in 32..1024")
    if min(args.batches) < 1 or len(set(args.batches)) != len(args.batches):
        p.error("--batches requires distinct positive curve counts")
    if min(args.workgroups) < 0 or len(set(args.workgroups)) != len(args.workgroups):
        p.error("--workgroups requires distinct nonnegative OpenCL sizes")
    if args.repeats < 1 or not args.target > 0:
        p.error("--repeats and --target must be positive")
    if args.phase == "confirm" and args.sweep is None:
        p.error("confirmation needs --sweep")
    collector = Collector(args)
    try:
        if args.phase == "sweep":
            run_sweep(collector, args)
        else:
            run_confirm(collector, args)
    finally:
        collector.telemetry.close()


if __name__ == "__main__":
    main()
