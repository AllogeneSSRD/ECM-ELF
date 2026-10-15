"""Compare production CUDA and exact Tensor NTTs across lengths; preserve raw evidence."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

MODES = {0: "CUDA production", 1: "Tensor64", 2: "Tensor40"}
OPERATIONS = {0: "forward", 1: "inverse_pointwise", 2: "convolution"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(text, label):
    return [dict(re.findall(r"(\w+)=([^\s]+)", row)) for row in re.findall(re.escape(label) + r": (.*)", text)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--sizes", type=int, nargs="+", default=[12, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    p.add_argument("--batches", type=int, nargs="+", default=[1])
    p.add_argument("--threads", type=int, nargs="+", default=[256])
    p.add_argument("--tile-bits", type=int, default=12)
    p.add_argument("--operations", type=int, nargs="+", choices=OPERATIONS, default=[0, 1, 2])
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--execution", choices=("native", "graph"), default="native")
    p.add_argument("--iterations", type=int, default=1, help="Chained operations/sample; graph mode amortizes host submission. All final words checked.")
    p.add_argument("--max-words", type=int, default=1 << 26, help="batch*length limit; four main arrays require 32 bytes/word")
    p.add_argument("--skip-gate", action="store_true", help="Record that this run does not repeat the independent GMP gate")
    a = p.parse_args()
    if any(k < 6 or k > 27 for k in a.sizes) or any(b < 1 or b > 64 for b in a.batches):
        p.error("length exponent must be 6..27; batch 1..64")
    if a.tile_bits < 6 or a.tile_bits > 12 or any(t not in (128, 256, 512) for t in a.threads) or not 1 <= a.repeats <= 101:
        p.error("tile bits 6..12; threads 128/256/512; repeats 1..101")
    if not 1 <= a.iterations <= 256:
        p.error("iterations must be 1..256")
    a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()):
        raise ValueError("Use a fresh output directory")
    repo = Path(__file__).resolve().parents[2]
    exe = a.exe.resolve()
    manifest = json.loads((exe.parent / "manifest.json").read_text(encoding="utf-8-sig"))
    identity = sha(exe)
    env = {k: v for k, v in os.environ.items() if not k.startswith("NTT_")}

    def verify():
        if identity != sha(exe) or identity != manifest["sha256"].lower():
            raise ValueError("Binary changed")
        for name, digest in manifest["sources"].items():
            if sha(repo / name) != digest.lower():
                raise ValueError("Source changed: " + name)
        if sha(exe.parent / "gmp-10.dll") != manifest["gmp_dll_sha256"].lower():
            raise ValueError("GMP DLL changed")

    verify()
    info = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.limit,memory.total,memory.used,utilization.gpu", "--format=csv"], capture_output=True, text=True, check=True).stdout
    (a.output / "devices_before.csv").write_text(info, encoding="utf-8")
    metadata = {"exe": str(exe), "sha256": identity, "manifest": manifest, "device": a.device,
                "script_sha256": sha(Path(__file__)), "date": datetime.now(timezone.utc).date().isoformat(), "recorded_utc": datetime.now(timezone.utc).isoformat(), "timing": "CUDA-event sample divided by chained operations; init, fill, CUDA reference, validation and graph capture excluded; no discarded warmup samples; reference executes CUDA before samples",
                "reference_note": "GMP full-transform oracle at k=6,8,10,12,13,14; every measured A/B word compared with production CUDA outside timing",
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "gate": {}, "runs": [], "skipped": []}
    (a.output / "provenance.json").write_text(json.dumps({k: v for k, v in metadata.items() if k != "runs"}, indent=2), encoding="utf-8")
    rows_out = []
    telemetry = None
    with (a.output / "telemetry.csv").open("w", encoding="utf-8") as stream:
        try:
            telemetry = subprocess.Popen(["nvidia-smi", f"--id={a.device}", "--query-gpu=timestamp,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu,memory.used", "--format=csv", "--loop-ms=500"], stdout=stream, stderr=subprocess.STDOUT)
            if not a.skip_gate:
                for flag, expected in [("--check", 0), ("--fault", 3)]:
                    verify()
                    command = [str(exe), str(a.device), flag]
                    result = subprocess.run(command, env=env, capture_output=True, timeout=300)
                    text = result.stdout.decode("utf-8", errors="replace") + result.stderr.decode("utf-8", errors="replace")
                    (a.output / (flag[2:] + ".log")).write_text(text, encoding="utf-8")
                    checks = fields(text, "tc_length_gate")
                    if result.returncode != expected or len(checks) != 1 or int(checks[0]["bad"]) != (1 if expected else 0):
                        raise RuntimeError((flag, result.returncode, text[-1600:]))
                    if not re.search(r"tc_length_memory: .*live_bytes=0 fuse_live_bytes=0", text):
                        raise RuntimeError("Gate allocation leak")
                    metadata["gate"][flag] = checks[0]
                    print("gate", flag, checks[0], flush=True)
            for k in a.sizes:
                for batch in a.batches:
                    if (1 << k) * batch > a.max_words:
                        metadata["skipped"].append({"k": k, "batch": batch, "reason": "max_words"})
                        continue
                    for threads in a.threads:
                        for op in a.operations:
                            verify()
                            name = f"k{k}_b{batch}_t{a.tile_bits}_cta{threads}_{OPERATIONS[op]}"
                            command = [str(exe), str(a.device), "--bench-graph" if a.execution == "graph" else "--bench", str(k), str(batch), str(a.tile_bits), str(threads), str(op), str(a.repeats), str(a.iterations)]
                            begin = time.monotonic()
                            result = subprocess.run(command, env=env, capture_output=True, timeout=300)
                            text = result.stdout.decode("utf-8", errors="replace") + result.stderr.decode("utf-8", errors="replace")
                            (a.output / (name + ".log")).write_text(text, encoding="utf-8")
                            samples = fields(text, "tc_length_bench")
                            plans = fields(text, "tc_length_plan")
                            if result.returncode or len(samples) != 3 * a.repeats or len(plans) != 1:
                                raise RuntimeError((name, result.returncode, text[-1600:]))
                            if any(int(s["bad"]) or int(s["k"]) != k or int(s["batch"]) != batch or int(s["operation"]) != op or int(s["checked_words"]) != (1 << k) * batch * 2 for s in samples):
                                raise RuntimeError("Invalid or incorrect sample: " + name)
                            if not re.search(r"tc_length_memory: .*live_bytes=0 fuse_live_bytes=0", text):
                                raise RuntimeError("Benchmark allocation leak")
                            item = {"name": name, "command": command, "plan": plans[0], "raw": samples, "driver_seconds": time.monotonic() - begin}
                            for mode in MODES:
                                group = [s for s in samples if int(s["mode"]) == mode]
                                if len(group) != a.repeats:
                                    raise RuntimeError("Unequal mode counts")
                                seconds = [float(s["seconds"]) for s in group]
                                median = statistics.median(seconds)
                                row = {"log2_length": k, "length": 1 << k, "batch": batch, "tile_bits": min(k, a.tile_bits), "tensor_threads": threads, "execution": a.execution, "iterations": a.iterations,
                                       "operation": OPERATIONS[op], "mode": MODES[mode], "median_seconds": median,
                                       "mad_seconds": statistics.median(abs(s - median) for s in seconds), "host_median_seconds": statistics.median(float(s["host_seconds"]) for s in group),
                                       "repeats": len(seconds), "batch_iter_per_s": 1 / median, "operations_per_s": batch / median,
                                       "transforms_per_s": (3 if op == 2 else 1) * batch / median, "checked_words": sum(int(s["checked_words"]) for s in group),
                                       "bad": 0, "data_bytes": int(plans[0]["data_bytes"]), "table_base_bytes": int(plans[0]["table_base_bytes"]),
                                       "root_bytes": int(plans[0]["root_bytes"]), "passes_forward": int(plans[0]["passes"]), "coop_outer": int(plans[0]["coop"])}
                                rows_out.append(row)
                            baseline = rows_out[-3]["median_seconds"]
                            for row in rows_out[-3:]:
                                row["speedup_vs_cuda"] = baseline / row["median_seconds"]
                                row["gain_percent"] = 100 * (1 - row["median_seconds"] / baseline)
                            item["summary"] = rows_out[-3:]
                            metadata["runs"].append(item)
                            verify()
                            (a.output / "measurements.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                            with (a.output / "summary.csv").open("w", newline="", encoding="utf-8") as dest:
                                writer = csv.DictWriter(dest, fieldnames=list(rows_out[0]))
                                writer.writeheader()
                                writer.writerows(rows_out)
                            print(name, " | ".join(f"{r['mode']}: {r['median_seconds'] * 1e3:.4f}ms ({r['speedup_vs_cuda']:.3f}x)" for r in rows_out[-3:]), flush=True)
        finally:
            if telemetry is not None:
                telemetry.terminate()
                telemetry.wait(timeout=10)
    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
