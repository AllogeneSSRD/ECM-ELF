"""Measure the two-slot queue on the 13 frozen small-B2 winners."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

from gpu_telemetry import TelemetryRecorder


MIB = 1 << 20
BUDGETS = [(6300, 256, 640), (4096, 128, 384), (3072, 64, 256), (2048, 32, 128)]
SMALLER_D = [(180180, 17280), (150150, 14400), (120120, 11520), (90090, 8640), (60060, 5760)]
PHASE = re.compile(r"stage2_phase_window: phase=\S+ start_ns=(\d+) end_ns=(\d+) wait_ns=\d+ gated=1")


class WindowNotQualified(ValueError):
    """The queue timing was rejected by the worker's live checks."""


def command(exe, row, folder, device, budgets, parallel=1):
    arena, batch, owner = budgets
    backend = lambda width: f"coop{width}" if width else "scalar"
    return [str(exe), "--config", str(folder/"ecm.toml"), "--save", str(folder/"points.save"),
            "--device", str(device), "--parallel", str(parallel), "--b2", str(row["B2"]), "--d", str(row["D"]),
            "--carrier-exponent", str(row["carrier"]), "--auto-carrier", "0", "--groot-layout", str(row["layout"]),
            "--giant-chain", str(row["chain"]), "--baby-backend", backend(row["baby"]),
            "--giant-backend", backend(row["giant"]), "--seed-backend", "scalar", "--reduce-backend", "scalar",
            "--short-calibration", "0", "--tune-runtime", "0", "--short-ntt-batch-tune", "0",
            "--relocation", "0", "--giant-tail-balance", "0", "--b2-round-chain", "0",
            "--weighted-ntt", "0", "--pack-zero", "1", "--factor-only", "--ntt-counts", "1", "--log-level", "phases",
            "--arena-mb", str(arena), "--batch-mb", str(batch), "--owner-budget-mb", str(owner),
            "--log", str(folder/"engine.log"), "--debug-log-file", str(folder/"debug.log"),
            "--results", str(folder/"result.jsonl"), "--submission", str(folder/"submission.txt"),
            "--tune-profile", str(folder/"tune.toml")]


def preflight(args, env, folder):
    probe = subprocess.run(args + ["--curves", "1", "--plan-only"], env=env, cwd=folder,
                           capture_output=True, text=True, encoding="utf-8", timeout=45)
    (folder/"plan.log").write_text(probe.stdout+probe.stderr, encoding="utf-8")
    for line in probe.stdout.splitlines():
        if not line.startswith("{"):
            continue
        report = json.loads(line)
        if "curve_workspace_memory" in report:
            memory = report["curve_workspace_memory"]
            reserved = memory["required_free_bytes"] + max(256*MIB, memory["peak_bytes"]//8)
            fits = (probe.returncode == 0 and memory["valid"] and memory["finished"]
                    and memory["initial_free_snapshot_fits"] and 2*reserved+256*MIB <= report["free_bytes"])
            return fits, reserved, report["free_bytes"]
    return False, 0, 0


def curve_outputs(folder, expected, row, count, workers, log):
    launches = re.findall(r"^curve_(start|done): (\d+)/", log, re.M)
    active = set()
    for event, index in launches:
        if event == "start":
            if index in active:
                raise ValueError("duplicate launch")
            active.add(index)
            if len(active) > workers:
                raise ValueError("more curves in flight than requested")
        else:
            if index not in active:
                raise ValueError("completion has no launch")
            active.remove(index)
    if active or len(launches) != 2*count or [x[0] for x in launches[:workers]] != ["start"]*workers:
        raise ValueError("incomplete launch sequence")
    rows = [json.loads(line) for line in (folder/"result.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != count or any((r["status"], int(r["N"]), r["b1"], r["b2"], r["d"], r["poly-size"]) !=
                                ("NF", expected, row["B1"], row["B2"], row["D"], row["P"]) for r in rows):
        raise ValueError("unexpected arithmetic result or geometry")
    route = (row["carrier"], row["chain"], row["baby"], row["giant"])
    route_keys = ("carrier_exponent", "giant_chain_block", "baby_coop", "giant_coop")
    if any(tuple(r["stage2-engine"][key] for key in route_keys) != route for r in rows):
        raise ValueError("unexpected arithmetic or point backend")
    for check in ("gmp_selftest_bad=0", "gmp_check_bad=0", "clean=1"):
        if log.count(check) != count:
            raise ValueError(f"missing arithmetic validation {check}")
    histogram = Counter(re.findall(r"^ntt_length_counts: (.+)$", log, re.M))
    if len(histogram) != 1 or histogram.total() != count:
        raise ValueError("NTT work changed across replays")
    names = ("debug.log.slot1", "debug.log.slot2") if workers == 2 else ("debug.log",)
    slots = [(folder/name).read_text(encoding="utf-8") for name in names]
    hashes = Counter(re.findall(r"descent_values: .*?hash=(\d+)", "\n".join(slots)))
    if len(hashes) != 1 or hashes.total() != count:
        raise ValueError("descent leaf fingerprint changed")
    if workers == 2:
        left, right = [[tuple(map(int, x)) for x in PHASE.findall(slot)] for slot in slots]
        if not left or not right or any(max(a[0], b[0]) < min(a[1], b[1]) for a in left for b in right):
            raise ValueError("gated phases overlapped")
    return {"leaf_hash": next(iter(hashes)), "ntt_counts": next(iter(histogram))}


def queue_sample(folder, expected, row, count, workers=2):
    log = (folder/"engine.log").read_text(encoding="utf-8")
    if count == 1:
        result = curve_outputs(folder, expected, row, count, workers, log)
        engine = re.findall(r"^stage2_full_wall: .*?\btotal=(\d+\.\d+)", log, re.M)
        if len(engine) != 1 or float(engine[0]) <= 0:
            raise ValueError("missing single-curve engine timing")
        return dict(result, curves=1, engine_seconds=float(engine[0]))
    decisions = re.findall(r"^stage2_parallel: (.+)$", log, re.M)
    windows = re.findall(r"^stage2_parallel_window: (.+)$", log, re.M)
    reason = "workers=2 reason=forced" if workers == 2 else "workers=1 reason=insufficient_samples"
    if len(decisions) != 1 or reason not in decisions[0] or len(windows) != 1:
        raise WindowNotQualified(f"no complete {workers}-slot window")
    timing = dict(re.findall(r"(\w+)=([0-9.e+-]+)", windows[0]))
    if int(timing["workers"]) != workers or int(timing["curves"]) != count or timing["checked"] != "1":
        raise WindowNotQualified("live route or memory differed from the preflight")
    result = curve_outputs(folder, expected, row, count, workers, log)
    middle_count = int(timing["middle_curves"])
    wall = float(timing["wall_seconds"])
    return dict(result, curves=count, wall_seconds=wall, wall_seconds_per_curve=wall/count,
                middle_seconds_per_curve=float(timing["middle_seconds"])/middle_count if middle_count else None)


def admitted_geometries(exe, row, device, folder, point, env, curves):
    layouts = (row["layout"], 1-row["layout"])
    degrees = [(row["D"], row["P"])] + [(d, p) for d, p in SMALLER_D if p < row["P"]]
    for d, p in degrees:
        found = False
        for layout in layouts:
            candidate = dict(row, D=d, P=p, layout=layout)
            for budgets in BUDGETS:
                name = f"preflight_d{d}_layout{layout}_{'_'.join(map(str, budgets))}"
                probe = folder/name
                probe.mkdir()
                (probe/"points.save").write_bytes((point+b"\n")*curves)
                (probe/"ecm.toml").write_text("[stage2]\nstage2_short_calibration=false\n",encoding="utf-8")
                args = command(exe,candidate,probe,device,budgets)
                fits, reserved, free = preflight(args,env,probe)
                print(f"{row['bits']} bit D={d} layout={layout} {budgets}: "
                      f"required per slot {reserved/MIB:.0f} MiB, free {free/MIB:.0f} MiB, fits={fits}",flush=True)
                if fits:
                    yield candidate,budgets,reserved,free
                    found = True
                    break
            if found:
                break


def run_sample(exe, row, folder, point, expected, device, budgets, env, count, workers=2):
    folder.mkdir()
    (folder/"points.save").write_bytes((point+b"\n")*count)
    (folder/"ecm.toml").write_text("[stage2]\nstage2_short_calibration=false\n",encoding="utf-8")
    # With a fresh tune and fewer than nine curves, auto stays on one slot
    # and emits the same native queue timing as forced two-slot mode.
    parallel = 1 if workers == 2 else 2 if count > 1 else 0
    args = command(exe,row,folder,device,budgets,parallel)
    (folder/"command.json").write_text(json.dumps(args,indent=2),encoding="utf-8")
    sampler = TelemetryRecorder(device,folder/"gpu.jsonl",.1)
    try:
        with (folder/"console.log").open("wb") as console:
            subprocess.run(args,env=env,cwd=folder,stdout=console,stderr=subprocess.STDOUT,check=True,timeout=900)
    finally:
        sampler.close()
    result = queue_sample(folder,expected,row,count,workers)
    gpu = [json.loads(line) for line in (folder/"gpu.jsonl").read_text().splitlines()]
    load = [s for s in gpu if (s["gpu_percent"] or 0)>=50 and s["power_mw"] is not None]
    result.update(device=sampler.identity,peak_mib=max(s["used_bytes"] for s in gpu)/MIB,
                  active_median_w=statistics.median(s["power_mw"] for s in load)/1000 if load else None,
                  folder=str(folder))
    return result


def benchmark_environment():
    env = {k:v for k,v in os.environ.items() if not k.startswith(("NTT_", "ECM_"))
           and k not in ("CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING")}
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", NTT_NO_PROGRESS="1", ECM_STAGE2_PHASE_TRACE="1")
    return env


def replay_point(inputs, row):
    bits = row["bits"]
    matches = list(inputs.glob(f"m{row['exponent']}_n{bits}_s*.save"))
    if len(matches) != 1:
        raise ValueError(f"expected one save for {bits} bits")
    point = next(line for line in matches[0].read_bytes().splitlines() if line.startswith(b"METHOD="))
    target = int(re.search(rb"(?:^|;\s*)N=([^;]+)",point)[1],0)
    if target.bit_length() != bits:
        raise ValueError("save width differs from matrix")
    return point, target


def main():
    # Result records include exact decimal moduli up to 16,384 bits.
    sys.set_int_max_str_digits(10000)
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("exe", "reference", "inputs", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--curves", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--widths", type=int, nargs="+")
    opt = parser.parse_args()
    if opt.curves < 2 or opt.repeats < 1:
        parser.error("use at least two curves and one repeat per point")
    exe, out = opt.exe.resolve(), opt.out.resolve()
    reference = json.loads(opt.reference.read_text(encoding="utf-8"))
    rows = sorted(reference["best"], key=lambda r: r["bits"])
    reference_widths = {row["bits"] for row in rows}
    if len(rows) != 13 or len(reference_widths) != 13:
        raise ValueError("expected the 13-point frozen small-B2 matrix")
    if opt.widths and (unknown := set(opt.widths) - reference_widths):
        parser.error(f"widths absent from the reference: {sorted(unknown)}")
    version = subprocess.check_output([str(exe), "--version"], text=True).strip().split()[-1]
    env = benchmark_environment()
    output = {"version": version, "curves_per_window": opt.curves, "repeats": opt.repeats, "rows": [], "failures": []}
    out.mkdir(parents=True)
    for row in rows:
        bits = row["bits"]
        if opt.widths is not None and bits not in opt.widths:
            continue
        point, target = replay_point(opt.inputs, row)
        try:
            folder = out/str(bits)
            folder.mkdir()
            rejected = []
            for chosen,budgets,reserved,free in admitted_geometries(exe,row,opt.device,folder,point,env,opt.curves):
                result = {k:chosen[k] for k in ("bits","exponent","B1","B2","D","P","carrier","layout","chain","baby","giant")}
                result.update(budgets=budgets,reservation_bytes=reserved,initial_free_bytes=free,folder=str(folder))
                if not opt.preflight_only:
                    try:
                        samples = [
                            run_sample(exe,chosen,folder/f"d{chosen['D']}_run{i+1}",point,target,opt.device,budgets,env,opt.curves)
                            for i in range(opt.repeats)
                        ]
                    except WindowNotQualified as error:
                        rejected.append({"D": chosen["D"], "reason": str(error)})
                        print(f"{bits} bit D={chosen['D']}: {error}",flush=True)
                        continue
                    if len({(s["leaf_hash"],s["ntt_counts"]) for s in samples}) != 1:
                        raise ValueError("curve outputs changed between runs")
                    result["samples"] = samples
                    print(bits,"bit", [round(s["wall_seconds_per_curve"],4) for s in samples],"s/curve",flush=True)
                result["rejected"] = rejected
                output["rows"].append(result)
                break
            else:
                raise ValueError("no candidate passed pair admission and timing checks")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            output["failures"].append({"bits":bits,"error":str(error)})
            print(f"{bits} bit: {error}",flush=True)
        (out/"summary.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    return int(bool(output["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
