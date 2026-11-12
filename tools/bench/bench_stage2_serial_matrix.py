"""Select and independently remeasure serial routes for the small-B2 matrix."""
import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
import sys

from bench_stage2_parallel_matrix import BUDGETS, benchmark_environment, command, replay_point, run_sample
from bench_stage2_version import load, write


SHAPE = ("D", "P", "carrier", "layout", "chain", "baby", "giant")
INPUT = ("bits", "exponent", "B1", "B2")
NEAR = 0.03


def shape_key(row):
    return tuple(row[key] for key in SHAPE) + tuple(row["budgets"])


def model_candidate(exe, row, folder, point, device, env):
    folder.mkdir()
    (folder/"points.save").write_bytes(point+b"\n")
    (folder/"ecm.toml").write_text("[stage2]\nstage2_short_calibration=false\n", encoding="utf-8")
    request = dict(row, D=0, layout="auto")
    args = command(exe, request, folder, device, row["budgets"], parallel=0) + ["--plan-only", "--curves", "1"]
    write(folder/"command.json", args)
    report = subprocess.run(args, env=env, cwd=folder, capture_output=True, text=True, encoding="utf-8", timeout=45)
    (folder/"plan.log").write_text(report.stdout+report.stderr, encoding="utf-8")
    report.check_returncode()
    lines = [json.loads(line) for line in report.stdout.splitlines() if line.startswith("{")]
    plan, = (line for line in lines if line.get("type") == "stage2_plan")
    if plan["carrier_exponent"] != row["carrier"]:
        raise ValueError("native plan changed arithmetic family")
    memory = plan["curve_workspace_memory"]
    if not (memory["valid"] and memory["finished"] and memory["initial_free_snapshot_fits"]):
        return None
    return dict(row, D=int(plan["D"]), P=int(plan["P"]), layout=int(plan["groot_balanced"]))


def screen_candidates(exe, historical, paired, folder, point, expected, device, env):
    candidates = {}

    def add(row, source, budgets):
        candidate = {key: row[key] for key in INPUT+SHAPE}
        candidate["budgets"] = budgets
        item = candidates.setdefault(shape_key(candidate), dict(candidate, sources=[], screening=[]))
        item["sources"].append(source)

    for reference in historical:
        family = reference["family"]
        row = dict(reference, budgets=BUDGETS[0])
        add(row, f"frozen_{family}", row["budgets"])
        model = model_candidate(exe, row, folder/f"model_{family}", point, device, env)
        if model is not None:
            add(model, f"model_{family}", row["budgets"])
    add(paired, "paired_shape", paired["budgets"])

    def screen(item, i, attempt):
        sample = run_sample(exe, item, folder/f"screen{i+1}_{attempt}", point, expected,
                            device, item["budgets"], env, count=1, workers=1)
        item["screening"].append(sample)
        print(f"  {','.join(item['sources'])}: D={item['D']} P={item['P']} "
              f"carrier={item['carrier']} layout={item['layout']} engine={sample['engine_seconds']:.4f}s", flush=True)

    choices = list(candidates.values())
    for i, item in enumerate(choices):
        screen(item, i, 1)
    best_time = min(item["screening"][0]["engine_seconds"] for item in choices)
    finalists = [i for i, item in enumerate(choices) if item["screening"][0]["engine_seconds"] <= best_time*(1+NEAR)]
    if len(finalists) > 1:
        for i in finalists:
            screen(choices[i], i, 2)
    for item in choices:
        if len({(s["leaf_hash"], s["ntt_counts"]) for s in item["screening"]}) != 1:
            raise ValueError("candidate output changed on screening replay")
        item["screen_engine_seconds"] = median(s["engine_seconds"] for s in item["screening"])
    winner = min((choices[i] for i in finalists), key=lambda item: item["screen_engine_seconds"])
    return winner, choices


def main():
    sys.set_int_max_str_digits(10000)
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("exe", "reference", "pair-summary", "inputs", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--widths", type=int, nargs="+")
    args = parser.parse_args()
    exe, out = args.exe.resolve(), args.out.resolve()
    paired = load(args.pair_summary)
    version = subprocess.check_output([str(exe), "--version"], text=True).strip().split()[-1]
    if version != paired["version"] or paired["failures"] or paired["curves_per_window"] != 2:
        raise ValueError("expected verified two-curve batches of the same version")
    reference = load(args.reference)["cells"]
    pairs = {r["bits"]: r for r in paired["rows"]}
    widths = sorted(args.widths or pairs)
    if len(pairs) != 13 or len(widths) != len(set(widths)) or set(widths)-pairs.keys():
        parser.error("expected 13 paired widths and a unique requested subset")
    env = benchmark_environment()
    repeats = paired["repeats"]
    summary = dict(version=version, curves_per_window=2, repeats=repeats, rows=[], failures=[])
    out.mkdir(parents=True)
    for bits in widths:
        print(f"{bits} bit", flush=True)
        folder = out/str(bits)
        folder.mkdir()
        pair = pairs[bits]
        sources = [r for r in reference if r["bits"] == bits]
        if {r["family"] for r in sources} != {"ordinary", "carrier"} or len(sources) != 2 or any(
            tuple(r[k] for k in INPUT) != tuple(pair[k] for k in INPUT) for r in sources
        ):
            raise ValueError("historical candidates do not cover both arithmetic families at the paired input")
        try:
            point, expected = replay_point(args.inputs, pair)
            best, screening = screen_candidates(exe, sources, pair, folder, point, expected, args.device, env)
            write(folder/"screening.json", screening)
            samples = [run_sample(exe, best, folder/f"holdout{i+1}", point, expected,
                                  args.device, best["budgets"], env, count=2, workers=1)
                       for i in range(repeats)]
            signature = (best["screening"][0]["leaf_hash"], best["screening"][0]["ntt_counts"])
            if any((s["leaf_hash"], s["ntt_counts"]) != signature for s in samples):
                raise ValueError("holdout replay changed the selected route's arithmetic or workload")
            if shape_key(best) == shape_key(pair) and any(
                (s["leaf_hash"], s["ntt_counts"]) != signature for s in pair["samples"]
            ):
                raise ValueError("the same route differs across serial and paired runs")
            result = {key: best[key] for key in INPUT+SHAPE}
            result.update(budgets=best["budgets"], candidates=len(screening), sources=best["sources"],
                          screen_engine_seconds=best["screen_engine_seconds"], samples=samples, folder=str(folder))
            summary["rows"].append(result)
            print(f"  selected {','.join(best['sources'])}: "
                  f"{[round(s['wall_seconds_per_curve'],4) for s in samples]} s/curve", flush=True)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            summary["failures"].append({"bits": bits, "error": str(error)})
            print(f"  failed: {error}", flush=True)
        write(out/"summary.json", summary)
    return int(bool(summary["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
