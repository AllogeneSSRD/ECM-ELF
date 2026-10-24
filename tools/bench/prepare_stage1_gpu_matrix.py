"""Prepare one Edwards/PARAM0 Stage1 batch per input, two blocks per SM.

Only --version / --gpu-info are queried. The user starts the prepared queue.
Geometry comes from the selected executable, not a duplicated TPI table.
The requested B1 depends on TPI; ECMSTAGE1 filenames carry curve and B1.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from analyze_prime95_stage1 import input_scope

ROOT = Path(__file__).resolve().parents[2]
B1_BY_TPI = {4: 1_000_000, 8: 200_000, 16: 50_000, 32: 10_000}
B1_TOKENS = {1_000_000: "1e6", 200_000: "2e5", 50_000: "5e4", 10_000: "1e4"}


def prepare(exe, device, output, source, template=None, models=("ed", "mont0")):
    exe = exe.resolve(strict=True)
    if output.exists():
        raise FileExistsError("Use a new output directory; existing measurements are preserved")
    template_paths = [] if template is None else [template] + [
        template.with_name(f"{template.stem}_{model}{template.suffix}") for model in models]
    for path in template_paths:
        if path.exists():
            raise FileExistsError(path)
    cases = input_scope(source)
    output.mkdir(parents=True)
    query_config = output / "metadata-query.toml"
    query_config.write_text("[shared]\n[stage1]\n", encoding="utf-8")
    version = subprocess.check_output([str(exe), "--version"], encoding="utf-8").strip()
    rows, lines, model_lines = [], [], {}
    identity = None
    for model in models:
        command = [str(exe), "--config", str(query_config.resolve()), "--gpu-info",
                   "--curve", "ed" if model == "ed" else "mont", "--target", "gpu", "-d", str(device)]
        if model == "mont0":
            command += ["--gpu-param", "0"]
        info = subprocess.check_output(command, encoding="utf-8")
        values = dict(re.findall(r"^(\w+)=(.*)$", info, re.M))
        expected = "ed" if model == "ed" else "mont"
        if values.get("backend") != "CUDA/CGBN" or values.get("curve_model") != expected:
            raise ValueError(f"Unexpected {model} metadata")
        if model == "mont0" and values.get("gpu_param") != "0":
            raise ValueError("Expected Montgomery PARAM0 metadata")
        sm_count = int(values["sm_count"])
        if sm_count <= 0:
            raise ValueError("Invalid SM count")
        current = (values["name"], sm_count)
        if identity is not None and identity != current:
            raise ValueError("Device changed between geometry queries")
        identity = current
        carry = int(values["carry_bits"])
        tiers = sorted(tuple(map(int, m)) for m in re.findall(
            r"^tier bits=(\d+) tpb=(\d+) tpi=(\d+)", info, re.M))
        model_lines[model] = []
        for exponent, case in sorted(cases.items()):
            tier = next((t for t in tiers if case["intended_cofactor_bits"] + carry <= t[0]), None)
            if tier is None:
                raise ValueError(f"No compiled {model} tier for M{exponent}")
            bits, tpb, tpi = tier
            if tpb % tpi or tpi not in B1_BY_TPI:
                raise ValueError("Unsupported TPB/TPI geometry")
            b1 = B1_BY_TPI[tpi]
            blocks = 2 * sm_count
            curves = blocks * (tpb // tpi)
            prefix = "e" if model == "ed" else "m0"
            save = f"{prefix}_m{exponent}_{B1_TOKENS[b1]}.save"
            factors = ",".join(case["intended_known_factors"])
            rows.append(dict(model=model, exponent=exponent, cofactor_bits=case["intended_cofactor_bits"],
                             container_bits=bits, carry_bits=carry, tpb=tpb, tpi=tpi, sm_count=sm_count,
                             launched_blocks=blocks, curves_per_batch=curves, b1=b1, b2=b1,
                             save_name=save, repeats=1, projection_to_b1_1e6=1_000_000 / b1))
            model_lines[model].append(f'ECMSTAGE1=1,2,{exponent},-1,{save},{b1},0,{curves},"{factors}"')
        lines.extend(model_lines[model])
        (output / f"gpu-info-{model}.txt").write_text(info, encoding="utf-8")

    def payload(entries):
        return ("\r\n".join(entries) + "\r\n").encode("ascii")

    (output / "worktodo.txt").write_bytes(payload(lines))
    for model, entries in model_lines.items():
        (output / f"worktodo_{model}.txt").write_bytes(payload(entries))
    if template is not None:
        template.parent.mkdir(parents=True, exist_ok=True)
        template.write_bytes(payload(lines))
        for model, path in zip(models, template_paths[1:]):
            path.write_bytes(payload(model_lines[model]))
    absolute = lambda name: json.dumps((output / name).resolve().as_posix())
    config = (f'[shared]\nworktodo = {absolute("worktodo.txt")}\npriority = 1\nskip = 2\n\n'
              f'[stage1]\ncurve = "mont"\ngpu_param = 0\ntarget = "gpu"\ndevice = {device}\nnaf_w = 6\nsigma = 20260922\n'
              'auto_stage2 = 0\nedwards_stage2_transfer = 0\nckpt_seconds = 0\nverbose = true\n'
              f'exp_cache = {absolute("exp_cache")}\n'
              f'tmp_dir = {absolute("saves")}\n'
              f'log_file = {absolute("screen_stage1_gpu.log")}\n'
              f'finished = {absolute("worktodo.finished.txt")}\n'
              f'stage1_results_file = {absolute("stage1_results.jsonl")}\n'
              f'stage1_submission_file = {absolute("results.json.txt")}\n')
    config_path = output / "ecm.toml"
    config_path.write_text(config, encoding="utf-8")
    (output / "program-version.txt").write_text(version + "\n", encoding="utf-8")
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    start_command = [str(exe), "--config", str(config_path.resolve()), "--worker", "1"]
    report = dict(schema=2, prepared_utc=datetime.now(timezone.utc).isoformat(),
                  program_version=version, executable=str(exe), gpu_model=identity[0],
                  device=device, sm_count=identity[1], b1_by_tpi=B1_BY_TPI,
                  blocks_per_sm=2, block_meaning="launch grid / SM count, not measured residency",
                  models=list(models), repeats=1, tasks=len(lines), cases=rows,
                  source_worktodo=str(source.resolve()), queue_format="ECMSTAGE1",
                  timing="main-loop progress work/time span; excludes setup, NAF dictionary, GCD and output",
                  projection="per_curve_at_B1 * target_B1 / measured_B1",
                  start_command=start_command, benchmark_started=False)
    (output / "prepared.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps(dict(program_version=version, gpu_model=identity[0], sm_count=identity[1],
                          tasks=len(lines), output=str(output.resolve())), ensure_ascii=False))
    print("& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in start_command))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--source", type=Path, default=ROOT / "tools/benchmarks/prime95/worktodo_ecm_stage1.txt")
    p.add_argument("--output", type=Path)
    p.add_argument("--models", choices=("ed", "mont0"), nargs="+", default=["ed", "mont0"])
    p.add_argument("--template", type=Path, help="Optional worktodo copy, plus _ed / _mont0 copies")
    args = p.parse_args()
    if args.device < 0:
        p.error("Device must be nonnegative")
    if len(set(args.models)) != len(args.models):
        p.error("Models must not repeat")
    output = args.output or ROOT / "data/experiments" / (
        f"stage1_gpu_tpi_2blocks_sm_d{args.device}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    prepare(args.exe, args.device, output, args.source, args.template, tuple(args.models))


if __name__ == "__main__":
    main()
