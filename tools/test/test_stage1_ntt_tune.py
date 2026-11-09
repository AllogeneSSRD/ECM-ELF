#!/usr/bin/env python3
"""Check Stage1 automatic NTT tune persistence, fixed-task identity and memory admission."""
import argparse
import json
from pathlib import Path
import subprocess

from test_stage1_ntt_production import saved_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    exe, out = args.exe.resolve(strict=True), args.output.resolve()
    out.mkdir(parents=True)
    p, b1, curves = 4423, 32, 16
    database = out / "auto.tune.json"
    fixture = f'[stage1]\ndevice = {args.device}\ngpu_param = 0\ngpu_algorithm = 2\n' \
              'log_file = ""\nckpt_seconds = 0\nexp_cache = "off"\n' \
              f'ntt_tune_file = "{database.as_posix()}"\n'

    def run(name, flags, cfg=fixture, expression=None, ok=True):
        directory = out / name
        directory.mkdir()
        config = directory / "ecm.toml"
        config.write_text(cfg, encoding="utf-8")
        process = subprocess.run([str(exe), "--config", str(config), *flags], input=expression,
                                 cwd=directory, text=True, encoding="utf-8", errors="replace",
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        (directory / "run.log").write_text(process.stdout, encoding="utf-8")
        assert (process.returncode == 0) == ok, (name, process.stdout[-2000:])
        return process.stdout

    flags = ["-gpucurves", str(curves), "-sigma", "26", "-save", "out.save", str(b1), str(b1)]
    expression = f"2^{p}-1\n"
    assert "source=short_tune" in run("first", flags, expression=expression)
    old = database.read_bytes()
    assert "source=tune" in run("cached", flags, expression=expression)
    assert database.read_bytes() == old
    assert saved_points(out / "first/out.save", p, b1) == saved_points(out / "cached/out.save", p, b1)
    assert len(saved_points(out / "cached/out.save", p, b1)) == curves
    print("PASS default auto short tune, durable cache hit and exact task curve count", flush=True)

    assert "stage1_ntt_tune:" in run("full", ["--tune", "ntt", "--bits", str(p)])
    profile, = json.loads(database.read_text(encoding="utf-8"))["profiles"].values()
    assert profile["p"] == p and profile["budget_bytes"] > 0
    assert str(curves) in profile["short"]
    for mode in ("linear", "weighted"):
        row = profile[mode]
        assert row["single_curve"]["batch"] == 1
        best = max(s["curves_per_second"] for s in row["samples"].values())
        recommend = min(s["batch"] for s in row["samples"].values() if s["curves_per_second"] >= best * .95)
        assert row["peak"]["curves_per_second"] == best
        assert row["recommended_batch"] == recommend
        assert row["recommended"]["plan"]["length"] > 0
        assert all(s["peak_bytes"] <= profile["budget_bytes"] for s in row["samples"].values())
    assert "source=tune" in run("full_cached", [*flags[:1], "1", *flags[2:]], expression=expression)
    print("PASS full tune CLI, per-backend 95% recommendations and production selection", flush=True)

    limited = fixture + 'ntt_convolution = "linear"\nntt_memory_mib = 1\n'
    log = run("low_budget", ["-gpucurves", "32", "-sigma", "26", "20"],
              cfg=limited, expression="2^262139-1\n", ok=False)
    assert "VRAM budget" in log
    print("PASS bounded VRAM refusal", flush=True)


if __name__ == "__main__":
    main()
