#!/usr/bin/env python3
"""Check Stage1 automatic NTT tune persistence, fixed-task identity and memory admission."""
import argparse
from pathlib import Path
import re
import subprocess
import tomllib

from test_stage1_ntt_production import saved_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    exe, out = args.exe.resolve(strict=True), args.output.resolve()
    out.mkdir(parents=True)
    p, b1, curves = 4423, 32, 128
    database = out / "tune.toml"
    database.write_text('''[profile]
format = 13
unit = "full_stage2"

[condition.0]
uuid_hex = "fixture"

[ntt.0]
matrix = [
    ["length", "batch", "mean_seconds", "stddev_seconds", "repeats", "verified", "measurement_ids"],
    [1024, 1, 0.01, 0, 1, 1, "4:test"],
]
''', encoding="utf-8")
    fixture = (f'config_version = 4\n[shared]\ntune_file = "{database.as_posix()}"\nlog_file = ""\n'
               f'[stage1]\ntarget = "gpu"\nparam = "m0"\nstage1_gpu_device = {args.device}\n'
               'gpu_algorithm = "ntt"\ncheckpoint_interval_seconds = 0\nexponent_cache_dir = "off"\n'
               'handoff_mode = "off"\nntt_max_batch = 128\n')

    def run(name, flags, cfg=fixture, expression=None, ok=True):
        directory = out / name
        directory.mkdir()
        config = directory / "ecm.toml"
        config.write_text(cfg, encoding="utf-8")
        process = subprocess.run([str(exe), "--config", str(config), *flags], input=expression,
                                 cwd=directory, text=True, encoding="utf-8", errors="replace",
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        (directory / "run.log").write_text(process.stdout, encoding="utf-8")
        assert (process.returncode == 0) == ok, (name, process.stdout[-2000:])
        return process.stdout

    flags = ["-gpucurves", str(curves), "-sigma", "26", "-save", "out.save", str(b1), str(b1)]
    expression = f"2^{p}-1\n"
    first = run("first", flags, expression=expression)
    assert "stage1_ntt_runtime_tune:" in first and "source=tune" in first
    old = database.read_bytes()
    assert "source=tune" in run("cached", flags, expression=expression)
    assert "source=tune" in run("same_packing", flags, expression="2^4421-1\n")
    assert database.read_bytes() == old
    assert saved_points(out / "first/out.save", p, b1) == saved_points(out / "cached/out.save", p, b1)
    assert len(saved_points(out / "cached/out.save", p, b1)) == curves
    print("PASS default runtime tune, durable cache hit and exact task curve count", flush=True)

    mirror = out / "screen.log"
    config = fixture.replace('log_file = ""', f'log_file = "{mirror.as_posix()}"')
    log = run("full", ["--tune", "ntt", "--bits", str(p)], cfg=config)
    legend = ("ntt_type = convolution:tile_bits:max_radix_bits:graph_bits:"
              "warp_tail:forward_pass_bits:inverse_pass_bits (0=linear, 1=weighted)")
    header = "length max_exp ntt_type batch iter/s"
    assert log.index(legend) < log.index(header)
    lengths = [int(n) for n in re.findall(r"^(?:\[[^\n]*\] )?(?:(?:Best|95%) +)?(\d+) +\d+ +[01]:", log, re.M)]
    assert len(set(lengths)) == 2, lengths
    best_lines = re.findall(r"Best +\d+ +\d+ +[01]:[\d:,]+ +\d+c +\d+ iter/s", log)
    mirrored = mirror.read_text(encoding="utf-8")
    assert legend in mirrored and len(best_lines) == 3 and all(line in mirrored for line in best_lines)
    profile = tomllib.loads(database.read_text(encoding="utf-8"))
    assert profile["profile"]["format"] == 18 and set(profile["ntt"]) == {"matrix"}
    columns, *values = profile["ntt"]["matrix"]
    assert columns[:5] == ["max_exp", "length", "iter_per_s", "batch", "ntt_type"]
    assert not {"p", "exponent", "measurement_ids", "mean_seconds", "log2_length"} & set(columns)
    rows = [dict(zip(columns, row)) for row in values]
    stage2, = [r for r in rows if r["ntt_type"] == "stage2_convolution"]
    assert stage2["iter_per_s"] == 100
    for mode in ("linear", "weighted"):
        group = [r for r in rows if r["ntt_type"] == f"stage1_{mode}"]
        assert {1, 128} <= {r["batch"] for r in group}
        assert all(r["max_exp"] >= r["reference_bits"] and
                   0 < r["ntt_memory_bytes"] <= r["peak_bytes"] <= r["budget_bytes"] and
                   r["program_version"] and r["updated_at"].endswith("Z") for r in group)
        for r in group:
            assert abs(r["iter_per_s"] * r["median_ms"] / (1000*r["batch"]*r["timed_steps"]) - 1) < 1e-7
            assert r["timed_steps"] % r["block_steps"] == 0 and r["stddev_seconds"] >= 0
    snapshot = database.read_bytes()
    hot = run("hot_matrix", ["--tune", "ntt", "--bits", str(p)], cfg=config)
    assert "measured=1 " in hot and hot.count("stage1_ntt_recheck:") == 3
    assert ("status=updated" in hot) == (database.read_bytes() != snapshot)
    assert "source=tune" in run("full_cached", [*flags[:1], "1", *flags[2:]], expression=expression)
    assert re.search(r"95% +\d+ +\d+ +[01]:[\d:,]+ +\d+c +\d+ iter/s", log)
    print("PASS full tune CLI, measured 95% batch and production selection", flush=True)

    direct = run("direct", ["--tune", "ntt,length=8"],
                 cfg=fixture + "ntt_window_bits = 0\nntt_length = 8\n")
    assert re.search(r"Best +8 +\d+ +[01]:\d+:\d+:0:", direct)
    assert set(re.findall(r"8 +\d+ +([01]):\d+:\d+:0:", direct)) == {"0", "1"}
    print("PASS both modes grouped at one length with direct submission", flush=True)

    p = 58631
    wide = f"2^{p}-1\n"
    one_curve = [*flags[:1], "1", *flags[2:]]
    assert "length=4096" in run("seed_binary", one_curve, cfg=fixture + "ntt_length = 4096\n", expression=wide)
    added = run("add_mixed_to_binary", one_curve, expression=wide)
    assert "source=tune" in added and "stage1_ntt_runtime_tune:" not in added
    run("seed_mixed", one_curve, cfg=fixture + 'ntt_convolution = "weighted"\nntt_length = 2560\n', expression=wide)
    columns, *values = tomllib.loads(database.read_text(encoding="utf-8"))["ntt"]["matrix"]
    rows = [dict(zip(columns, row)) for row in values]
    assert any(row["ntt_type"] == "stage2_convolution" for row in rows)
    mixed = next(row for row in rows if row["length"] == 2560 and row["batch"] == 1)
    assert mixed["ntt_type"] == "stage1_weighted" and mixed["max_exp"] == 64000
    assert mixed["forward"].startswith("r5+") and mixed["inverse"].endswith("+r5")
    cached = database.read_bytes()
    assert "source=tune" in run("mixed_cached", one_curve, expression=wide)
    assert database.read_bytes() == cached
    weighted = fixture + 'ntt_convolution = "weighted"\n'
    assert "length=2560" in run("mixed_default", one_curve, cfg=weighted, expression=wide)
    assert "length=4096" in run("binary_for_tile", one_curve, cfg=weighted + "ntt_tile = 10\n", expression=wide)
    assert saved_points(out / "mixed_default/out.save", p, b1) == saved_points(out / "seed_binary/out.save", p, b1)
    print("PASS radix-5, explicit tile fallback and reuse of a covered binary plan", flush=True)

    # This matrix checks mixed-layout coverage at every shortlisted batch.
    # The preceding full matrix already verifies the default timing and reuse.
    log = run("full_mixed", ["--tune", f"ntt,exp={p},time=0.25"])
    lengths = [int(n) for n in re.findall(r"^(?:\[[^\n]*\] )?Best +(\d+) +\d+ +[01]:", log, re.M)]
    assert lengths == [4096, 2048, 2560, 4096], lengths
    columns, *values = tomllib.loads(database.read_text(encoding="utf-8"))["ntt"]["matrix"]
    mixed_rows = [dict(zip(columns,row)) for row in values if row[columns.index("length")] == 2560]
    assert {row["dataflow"] & 384 for row in mixed_rows} == {0,128,384}
    print("PASS full tune compares four capacity entries and all mixed layouts", flush=True)

    limited = fixture + 'ntt_convolution = "linear"\nntt_memory_mib = 1\n'
    log = run("low_budget", ["-gpucurves", "128", "-sigma", "26", "20"],
              cfg=limited, expression="2^262139-1\n", ok=False)
    assert "VRAM budget" in log
    print("PASS bounded VRAM refusal", flush=True)


if __name__ == "__main__":
    main()
