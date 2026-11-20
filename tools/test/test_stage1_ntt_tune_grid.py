"""Check the Stage1 tune grid, timed groups and read-only/runtime production selection."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tomllib

from test_stage1_ntt_production import edwards_stage1, ntt_executions, saved_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    exe = args.exe.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True)
    database = out / "tune.toml"

    def run(name, flags, *, stdin=None, runtime=False, settings="", convolution="linear", success=True):
        directory = out / name
        directory.mkdir()
        cfg = directory / "ecm.toml"
        cfg.write_text(f'config_version = 4\n[shared]\nlog_file = ""\ntune_file = "{database.as_posix()}"\n'
                       f'[stage1]\ntarget = "gpu"\ngpu_algorithm = "ntt"\nstage1_gpu_device = {args.device}\n'
                       f'ntt_convolution = "{convolution}"\nexponent_cache_dir = "off"\nhandoff_mode = "off"\n{settings}\n'
                       f'[stage2]\ntune_runtime_enabled = {str(runtime).lower()}\n', encoding="utf-8")
        result = subprocess.run([str(exe), "--config", str(cfg), *flags], input=stdin, cwd=directory,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", timeout=240)
        (directory / "run.log").write_text(result.stdout, encoding="utf-8")
        assert (result.returncode == 0) == success, (name, result.stdout[-2500:])
        lines = result.stdout.splitlines()
        prefix = "" if "--no-log-timestamp" in flags else r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] "
        for i, line in enumerate(lines):
            if "stage1_ntt_round:" in line or "stage1_ntt_graph_scan:" in line:
                assert re.fullmatch(prefix + "length max_exp ntt_type batch iter/s", lines[i+1]), name
        return result.stdout

    def rows():
        fields, *values = tomllib.loads(database.read_text(encoding="utf-8"))["ntt"]["matrix"]
        return [dict(zip(fields, row)) for row in values]

    def write_matrix(header, fields, values):
        matrix = "".join(json.dumps(row) + ",\n" for row in [fields, *values])
        database.write_text(header.split("matrix = [", 1)[0] + "matrix = [\n" + matrix + "]\n", encoding="utf-8")

    for index, spec in enumerate(("exp=61,length=8", "naf=12-6", "batch=0", "batch=513",
                                  "exp=31", "time=0", "ntt,naf=6,naf=7", "typo=1",
                                  "repeat=2", "repeat=1:0", "repeat=:2", "repeat=21:1",
                                  "repeat=1:2:3", "repeat=2:2,repeat=1:1")):
        log = run(f"bad{index}", ["-tune", spec], success=False)
        assert "INPUT[cli]" in log
        if index == 0:
            assert "conflict" in log and "length" in log and "exp" in log
    assert not database.exists()

    # Set a valid but deliberately slow/fast device time so both record paths
    # are deterministic even with normal fluctuations in GPU clocks.
    for mode, batch in (("compact", ""), ("explicit", ",batch=1")):
        database = out / f"recheck_{mode}.toml"
        flags = ["-tune", f"exp=127,naf=6,time=0.04{batch}", "32"]
        settings = "ntt_max_batch = 1\nntt_profile = 1"
        run(f"recheck_{mode}_seed", flags, settings=settings)
        for scale, status in ((1e-5, "updated"), (1e5, "kept")):
            text = database.read_text(encoding="utf-8")
            fields, *values = tomllib.loads(text)["ntt"]["matrix"]
            coarse, = (r for r in values if r[fields.index("sample_phase")] == 1)
            fine, = (r for r in values if r[fields.index("sample_phase")] == 2)
            fine[fields.index("iter_per_s")] *= scale
            for key in ("total_device_ms", "median_ms", "stddev_seconds"):
                fine[fields.index(key)] /= scale
            write_matrix(text, fields, values)
            previous = database.read_bytes()
            shorter = [flag.replace("time=0.04", "time=0.01") for flag in flags]
            if mode == "explicit" and status == "kept":
                shorter.insert(0, "--no-log-timestamp")
            log = run(f"recheck_{mode}_{status}", shorter, settings=settings)
            assert "measured=1 " in log and log.count("stage1_ntt_recheck:") == 1
            assert f"status={status}" in log and (previous != database.read_bytes()) == (status == "updated")
            updated = rows()
            assert len(updated) == 2
            assert next(r for r in updated if r["sample_phase"] == 1) == dict(zip(fields, coarse))
            if status == "updated":
                sample, = (r for r in updated if r["sample_phase"] == 2)
                assert sample["sample_target_ms"] == 4
                assert sample["iter_per_s"] > fine[fields.index("iter_per_s")]
                assert re.search(rf'Best +\d+ +\d+ +\S+ +1c +{sample["iter_per_s"]:.0f} iter/s', log)

    database = out / "recheck_roles.toml"
    flags = ["-tune", "exp=127,naf=6,time=0.02", "32"]
    limits = "ntt_max_batch = 2\nntt_profile = 1"
    run("roles_seed", flags, settings=limits)
    text = database.read_text(encoding="utf-8")
    fields, *values = tomllib.loads(text)["ntt"]["matrix"]
    for row in values:
        if row[fields.index("sample_phase")] == 2:
            b = row[fields.index("batch")]
            rate = 98 + b
            scale = row[fields.index("iter_per_s")] / rate
            for key in ("total_device_ms", "median_ms", "stddev_seconds"):
                row[fields.index(key)] *= scale
            row[fields.index("iter_per_s")] = rate
            row[fields.index("batch_roles")] = (1 | 8) if b == 1 else (2 | 4)
    write_matrix(text, fields, values)
    log = run("roles_raised", flags, settings=limits)
    assert log.count("stage1_ntt_recheck:") == 1 and "status=updated" in log
    assert re.search(r"95% +\d+ +\d+ +\S+ +2c ", log)
    assert {r["batch"]: r["batch_roles"] for r in rows() if r["sample_phase"] == 2} == {1: 1, 2: 14}

    for cap in (0, 513):
        log = run(f"invalid_cap_{cap}", ["-tune", "exp=127,batch=1"], settings=f"ntt_max_batch = {cap}", success=False)
        assert "ntt_max_batch" in log
    log = run("above_cap", ["-tune", "exp=127,batch=129"], settings="ntt_max_batch = 128", success=False)
    assert "exceeds [stage1] ntt_max_batch" in log

    database = out / "wide.toml"
    wide = ["-tune", "exp=127,naf=6,batch=128-129,time=0.02", "--tune-exhaustive", "32"]
    run("capped_range", wide, settings="ntt_max_batch = 128\nntt_profile = 1")
    assert {r["batch"] for r in rows()} == {128}
    # With only batch 128 in the table the scheduler runs four reference batches.
    flags = ["--param", "e", "-gpucurves", "512", "-sigma", "20260922", "-save", "out.save", "32", "32"]
    for name, settings, expected in (("wide_full", "", [512]), ("wide_split", "ntt_max_batch = 128", [128]*4)):
        log = run(name, flags, stdin="2^127-1\n", settings="edwards_naf_window_bits = 6\n" + settings)
        assert [int(row["ntt_batch"]) for row in ntt_executions(log)] == expected
    points = saved_points(out / "wide_full/out.save", 127, 32, edwards=True)
    assert list(points) == list(range(20260922, 20260922 + 512))
    modulus = (1 << 127) - 1
    for sigma, point in points.items():
        _, x, z = edwards_stage1(modulus, sigma, 32)
        assert point == x * pow(z, -1, modulus) % modulus, sigma
    assert points == saved_points(out / "wide_split/out.save", 127, 32, edwards=True)
    log = run("wide_memory", flags, stdin="2^127-1\n", settings="edwards_naf_window_bits = 6\nnaf_memory_mb = 1")
    batches = [int(row["ntt_batch"]) for row in ntt_executions(log)]
    assert sum(batches) == 512 and max(batches) < 128, batches
    assert points == saved_points(out / "wide_memory/out.save", 127, 32, edwards=True)

    # This weighted radix-5 length needs the native Edwards coefficient bound.
    database = out / "native_only.toml"
    native = "ntt_length = 2560\nntt_profile = 1"
    run("native_only_tune", ["-tune", "exp=65521,naf=6,batch=2,time=0.02", "20"],
        convolution="weighted", settings=native)
    log = run("native_only_direct", ["--param", "e", "-gpucurves", "2", "-sigma", "26",
              "-save", "out.save", "20", "20"], stdin="2^65521-1\n", convolution="weighted", settings=native)
    record, = ntt_executions(log)
    assert (record["naf"], record["ntt_length"], record["ntt_batch"]) == ("6", "2560", "2")

    database = out / "compact.toml"
    settings = "ntt_profile = 1\nnaf_memory_mb = 3"
    automatic = ["-tune", "ntt,exp=3557,naf=6,batch=,time=0.05", "--tune-exhaustive", "16"]
    cold = run("compact", automatic, settings=settings)
    assert "stage1_ntt_graph_scan:" not in cold
    compact = rows()
    def aligned_report(log):
        pattern = r"(Single|Max|Best|95%) +(\d+) +(\d+) +(\S+) +(\d+)c +(\d+) iter/s dataflow=\d+"
        matches = list(re.finditer(pattern, log))
        assert [m[1] for m in matches] == ["Single", "Max", "Best", "95%"]
        columns = {(m.end(2)-m.start(), m.end(3)-m.start(), m.start(4)-m.start(),
                    m.end(5)-m.start(), m.end(6)-m.start()) for m in matches}
        assert len(columns) == 1, matches
        return {m[1]: int(m[5]) for m in matches}

    reports = aligned_report(cold)
    assert set(reports) == {"Single", "Max", "Best", "95%"}
    assert reports["Single"] == 1 and reports["Max"] == int(re.search(r"capacity=(\d+)", cold)[1]) > 4
    assert {r["batch"] for r in compact} == set(reports.values())
    for role, bit in (("Single", 1), ("Max", 2), ("Best", 4), ("95%", 8)):
        assert [r["batch"] for r in compact if r["batch_roles"] & bit] == [reports[role]]
    snapshot = database.read_bytes()
    hot = run("compact_hot", automatic, settings=settings)
    aligned_report(hot)
    assert "measured=1 " in hot and hot.count("stage1_ntt_recheck:") == 1
    assert ("status=updated" in hot) == (database.read_bytes() != snapshot)
    compatible = [arg.replace("3557", "3558") for arg in automatic[:-1]] + ["500"]
    reused = run("compact_other_p_b1", compatible, settings=settings)
    assert "measured=1 " in reused and reused.count("stage1_ntt_naf_sample:") == 1

    explicit = ["-tune", "ntt,exp=3557,naf=6,batch=1-128,time=0.05", "--tune-exhaustive", "16"]
    run("compact_explicit", explicit, settings=settings)
    manual = [r for r in rows() if not r["batch_scope"]]
    assert {r["batch"] for r in manual if r["sample_phase"] == 2} == set(range(1,reports["Max"]+1))
    run("compact_grow", automatic, settings=settings.replace("mb = 3", "mb = 5"))
    assert len({r["batch"] for r in rows() if r["batch_scope"]}) <= 4
    assert [r for r in rows() if not r["batch_scope"]] == manual

    database = out / "interrupted.toml"
    timed = ["-tune", "ntt,exp=3557,naf=6,time=1", "--tune-exhaustive"]
    stopped = run("compact_stop", [*timed, "--tune-seconds", "2", "16"], settings=settings)
    assert "status=time_limit" in stopped and len({r["batch"] for r in rows()}) <= 4
    checkpoints = list(out.glob("interrupted.toml.*.partial"))
    assert len(checkpoints) == 1
    resumed = run("compact_resume", [*timed, "16"], settings=settings)
    counts = re.search(r"measured=(\d+) reused=(\d+)", resumed)
    assert counts and all(int(n) > 0 for n in counts.groups())
    assert len({r["batch"] for r in rows()}) <= 4 and not checkpoints[0].exists()

    database = out / "range.toml"
    crossing = ["-tune", "ntt,exp=3554-3560,naf=6,batch=1,time=0.01", "16"]
    for name in ("crossing_cold", "crossing_hot"):
        log = run(name, crossing, settings="ntt_profile = 1")
        assert re.findall(r"stage1_ntt_plan: p=(\d+) length=(\d+)", log) == [
            ("3554", "256"), ("3557", "512")], log
        summaries = re.findall(r"stage1_ntt_tune: p=\d+ measured=(\d+) reused=(\d+)", log)
        assert len(summaries) == 2
        assert log.count("stage1_ntt_naf_sample:") == 1
        assert log.count("length max_exp ntt_type batch iter/s") >= 4
        assert re.findall(r"stage1_ntt_length_done: p=\d+ naf=6 length=(\d+) phase=fine", log) == ["256", "512"]
        assert "\x1b[" not in log
        if name == "crossing_cold":
            assert all(int(measured) and int(reused) == 0 for measured, reused in summaries)
        else:
            assert all(int(measured) == 1 and int(reused) for measured, reused in summaries)

    def planned(log):
        return [(int(p), int(length), int(w)) for p, length, w in re.findall(
            r"stage1_ntt_plan: p=(\d+) length=(\d+) bpw=\d+ naf=(\d+)", log)]

    # The shortest native weighted length drops at this isolated even exponent.
    isolated = run("isolated", ["-tune", "exp=66049-66051,naf=6,batch=1,time=0.01", "16"],
                   convolution="auto", settings="ntt_profile = 1")
    assert planned(isolated) == [(66049, 8192, 6), (66049, 4096, 6), (66050, 2560, 6)]

    # Exercise the entire reported range. An infeasible batch exposes the full
    # production admission plan without benchmarking a hundred matrices.
    broad = run("broad", ["-tune", "ntt,exp=2000-1000000,batch=128"],
                convolution="auto", settings="ntt_memory_mib = 1", success=False)
    expected = sorted([2**i for i in range(8, 18)] + [2**i for i in range(7, 17)]
                      + [2560, 5120, 10240, 20480, 40960])
    for window in (6, 8, 10, 12):
        assert sorted(n for p, n, w in planned(broad) if w == window) == expected
    assert "stage1_ntt_reuse" not in broad and "fits the requested grid" in broad

    database = out / "tune.toml"
    spec = "ntt,exp=61-62,naf=6-7,batch=1-2"
    grid_log = run("grid", ["-tune", spec, "16"])
    assert grid_log.count("stage1_ntt_naf_sample:") == 2
    data = rows()
    assert {r["naf_window"] for r in data} == {6, 7}
    assert {r["batch"] for r in data} == {1, 2}
    assert {r["graph_bits"] for r in data} == {4, 8, 16, 32, 64}
    assert {(r["sample_phase"], r["repeats"]) for r in data} == {(1, 2), (2, 2)}
    scans = {int(w): int(b) for w, b in re.findall(r"stage1_ntt_graph_scan: p=\d+ naf=(\d+) batch=(\d+)", grid_log)}
    assert set(scans) == {6, 7}
    for w, batch in scans.items():
        for b in (1, 2):
            fine = {r["graph_bits"] for r in data if r["naf_window"] == w and r["batch"] == b and r["sample_phase"] == 2}
            assert fine == ({4, 8, 16, 32, 64} if b == batch else {16})
    assert {r["graph_bits"] for r in data if r["sample_phase"] == 1} == {16}
    for row in data:
        assert row["sample_target_ms"] == 100 and 61 <= row["reference_bits"] <= 62
        assert row["benchmark_kind"] == "stage1_naf_stratified"
        assert 0 < row["block_steps"] <= row["sample_pool_steps"] == row["naf_columns"] <= 64
        rate = 1000 * row["batch"] * row["repeats"] * row["timed_steps"] / row["total_device_ms"]
        assert abs(row["iter_per_s"] / rate - 1) < 1e-8
    snapshot = database.read_bytes()
    hot = run("hot", ["--tune", spec, "16"])
    assert hot.count("measured=1 ") == 2 and hot.count("stage1_ntt_recheck:") == 2
    assert ("status=updated" in hot) == (database.read_bytes() != snapshot)
    run("stronger", ["-tune", "exp=61,naf=6,batch=1,time=2", "16"])
    stronger = [r for r in rows() if r["sample_target_ms"] == 200]
    assert {(r["sample_phase"], r["repeats"]) for r in stronger} == {(1, 2), (2, 2)}
    assert all(r["batch"] == 1 and r["naf_window"] == 6 for r in stronger)

    production = ["--param", "e", "-sigma", "26", "-gpucurves", "1", "-save", "result.save"]
    snapshot = database.read_bytes()
    cached = run("production", production + ["16"], stdin="2^61-1\n")
    best = max((r for r in rows() if r["batch"] == 1 and r["sample_phase"] == 2), key=lambda r: r["iter_per_s"])
    assert f'naf={best["naf_window"]} source=tune' in cached
    assert database.read_bytes() == snapshot and "stage1_ntt_runtime_tune:" not in cached
    cached = run("production_cross_b1", production + ["20"], stdin="2^61-1\n", runtime=True)
    assert f'naf={best["naf_window"]} source=tune' in cached and "stage1_ntt_runtime_tune:" not in cached
    assert database.read_bytes() == snapshot

    # Archived middle-slice measurements retain their original B1 scope.
    header = database.read_text(encoding="utf-8").split("matrix = [", 1)[0]
    old = rows()
    for row in old:
        row["repeats"] = 3 if row.pop("sample_phase") == 1 else 5
        for column in ("naf_columns", "sample_nonzero", "total_device_ms", "sample_pool_steps"):
            del row[column]
        row["benchmark_kind"] = "stage1_graph_replay"
        row["program_version"] = "0.71.1"
        row["iter_per_s"] = 1000 * row["batch"] * row["timed_steps"] / row["median_ms"]
    database = out / "legacy.toml"
    write_matrix(header, list(old[0]), [list(row.values()) for row in old])
    assert "source=tune" in run("legacy_same", production + ["16"], stdin="2^61-1\n")
    assert "naf=0 source=default" in run("legacy_other", production + ["20"], stdin="2^61-1\n")

    database = out / "runtime.toml"
    cold = run("read_only", production + ["20"], stdin="2^61-1\n")
    assert "naf=0 source=default" in cold and not database.exists()
    learned = run("runtime", production + ["20"], stdin="2^61-1\n", runtime=True,
                  settings="ntt_profile = 1")
    assert "stage1_ntt_runtime_tune:" in learned
    assert {r["naf_window"] for r in rows() if r["b1"] == 20} == {6, 8, 10, 12}
    assert saved_points(out/"read_only/result.save", 61, 20, True) == saved_points(
        out/"runtime/result.save", 61, 20, True)
    snapshot = database.read_bytes()
    replay = run("runtime_hit", production + ["20"], stdin="2^61-1\n", runtime=True,
                 settings="ntt_profile = 1")
    assert "stage1_ntt_runtime_tune:" not in replay and database.read_bytes() == snapshot
    log = run("length", ["-tune", ",length=8-9,naf=6,batch=2", "16"], settings="ntt_profile = 1")
    assert set(re.findall(r"stage1_ntt_plan: p=\d+ length=(\d+)", log)) == {"8"}
    run("mont", ["--param", "m0", "-tune", "ntt,exp=127,naf=,batch=1,time=0.25,repeat=3:1", "16"])
    mont = [r for r in rows() if r["curve"] == "mont0"]
    assert {(r["sample_phase"], r["repeats"]) for r in mont} == {(1, 3), (2, 1)}
    assert all(r["sample_target_ms"] == 25 for r in mont)
    assert {r["graph_bits"] for r in mont} == {4, 8, 16, 32, 64}

    # At this cost even a single full block misses a 1 ms budget. A 64-step
    # Graph window must still run NAF zero runs and short binary tail Graphs.
    previous_db = database
    database = out / "short_blocks.toml"
    for curve, spec, pool in (("e", ",naf=6", 186), ("m0", "", 64)):
        log = run(f"short_blocks_{curve}", ["--param", curve, "-tune",
            "ntt,exp=27622,batch=32,time=0.01" + spec, "128"],
            settings="ntt_profile = 1\nntt_window_bits = 64")
        assert len(re.findall(r"block_steps=\d+ group_ms=\d+\.\d+", log)) == 2
        tested = [r for r in rows() if r["curve"] == ("edwards" if curve == "e" else "mont0")]
        assert {r["sample_phase"] for r in tested} == {1, 2}
        for r in tested:
            assert r["graph_bits"] == 64 and r["sample_pool_steps"] == pool
            assert 0 < r["block_steps"] < 64 and r["timed_steps"] % r["block_steps"] == 0
            if curve == "e":
                assert abs(r["sample_nonzero"] - r["block_steps"]*26/186) <= 1 and r["sample_nonzero"] > 0
    database = previous_db
    database = out / "default_windows.toml"
    run("default_windows", ["-tune", "exp=61,batch=1,time=0.01", "32"],
        settings="edwards_naf_window_bits = 3\nntt_profile = 1")
    assert {r["naf_window"] for r in rows() if r["b1"] == 32} == {6, 8, 10, 12}
    assert len({r["naf_window"] for r in rows() if r["sample_phase"] == 2}) == 1
    database = out / "tiered.toml"
    limits = "ntt_max_batch = 12\nnaf_memory_mb = 20"
    tiers = ["-tune", "ntt,exp=2000,time=0.02,repeat=1:2", "32"]
    log = run("tiered", tiers, settings=limits)
    counters = re.findall(r"stage1_ntt_naf_screen: p=2000 naf=(\d+) capacity=(\d+) iter_per_s=\S+ refine=(\d+)", log)
    caps = {int(w): int(cap) for w, cap, _ in counters}
    winners = {int(w) for w, _, advance in counters if advance == "1"}
    assert set(caps) == {6, 8, 10, 12} and caps[6] == caps[8] > caps[10] > caps[12] > 0, caps
    assert sorted(caps[w] for w in winners) == sorted(set(caps.values())), counters
    samples = rows()
    scans = [(int(w), int(b)) for w, b in re.findall(r"stage1_ntt_graph_scan: p=2000 naf=(\d+) batch=(\d+)", log)]
    assert len(scans) == len(winners) and {w for w, b in scans} == winners
    for w, cap in caps.items():
        candidate = [r for r in samples if r["naf_window"] == w]
        measured = {r["batch"] for r in candidate}
        assert 1 in measured and max(measured) == cap and len(measured) <= 4
        phases = {(r["sample_phase"], r["repeats"]) for r in candidate}
        assert phases == ({(1, 1), (2, 2)} if w in winners else {(1, 1)})
        scanned = {r["batch"] for r in candidate if r["graph_bits"] != 16}
        assert scanned == {b for window, b in scans if window == w}
    def endpoint(w):
        return max(r["iter_per_s"] for r in samples if r["naf_window"] == w
                   and r["batch"] == caps[w] and r["sample_phase"] == 1)
    assert max((6, 8), key=endpoint) in winners
    snapshot = database.read_bytes()
    hot = run("tiered_hot", tiers, settings=limits)
    assert hot.count("measured=0 ") == 1 and hot.count("measured=1 ") == len(winners)
    assert hot.count("stage1_ntt_recheck:") == len(winners)
    assert ("status=updated" in hot) == (database.read_bytes() != snapshot)
    loser, = set(caps) - winners
    run("tiered_promote", [tiers[0], tiers[1] + f",naf={loser}", tiers[2]], settings=limits)
    assert any(r["naf_window"] == loser and r["sample_phase"] == 2 for r in rows())
    new_repeat = run("tiered_repeat", [tiers[0], tiers[1].replace("1:2", "1:3") + f",naf={loser}", tiers[2]],
                     settings=limits)
    assert "stage1_ntt_reuse:" in new_repeat
    updated = [r for r in rows() if r["naf_window"] == loser]
    assert {(r["sample_phase"], r["repeats"]) for r in updated} == {(1, 1), (2, 3)}
    wide = run("wide", ["-tune", "exp=900001,naf=6,batch=1,time=0.25", "16"],
               convolution="auto", settings="ntt_memory_mib = 128")
    assert "Best " in wide
    candidates = [r for r in rows() if r["reference_bits"] == 900001]
    assert {r["length"] for r in candidates} == {40960, 65536, 131072}
    assert {r["dataflow"] & 384 for r in candidates if r["length"] == 40960} == {0, 128, 384}
    bare = run("bare", ["-tune", "--tune-seconds", "1"], settings="ntt_profile = 1")
    assert "stage1_ntt_plan: p=32" in bare and "status=time_limit" in bare
    print(f"PASS GPU {args.device}: CLI, bounds, Graph grid, timing, cache, runtime and endpoints")


if __name__ == "__main__":
    main()
