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
        if "ntt_memory_mib" not in settings:
            settings += "\nntt_memory_mib = 2048\n"
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
        first_measurement = True
        for i, line in enumerate(lines):
            if "stage1_ntt_tune:" in line:
                assert re.fullmatch(prefix + r"=+", lines[i+1]), name
            if "stage1_ntt_round:" in line or "stage1_ntt_graph_scan:" in line or "stage1_ntt_find95:" in line:
                first_measurement = True
            header = prefix + "length max_exp ntt_type batch iter/s dataflow actual/target steps/group steps/block radix5_layout"
            if re.fullmatch(header, line):
                assert re.match(prefix + r"\d+ \d+ [01]:", lines[i+1]), name
            if re.match(prefix + r"\d+ \d+ [01]:", line):
                assert bool(re.fullmatch(header, lines[i-1])) == first_measurement, name
                first_measurement = False
        return result.stdout

    def rows():
        fields, *values = tomllib.loads(database.read_text(encoding="utf-8"))["ntt"]["matrix"]
        return [dict(zip(fields, row)) for row in values]

    def coverage(row):
        tested = {(row["sample_phase"], row["repeats"]): row["sample_target_ms"]}
        for pair in filter(None, row.get("tested_groups", "").split(",")):
            phase, groups, ms = map(int, re.split("[/:]", pair))
            tested[phase, groups] = max(tested.get((phase, groups), 0), ms)
        return tested

    def write_matrix(header, fields, values):
        matrix = "".join(json.dumps(row) + ",\n" for row in [fields, *values])
        database.write_text(header.split("matrix = [", 1)[0] + "matrix = [\n" + matrix + "]\n", encoding="utf-8")

    for index, spec in enumerate(("exp=61,length=8", "naf=12-6", "batch=0", "batch=513",
                                  "exp=31", "time=0", "ntt,naf=6,naf=7", "typo=1",
                                  "repeat=2", "repeat=1:0", "repeat=:2", "repeat=21:1",
                                  "find95=2", "find95,find95=1", "repeat=1:2:3", "repeat=2:2,repeat=1:1")):
        log = run(f"bad{index}", ["-tune", spec], success=False)
        assert "INPUT[cli]" in log
        if index == 0:
            assert "conflict" in log and "length" in log and "exp" in log
    assert not database.exists()

    for i, value in enumerate(("-1", "5242882", '"invalid"')):
        log = run(f"bad_dictionary_threshold_{i}", ["-tune", "ntt,length=64"],
                  settings=f"ntt_naf_gpu_min_bits = {value}", success=False)
        assert "ntt_naf_gpu_min_bits" in log

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
            fine, = values
            assert coverage(dict(zip(fields, fine))) == {(1, 2): 4, (2, 2): 4}
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
            comparison = re.search(r"stage1_ntt_recheck: p=\d+ naf=6 length=\d+ batch=1 graph=\d+\n"
                                   r"(?:\[[^\n]+?\] )?(\S+) vs (\S+) \(old\)  ([+-][\d.]+)%, status="
                                   + status + r"$", log, re.M)
            assert comparison, log
            measured, historical, percent = map(float, comparison.groups())
            assert abs(historical / fine[fields.index("iter_per_s")] - 1) < 1e-8
            assert abs(percent - 100 * (measured / historical - 1)) <= 0.0051 + abs(percent) * 1e-8
            assert (comparison[3][0] == "+") == (status == "updated")
            comparison95 = re.search(
                r"stage1_ntt_95_compare: p=\d+ naf=6 length=\d+ batch=1 graph=\d+\n"
                r"(?:\[[^\n]+?\] )?1c, (\S+) vs 1c, (\S+) \(old\)  ([+-]?[\d.]+)%, status=" + status +
                r"\n(?:\[[^\n]+?\] )?Single ", log)
            assert comparison95, log
            published, old95, delta95 = map(float, comparison95.groups())
            assert old95 == historical and published == measured
            assert abs(delta95 - 100 * (published/old95-1)) <= 0.0051 + abs(delta95)*1e-8
            updated = rows()
            assert len(updated) == 1
            if status == "updated":
                sample, = (r for r in updated if r["sample_phase"] == 2)
                assert sample["sample_target_ms"] == 4
                assert sample["iter_per_s"] > fine[fields.index("iter_per_s")]
                assert re.search(rf'Best +\d+ +\d+ +\S+ +1c +{sample["iter_per_s"]:.0f} iter/s', log)

    # Import duplicate old decisions in either order. The winning provenance
    # stays with the original rate even if the slower record is newer.
    source = tomllib.loads(database.read_text(encoding="utf-8"))
    modern, best = source["ntt"]["matrix"]
    baseline = dict(zip(modern, best))
    for key in ("max_exp", "explicit_batch", "tested_groups", "budget_limit_bytes", "naf_memory_mb"):
        baseline.pop(key, None)
    baseline["owned_bytes"] = baseline.pop("ntt_memory_bytes")
    baseline.pop("naf_memory_bytes", None)
    baseline["log2_length"] = baseline["length"].bit_length()-1
    baseline.update(program_version="0.71.6", b1=32)
    baseline.update(budget_limit_bytes=6 * 1048576, naf_memory_mb=3)
    slow = dict(baseline, budget_limit_bytes=8 * 1048576, naf_memory_mb=5, b1=64, program_version="0.71.9", updated_at="2026-11-20T23:59:59.000000000Z")
    slow["carry_bits"] = 1 if slow["carry_bits"] != 1 else 2
    slow["iter_per_s"] /= 2
    for key in ("median_ms", "total_device_ms", "stddev_seconds"):
        slow[key] *= 2
    for name, duplicates in (("first", [baseline, slow]), ("last", [slow, baseline])):
        database = out / f"legacy_{name}.toml"
        header = text.split("matrix = [",1)[0].replace("format = 18","format = 16")
        write_matrix(header,list(baseline),[list(r.values()) for r in duplicates])
        run("migrate_"+name, ["-tune","length=8,naf=6,batch=1,time=0.01","32"])
        converted = tomllib.loads(database.read_text(encoding="utf-8"))
        cols = converted["ntt"]["matrix"][0]
        assert converted["profile"]["format"] == 18 and cols[:5] == ["max_exp","length","iter_per_s","batch","ntt_type"]
        assert "log2_length" not in cols and all(any(r.get(k)!="" for r in rows()) for k in cols)
        assert cols[cols.index("program_version")+1] == "updated_at" and cols.index("b1") > cols.index("updated_at")
        winner, = (r for r in rows() if r["length"] == baseline["length"])
        assert "budget_limit_bytes" not in cols and "naf_memory_mb" not in cols
        assert winner["ntt_memory_bytes"] == baseline["owned_bytes"]
        assert not winner.get("naf_memory_bytes")
        assert winner["carry_bits"] == baseline["carry_bits"]
        assert cols.index("segments") > cols.index("dataflow")
        assert winner["b1"] == 32 and winner["program_version"] == "0.71.6"
        assert winner["iter_per_s"] == baseline["iter_per_s"] and winner["updated_at"] == baseline["updated_at"]

    # Same dispatch, independent NTT/dictionary requirements. Use controlled
    # measurements so the winners do not depend on the GPU's clock frequency.
    mib = 1024**2
    timed = dict(zip(modern, best))
    runtime = {k: v for k, v in timed.items() if k not in
               ("sample_target_ms", "sample_phase", "batch_scope", "batch_roles", "batch_search",
                "tested_groups", "explicit_batch", "sample_pool_steps", "naf_columns", "sample_nonzero",
                "median_ms", "timed_steps", "block_steps", "total_device_ms", "reference_bits")}
    runtime.update(benchmark_kind="stage1_runtime", arithmetic_bits=127, target_bits=127,
                   lcm_multiplier=48, valid_curves=1, curves_per_s=1)
    def footprint(record, ntt, naf, rate):
        return dict(record, ntt_memory_bytes=64*ntt*mib, naf_memory_bytes=naf*1_000_000,
                    iter_per_s=rate, budget_bytes=512*mib, peak_bytes=64*ntt*mib)
    middle = footprint(timed, 4, 3, 300)
    for k in ("median_ms", "total_device_ms", "stddev_seconds"):
        middle[k] *= timed["iter_per_s"] / 300
    scoped = footprint(middle, 7, 1, 300)
    scoped.update(batch_scope="frontier", batch_search="full", batch_roles=15)
    unknown = footprint(runtime, 6, 1, 10)
    del unknown["naf_memory_bytes"]
    frontier = [footprint(runtime, 2, 1, 100), middle, footprint(runtime, 7, 1, 500), scoped, unknown,
                footprint(runtime, 5, 4, 300), footprint(runtime, 4, 3, 299),
                footprint(dict(runtime, naf_window=8), 2, 1, 250)]
    for label, records in (("forward", frontier), ("reverse", frontier[::-1])):
        database = out / f"frontier_{label}.toml"
        fields = list(dict.fromkeys(k for r in records for k in r))
        write_matrix(text, fields, [[r.get(k, "") for k in fields] for r in records])
        run("frontier_import_"+label, ["-tune", "length=8,naf=6,batch=1,time=0.01", "32"])
        stored = [r for r in rows() if r["length"] == timed["length"] and r["naf_window"] == 6]
        assert {r["iter_per_s"] for r in stored} == {10, 100, 300, 500} and len(stored) == 4
        assert not next(r for r in stored if r["iter_per_s"] == 10).get("naf_memory_bytes")
        assert all(r["reference_bits"] == 127 for r in stored if r["benchmark_kind"] == "stage1_runtime")
        assert all(k not in stored[0] for k in ("arithmetic_bits", "target_bits", "lcm_multiplier", "valid_curves"))
        assert list(stored[0]).index("benchmark_kind") > list(stored[0]).index("naf_memory_bytes")
        assert next(r for r in stored if r["iter_per_s"] == 300)["benchmark_kind"] == timed["benchmark_kind"]
    for ntt, naf, expected in ((3, 3, 8), (4, 3, 6), (4, 2, 8), (8, 2, 6)):
        log = run(f"frontier_select_{ntt}_{naf}", ["--param", "e", "-gpucurves", "1", "-sigma", "26", "32", "32"],
                  stdin="2^127-1\n", settings=f"ntt_profile = 1\nntt_memory_mib = {64*ntt}\nnaf_memory_mb = {naf}")
        assert f"p=127 naf={expected} source=tune" in log, log

    # The 1M release capped these three families, including forced 131072-point
    # weighted transforms. Loading a wide search must preserve their old scope.
    wide_rows = []
    for n, kind in ((65536, "stage1_weighted"), (131072, "stage1_linear"), (131072, "stage1_weighted")):
        row = dict(zip(modern, best))
        layers = [4, 12] if n == 65536 else [1, 4, 12]
        row.update(length=n, ntt_type=kind, max_exp=1048576, reference_bits=1048576,
                   tile=12, max_radix_bits=4, passes_forward=len(layers),
                   forward="+".join(map(str, layers)), inverse="+".join(map(str, reversed(layers))),
                   segments=128, bpw=24 if kind == "stage1_linear" else 1048576 // n,
                   uniform_digits=int(kind == "stage1_weighted"), program_version="0.71.14")
        wide_rows.append(row)
    for version in (16, 17):
        database = out / f"legacy_1m_v{version}.toml"
        columns = [k for k in modern if version == 17 or k != "max_exp"]
        write_matrix(text.replace("format = 18", f"format = {version}"), columns,
                     [[r[k] for k in columns] for r in wide_rows])
        run(f"legacy_1m_load_{version}", ["-tune", "length=8,naf=6,batch=1,time=0.01", "32"])
        preserved = [r for r in rows() if r["length"] >= 65536]
        assert len(preserved) == 3 and all(r["max_exp"] == r["reference_bits"] == 1048576 and
            r["program_version"] == "0.71.14" and r["iter_per_s"] == best[modern.index("iter_per_s")]
            for r in preserved)
    for tag, changes in (("wrong_limit", {"max_exp": 1048577}),
                         ("wrong_family", {"length": 81920}),
                         ("outside_old_range", {"reference_bits": 1048577, "segments": 129})):
        database = out / f"bad_1m_{tag}.toml"
        damaged = dict(wide_rows[0], **changes)
        write_matrix(text, modern, [[damaged[k] for k in modern]])
        assert "invalid Stage1 NTT max_exp" in run("reject_1m_" + tag,
            ["-tune", "ntt,exp=3000000,time=1"], success=False)

    database = out / "recheck_roles.toml"
    flags = ["-tune", "exp=127,naf=6,time=0.02", "32"]
    limits = "ntt_max_batch = 2\nntt_profile = 1"
    run("roles_seed", flags, settings=limits)
    text = database.read_text(encoding="utf-8")
    fields, *values = tomllib.loads(text)["ntt"]["matrix"]
    for row in values:
        if (2, 2) in coverage(dict(zip(fields, row))):
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
    assert re.search(r"2c, [\d.e+]+ vs 1c, 99 \(old\)  \+[\d.]+%, status=updated", log)
    assert {r["batch"]: r["batch_roles"] for r in rows() if (2, 2) in coverage(r)} == {1: 1, 2: 14}

    # The old 95% batch is a cache hit. Only Best gets a fresh timing.
    for row in values:
        row[fields.index("iter_per_s")] *= 10000
        for key in ("total_device_ms", "median_ms", "stddev_seconds"):
            row[fields.index(key)] /= 10000
    write_matrix(text, fields, values)
    log = run("roles_kept_cached95", [flags[0], flags[1] + ",find95", flags[2]], settings=limits)
    fresh = re.search(r"([\d.e+]+) vs 1000000 \(old\)  -[\d.]+%, status=kept", log)[1]
    assert f"2c, {fresh} vs 1c, 990000 (old)" in log
    assert re.search(r"95% +\d+ +\d+ +\S+ +1c +990000 iter/s", log)

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
    automatic = ["-tune", "ntt,length=512,naf=6,batch=,time=0.05", "--tune-exhaustive", "16"]
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
    compatible = [arg.replace("length=512", "exp=5000-5100") for arg in automatic[:-1]] + ["500"]
    reused = run("compact_other_p_b1", compatible, settings=settings)
    assert "measured=1 " in reused and reused.count("stage1_ntt_naf_sample:") == 1

    explicit = ["-tune", "ntt,length=512,naf=6,batch=1-128,time=0.05", "--tune-exhaustive", "16"]
    run("compact_explicit", explicit, settings=settings)
    manual = [r for r in rows() if r.get("explicit_batch")]
    assert {r["batch"] for r in manual if (2, 2) in coverage(r)} == set(range(1,reports["Max"]+1))
    run("compact_grow", automatic, settings=settings.replace("mb = 3", "mb = 5"))
    assert all("budget_limit_bytes" not in r and "naf_memory_mb" not in r for r in rows())
    assert all(0 < r["naf_memory_bytes"] <= r["ntt_memory_bytes"] for r in rows())
    by_batch = {r["batch"]: r["iter_per_s"] for r in rows() if r.get("explicit_batch")}
    assert all(by_batch[r["batch"]] >= r["iter_per_s"] for r in manual)

    database = out / "interrupted.toml"
    timed = ["-tune", "ntt,length=512,naf=6,time=1", "--tune-exhaustive"]
    stopped = run("compact_stop", [*timed, "--tune-seconds", "2", "16"], settings=settings)
    assert "status=time_limit" in stopped and len({r["batch"] for r in rows()}) <= 4
    checkpoints = list(out.glob("interrupted.toml.*.partial"))
    assert len(checkpoints) == 1
    resumed = run("compact_resume", [*timed, "16"], settings=settings)
    counts = re.search(r"measured=(\d+) reused=(\d+)", resumed)
    assert counts and all(int(n) > 0 for n in counts.groups())
    assert len({r["batch"] for r in rows()}) <= 4 and not checkpoints[0].exists()

    database = out / "find95_first_run.toml"
    log = run("find95_first_run", ["-tune", "length=16,naf=6,time=0.01,find95", "16"],
              settings="ntt_max_batch = 8")
    assert log.index("stage1_ntt_graph_scan:") < log.index("stage1_ntt_find95:") < log.index("Single ")
    best, = (r for r in rows() if r["batch_roles"] & 4)
    economical, = (r for r in rows() if r["batch_roles"] & 8)
    assert economical["iter_per_s"] >= 0.95 * best["iter_per_s"]

    # A complete default search can seed the optional post-Graph search.
    for label, batch_arg in (("compact", ""), ("explicit", ",batch=2-8")):
        database = out / f"find95_{label}.toml"
        settings = "ntt_max_batch = 8"
        spec = "ntt,length=16,naf=6,time=0.01" + batch_arg
        def search(tag, extra):
            return run(f"find95_{label}_{tag}", ["-tune", spec + extra, "16"], settings=settings)
        def best_batch(log):
            return int(re.search(r"Best +\d+ +\d+ +\S+ +(\d+)c ", log)[1])
        cold = search("cold", ",find95=0")
        assert "stage1_ntt_find95:" not in cold
        for tag, extra in (("enabled", ",find95"), ("repeat", ",find95=1")):
            log = search(tag, extra)
            start = re.search(r"stage1_ntt_find95: .*best_batch=(\d+) graph=(\d+)", log)
            assert start and int(start[1]) == best_batch(cold)
            assert log.find("stage1_ntt_graph_scan:") < start.start()
            b = best_batch(log)
            report = re.search(r"95% +\d+ +\d+ +\S+ +(\d+)c ", log)
            assert (2 if batch_arg else 1) <= int(report[1]) <= b <= 8
            saved = rows()
            if not batch_arg:
                assert {r["batch"] for r in saved if r["batch_roles"] & 4} == {b}
                assert {r["batch"] for r in saved if r["batch_roles"] & 8} == {int(report[1])}
                assert len({r["batch"] for r in saved}) <= 4
            cold = log

    database = out / "range.toml"
    crossing = ["-tune", "ntt,exp=3554-3560,naf=6,batch=1,time=0.01", "16"]
    for name in ("crossing_cold", "crossing_hot"):
        log = run(name, crossing, settings="ntt_profile = 1")
        assert re.findall(r"stage1_ntt_plan: p=(\d+) length=(\d+)", log) == [
            ("3556", "256"), ("7140", "512")], log
        summaries = re.findall(r"stage1_ntt_tune: p=\d+ measured=(\d+) reused=(\d+)", log)
        assert len(summaries) == 2
        assert log.count("stage1_ntt_naf_sample:") == 1
        assert log.count("length max_exp ntt_type batch iter/s") == (4 if name == "crossing_cold" else 2)
        assert re.findall(r"stage1_ntt_length_done: p=\d+ naf=6 length=(\d+) phase=fine", log) == ["256", "512"]
        assert "\x1b[" not in log
        if name == "crossing_cold":
            assert all(int(measured) and int(reused) == 0 for measured, reused in summaries)
        else:
            assert all(int(measured) == 1 and int(reused) for measured, reused in summaries)

    def planned(log):
        return [(int(p), int(length), int(w)) for p, length, w in re.findall(
            r"stage1_ntt_plan: p=(\d+) length=(\d+) bpw=\d+ naf=(\d+)", log)]

    # The expanded range is ordered by continuous capacity, with independent families.
    isolated = run("isolated", ["-tune", "exp=66049-66051,naf=6,batch=1,time=0.01", "16"],
                   convolution="auto", settings="ntt_profile = 1")
    assert planned(isolated) == [(53222, 4096, 6), (53248, 2048, 6), (66048, 2560, 6), (104448, 4096, 6)]

    # Exact scaled thresholds are strict. 0.8*66560 = 53248 and 1.2*5950 = 7140.
    for tag, exp, mode, expected in (
        ("strict_lower", 66560, "auto", [(66048, 2560, 6), (104448, 4096, 6)]),
        ("strict_upper", 5950, "linear", [(7140, 512, 6), (13797, 1024, 6)]),
    ):
        log = run(tag, ["-tune", f"exp={exp},naf=6,batch=1,time=0.01", "16"],
                  convolution=mode, settings="ntt_profile = 1")
        assert planned(log) == expected

    # Exercise the entire reported range. An infeasible batch exposes the full
    # production admission plan without benchmarking a hundred matrices.
    broad = run("broad", ["-tune", "ntt,exp=2000-1000000,batch=128"],
                convolution="auto", settings="ntt_memory_mib = 1", success=False)
    expected = sorted([2**i for i in range(7, 17)] + [2**i for i in range(6, 17)]
                      + [2560, 5120, 10240, 20480, 40960])
    for window in (6, 8, 10, 12):
        assert sorted(n for p, n, w in planned(broad) if w == window) == expected
    assert "stage1_ntt_reuse" not in broad and "fits the requested grid" in broad

    upper = run("upper_bound", ["-tune", "ntt,exp=5242880,naf=4,batch=128"],
                convolution="auto", settings="ntt_memory_mib = 1", success=False)
    assert sorted(planned(upper)) == [(5242880, 262144, 4), (5242880, 524288, 4)]

    database = out / "tune.toml"
    spec = "ntt,exp=61-62,naf=6-7,batch=1-2"
    grid_log = run("grid", ["-tune", spec, "16"])
    assert grid_log.count("stage1_ntt_naf_sample:") == 2
    data = rows()
    assert {r["naf_window"] for r in data} == {6, 7}
    assert {r["batch"] for r in data} == {1, 2}
    assert {r["graph_bits"] for r in data} == {4, 8, 16, 32, 64}
    assert {pair for r in data for pair in coverage(r)} == {(1, 2), (2, 2)}
    scans = {int(w): int(b) for w, b in re.findall(r"stage1_ntt_graph_scan: p=\d+ naf=(\d+) batch=(\d+)", grid_log)}
    assert set(scans) == {6, 7}
    for w, batch in scans.items():
        for b in (1, 2):
            fine = {r["graph_bits"] for r in data if r["naf_window"] == w and r["batch"] == b and (2, 2) in coverage(r)}
            assert fine == ({4, 8, 16, 32, 64} if b == batch else {16})
    assert {r["graph_bits"] for r in data if (1, 2) in coverage(r)} == {16}
    for row in data:
        assert row["sample_target_ms"] == 100 and row["reference_bits"] == row["max_exp"] == 93
        assert row["benchmark_kind"] == "stage1_naf_stratified"
        assert 0 < row["block_steps"] <= row["sample_pool_steps"] == row["naf_columns"] <= 64
        rate = 1000 * row["batch"] * row["repeats"] * row["timed_steps"] / row["total_device_ms"]
        assert abs(row["iter_per_s"] / rate - 1) < 1e-8
    snapshot = database.read_bytes()
    hot = run("hot", ["--tune", spec, "16"])
    assert hot.count("measured=1 ") == 2 and hot.count("stage1_ntt_recheck:") == 2
    assert ("status=updated" in hot) == (database.read_bytes() != snapshot)
    run("stronger", ["-tune", "exp=61,naf=6,batch=1,time=2", "16"])
    stronger = [r for r in rows() if 200 in coverage(r).values()]
    assert {k for r in stronger for k,v in coverage(r).items() if v == 200} == {(1, 2), (2, 2)}
    assert all(r["batch"] == 1 and r["naf_window"] == 6 for r in stronger)

    production = ["--param", "e", "-sigma", "26", "-gpucurves", "1", "-save", "result.save"]
    snapshot = database.read_bytes()
    cached = run("production", production + ["16"], stdin="2^93-1\n")
    best = max((r for r in rows() if r["batch"] == 1 and (2, 2) in coverage(r)), key=lambda r: r["iter_per_s"])
    assert f'naf={best["naf_window"]} source=tune' in cached
    assert database.read_bytes() == snapshot and "stage1_ntt_runtime_tune:" not in cached
    cached = run("production_cross_b1", production + ["20"], stdin="2^93-1\n", runtime=True)
    assert f'naf={best["naf_window"]} source=tune' in cached and "stage1_ntt_runtime_tune:" not in cached
    assert database.read_bytes() == snapshot

    # B1 is measurement provenance, including for archived middle-slice samples.
    header = database.read_text(encoding="utf-8").split("matrix = [", 1)[0]
    old = rows()
    for row in old:
        row.pop("tested_groups", None)
        row["repeats"] = 3 if row.pop("sample_phase") == 1 else 5
        for column in ("naf_columns", "sample_nonzero", "total_device_ms", "sample_pool_steps"):
            del row[column]
        row["benchmark_kind"] = "stage1_graph_replay"
        row["program_version"] = "0.71.1"
        row["iter_per_s"] = 1000 * row["batch"] * row["timed_steps"] / row["median_ms"]
    database = out / "legacy.toml"
    write_matrix(header, list(old[0]), [list(row.values()) for row in old])
    assert "source=tune" in run("legacy_same", production + ["16"], stdin="2^93-1\n")
    assert "source=tune" in run("legacy_other", production + ["20"], stdin="2^93-1\n")

    database = out / "runtime.toml"
    cold = run("read_only", production + ["20"], stdin="2^93-1\n")
    assert "naf=0 source=default" in cold and not database.exists()
    learned = run("runtime", production + ["20"], stdin="2^93-1\n", runtime=True,
                  settings="ntt_profile = 1")
    assert "stage1_ntt_runtime_tune:" in learned
    assert {r["naf_window"] for r in rows() if r["b1"] == 20} == {6, 8, 10, 12}
    assert saved_points(out/"read_only/result.save", 93, 20, True) == saved_points(
        out/"runtime/result.save", 93, 20, True)
    snapshot = database.read_bytes()
    replay = run("runtime_hit", production + ["20"], stdin="2^93-1\n", runtime=True,
                 settings="ntt_profile = 1")
    assert "stage1_ntt_runtime_tune:" not in replay and database.read_bytes() == snapshot
    log = run("length", ["-tune", ",length=8-9,naf=6,batch=2", "16"], settings="ntt_profile = 1")
    assert set(re.findall(r"stage1_ntt_plan: p=\d+ length=(\d+)", log)) == {"8"}
    run("mont", ["--param", "m0", "-tune", "ntt,exp=127,naf=,batch=1,time=0.25,repeat=3:1", "16"])
    mont = [r for r in rows() if r["curve"] == "mont0"]
    assert {pair for r in mont for pair in coverage(r)} == {(1, 3), (2, 1)}
    assert all(r["sample_target_ms"] == 25 for r in mont)
    assert {r["graph_bits"] for r in mont} == {4, 8, 16, 32, 64}

    # At this cost even a single full block misses a 1 ms budget. A 64-step
    # Graph window must still run NAF zero runs and short binary tail Graphs.
    previous_db = database
    database = out / "short_blocks.toml"
    for curve, spec, pool in (("e", ",naf=6", 186), ("m0", "", 64)):
        log = run(f"short_blocks_{curve}", ["--param", curve, "-tune",
            "ntt,length=4096,batch=32,time=0.01" + spec, "128"],
            settings="ntt_profile = 1\nntt_window_bits = 64")
        samples = re.findall(r" iter/s (\d+)bit (\d+\.\d{3})/1 ms (\d+) (\d+) -$", log, re.M)
        assert len(samples) == 2
        for _, actual, steps, block in samples:
            assert float(actual) > 0 and 0 < int(block) < 64 and int(steps) % int(block) == 0
        tested = [r for r in rows() if r["curve"] == ("edwards" if curve == "e" else "mont0")]
        assert {phase for r in tested for phase, _ in coverage(r)} == {1, 2}
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
    assert len({r["naf_window"] for r in rows() if (2, 2) in coverage(r)}) == 1
    database = out / "tiered.toml"
    limits = "ntt_max_batch = 12\nnaf_memory_mb = 20"
    tiers = ["-tune", "ntt,length=256,time=0.02,repeat=1:2", "32"]
    log = run("tiered", tiers, settings=limits)
    counters = re.findall(r"stage1_ntt_naf_screen: p=3556 naf=(\d+) capacity=(\d+) iter_per_s=\S+ refine=(\d+)", log)
    caps = {int(w): int(cap) for w, cap, _ in counters}
    winners = {int(w) for w, _, advance in counters if advance == "1"}
    assert set(caps) == {6, 8, 10, 12} and caps[6] == caps[8] > caps[10] > caps[12] > 0, caps
    assert sorted(caps[w] for w in winners) == sorted(set(caps.values())), counters
    samples = rows()
    scans = [(int(w), int(b)) for w, b in re.findall(r"stage1_ntt_graph_scan: p=3556 naf=(\d+) batch=(\d+)", log)]
    assert len(scans) == len(winners) and {w for w, b in scans} == winners
    for w, cap in caps.items():
        candidate = [r for r in samples if r["naf_window"] == w]
        measured = {r["batch"] for r in candidate}
        assert 1 in measured and max(measured) == cap and len(measured) <= 4
        phases = {pair for r in candidate for pair in coverage(r)}
        assert phases == ({(1, 1), (2, 2)} if w in winners else {(1, 1)})
        scanned = {r["batch"] for r in candidate if r["graph_bits"] != 16}
        assert scanned == {b for window, b in scans if window == w}
    endpoints = {int(w): float(rate) for w, rate in re.findall(
        r"stage1_ntt_naf_screen: p=3556 naf=(\d+) capacity=\d+ iter_per_s=(\S+)", log)}
    assert max((6, 8), key=endpoints.get) in winners
    snapshot = database.read_bytes()
    hot = run("tiered_hot", tiers, settings=limits)
    assert hot.count("measured=0 ") == 1 and hot.count("measured=1 ") == len(winners)
    assert hot.count("stage1_ntt_recheck:") == len(winners)
    assert ("status=updated" in hot) == (database.read_bytes() != snapshot)
    loser, = set(caps) - winners
    run("tiered_promote", [tiers[0], tiers[1] + f",naf={loser}", tiers[2]], settings=limits)
    assert any(r["naf_window"] == loser and (2, 2) in coverage(r) for r in rows())
    new_repeat = run("tiered_repeat", [tiers[0], tiers[1].replace("1:2", "1:3") + f",naf={loser}", tiers[2]],
                     settings=limits)
    assert "stage1_ntt_reuse:" in new_repeat
    updated = [r for r in rows() if r["naf_window"] == loser]
    assert {(1, 1), (2, 3)} <= {pair for r in updated for pair in coverage(r)}
    wide = run("wide", ["-tune", "exp=900001,naf=6,batch=1,time=0.25", "16"],
               convolution="auto", settings="ntt_memory_mib = 128")
    assert {line.split()[-1] for line in wide.splitlines() if " 40960 " in line and "bit " in line} == {"twiddle", "pfa-direct", "pfa-coalesced"}
    assert "Best " in wide
    candidates = [r for r in rows() if r["reference_bits"] >= 720001]
    assert {r["length"] for r in candidates} == {32768, 40960, 65536}
    assert {r["dataflow"] & 384 for r in candidates if r["length"] == 40960} == {0, 128, 384}
    bare = run("bare", ["-tune", "--tune-seconds", "1"], settings="ntt_profile = 1")
    assert "stage1_ntt_plan: p=93" in bare and "status=time_limit" in bare
    print(f"PASS GPU {args.device}: CLI, bounds, Graph grid, timing, cache, runtime and endpoints")


if __name__ == "__main__":
    main()
