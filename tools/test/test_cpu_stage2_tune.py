"""Exercise CPU Stage2 runtime tuning with real save and worktodo curves."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import tomllib

from test_cpu_stage2_queue import HIT, POINTS


GPU_TOML = """[profile]
format = 13
unit = "full_stage2"

[condition.0]
uuid_hex = "fixture"

[ntt.0]
matrix = [
    ["length", "batch", "mean_seconds", "stddev_seconds", "repeats", "verified", "measurement_ids"],
    [1024, 1, 0.01, 0, 1, 1, "4:test"],
]
"""
B1, B2, MEMORY_MIB = 1000, 40000, 512
TIMINGS = (
    "mean_seconds", "init_seconds", "main_seconds", "baby_seconds",
    "ftree_seconds", "reciprocal_seconds", "giant_seconds", "gtree_seconds",
    "fold_seconds", "scale_seconds", "descent_seconds", "leaf_gcd_seconds",
)


def read_tune(path):
    return tomllib.loads(path.read_text(encoding="utf-8-sig"))


def read_results(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def verify_tune(row, memory=MEMORY_MIB):
    assert row["cpu"] and row["build"] and row["fft_description"] and row["isa"]
    assert (row["bits"], row["exponent"], row["b1"], row["b2"],
            row["memory_mib"], row["threads"]) == (127, 127, B1, B2, memory, 1)
    assert row["d"] >= 6 and row["d"] % 2 == 0
    assert row["p"] == sum(math.gcd(j, row["d"]) == 1 for j in range(1, row["d"] // 2 + 1))
    assert row["blocks"] > 0 and row["giants"] > 0 and row["fft_length"] > 0
    assert 0 < row["estimated_memory_mib"] <= memory
    assert all(math.isfinite(row[key]) and row[key] >= 0 for key in TIMINGS)
    assert all(row[key] > 0 for key in ("mean_seconds", "init_seconds", "main_seconds",
                                        "giant_seconds", "gtree_seconds", "fold_seconds"))


class Fixture:
    def __init__(self, executable, output):
        self.executable = executable
        self.output = output
        self.workflow = output / "workflow"
        self.workflow.mkdir()
        self.config = self.workflow / "ecm.toml"
        self.tune = self.workflow / "shared-profile.toml"
        self.tune.write_text(GPU_TOML, encoding="utf-8")
        self.config.write_text(
            'config_version = 4\n[shared]\ntune_file = "shared-profile.toml"\n'
            '[stage2]\nstage2_cpu_threads = 1\ncpu_backend = "gwpoly"\ncpu_memory_mib = 512\n'
            'tune_runtime_enabled = true\ntune_min_repeats = 2\ntune_runtime_sample_limit = 2\n',
            encoding="utf-8",
        )
        self.save = self.workflow / "points.save"
        self.save.write_text(POINTS * 3, encoding="utf-8")
        self.hit = self.workflow / "hit.save"
        self.hit.write_text(HIT, encoding="utf-8")

    def start(self, name, *flags, config=None, folder=None):
        if folder is None:
            folder = self.output / name
            folder.mkdir()
        args = [self.executable, "--config", config or self.config,
                "--results", folder / "local.jsonl", "--submission", folder / "public.json.txt",
                *flags]
        args = list(map(str, args))
        (folder / "command.json").write_text(json.dumps(args, indent=2) + "\n", encoding="utf-8")
        log_path = folder / "run.log"
        with log_path.open("wb") as log:
            proc = subprocess.Popen(args, cwd=folder, stdout=log, stderr=subprocess.STDOUT)
        return proc, log_path, folder / "local.jsonl"

    def finish(self, started, expected=0):
        proc, log, results = started
        try:
            rc = proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        assert rc == expected, (rc, log, log.read_text(encoding="utf-8", errors="replace")[-2000:])
        return read_results(results)

    def save_flags(self, curves):
        return ("--save", self.save, "--exponent", 127, "--b2", B2, "--curves", curves)

    def save_run(self, name, *flags, curves=1):
        return self.finish(self.start(name, *self.save_flags(curves), *flags))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--portable-fixture", type=Path,
                        help="Compiled stage2_portable_fixture for GPU reader round-trip checks")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    f = Fixture(args.cpu.resolve(strict=True), out)
    original = f.tune.read_bytes()
    cold = f.start("cold_read_only", *f.save_flags(1), "--tune-runtime", 0)
    assert len(f.finish(cold)) == 1
    assert "gwpoly-tune: D=" not in cold[1].read_text(encoding="utf-8")
    assert f.tune.read_bytes() == original
    assert f.save_run("dry", "--dry-run") == []
    assert f.tune.read_bytes() == original
    checked, = f.save_run("check", "--gw-check")
    assert checked["stage2-engine"]["checked-values"] == 3
    assert f.tune.read_bytes() == original
    hit, = f.finish(f.start("factor", "--save", f.hit, "--b2", B2))
    assert (hit["status"], hit["stage"], hit["factors"]) == ("F", 2, ["10007"])
    assert f.tune.read_bytes() == original
    print("PASS dry-run, checked work and factor work produce no tune observations", flush=True)

    results = f.save_run("train", curves=6)
    assert len(results) == 6 and all(row["status"] == "NF" for row in results)
    assert [row["stage2-engine"]["save-line"] for row in results] == list(range(1, 7))
    assert all(row["b2"] == B2 for row in results)
    observed = Counter((row["d"], row["poly-size"], row["fft-length"]) for row in results)
    assert 2 <= len(observed) <= 3, observed
    document = read_tune(f.tune)
    tuned = document.pop("cpu_stage2")
    assert document == tomllib.loads(GPU_TOML)
    assert {(row["d"], row["p"], row["fft_length"]) for row in tuned} == set(observed)
    for row in tuned:
        verify_tune(row)
        assert row["repeats"] == min(2, observed[(row["d"], row["p"], row["fft_length"])]) == 2
    print("PASS fixed workload, candidate cycling, phase schema and preserved GPU profile", flush=True)

    state = f.tune.read_bytes()
    winner, = f.save_run("read_only", "--tune-runtime", 0)
    best = min(row["mean_seconds"] for row in tuned)
    assert winner["d"] in {row["d"] for row in tuned if row["mean_seconds"] == best}
    assert f.tune.read_bytes() == state
    forced, = f.save_run("forced", "--d", 210)
    assert forced["d"] == 210
    assert f.tune.read_bytes() == state
    budget = 256
    f.save_run("other_budget", "--memory-mib", budget)
    combined = read_tune(f.tune)["cpu_stage2"]
    assert [row for row in combined if row["memory_mib"] == MEMORY_MIB] == tuned
    different, = [row for row in combined if row["memory_mib"] == budget]
    verify_tune(different, budget)
    assert different["repeats"] == 1
    print("PASS read-only fastest reuse, explicit D and memory-specific scope", flush=True)

    save_bytes = f.save.read_bytes()
    assert f.finish(f.start("save_alias", *f.save_flags(1), "--tune-profile", f.save), expected=1) == []
    assert f.save.read_bytes() == save_bytes
    config_bytes = f.config.read_bytes()
    assert f.finish(f.start("config_alias", *f.save_flags(1), "--tune-profile", f.config), expected=1) == []
    assert f.config.read_bytes() == config_bytes
    folder = out / "ledger_alias"
    folder.mkdir()
    ledger_marker = folder / "data/public.json.txt.ecm-ledger/format"
    assert f.finish(f.start("ledger_alias", *f.save_flags(1), "--tune-profile", ledger_marker,
                            folder=folder), expected=1) == []
    assert not ledger_marker.exists()
    print("PASS tune path aliases cannot overwrite input files", flush=True)

    shared = out / "concurrent.toml"
    shared.write_text(GPU_TOML, encoding="utf-8")
    pair = [f.start(f"writer_{i}", *f.save_flags(1), "--skip", i,
                    "--tune-profile", shared, "--tune-min-repeats", 1)
            for i in range(2)]
    assert all(len(f.finish(run)) == 1 for run in pair)
    concurrent = read_tune(shared)
    assert sum(row["repeats"] for row in concurrent.pop("cpu_stage2")) == 2
    assert concurrent == tomllib.loads(GPU_TOML)
    assert len(read_tune(f.tune)["cpu_stage2"]) == len(combined)
    print("PASS CLI overrides and concurrent tune writers", flush=True)

    queue = out / "queue"
    queue.mkdir()
    queue_config = queue / "ecm.toml"
    queue_config.write_text(
        'config_version = 4\n[shared]\nworktodo_file = "worktodo.txt"\n'
        'finished_file = "finished.txt"\nlog_file = "worker.log"\n'
        '[stage2]\nstage2_save_dir = "."\ncpu_backend = "gwpoly"\nstage2_cpu_threads = 1\n'
        '[worker.3]\ntune_file = "tune-cpu.toml"\n',
        encoding="utf-8",
    )
    (queue / "points.save").write_text(POINTS, encoding="utf-8")
    task = f'ECMSTAGE2=1,2,127,-1,"points.save",{B2},0,2'
    worktodo = queue / "worktodo.txt"
    worktodo.write_text("[Worker #3]\n" + task + "\n", encoding="utf-8")
    results = f.finish(f.start("queue", "--worker", 3, "--once", "--gui-events",
                               "--log-level", "batches", config=queue_config, folder=queue))
    assert [row["Edwards"]["sigma"] for row in results] == [17, 18]
    assert task not in worktodo.read_text(encoding="utf-8-sig")
    assert (queue / "finished.txt").read_text(encoding="utf-8-sig").count(task) == 1
    queue_tune = queue / "tune-cpu.toml"
    queue_rows = read_tune(queue_tune)["cpu_stage2"]
    assert not (queue / "tune.toml").exists()
    for row in queue_rows:
        verify_tune(row)
    assert sum(row["repeats"] for row in queue_rows) == 2
    for path in (queue / "run.log", queue / "worker.log"):
        events = path.read_text(encoding="utf-8")
        for name in ("Plan Stage2", "Generate and normalize", "Build baby-point",
                     "Prepare polynomial inverse", "Giant points", "Normalize fold", "Accumulate leaf"):
            assert f"stage2_phase: {name}" in events, (path, name)
        assert "stage2_real: B1=1000 B2=40000 D=" in events, path
        assert "batched_progress: batch=" in events and "left=" in events, path
    print("PASS worker tune override, live GWNUM phases and batches, and exact queue completion", flush=True)

    if args.portable_fixture:
        executable = args.portable_fixture.resolve(strict=True)
        for path in (queue_tune, shared):
            serialized = subprocess.check_output([executable, "--roundtrip", path], timeout=60)
            restored = tomllib.loads(serialized.decode("utf-8"))
            current = read_tune(path)
            assert restored["cpu_stage2"] == current["cpu_stage2"]
            if "ntt" in current:
                assert "ntt" in restored
        print("PASS CPU-only and mixed profiles survive the CUDA profile serializer", flush=True)

    (out / "summary.json").write_text(json.dumps(
        {"curves": sum(observed.values()), "sample_count": sum(row["repeats"] for row in tuned),
         "candidate_d": sorted(row["d"] for row in tuned), "reused_d": winner["d"],
         "queue_completed": len(results)}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
