"""Exercise P-1 GPU timing, portable matrix round trips, and mixed-task batching."""
import argparse
import json
import math
import re
import struct
import subprocess
import time
import tomllib
import zlib
from pathlib import Path

from test_pm1_stage1_queue import saved_residue
from test_stage1_ntt_task_batches import pausable_process


def check(exe, root):
    root.mkdir(parents=True)
    profile = root / "tune.toml"
    queue = root / "worktodo.txt"
    cfg = root / "ecm.toml"
    cfg.write_text(f'config_version=4\n[shared]\nlog_file=""\ntune_file={json.dumps(profile.as_posix())}\n'
                   'finished_file="finished.txt"\n[stage1]\nntt_max_batch=2\nntt_batch_target_percent=95\n'
                   'handoff_mode_pm1="off"\n[stage2]\ntune_runtime_enabled=false\n', encoding="utf-8")

    def run(name, *args, success=True):
        result = subprocess.run([exe, "--config", cfg, *args], cwd=root, capture_output=True,
                                stdin=subprocess.DEVNULL, timeout=150)
        log = (result.stdout + result.stderr).decode(errors="replace")
        (root / (name + ".log")).write_text(log, encoding="utf-8")
        assert (result.returncode == 0) == success, log
        return log

    run("reject_naf", "-tune", "pm1,naf=0", success=False)
    spec = "pm1,ntt,length=2,batch=1-2,time=0.02,repeat=1:2,find=95"
    cold = run("cold", "-tune", spec)
    assert "\x1b[" not in cold
    assert "phase=coarse" in cold and "phase=fine" in cold
    assert cold.index("phase=graph") < cold.index("pm1_stage1_graph:") < cold.index("phase=fine")
    assert cold.count("phase=graph") == 1
    graph = int(re.search(r"pm1_stage1_graph: length=2 batch=1 graph_bits=([0124]) mode=tuned", cold)[1])
    best_batch = int(re.search(rf"pm1_stage1_length_done: .*length=2 phase=fine batch=(\d+) graph={graph} ", cold)[1])
    assert all(re.search(rf"\b{label}\s+2\s+62\s", cold) for label in ("Single","Max","Best","95%"))
    original = tomllib.loads(profile.read_text())["pm1"]["stage1"]["matrix"]
    columns, *rows = original
    assert "batch" in columns and not any("naf" in c or c == "curve" for c in columns)
    def decisions(matrix):
        keys = ("condition", "max_exp", "length", "batch", "ntt_type", "bpw", "tile", "max_radix_bits",
                "passes_forward", "forward", "inverse", "graph_bits", "warp_tail", "dataflow", "ntt_memory_bytes")
        return [tuple(row[matrix[0].index(k)] for k in keys) for row in matrix[1:]]

    records = [dict(zip(columns, row)) for row in rows]
    assert len(set(decisions(original))) == len(rows)
    assert {r["graph_bits"] for r in records if r["batch"] == 1} == {0, 1, 2, 4}
    assert all(r["graph_bits"] == graph for r in records if r["batch"] != 1)
    fine = [r for r in records if "2/2:2" in r.get("tested_groups", "").split(",")]
    assert {r["batch"] for r in fine} == {1, 2}
    for r in records:
        expected = 1000 * r["batch"] * r["timed_steps"] / r["median_ms"]
        assert math.isclose(r["iter_per_s"], expected, rel_tol=1e-9)
    hot = run("hot", "-tune", spec)
    assert hot.count("pm1_stage1_round:") == 1 and "phase=recheck" in hot
    assert f"pm1_stage1_recheck: length=2 batch={best_batch} " in hot
    assert re.search(r"pm1_stage1_tune: .*measured=1 reused=[1-9]", hot)
    assert decisions(tomllib.loads(profile.read_text())["pm1"]["stage1"]["matrix"]) == decisions(original)
    forced = run("force", "-tune", spec + ",force")
    assert "phase=graph" in forced and "phase=fine" in forced
    pm1 = tomllib.loads(profile.read_text())["pm1"]
    run("ecm", "-tune", "ntt,length=8,naf=2,batch=1,time=0.01,repeat=1:1", "--param", "e")
    db = tomllib.loads(profile.read_text())
    assert "ntt" in db and db["pm1"] == pm1
    run("pm1_again", "-tune", "pm1,exp=37,batch=1,time=0.01,repeat=1:1,force")
    assert tomllib.loads(profile.read_text())["ntt"] == db["ntt"]

    # Deterministic selectors exercise the 95% decision independently of GPU clocks.
    def write_matrix(matrix):
        section = "[pm1.stage1]\nmatrix = [\n" + "".join("    " + json.dumps(r) + ",\n" for r in matrix) + "]"
        profile.write_text(re.sub(r"\[pm1\.stage1\]\s*matrix = \[.*?\n\]", lambda _: section, profile.read_text(), flags=re.S))

    def set_rates(single, double):
        matrix = tomllib.loads(profile.read_text())["pm1"]["stage1"]["matrix"]
        header = matrix[0]
        for row in matrix[1:]:
            rate = single if row[header.index("batch")] == 1 else double
            row[header.index("iter_per_s")] = rate
            row[header.index("median_ms")] = 1000 * row[header.index("batch")] * row[header.index("timed_steps")] / rate
        write_matrix(matrix)

    # Start the second independent exponent from a nonzero GpuOwl checkpoint.
    p, b1, k = 41, 31, 11
    exponent = 2*p*math.lcm(*range(1, b1+1))
    initial = pow(3, exponent >> (exponent.bit_length()-k), 2**p-1)
    checkpoint = root / "pm1" / f"M{p}-B1-{b1}" / f"p1-{p}"
    checkpoint.mkdir(parents=True)
    limbs = initial.to_bytes(4*((p+31)//32), "little")
    (checkpoint / f"{p}-{k:09d}.p1").write_bytes(
        f"OWL P1 2 N=1*2^{p}-1 B1={b1} B2=0 k={k} bits={exponent.bit_length()} err=0 time=0\n".encode()
        + struct.pack("<I", zlib.crc32(limbs)) + limbs)
    tasks = [(37,17), (41,31), (43,19), (67,13), (47,11)]
    lines = [f"Pminus1=1,2,{p},-1,{b1},0" for p,b1 in tasks]
    queue.write_text("\n".join(lines) + "\nECMSTAGE2=1,2,127,-1,unread.save,200\nPminus1=1,2,53,-1,7,0\n")
    set_rates(70,100)
    log = run("mixed")
    assert re.findall(r"pm1_stage1_batch: length=(\d+) batch=(\d+) available=(\d+)",log) == [
        ("2","2","2"), ("2","1","1"), ("4","1","1"), ("2","1","1")], log
    assert f"k={k}/{exponent.bit_length()}" in log
    mixed_batch = log.split("pm1_stage1_batch:", 2)[1]
    counts = [(int(done), int(total)) for done, total in re.findall(
        r"pm1_stage1_progress: iter=(\d+)/(\d+)", mixed_batch)]
    total = sum((2*p*math.lcm(*range(1, b1+1))).bit_length()-1 for p,b1 in tasks[:2])
    assert counts and counts == sorted(counts) and counts[-1] == (total, total)
    assert counts[0][0] >= k-1 and all(limit == total for _,limit in counts)
    first = (2*tasks[0][0]*math.lcm(*range(1, tasks[0][1]+1))).bit_length()-1
    assert {int(value) for value in re.findall(r" first_iter=(\d+)", mixed_batch)} == {first}
    assert re.search(rf"iter={total}/{total} 100.00% iter_per_s=[\d.]+ elapsed=[\d.]+s ETA=0.0s", mixed_batch)
    assert re.findall(r"stage1_task_done: processed=(\d+)", log) == ["1","2","3","4","5"]
    assert "unread.save" in queue.read_text() and "1,2,53,-1" in queue.read_text()
    finished = (root / "finished.txt").read_text().splitlines()
    assert sorted(r.split(" #")[0] for r in finished) == sorted(lines)
    for p,b1 in tasks:
        residue = saved_residue(root/"pm1"/f"M{p}-B1-{b1}", p, b1)
        assert residue == pow(3, 2*p*math.lcm(*range(1,b1+1)), 2**p-1)
    set_rates(100,94)
    queue.write_text("Pminus1=1,2,53,-1,7,0\nPminus1=1,2,59,-1,8,0\n")
    assert "batch=1 available=2" in run("threshold")
    run("radix5_tune", "-tune", "pm1,length=2560,batch=2,time=0.01,repeat=1:1")
    wide = [(60013,31),(63029,43)]
    queue.write_text("".join(f"Pminus1=1,2,{p},-1,{b1},0\n" for p,b1 in wide))
    assert "length=2560 batch=2 available=2" in run("radix5_queue")
    for p,b1 in wide:
        assert saved_residue(root/"pm1"/f"M{p}-B1-{b1}",p,b1) == pow(3, 2*p*math.lcm(*range(1,b1+1)), 2**p-1)
    initial_config = cfg.read_text()
    tune_two = ("-tune","pm1,length=2,batch=2,time=0.02,repeat=1:2")
    def limits(common, specific, worker):
        cfg.write_text(initial_config.replace("ntt_max_batch=2",f"ntt_max_batch={common}\nntt_max_batch_pm1={specific}")
                       + "\n[worker.1]\n" + worker)
    def run_two(name):
        queue.write_text("Pminus1=1,2,37,-1,17,0\nPminus1=1,2,41,-1,31,0\n")
        return run(name)
    set_rates(70,100)
    check_batch_completion(exe, root / "batch_completion", profile)
    limits(2,0,"ntt_max_batch_pm1=1")
    run("lowered_tune", *tune_two, success=False)
    assert "batch=2" not in run_two("lowered_queue")
    limits(1,0,"ntt_max_batch_pm1=2")
    run("raised_tune", *tune_two)
    assert "batch=2 available=2" in run_two("raised_queue")
    limits(1,0,'ntt_max_batch_pm1=2\nhandoff_mode_pm1="prime95"\nstage2_after_factored=0')
    miss = "Pminus1=1,2,41,-1,3,1000"
    queue.write_text("Pminus1=1,2,37,-1,3,1000\n" + miss + "\n")
    assert "batch=2 available=2" in run("mixed_handoff")
    assert not (root/"pm1"/"M37-B1-3"/"worktodo.add").exists()
    assert (root/"pm1"/"M41-B1-3"/"worktodo.add").read_text().strip() == miss
    run("ecm_limit", "-tune","ntt,length=8,naf=2,batch=2", "--param","e", success=False)
    limits(2,2,"ntt_max_batch_pm1=0\nntt_max_batch=1")
    run("inherit_worker_limit", *tune_two, success=False)
    assert "batch=2" not in run_two("inherit_worker_queue")
    for value in (-1,513):
        limits(2,value,"")
        run(f"invalid_limit_{value}", *tune_two, success=False)
    # Import overlapping format-19 observations. Only the timing tuple of the
    # winner survives; a different allocation is a separate decision.
    matrix = tomllib.loads(profile.read_text())["pm1"]["stage1"]["matrix"]
    col = matrix[0].index
    duplicate = matrix[1].copy()
    duplicate[col("repeats")] = 3
    duplicate[col("sample_phase")] = 2
    duplicate[col("sample_target_ms")] = 13
    duplicate[col("tested_groups")] = ""
    duplicate[col("iter_per_s")] *= 2
    duplicate[col("median_ms")] /= 2
    other_b1 = duplicate.copy()
    other_b1[col("b1")] += 1
    other_b1[col("repeats")] = 4
    other_b1[col("iter_per_s")] /= 2
    other_b1[col("median_ms")] *= 2
    distinct = duplicate.copy()
    distinct[col("ntt_memory_bytes")] -= 8
    write_matrix(matrix + [duplicate, other_b1, distinct])
    cfg.write_text(initial_config.replace("ntt_max_batch=2", "ntt_max_batch=2\nntt_window_bits=0"))
    fixed = run("fixed_multiple", "-tune", "pm1,length=4-8,batch=1,time=0.01,repeat=1:2")
    assert "phase=graph" not in fixed
    for length in (4, 8):
        assert f"pm1_stage1_graph: length={length} batch=1 graph_bits=0 mode=fixed" in fixed
        assert re.search(rf"pm1_stage1_length_done: .*length={length} phase=fine batch=1 graph=0 ", fixed)
    merged = tomllib.loads(profile.read_text())["pm1"]["stage1"]["matrix"]
    keys = decisions(merged)
    assert len(keys) == len(set(keys))
    for key in decisions([matrix[0], duplicate, distinct]):
        winner = dict(zip(merged[0], merged[1 + keys.index(key)]))
        assert winner["iter_per_s"] == duplicate[col("iter_per_s")]
        assert math.isclose(winner["median_ms"], duplicate[col("median_ms")])
        assert winner["repeats"] == 3 and winner["b1"] == duplicate[col("b1")]
        assert "2/4:13" not in winner.get("tested_groups", "")
    print("P-1 tune matrix, independent resume, queue boundaries and batch threshold passed")


def check_batch_completion(exe, directory, profile):
    directory.mkdir()
    config = directory / "ecm.toml"
    queue = directory / "worktodo.txt"
    finished = directory / "finished.txt"
    receiver = directory / "receiver"
    cfg = ('config_version=4\n[shared]\nlog_file=""\nfinished_file="finished.txt"\n'
           f'tune_file={json.dumps(profile.as_posix())}\n[stage1]\nntt_max_batch=2\n'
           f'prime95_worktodo_file={json.dumps((receiver / "worktodo.txt").as_posix())}\n'
           '[stage2]\nstage2_after_factored=1\n')
    config.write_text(cfg)
    slow, fast = "Pminus1=1,2,37,-1,100000,100001", "Pminus1=1,2,41,-1,3,1000"
    queue.write_text(slow + "\n" + fast + "\n")
    log = directory / "stop.log"
    cmd = [exe, "--config", config]
    with pausable_process(cmd, directory, None, log) as (proc, stop):
        until = time.monotonic() + 30
        while b"P-1 Stage1 complete M41 " not in log.read_bytes():
            assert time.monotonic() < until and proc.poll() is None, log.read_bytes()
            time.sleep(.01)
        stop()
        assert proc.wait(timeout=90) != 0
    assert b"batch=2 available=2" in log.read_bytes()
    assert finished.exists(), "completed lane was discarded after an earlier lane stopped"
    assert finished.read_text().splitlines() == [fast + " # P-1 Stage1 no factor"]
    add = receiver / "worktodo.add"
    assert add.read_text().strip() == fast and queue.read_text().strip() == slow

    add.unlink()
    add.mkdir()
    other = "Pminus1=1,2,43,-1,2,1000"
    queue.write_text(other + "\n" + fast + "\n")
    blocked = subprocess.run(cmd, cwd=directory, capture_output=True, timeout=90)
    (directory / "blocked.log").write_bytes(blocked.stdout + blocked.stderr)
    assert blocked.returncode != 0 and b"batch=2 available=2" in blocked.stdout
    assert sorted(line.split(" #")[0] for line in finished.read_text().splitlines()) == sorted((fast, other))
    assert len(list(directory.glob("worktodo.txt.stage1_completed_worker1*"))) == 2
    assert queue.read_text().splitlines() == [other, fast]
    add.rmdir()
    add.write_text("# existing\n")
    replay = subprocess.run(cmd, cwd=directory, capture_output=True, timeout=90)
    (directory / "replay.log").write_bytes(replay.stdout + replay.stderr)
    assert replay.returncode == 0 and b"START:" not in replay.stdout, replay.stdout + replay.stderr
    header, *rows = add.read_text().splitlines()
    assert header == "# existing" and sorted(rows) == sorted((fast, other))
    assert not (receiver / "worktodo.add.txt").exists() and not list(receiver.glob("*.ready"))
    assert len(finished.read_text().splitlines()) == 2 and not queue.read_text().strip()
    third = "Pminus1=1,2,47,-1,2,1000"
    published = add.read_bytes()
    queue.write_text(third + "\n")
    appended = subprocess.run(cmd, cwd=directory, capture_output=True, timeout=90)
    assert appended.returncode == 0, appended.stdout + appended.stderr
    assert add.read_bytes() == published + (third + "\n").encode()
    assert len(finished.read_text().splitlines()) == 3 and not queue.read_text().strip()
    assert not (receiver / "worktodo.add.txt").exists() and not list(receiver.glob("*.ready"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    check(args.exe.resolve(strict=True), args.output.resolve())
