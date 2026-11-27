"""Exercise P-1 GPU timing, portable matrix round trips, and mixed-task batching."""
import argparse
import json
import math
import re
import struct
import subprocess
import tomllib
import zlib
from pathlib import Path

from test_pm1_stage1_queue import saved_residue


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
    run("cold", "-tune", spec)
    original = tomllib.loads(profile.read_text())["pm1"]["stage1"]["matrix"]
    columns, *rows = original
    assert "batch" in columns and not any("naf" in c or c == "curve" for c in columns)
    fine = [dict(zip(columns, row)) for row in rows if row[columns.index("sample_phase")] == 2]
    assert {r["batch"] for r in fine} == {1, 2} and all(r["repeats"] == 2 for r in fine)
    for r in fine:
        expected = 1000 * r["batch"] * r["timed_steps"] / r["median_ms"]
        assert math.isclose(r["iter_per_s"], expected, rel_tol=1e-9)
    run("hot", "-tune", spec)
    assert tomllib.loads(profile.read_text())["pm1"]["stage1"]["matrix"] == original
    run("force", "-tune", spec + ",force")
    pm1 = tomllib.loads(profile.read_text())["pm1"]
    run("ecm", "-tune", "ntt,length=8,naf=2,batch=1,time=0.01,repeat=1:1", "--param", "e")
    db = tomllib.loads(profile.read_text())
    assert "ntt" in db and db["pm1"] == pm1
    run("pm1_again", "-tune", "pm1,exp=37,batch=1,time=0.01,repeat=1:1,force")
    assert tomllib.loads(profile.read_text())["ntt"] == db["ntt"]

    # Deterministic selectors exercise the 95% decision independently of GPU clocks.
    def set_rates(single, double):
        raw = profile.read_text()
        matrix = tomllib.loads(raw)["pm1"]["stage1"]["matrix"]
        header = matrix[0]
        for row in matrix[1:]:
            rate = single if row[header.index("batch")] == 1 else double
            row[header.index("iter_per_s")] = rate
            row[header.index("median_ms")] = 1000 * row[header.index("batch")] * row[header.index("timed_steps")] / rate
        section = "[pm1.stage1]\nmatrix = [\n" + "".join("    " + json.dumps(r) + ",\n" for r in matrix) + "]"
        profile.write_text(re.sub(r"\[pm1\.stage1\]\s*matrix = \[.*?\n\]", lambda _: section, raw, flags=re.S))

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
    assert "unread.save" in queue.read_text() and "1,2,53,-1" in queue.read_text()
    finished = (root / "finished.txt").read_text().splitlines()
    assert [r.split(" #")[0] for r in finished] == lines
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
    print("P-1 tune matrix, independent resume, queue boundaries and batch threshold passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    check(args.exe.resolve(strict=True), args.output.resolve())
