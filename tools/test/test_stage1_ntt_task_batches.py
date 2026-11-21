#!/usr/bin/env python3
"""Verify production NTT queue splitting, tune samples, resume, and Stage2 delivery."""

import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import csv
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time
import tomllib

from test_cuda_queue_stop import SecurityAttributes
from test_stage1_ntt_production import edwards_stage1, ntt_executions, saved_points
from stage1_checkpoint_decode import ntt_inflight


EXPONENT = 4423  # Prime modulus with a valid CUDA Stage2 handoff.
SIGMA = 20260922
BOUND = 1024
STAGE2_BOUND = 4096
CURVES = 1000
CHECKPOINT = f".e_s1_m{EXPONENT}_{BOUND}.ckpt"


def read_log(path):
    return path.read_text(encoding="utf-8", errors="replace")


def log_records(log, label):
    return [dict(re.findall(r"([\w]+)=(\S+)", line.split(label, 1)[1]))
            for line in log.splitlines() if label in line]


def batch_launches(log):
    keys = ("offset", "total", "curves", "capacity", "resumed")
    return [{key: int(row[key]) for key in keys}
            for row in log_records(log, "stage1_ntt_batch:")]


def tune_rows(tune, p):
    columns, *values = tomllib.loads(tune.read_text(encoding="utf-8"))["ntt"]["matrix"]
    rows = [dict(zip(columns, row, strict=True)) for row in values]
    return [row for row in rows if row.get("benchmark_kind") == "stage1_runtime"
            and row["curve"] == "edwards" and row["arithmetic_bits"] == p
            and row["naf_window"] == 6 and row["b1"] == BOUND]


def read_series(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assert_finished(work, requested):
    todo = (work / "worktodo.txt").read_text(encoding="utf-8").splitlines()
    handoff, = [line for line in todo if line.startswith("ECMSTAGE2=")]
    assert all(line.startswith("ECMSTAGE2=") or not line.strip() for line in todo), todo
    fields, = list(csv.reader([handoff.removeprefix("ECMSTAGE2=")]))
    assert list(map(int, fields[-3:])) == [STAGE2_BOUND, 0, requested], fields
    save = Path(fields[-4])
    points = saved_points(save, EXPONENT, BOUND, edwards=True)
    assert list(points) == list(range(SIGMA, SIGMA + requested)), save
    receipts = read_series(work / "results.json.txt")
    assert len(receipts) == 1 and int(receipts[0]["curves"]) == requested, receipts
    assert receipts[0]["status"] == "NF" and receipts[0]["b1"] == receipts[0]["b2"] == BOUND
    assert not (work / CHECKPOINT).exists(), work
    finished = (work / "finished.txt").read_text(encoding="utf-8").splitlines()
    assert len(finished) == 1 and finished[0].startswith("ECMSTAGE1="), finished
    finished_fields, = list(csv.reader([finished[0].removeprefix("ECMSTAGE1=")]))
    assert finished_fields == fields, (finished, handoff)
    results = read_series(work / "stage1_results.jsonl")
    assert [int(row["Edwards"]["sigma"]) for row in results] == list(points), work
    (work / "handoff.txt").write_text(handoff + "\n", encoding="utf-8")
    return save, handoff


def assert_schedule(launches, requested, expect_pad):
    assert len(launches) > 1, launches
    offset = 0
    for row in launches:
        assert row["offset"] == offset and row["total"] == requested, launches
        assert offset < requested and 0 < row["curves"] == row["capacity"] <= 128, launches
        offset += row["curves"]
    if expect_pad:
        assert requested < offset < requested + 128, launches
    else:
        assert offset == requested, launches
    return offset


@contextmanager
def pausable_process(command, work, env, log_path):
    with log_path.open("wb") as output:
        if os.name != "nt":
            process = subprocess.Popen(command, cwd=work, env=env, stdout=output,
                                       stderr=output, start_new_session=True)
            stop = lambda: process.send_signal(signal.SIGINT)
            cleanup = lambda: None
        else:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateEventW.argtypes = [ctypes.POINTER(SecurityAttributes), wintypes.BOOL,
                                            wintypes.BOOL, wintypes.LPCWSTR]
            kernel.CreateEventW.restype = wintypes.HANDLE
            kernel.SetEvent.argtypes = [wintypes.HANDLE]
            kernel.SetEvent.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), None, True)
            event = kernel.CreateEventW(ctypes.byref(attributes), True, False, None)
            if not event:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                process = subprocess.Popen([*command, "--stage1-stop-handle", str(event)],
                                           cwd=work, env=env, stdout=output, stderr=output,
                                           close_fds=False, creationflags=subprocess.CREATE_NO_WINDOW)
            except BaseException:
                kernel.CloseHandle(event)
                raise

            def stop():
                if not kernel.SetEvent(event):
                    raise ctypes.WinError(ctypes.get_last_error())

            def cleanup():
                kernel.CloseHandle(event)

        try:
            yield process, stop
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            cleanup()


def checkpoint_cursor(path):
    # The task checkpoint embeds the production Edwards NAF payload unchanged.
    try:
        data = path.read_bytes()
    except (FileNotFoundError, PermissionError):
        return 0
    marker = b"ECMNTTE1"
    start = data.find(marker)
    if start < 0 or len(data) < start + len(marker) + 12:
        return 0
    length, window, cursor = struct.unpack_from("<III", data, start + len(marker))
    return cursor if window == 6 and 0 < cursor < length else 0


def stop_inside_later_batch(work, command, env, timeout, *, extended_tail=False):
    log_path = work / "stop.log"
    checkpoint = work / CHECKPOINT
    with pausable_process(command, work, env, log_path) as (process, stop):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and process.poll() is None:
            log = read_log(log_path)
            launches = batch_launches(log)
            if extended_tail and not (launches and launches[-1]["offset"] + launches[-1]["curves"]
                                      > launches[-1]["total"]):
                time.sleep(0.01)
                continue
            # Status lines are flushed before GPU execution. Wait for both a
            # completed batch and a positive cursor inside the next batch.
            executions = ntt_executions(log)
            if (len(launches) >= 2 and len(executions) == len(launches)
                    and checkpoint_cursor(checkpoint)):
                stop()
                assert process.wait(timeout=60) != 0, read_log(log_path)[-4000:]
                stopped_batch = batch_launches(read_log(log_path))[-1]
                cursor = checkpoint_cursor(checkpoint)
                assert stopped_batch["offset"] and cursor, read_log(log_path)[-4000:]
                return stopped_batch, cursor
            time.sleep(0.01)
        raise AssertionError(f"could not interrupt a later batch: {read_log(log_path)[-4000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--stage2-exe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    exe, out = args.exe.resolve(strict=True), args.output.resolve()
    out.mkdir(parents=True)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("ECM_", "NTT_"))}
    version = subprocess.check_output([str(exe), "--version"], cwd=out, env=env,
                                      text=True, encoding="utf-8", timeout=15).strip()
    report = {"program": version, "device": args.device, "curves": CURVES,
              "p": EXPONENT, "b1": BOUND, "b2": STAGE2_BOUND, "checks": []}

    def passed(name, **data):
        report["checks"].append({"name": name, **data})
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("PASS", name, flush=True)

    def fixture(name, requested=0, pad=None, tune=None, checkpoint_period=0,
                naf_window=6, extra_stage1=""):
        work = out / name
        work.mkdir()
        database = tune or work / "tune.toml"
        config = work / "ecm.toml"
        # Keep the 128+tail boundaries used by the scheduling and resume assertions.
        config.write_text(
            f'config_version = 4\n[shared]\nworktodo_file = "worktodo.txt"\n'
            f'finished_file = "finished.txt"\nlog_file = ""\ntune_file = "{database.as_posix()}"\n'
            f'[stage1]\nparam = "e"\ntarget = "gpu"\nstage1_gpu_backend = "cuda"\nstage1_gpu_device = {args.device}\n'
            f'stage1_save_dir = "{work.as_posix()}"\nhandoff_mode = "ecm-elf"\nedwards_naf_window_bits = {naf_window}\n'
            f'exponent_cache_dir = "off"\ncheckpoint_interval_seconds = {checkpoint_period}\nntt_max_batch = 128\n'
            + (f'gpu_algorithm = "ntt"\nntt_pad_batch = {pad}\n' if pad is not None else "") + extra_stage1,
            encoding="utf-8",
        )
        if requested:
            # Give an explicit sigma to verify task-relative offsets through
            # adaptive batches and checkpoints.
            row = f"ECM=1,2,{EXPONENT},-1,{BOUND},{STAGE2_BOUND},{requested},{SIGMA}\n"
            (work / "worktodo.txt").write_text(row, encoding="ascii")
        return work, [str(exe), "--config", str(config)], database

    def run(work, command, label="run", input=None):
        start = time.monotonic()
        process = subprocess.run(command, cwd=work, env=env, input=input,
                                 text=True, encoding="utf-8", errors="replace",
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=args.timeout)
        (work / f"{label}.log").write_text(process.stdout, encoding="utf-8")
        assert process.returncode == 0, (label, process.returncode, process.stdout[-4000:])
        return process.stdout, time.monotonic() - start

    # Force neither backend. The upper case lies beyond the configured cutoff
    # and above CUDA Stage2's width limit, so it is tested in direct Stage1 mode.
    for p, backend in ((127, "cgbn"), (19937, "ntt")):
        work, command, _ = fixture(f"default_{backend}")
        count_flags = ["-gpucurves", "1"] if backend == "cgbn" else []
        log, wall = run(work, [*command, *count_flags, "-sigma", str(SIGMA),
                               "-save", str(work / "direct.save"), "64", "64"],
                        input=f"2^{p}-1\n")
        assert any(row.get("backend") == backend for row in log_records(log, "Stage1 execution:")), log
        assert list(saved_points(work / "direct.save", p, 64, edwards=True)) == [SIGMA]
        passed(f"default_{backend}", p=p, seconds=wall)

    # Establish small production samples without trying larger batches. The
    # live tuner may split 35 into a preferred width and a short tail.
    work, command, small_tune = fixture("trial_seed_35", requested=35, pad=0)
    log, _ = run(work, command)
    seed_launches = batch_launches(log)
    offset = 0
    for row in seed_launches:
        assert row["offset"] == offset and row["total"] == 35 and 0 < row["curves"] <= 35, seed_launches
        offset += row["curves"]
    assert offset == 35, seed_launches
    assert_finished(work, 35)
    incumbents = {row["batch"] for row in tune_rows(small_tune, EXPONENT)}
    assert incumbents == {row["curves"] for row in seed_launches}, incumbents

    work, command, _ = fixture("trial_at_capacity", requested=128, pad=0, tune=small_tune)
    log, _ = run(work, command)
    selections = log_records(log, "stage1_ntt_batch_selection:")
    assert (selections[0]["remaining"], selections[0]["capacity"], selections[0]["source"]) == ("128", "128", "production")
    assert int(selections[0]["batch"]) in incumbents, selections
    assert all(row["source"] != "production_trial" for row in selections)
    assert_finished(work, 128)

    work, command, _ = fixture("trial_above_capacity", requested=200, pad=0, tune=small_tune)
    log, wall = run(work, command)
    selection = log_records(log, "stage1_ntt_batch_selection:")[0]
    assert (selection["source"], selection["remaining"], selection["capacity"], selection["batch"]) == (
        "production_trial", "200", "128", "128"), selection
    launches = batch_launches(log)
    assert [row["curves"] for row in launches] == [128, 72], launches
    assert [int(row["ntt_batch"]) for row in ntt_executions(log)] == [128, 72]
    assert any(row["batch"] == 128 for row in tune_rows(small_tune, EXPONENT))
    assert_finished(work, 200)
    passed("capacity_trial_128_of_200", selection=selection, launches=launches, seconds=wall)

    work, command, _ = fixture("direct_split", pad=0)
    save = work / "direct.save"
    log, wall = run(work, [*command, "-gpucurves", "129", "-sigma", str(SIGMA),
                           "-save", str(save), "64", "64"], input="2^127-1\n")
    schedule = batch_launches(log)
    assert_schedule(schedule, 129, expect_pad=False)
    assert [row["curves"] for row in schedule] == [128, 1], schedule
    assert list(saved_points(save, 127, 64, edwards=True)) == list(range(SIGMA, SIGMA + 129))
    passed("direct_splits_129", p=127, launches=schedule, seconds=wall)

    # Reproduce plan-specific memory caps with a short scalar. The live tuner
    # must use a weighted batch larger than the linear plan could admit.
    for memory, expected in ((6144, (54, 98)), (1024, (14, 25))):
        work, command, _ = fixture(f"per_plan_capacity_{memory}", pad=0, naf_window=12,
                                  extra_stage1=f"naf_memory_mb = 4096\nntt_memory_mib = {memory}\n")
        (work / "worktodo.txt").write_text(f'ECM=1,2,15919,-1,1024,1024,95,{SIGMA},"1782929"\n',
                                         encoding="ascii")
        log, wall = run(work, command)
        selection = log_records(log, "stage1_ntt_batch_selection:")[0]
        assert tuple(int(selection[f"{mode}_capacity"]) for mode in ("linear", "weighted")) == expected, selection
        assert int(selection["capacity"]) == min(95, max(expected)), selection
        assert 0 < int(selection["batch"]) <= int(selection["capacity"]), selection
        launch = log_records(log, "Stage1 execution:")[0]
        assert launch["ntt_type"] == "weighted" and int(launch["ntt_batch"]) > expected[0], launch
        assert int(launch["ntt_batch"]) == int(selection["batch"]), (selection, launch)
        assert int(launch["ntt_memory_per_curve_bytes"]) == 41867520, launch
        assert int(launch["ntt_memory_config_mib"]) == memory and int(launch["ntt_naf_memory_mb"]) == 4096, launch
        assert int(launch["ntt_memory_device_bytes"]) >= int(launch["ntt_batch"]) * 41867520, launch
        assert int(launch["max_curves_in_memory"]) == expected[1], launch
        assert_schedule(batch_launches(log), 95, expect_pad=False)
        passed(f"per_plan_capacity_{memory}", selection=selection, execution=launch, seconds=wall)

    work, command, tune = fixture("uninterrupted", CURVES, pad=0)
    log, wall = run(work, command)
    schedule = batch_launches(log)
    assert_schedule(schedule, CURVES, expect_pad=False)
    save, _ = assert_finished(work, CURVES)
    points = save.read_bytes()
    n = (1 << EXPONENT) - 1
    saved = saved_points(save, EXPONENT, BOUND, edwards=True)
    sample_lanes = (0, schedule[0]["curves"] - 1, schedule[0]["curves"], CURVES - 1)
    for lane in sample_lanes:
        sigma = SIGMA + lane
        _, x, z = edwards_stage1(n, sigma, BOUND)
        assert saved[sigma] == x * pow(z, -1, n) % n, sigma
    measured = tune_rows(tune, EXPONENT)
    assert any(row["batch"] == row["valid_curves"] for row in measured), measured
    assert all(math.isfinite(row["iter_per_s"]) and row["iter_per_s"] > 0
               and math.isfinite(row["curves_per_s"]) and row["curves_per_s"] > 0
               and row["repeats"] > 0 for row in measured), measured
    passed("uninterrupted_1000", launches=schedule, seconds=wall, production_rows=measured,
           oracle_sigmas=[SIGMA + lane for lane in sample_lanes])

    work, command, _ = fixture("interrupted", CURVES, pad=0, tune=tune, checkpoint_period=0.01)
    todo = work / "worktodo.txt"
    original = todo.read_bytes()
    interrupted, cursor = stop_inside_later_batch(work, command, env, args.timeout)
    assert todo.read_bytes() == original
    assert all(path.stat().st_size == 0 for path in work.glob("*.save"))
    assert not (work / "finished.txt").exists() or (work / "finished.txt").stat().st_size == 0
    assert not (work / "results.json.txt").exists() or (work / "results.json.txt").stat().st_size == 0
    config = work / "ecm.toml"
    config.write_text(config.read_text(encoding="utf-8").replace(
        'gpu_algorithm = "ntt"', 'gpu_algorithm = "cgbn[]"').replace(
        "edwards_naf_window_bits = 6", "edwards_naf_window_bits = 8"), encoding="utf-8")
    log, wall = run(work, command, "resumed")
    resumed = batch_launches(log)
    assert resumed and resumed[0]["resumed"] == 1
    for key in ("offset", "total", "capacity", "curves"):
        assert resumed[0][key] == interrupted[key], (resumed, interrupted)
    restored, *fresh = ntt_executions(log)
    assert int(restored["ntt_resume"]) == cursor and restored["naf"] == "6", log[:4000]
    assert fresh and all(row["ntt_resume"] == "0" and row["naf"] == "8" for row in fresh), fresh
    assert f"stage1_ntt_naf_resume: saved_window=6 configured_window=8 cursor={cursor}" in log
    prior = [row for row in batch_launches(read_log(work / "stop.log"))
             if row["offset"] < interrupted["offset"]]
    assert_schedule(prior + resumed, CURVES, expect_pad=False)
    save, handoff = assert_finished(work, CURVES)
    assert save.read_bytes() == points
    assert any(row.get("source") == "production"
               for row in log_records(read_log(work / "stop.log") + log, "stage1_ntt_batch_selection:"))
    published = [save, todo, work / "stage1_results.jsonl", work / "results.json.txt", work / "finished.txt"]
    before = [path.read_bytes() for path in published]
    empty, _ = run(work, command, "already_finished")
    assert not batch_launches(empty) and [path.read_bytes() for path in published] == before
    passed("resume_1000_and_single_handoff", interrupted=interrupted, cursor=cursor,
           launches=resumed, seconds=wall, handoff=handoff)

    # Choose a prime assignment size above 128 so a stable multi-curve batch
    # has a real short tail while leaving plenty of points to compare.
    requested = 131
    work, command, _ = fixture("padded_tail", requested, pad=1, tune=tune)
    log, wall = run(work, command)
    padded = batch_launches(log)
    actual = assert_schedule(padded, requested, expect_pad=True)
    save, expanded_handoff = assert_finished(work, actual)
    assert save.read_bytes().splitlines() == points.splitlines()[:actual]
    passed("padded_tail_adds_real_curves", requested=requested, actual=actual,
           launches=padded, seconds=wall, handoff=expanded_handoff)

    work, command, _ = fixture("padded_tail_resume", requested, pad=1, tune=tune, checkpoint_period=0.01)
    interrupted, cursor = stop_inside_later_batch(work, command, env, args.timeout, extended_tail=True)
    resumed_actual = interrupted["offset"] + interrupted["curves"]
    sigmas, _ = ntt_inflight(work / CHECKPOINT)
    assert sigmas == list(range(SIGMA + interrupted["offset"], SIGMA + resumed_actual))
    config = work / "ecm.toml"
    config.write_text(config.read_text(encoding="utf-8").replace("ntt_pad_batch = 1", "ntt_pad_batch = 0"),
                      encoding="utf-8")
    log, wall = run(work, command, "resumed")
    resumed, = batch_launches(log)
    assert resumed == {**interrupted, "resumed": 1}, resumed
    assert int(ntt_executions(log)[0]["ntt_resume"]) == cursor, log[:4000]
    save, _ = assert_finished(work, resumed_actual)
    assert save.read_bytes().splitlines() == points.splitlines()[:resumed_actual]
    passed("padded_tail_resume_preserves_extra_curves", cursor=cursor, actual=resumed_actual, seconds=wall)

    work, command, _ = fixture("factors", pad=0)
    config = work / "ecm.toml"
    config.write_text(config.read_text(encoding="utf-8").replace('param = "e"', 'param = "m0"')
                      + 'ntt_convolution = "linear"\n', encoding="utf-8")
    todo = work / "worktodo.txt"
    todo.write_text("ECM=1,2,65,-1,40,100,2,30\n", encoding="utf-8")
    log, wall = run(work, command)
    assert len(batch_launches(log)) == 1 and not todo.read_text(encoding="utf-8").strip()
    submitted = read_series(work / "results.json.txt")
    assert [(int(row["sigma"]), row["status"], row["curves"], list(map(int, row["factors"])))
            for row in submitted] == [(30, "F", 1, [253921]), (31, "F", 1, [31])], submitted
    assert len(read_series(work / "stage1_results.jsonl")) == 2
    assert len((work / "finished.txt").read_text(encoding="utf-8").splitlines()) == 1
    assert saved_points(next(work.glob("*.save")), 65, 40) == {30: 253921, 31: 31}
    passed("all_factor_queue_publishes_once", seconds=wall)

    if args.stage2_exe:
        stage2 = args.stage2_exe.resolve(strict=True)
        work = out / "stage2_receive"
        work.mkdir()
        # Execute the last extra curve from a copy of the expanded handoff.
        (work / "worktodo.txt").write_text(
            f"{expanded_handoff.rsplit(',', 2)[0]},{actual - 1},1\n", encoding="utf-8")
        config = work / "ecm.toml"
        # Exercise the exact small handoff bound without rounding it to a full chain.
        config.write_text('config_version = 4\n[shared]\nworktodo_file = "worktodo.txt"\n'
                          'finished_file = "finished.txt"\nlog_file = "stage2.log"\n'
                          f'[stage2]\ndevice = "gpu"\nstage2_gpu_device = {args.device}\n'
                          'b2_round_chain = false\n'
                          'stage2_results_file = "stage2_results.jsonl"\ntune_runtime_enabled = false\n',
                          encoding="utf-8")
        log, wall = run(work, [str(stage2), "--config", str(config), "--device", str(args.device),
                               "--d", "210", "--factor-only"])
        entry, = read_series(work / "stage2_results.jsonl")
        assert int(entry["Edwards"]["sigma"]) == SIGMA + actual - 1, (entry, log[-4000:])
        assert not (work / "worktodo.txt").read_text(encoding="utf-8").strip()
        passed("cuda_stage2_reads_extra_tail_curve", sigma=SIGMA + actual - 1, seconds=wall)

    report["complete"] = True
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
