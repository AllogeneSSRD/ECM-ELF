"""Validate Mersenne-carried Stage1 against cofactor arithmetic and queue outputs."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time
import tomllib

from test_stage1_ntt_production import edwards_stage1, edwards_checksum
from test_stage1_ntt_task_batches import pausable_process, checkpoint_cursor, read_log
from stage1_checkpoint_decode import batch, ntt_inflight
from suyama_mont_ref import suyama_curve, lcm_1_to, ladder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--stage2-exe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    exe, root = args.exe.resolve(strict=True), args.output.resolve()
    root.mkdir(parents=True)
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("ECM_", "NTT_"))}
    report = {"program": subprocess.check_output([exe, "--version"], text=True).strip(),
              "device": args.device, "checks": []}

    def passed(name, **values):
        report["checks"].append({"name": name, **values})
        (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("PASS", name, flush=True)

    def fixture(name, model="e", p=15919, known=1782929, b1=64, count=2, sigma=20260922,
                algorithm="ntt", mode="weighted", window=6, allow=None):
        work = root / name
        work.mkdir()
        config = work / "ecm.toml"
        config.write_text(
            'config_version = 4\n[shared]\nworktodo_file = "worktodo.txt"\n'
            'finished_file = "finished.txt"\nlog_file = ""\n'
            f'tune_file = "{(work / "tune.toml").as_posix()}"\n'
            f'[stage1]\nparam = "{model}"\ntarget = "gpu"\nstage1_gpu_backend = "cuda"\n'
            f'stage1_gpu_device = {args.device}\ngpu_algorithm = "{algorithm}"\n'
            f'stage1_save_dir = "{work.as_posix()}"\nhandoff_mode = "ecm-elf"\n'
            f'edwards_naf_window_bits = {window}\nntt_convolution = "{mode}"\n'
            'exponent_cache_dir = "off"\ncheckpoint_interval_seconds = 0.01\n'
            + ("" if allow is None else f"ntt_allow_known_factors = {allow}\n"),
            encoding="utf-8",
        )
        (work / "worktodo.txt").write_text(
            f'ECM2=1,2,{p},-1,{b1},{b1*10},{count},{sigma},"{known}"\n', encoding="ascii"
        )
        return work, [exe, "--config", config], ((1 << p) - 1) // known

    def run(work, command, label="run", dictionary="gpu", success=True):
        process = subprocess.run(command, cwd=work, env={**environment, "ECM_EDWARDS_NTT_DICTIONARY": dictionary},
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        log = process.stdout + process.stderr
        (work / f"{label}.log").write_text(log, encoding="utf-8")
        assert (process.returncode == 0) == success, (process.returncode, log[-5000:])
        return log

    def inspect(work, n, model, b1=64, known=1782929, count=2, oracle=True, p=15919):
        points = {}
        for line in next(work.glob("*.save")).read_text(encoding="utf-8").splitlines():
            row = dict(v.strip().split("=", 1) for v in line.split(";") if "=" in v)
            assert int(row["B1"]) == b1
            if model == "m0":
                assert row["N"] == f"(1*2^{p}-1)/({known})"
                assert int(row["CHECKSUM"]) == (b1 * int(row["SIGMA"]) * n * int(row["X"], 0)) % 4294967291
            saved_n = int(row["N"], 0) if model == "e" else n
            assert saved_n == n
            sigma, x = int(row["SIGMA"]), int(row["X"], 0)
            assert 0 <= x < n and sigma not in points
            points[sigma] = (row.get("STATUS", "POINT"), x)
            if model == "e":
                assert int(row["CHECKSUM"]) == edwards_checksum(n, x, sigma, b1, row["STATUS"])
            if oracle and row.get("STATUS", "POINT") != "FACTOR":
                if model == "e":
                    _, rx, rz = edwards_stage1(n, sigma, b1)
                else:
                    _, a24, rx, rz = suyama_curve(sigma, n)
                    rx, rz = ladder(lcm_1_to(b1), rx, rz, a24, n)
                assert math.gcd(rz, n) == 1 and x * rz % n == rx
        assert len(points) == count, points
        submissions = [json.loads(line) for line in (work / "results.json.txt").read_text().splitlines()]
        assert all(str(known) in row["known-factors"] for row in submissions), submissions
        for row in submissions:
            for factor in row.get("factors", []):
                factor = int(factor)
                assert 1 < factor < n and n % factor == 0 and factor != known, row
        finished = (work / "finished.txt").read_text().strip()
        fields = next(csv.reader([finished.split("=", 1)[1]]))
        assert int(fields[-2]) == count, fields
        handoffs = [line for line in (work / "worktodo.txt").read_text().splitlines()
                    if line.startswith("ECMSTAGE2=")]
        if all(row["status"] == "NF" for row in submissions):
            assert len(handoffs) == 1 and int(submissions[0]["curves"]) == count
            handoff = next(csv.reader([handoffs[0].split("=", 1)[1]]))
            assert int(handoff[-2]) == count and handoff[-1] == str(known), handoff
        return points, handoffs

    reference = {}
    for model in ("e", "m0"):
        work, command, n = fixture(f"{model}_cgbn", model=model, algorithm="cgbn", window=6)
        run(work, command)
        reference[model], _ = inspect(work, n, model)
    for model, window, dictionary, mode in (
        ("e", 0, "host", "linear"), ("e", 6, "host", "weighted"),
        ("e", 8, "gpu", "weighted"), ("m0", 0, "host", "linear"), ("m0", 0, "host", "weighted")
    ):
        name = f"{model}_w{window}_{dictionary}_{mode}"
        algorithm = "ntt[14336:], cgbn[]" if model == "e" and window == 0 else "ntt"
        work, command, n = fixture(name, model=model, window=window, algorithm=algorithm, mode=mode)
        log = run(work, command, dictionary=dictionary)
        assert "backend=ntt" in log and "target_bits=15899" in log and "carrier_exponent=15919" in log
        points, handoffs = inspect(work, n, model)
        assert points == reference[model]
        columns, *values = tomllib.loads((work / "tune.toml").read_text(encoding="utf-8"))["ntt"]["matrix"]
        measurements = [dict(zip(columns, row, strict=True)) for row in values]
        assert any(row.get("benchmark_kind") == "stage1_runtime" and row.get("target_bits") == 15899
                   and row.get("arithmetic_bits") == 15919 for row in measurements), measurements
        passed(name, target_bits=n.bit_length(), arithmetic_bits=15919)
        if model == "e":
            stage2_handoff = handoffs[0]

    work, command, n = fixture("disabled_policy", algorithm="ntt[14336:], cgbn[]", allow=0)
    assert "backend=cgbn" in run(work, command)
    assert inspect(work, n, "e")[0] == reference["e"]
    passed("disabled_policy_uses_cgbn")
    for value, reason in ((0, "Mersenne carrier"), (2, "ntt_allow_known_factors")):
        work, command, _ = fixture(f"rejected_{value}", allow=value)
        before = (work / "worktodo.txt").read_bytes()
        assert reason in run(work, command, success=False)
        assert (work / "worktodo.txt").read_bytes() == before
    passed("disabled_explicit_and_invalid_config_preserve_queue")

    small = {}
    for model, sigma, bound in (("m0", 30, 40), ("e", 20260922, 2)):
        for dictionary in ("host", "gpu"):
            name = f"known31_{model}_{dictionary}"
            work, command, n = fixture(name, model=model, p=65, known=31, sigma=sigma,
                                      b1=bound, window=6, mode="linear")
            run(work, command, dictionary=dictionary)
            points, _ = inspect(work, n, model, b1=bound, known=31, oracle=False, p=65)
            if model in small:
                assert points == small[model]
            small[model] = points
            submitted = [json.loads(row) for row in (work / "results.json.txt").read_text().splitlines()]
            passed(name, factors=[int(factor) for row in submitted for factor in row.get("factors", [])])
    assert any(x == 8191 for _, x in small["m0"].values()), small

    known = ((1 << 244) - 1) // ((1 << 61) - 1)
    work, command, n = fixture("small_target_large_carrier", p=244, known=known)
    log = run(work, command)
    assert "target_bits=61" in log and "carrier_exponent=244" in log
    inspect(work, n, "e", known=known, p=244)
    passed("small_target_large_carrier", target_bits=61, arithmetic_bits=244)

    work, command, n = fixture("resume", p=244, known=known, b1=1024, count=130, window=6, allow=1)
    checkpoint = work / ".e_s1_m244_1024.ckpt"
    env = {**environment, "ECM_EDWARDS_NTT_DICTIONARY": "gpu"}
    with pausable_process(command, work, env, work / "stop.log") as (process, stop):
        deadline = time.monotonic() + 120
        while process.poll() is None and time.monotonic() < deadline:
            if checkpoint.exists() and checkpoint_cursor(checkpoint):
                stop()
                assert process.wait(timeout=60) != 0
                break
            time.sleep(0.01)
        else:
            raise AssertionError(read_log(work / "stop.log")[-4000:])
    identity, _ = batch(checkpoint)
    assert "|ntt-carrier-v1|244" in identity, identity
    sigmas, snapshot = ntt_inflight(checkpoint)
    assert sigmas and snapshot.startswith(b"ECMNTTE1")
    saved = checkpoint.read_bytes()
    config = work / "ecm.toml"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "ntt_allow_known_factors = 1", "ntt_allow_known_factors = 0"), encoding="utf-8")
    run(work, command, label="disabled_resume", success=False)
    assert checkpoint.read_bytes() == saved
    config.write_text(config.read_text(encoding="utf-8").replace(
        "ntt_allow_known_factors = 0", "ntt_allow_known_factors = 1"), encoding="utf-8")
    run(work, command, label="resumed")
    resumed, _ = inspect(work, n, "e", b1=1024, count=130, oracle=False, known=known, p=244)
    full, full_command, _ = fixture("continuous", p=244, known=known, b1=1024, count=130, window=6)
    run(full, full_command)
    assert inspect(full, n, "e", b1=1024, count=130, oracle=False, known=known, p=244)[0] == resumed
    passed("cofactor_checkpoint_resume_matches_continuous", curves=130, target_bits=61, arithmetic_bits=244)

    if args.stage2_exe:
        stage2 = args.stage2_exe.resolve(strict=True)
        work = root / "stage2"
        work.mkdir()
        (work / "worktodo.txt").write_text(stage2_handoff + "\n", encoding="utf-8")
        (work / "ecm.toml").write_text(
            'config_version = 4\n[shared]\nlog_file = ""\n'
            '[stage1]\nntt_allow_known_factors = 0\n'
            '[stage2]\ndevice = "gpu"\nstage2_results_file = "stage2_results.jsonl"\ntune_runtime_enabled = false\n', encoding="utf-8")
        run(work, [stage2, "--config", work / "ecm.toml", "--device", str(args.device),
                   "--d", "210", "--factor-only"])
        rows = [json.loads(row) for row in (work / "stage2_results.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all("1782929" in row["known-factors"] for row in rows)
        passed("stage2_consumes_cofactor_save")
    report["complete"] = True
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
