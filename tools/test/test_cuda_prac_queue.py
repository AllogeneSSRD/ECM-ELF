"""Verify CUDA PARAM0 PRAC configuration, checkpoint recovery and Stage2 handoff."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from test_cuda_prac import checkpoints, rows


def run(exe, folder, name, env, args=(), success=True):
    log = folder / (name + ".log")
    with log.open("wb") as output:
        proc = subprocess.run([str(exe), "--config", str(folder / "ecm.toml"), *args],
                              cwd=folder, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=180)
    text = log.read_text(encoding="utf-8", errors="replace")
    assert (proc.returncode == 0) == success, (log, proc.returncode, text[-2000:])
    return text


def write_config(folder, device, chain, search, stage2=False):
    text = [
        "config_version = 3", "[shared]", 'log_file = ""', 'worktodo_file = "worktodo.txt"',
        "[stage1]", 'target = "gpu"', 'gpu_backend = "cuda"', 'gpu_algorithm = "cgbn"',
        'param = "m0"', f"gpu_device = {device}", "sigma = 26", "verbose = true",
        f'save_dir = "{folder.as_posix()}"', "checkpoint_interval_seconds = 0",
        'exponent_cache_dir = "off"', "cuda_slice_depth = 3",
        'handoff.mode = "ecm-elf"' if stage2 else 'handoff.mode = "off"',
    ]
    if chain is not None:
        text.append(f'cuda_montgomery_chain = "{chain}"')
    if search is not None:
        text.extend(["[worker.1]", f"stage1.cuda_prac_search = {search}"])
    (folder / "ecm.toml").write_text("\n".join(text) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--stage2", type=Path)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    exe = args.exe.resolve(strict=True)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("ECM_", "NTT_"))}

    def queue(name, lines, exponent=127):
        folder = root / name
        folder.mkdir()
        text = "\n".join(
            f'ECMSTAGE1=1,2,{exponent},-1,"{(folder / f"{param}_m{exponent}_1000.save").as_posix()}",10000,0,8'
            for param in lines) + "\n"
        (folder / "worktodo.txt").write_text(text, encoding="ascii")
        return folder, text

    reference, _ = queue("reference", ["m0"])
    write_config(reference, args.device, None, None)
    assert "Stage1 ladder completed" in run(exe, reference, "ladder", env)
    expected = rows(reference)
    assert len(expected) == 8

    resumed, original_queue = queue("resumed", ["m0"])
    write_config(resumed, args.device, "prac", 50, stage2=True)
    contradict = dict(env, ECM_GPU_STAGE1_ALGO="ladder", ECM_PRAC_SEARCH="1")
    sampled = run(exe, resumed, "stop", dict(contradict, ECM_GPU_STAGE1_SAMPLE_SECONDS="0.000001"),
                  success=False)
    assert "search=50" in sampled and "sample limit reached" in sampled
    saved_checkpoint, = checkpoints(resumed)
    snapshot = saved_checkpoint.read_bytes()
    assert not rows(resumed) and (resumed / "worktodo.txt").read_text() == original_queue

    write_config(resumed, args.device, "ladder", 50, stage2=True)
    rejected = run(exe, resumed, "wrong-chain", env, success=False)
    assert "checkpoint belongs to different" in rejected
    write_config(resumed, args.device, "prac", 1, stage2=True)
    rejected = run(exe, resumed, "wrong-search", env, success=False)
    assert "resident checkpoint request/layout mismatch" in rejected
    assert saved_checkpoint.read_bytes() == snapshot and not rows(resumed)
    assert (resumed / "worktodo.txt").read_text() == original_queue

    write_config(resumed, args.device, "prac", 50, stage2=True)
    completion = run(exe, resumed, "resume", contradict)
    assert "resident checkpoint resumed" in completion and "Stage1 PRAC completed" in completion
    assert not checkpoints(resumed) and rows(resumed) == expected
    handoff = (resumed / "worktodo.txt").read_text().strip()
    assert handoff.count("ECMSTAGE2=") == 1 and "ECMSTAGE1=" not in handoff
    assert handoff.endswith(",10000,0,8"), handoff

    mixed, _ = queue("mixed", ["m0", "m2", "m3", "e"], exponent=347)
    write_config(mixed, args.device, "prac", 7)
    completion = run(exe, mixed, "mixed", env)
    assert completion.count("Stage1 PRAC completed") == 1
    assert completion.count("Stage1 ladder completed") == 2
    assert "GPU Edwards Stage1:" in completion
    assert not (mixed / "worktodo.txt").read_text().strip()

    legacy, _ = queue("legacy", ["m0"])
    write_config(legacy, args.device, None, None)
    completion = run(exe, legacy, "legacy", dict(env, ECM_GPU_STAGE1_ALGO="prac", ECM_PRAC_SEARCH="3"))
    assert "search=3" in completion and rows(legacy) == expected

    invalid, original_queue = queue("invalid", ["m0"])
    for chain, search, key in (("unknown", 7, "cuda_montgomery_chain"), ("prac", 0, "cuda_prac_search"),
                               ("prac", 51, "cuda_prac_search")):
        write_config(invalid, args.device, chain, search)
        assert key in run(exe, invalid, f"invalid-{chain}-{search}", env, success=False)
        assert (invalid / "worktodo.txt").read_text() == original_queue and not checkpoints(invalid)

    factor = root / "factor"
    factor.mkdir()
    write_config(factor, args.device, "prac", 50)
    direct = subprocess.run(
        [str(exe), "--config", str(factor / "ecm.toml"), "--gpu-param", "0", "-gpucurves", "8", "-sigma",
         "0:4611686018427511360", "-savea", "completed.save", "1000", "0"],
        input=b"(1009*(2^127-1))\n", cwd=factor, env=env, capture_output=True, timeout=60)
    (factor / "run.log").write_bytes(direct.stdout + direct.stderr)
    assert direct.returncode == 0 and b"Stage1 PRAC completed" in direct.stdout
    results = [json.loads(line) for line in (factor / "stage1_results.jsonl").read_text().splitlines()]
    hits = [row for row in results if row["status"] == "F"]
    assert len(results) == 8 and hits
    assert all([str(f) for f in row["factors"]] == ["1009"] for row in hits)
    submission = [json.loads(line) for line in (factor / "results.json.txt").read_text().splitlines()]
    assert sum(row["status"] == "F" and row["curves"] == 1 for row in submission) == len(hits)

    if args.stage2:
        completion = run(args.stage2.resolve(strict=True), resumed, "stage2", env,
                         args=["--device", str(args.device), "--once",
                               "--results", str(resumed / "stage2_results.jsonl"),
                               "--submission", str(resumed / "results.json_2.txt")])
        assert not (resumed / "worktodo.txt").read_text().strip(), completion[-3000:]
        results = [json.loads(line) for line in (resumed / "stage2_results.jsonl").read_text().splitlines()]
        assert len(results) == 8 and all(row["status"] == "NF" for row in results)
        submission, = (json.loads(line) for line in (resumed / "results.json_2.txt").read_text().splitlines())
        assert (submission["status"], submission["curves"], submission["b1"], submission["b2"]) == (
            "NF", 8, 1000, 10000)

    print("PASS CUDA PRAC TOML, legacy overrides, mixed-curve queue, checkpoint, factor and handoff")


if __name__ == "__main__":
    main()
