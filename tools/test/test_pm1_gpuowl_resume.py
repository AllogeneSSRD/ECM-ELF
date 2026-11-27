"""Run both directions of a P-1 Stage 1 checkpoint handoff against a local GpuOwl."""
import argparse
import re
import struct
import subprocess
import tempfile
import time
import zlib
from pathlib import Path
from test_stage1_ntt_task_batches import pausable_process


def run(program, args, work, name):
    process = subprocess.run([str(program), *args], cwd=work, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=300)
    (work / (name + ".log")).write_bytes(process.stdout)
    assert process.returncode == 0, process.stdout[-2500:]
    return process.stdout.decode("utf-8", errors="replace")


def checkpoint(path):
    header, payload = path.read_bytes().split(b"\n", 1)
    assert len(payload) == 4 + 4 * ((400381 + 31) // 32)
    assert struct.unpack_from("<I", payload)[0] == zlib.crc32(payload[4:])
    k = int(re.search(rb" k=(\d+) bits=744 ", header).group(1))
    assert k == int(path.stem.split("-")[1])
    return k, payload[4:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--gpuowl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    worker = args.exe.resolve(strict=True)
    gpuowl = args.gpuowl.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="resume-", dir=args.output)).resolve()
    print(f"P-1 interop evidence: {root}", flush=True)

    def owl_task(name):
        work = root / name
        work.mkdir()
        (work / "worktodo-0.txt").write_text("Pminus1=1,2,400381,-1,500,0\n", encoding="ascii")
        return work

    source, reference, destination = (owl_task(s) for s in ("source", "reference", "destination"))
    owl_args = ["-device", str(args.device), "-noclean", "-log", "1000"]
    run(gpuowl, [*owl_args, "-iters", "110"], source, "partial")
    incoming = max((source / "p1-400381").glob("*.p1"), key=lambda path: checkpoint(path)[0])
    initial_k, _ = checkpoint(incoming)
    assert 1 < initial_k < 744
    run(gpuowl, owl_args, reference, "complete")
    end = "p1-400381/400381-000000744.p1"
    _, expected = checkpoint(reference / end)

    work = root / "ecm"
    task = work / "pm1/M400381-B1-500"
    saves = task / "p1-400381"
    saves.mkdir(parents=True)
    (saves / incoming.name).write_bytes(incoming.read_bytes())
    (work / "worktodo.txt").write_text("Pminus1=1,2,400381,-1,500,1000\n", encoding="ascii")
    cfg = work / "ecm.toml"
    cfg.write_text(f'''config_version = 4
[shared]
worktodo_file = "{work.as_posix()}/worktodo.txt"
finished_file = "{work.as_posix()}/finished.txt"
log_file = "{work.as_posix()}/screen.log"
[stage1]
target = "gpu"
stage1_gpu_device = {args.device}
checkpoint_interval_seconds = 0.001
exponent_cache_dir = "off"
ntt_convolution = "weighted"
''', encoding="utf-8")
    stopped = work / 'worker_partial.log'
    with pausable_process([worker, '--config', cfg], work, None, stopped) as (process, stop):
        deadline = time.monotonic() + 60
        while b'pm1_stage1_progress:' not in stopped.read_bytes():
            assert time.monotonic() < deadline and process.poll() is None
            time.sleep(.002)
        stop()
        assert process.wait(timeout=90) != 0
    assert f"k={initial_k}/744" in stopped.read_text(errors='replace')

    middle = sorted((checkpoint(file)[0], file) for file in saves.glob("*.p1"))
    outgoing_k, outgoing = next(pair for pair in middle if initial_k < pair[0] < 744)
    (destination / "p1-400381").mkdir()
    (destination / "p1-400381" / outgoing.name).write_bytes(outgoing.read_bytes())
    run(worker, ["--config", str(cfg)], work, "worker")
    native = (task / 'm0400381').read_bytes()
    assert int.from_bytes(native[80:], 'little') == int.from_bytes(expected, 'little')
    assert not saves.exists()
    log = run(gpuowl, owl_args, destination, "resumed")
    assert re.search(rf"OK\s+{outgoing_k} P-1 loaded:", log), log[-2000:]
    assert checkpoint(destination / end)[1] == expected
    print(f"GpuOwl -> ECM-ELF at k={initial_k}, ECM-ELF -> GpuOwl at k={outgoing_k}; "
          f"all {len(expected)} residue bytes match.")


if __name__ == "__main__":
    main()
