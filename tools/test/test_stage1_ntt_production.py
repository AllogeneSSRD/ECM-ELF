#!/usr/bin/env python3
"""Exercise the production CUDA Stage1 NTT route, result files and resumable batch."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

from stage1_checkpoint_decode import batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stat"))
from suyama_mont_ref import stage1


def saved_points(path, p, b1):
    n = (1 << p) - 1
    points = {}
    for row in path.read_text(encoding="utf-8").splitlines():
        fields = dict(v.strip().split("=", 1) for v in row.split(";") if "=" in v)
        sigma, x = int(fields["SIGMA"]), int(fields["X"], 0)
        assert fields["METHOD"] == "ECM" and int(fields.get("PARAM", 0)) == 0
        assert int(fields["B1"]) == b1 and 0 <= x < n
        assert int(fields["CHECKSUM"]) == (b1 * sigma * n * x) % 4294967291
        assert sigma not in points
        points[sigma] = x
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    exe, out = args.exe.resolve(strict=True), args.output.resolve()
    out.mkdir(parents=True)

    def fixture(name, algorithm="2", extra=""):
        work = out / name
        work.mkdir()
        config = work / "ecm.toml"
        if "ntt_convolution" not in extra:
            extra += 'ntt_convolution = "linear"\n'
        config.write_text(f'[shared]\nworktodo = "{(work / "worktodo.txt").as_posix()}"\n'
                          f'[stage1]\ntarget = "gpu"\ncurve = "mont"\ngpu_param = 0\n'
                          f'gpu_algorithm = {algorithm}\ndevice = {args.device}\n'
                          f'log_file = "{(work / "screen.log").as_posix()}"\n'
                          f'tmp_dir = "{work.as_posix()}"\n'
                          f'finished = "{(work / "finished.txt").as_posix()}"\n'
                          'exp_cache = "off"\nckpt_seconds = 0.01\n' + extra, encoding="utf-8")
        return work, [str(exe), "--config", str(config)]

    def run(work, cmd, label="run", text=None, ok=True):
        process = subprocess.run(cmd, input=text, text=True, encoding="utf-8", errors="replace",
                                 cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=120)
        (work / (label + ".log")).write_text(process.stdout, encoding="utf-8")
        assert (process.returncode == 0) == ok, (work, label, process.stdout[-2000:])
        return process.stdout

    def direct(work, cmd, p, b1, sigma, curves, *flags):
        return cmd + ["--ckpt", "0", "-save", str(work / "out.save"), "-sigma", str(sigma),
                      "-gpucurves", str(curves), *flags, str(b1), str(b1)], f"2^{p}-1\n"

    work, cmd = fixture("graph", '"NtT"', "ntt_tile = 8\nntt_outer_max = 2\n")
    call, data = direct(work, cmd, 4003, 128, 4294967311, 4, "--lcm", "12")
    log = run(work, call, text=data)
    mirror = (work / "screen.log").read_text(encoding="utf-8")
    assert "length=512" in log and "forward_layers=1+8" in log and "inverse_layers=8+1" in log
    assert "stage1_ntt_plan:" in mirror and "stage1_ntt_timing:" in mirror
    n = (1 << 4003) - 1
    assert saved_points(work / "out.save", 4003, 128) == {
        s: stage1(s, 128, n, torsion=12)["x"] for s in range(4294967311, 4294967315)}
    print("PASS mixed-case algorithm, explicit plan, Graph, log mirror and 64-bit sigma", flush=True)

    work, cmd = fixture("direct", "1", "ntt_window = 0\nntt_bpw = 27\nntt_length = 1024\n")
    call, data = direct(work, cmd, 4003, 32, 26, 2, "--gpu-algorithm", "2")
    log = run(work, call, text=data)
    assert "length=1024 bpw=27" in log and "graph_bits=0" in log
    expected = {s: stage1(s, 32, n)["x"] for s in (26, 27)}
    assert saved_points(work / "out.save", 4003, 32) == expected
    cgbn, cgbn_cmd = fixture("cgbn", '"CgBn"')
    call, data = direct(cgbn, cgbn_cmd, 4003, 32, 26, 2)
    assert "stage1_ntt_plan:" not in run(cgbn, call, text=data)
    assert saved_points(cgbn / "out.save", 4003, 32) == expected
    print("PASS numeric CLI override, direct launch, custom packing, CGBN comparison", flush=True)

    work, cmd = fixture("weighted_direct", extra='ntt_convolution = "WeIgHtEd"\nntt_window = 0\n')
    call, data = direct(work, cmd, 4003, 32, 26, 2)
    log = run(work, call, text=data)
    assert "length=256 bpw=16" in log and "convolution=weighted" in log and "graph_bits=0" in log
    assert saved_points(work / "out.save", 4003, 32) == expected
    print("PASS weighted direct launch against independent Python", flush=True)

    for mode, length in (("linear", 8), ("weighted", 4)):
        work, cmd = fixture(f"factors_{mode}", extra=f'ntt_convolution = "{mode}"\n')
        call, data = direct(work, cmd, 65, 40, 30, 2)
        log = run(work, call, text=data)
        assert f"p=65 curves=1 length={length}" in log and f"convolution={mode}" in log
        assert re.findall(r"factor\[\d+\]=(\d+) curve=\d+ sigma=(\d+)", log) == [
            ("253921", "30"), ("31", "31")]
        assert saved_points(work / "out.save", 65, 40) == {30: 253921, 31: 31}
        submission = [json.loads(row) for row in (work / "results.json.txt").read_text().splitlines()]
        assert len(submission) == 2 and all(row["status"] == "F" and row["curves"] == 1 and
                                            "checksum" in row for row in submission)
    print("PASS endpoint and construction factors, submission records", flush=True)

    for name, extra, data, curves, reason in (
        ("cofactor", "", "2^65-3\n", 1, "full 2^p-1"),
        ("too_many_curves", "", "2^127-1\n", 33, "at most 32 curves"),
        ("undersized_transform", "ntt_length = 8\n", "2^4003-1\n", 1, "covering 2*digits+1"),
        ("unsafe_digits", "ntt_bpw = 31\n", "2^4003-1\n", 1, "coefficient bound"),
        ("unsafe_weighted", 'ntt_convolution = "weighted"\nntt_length = 2\n',
         "2^4003-1\n", 1, "unsafe weighted NTT length"),
        ("too_many_passes", "ntt_length = 65536\nntt_tile = 1\nntt_outer_max = 1\n",
         "2^4003-1\n", 1, "more than eight outer passes"),
    ):
        work, cmd = fixture(name, extra=extra)
        call, _ = direct(work, cmd, 65, 20, 26, curves)
        assert reason in run(work, call, text=data, ok=False)
        assert not (work / "out.save").exists()
    print("PASS unsupported route and unsafe-plan rejection", flush=True)

    p, b1, sigma = 4423, 32, 26
    n = (1 << p) - 1
    expected = {seed: stage1(seed, b1, n)["x"] for seed in range(sigma, sigma + 32)}
    for curves in (16, 32):
        for mode in ("linear", "weighted"):
            work, cmd = fixture(f"batch_{curves}_{mode}", extra=f'ntt_convolution = "{mode}"\n')
            call, data = direct(work, cmd, p, b1, sigma, curves)
            log = run(work, call, text=data)
            assert f"curves={curves}" in log and f"convolution={mode}" in log
            owned, observed, budget = map(int, re.search(
                r"stage1_ntt_memory: owned_bytes=(\d+) observed_bytes=(\d+) budget_bytes=(\d+)", log).groups())
            assert 0 < owned <= budget and 0 <= observed <= budget
            assert saved_points(work / "out.save", p, b1) == {
                seed: expected[seed] for seed in range(sigma, sigma + curves)}
            print(f"PASS {mode} {curves} curves against Python, owned={owned}, observed={observed}", flush=True)

    for curves in (16, 32):
        work, cmd = fixture(f"resume_{curves}", extra="auto_stage2 = 1\nntt_tile = 8\n")
        p, b1, sigma = 9689, 4096, 26
        save = work / f"m0_m{p}_{b1}.save"
        row = f'ECMSTAGE1=1,2,{p},-1,"{save.as_posix()}",{10*b1},0,{curves}\n'
        todo = work / "worktodo.txt"
        todo.write_text(row, encoding="ascii")
        config = work / "ecm.toml"
        config.write_text(config.read_text() + f"sigma = {sigma}\n", encoding="utf-8")
        checkpoint = work / f".m0_s1_m{p}_{b1}.ckpt"

        class Security(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("descriptor", wintypes.LPVOID),
                        ("inherit", wintypes.BOOL)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateEventW.argtypes = [ctypes.POINTER(Security), wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.CreateEventW.restype = wintypes.HANDLE
        kernel.SetEvent.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        sa = Security(ctypes.sizeof(Security), None, True)
        event = kernel.CreateEventW(ctypes.byref(sa), True, False, None)
        assert event
        with (work / "stop.log").open("wb") as log:
            process = subprocess.Popen(cmd + ["--stage1-stop-handle", str(event)], cwd=work,
                                       stdout=log, stderr=log, close_fds=False,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and process.poll() is None and not checkpoint.exists():
                    time.sleep(0.01)
                assert process.poll() is None and checkpoint.is_file()
                assert kernel.SetEvent(event) and process.wait(timeout=30) != 0
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                kernel.CloseHandle(event)
        assert todo.read_text() == row and (not save.exists() or not save.stat().st_size)
        identity, lanes = batch(checkpoint)
        assert identity.endswith("|ntt-v1") and [lane[0] for lane in lanes] == list(range(sigma, sigma + curves))
        magic, count, cursor, mask = struct.unpack_from("<8sIII", lanes[0][1])
        assert magic == b"ECMNTTS1" and 0 < cursor < count and mask == (1 << curves) - 1
        assert not lanes[1][1]
        config.write_text(config.read_text().replace("ntt_tile = 8", "ntt_tile = 9")
                          .replace('ntt_convolution = "linear"', 'ntt_convolution = "weighted"')
                          + "ntt_window = 32\n", encoding="utf-8")
        log = run(work, cmd, "resume")
        assert f"resume={cursor}" in log and "graph_bits=32" in log and "convolution=weighted" in log
        assert "ECMSTAGE2=" in todo.read_text() and not checkpoint.exists()
        ref, ref_cmd = fixture(f"uninterrupted_{curves}")
        call, data = direct(ref, ref_cmd, p, b1, sigma, curves)
        run(ref, call, text=data)
        assert saved_points(save, p, b1) == saved_points(ref / "out.save", p, b1)
        print(f"PASS real stop and resume with {curves} curves at bit {cursor}/{count}, linear to weighted, Stage2 handoff", flush=True)


if __name__ == "__main__":
    main()
