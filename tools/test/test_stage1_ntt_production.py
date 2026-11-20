#!/usr/bin/env python3
"""Exercise the production CUDA Stage1 NTT route, result files and resumable batch."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

from stage1_checkpoint_decode import ntt_inflight
from test_edwards_gpu_pipeline import checksum as edwards_checksum, stage1 as edwards_stage1

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stat"))
from suyama_mont_ref import stage1

DEFERRED_PACK = 1 << 6
EDWARDS_2XY_FREQUENCY = 1 << 15
EDWARDS_2XY_INVERSE_INPUT = 1 << 16
TIGHT_CARRY_SPANS = 1 << 17
FIXED_WIDE_INVERSE = 1 << 18
FIXED_WIDE_FORWARD = 1 << 19


def ntt_executions(log):
    marker = "Stage1 execution: "
    records = []
    for line in log.splitlines():
        if marker in line:
            fields = dict(token.split("=", 1) for token in line.split(marker, 1)[1].split())
            if fields["backend"] == "ntt":
                records.append(fields)
    return records


def wait_for_ntt_snapshot(process, checkpoint, magic, cursor_offset):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and process.poll() is None:
        try:
            _, payload = ntt_inflight(checkpoint)
            if (payload.startswith(magic) and len(payload) >= cursor_offset + 4
                    and struct.unpack_from("<I", payload, cursor_offset)[0]):
                return
        except (OSError, AssertionError, ValueError, struct.error):
            pass
        time.sleep(0.01)
    raise AssertionError("did not get an in-flight NTT snapshot")


def saved_points(path, p, b1, edwards=False):
    n = (1 << p) - 1
    points = {}
    for row in path.read_text(encoding="utf-8").splitlines():
        fields = dict(v.strip().split("=", 1) for v in row.split(";") if "=" in v)
        sigma, x = int(fields["SIGMA"]), int(fields["X"], 0)
        assert int(fields["B1"]) == b1 and 0 <= x < n
        if edwards:
            assert fields["METHOD"] == "ECM-EDWARDS-AM" and "PARAM" not in fields
            assert fields["SCALAR"] == "48LCM" and fields["VERSION"] == "1"
            assert int(fields["CHECKSUM"]) == edwards_checksum(n, x, sigma, b1, fields["STATUS"])
        else:
            assert fields["METHOD"] == "ECM" and int(fields.get("PARAM", 0)) == 0
            assert int(fields["CHECKSUM"]) == (b1 * sigma * n * x) % 4294967291
        assert sigma not in points
        points[sigma] = x
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--edwards-naf-only", action="store_true")
    parser.add_argument("--edwards-dictionary", choices=("host", "gpu"))
    args = parser.parse_args()
    exe, out = args.exe.resolve(strict=True), args.output.resolve()
    out.mkdir(parents=True)
    env = dict(os.environ)
    if args.edwards_dictionary:
        env["ECM_EDWARDS_NTT_DICTIONARY"] = args.edwards_dictionary

    def fixture(name, algorithm="2", extra="", model="mont"):
        work = out / name
        work.mkdir()
        config = work / "ecm.toml"
        if "ntt_convolution" not in extra:
            extra += 'ntt_convolution = "linear"\n'
        handoff = '' if 'handoff_mode' in extra else 'handoff_mode = "off"\n'
        param = 'e' if model == 'ed' else 'm0'
        config.write_text(f'config_version = 4\n[shared]\nworktodo_file = "{(work / "worktodo.txt").as_posix()}"\n'
                          f'log_file = "{(work / "screen.log").as_posix()}"\n'
                          f'finished_file = "{(work / "finished.txt").as_posix()}"\n'
                          f'[stage1]\ntarget = "gpu"\nparam = "{param}"\n'
                          f'gpu_algorithm = {algorithm}\nstage1_gpu_device = {args.device}\n'
                          f'stage1_save_dir = "{work.as_posix()}"\n'
                          'exponent_cache_dir = "off"\ncheckpoint_interval_seconds = 0.01\n'
                          + handoff + extra, encoding="utf-8")
        return work, [str(exe), "--config", str(config)]

    def run(work, cmd, label="run", text=None, ok=True, env_override=None):
        process = subprocess.run(cmd, input=text, text=True, encoding="utf-8", errors="replace",
                                 cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 env=env if env_override is None else env_override,
                                 timeout=120)
        (work / (label + ".log")).write_text(process.stdout, encoding="utf-8")
        assert (process.returncode == 0) == ok, (work, label, process.stdout[-2000:])
        return process.stdout

    def direct(work, cmd, p, b1, sigma, curves, *flags):
        return cmd + ["--ckpt", "0", "-save", str(work / "out.save"), "-sigma", str(sigma),
                      "-gpucurves", str(curves), *flags, str(b1), str(b1)], f"2^{p}-1\n"

    if args.edwards_naf_only:
        for p, b1, sigma, curves, mode, length, window, naf in (
            (127, 20, 26, 2, "linear", 0, 0, 2),
            (127, 128, 26, 3, "linear", 0, 0, 6),
            (4253, 64, 4294967311, 4, "weighted", 0, 4, 6),
            (53281, 32, 26, 2, "weighted", 2560, 8, 8),
            (65521, 20, 26, 2, "weighted", 2560, 8, 6),
            (262139, 20, 26, 1, "weighted", 0, 4, 6),
            (65, 8, 9, 1, "linear", 0, 0, 6),
            (35, 128, 16, 1, "weighted", 0, 4, 6),
        ):
            expected = None
            for kind in (0, naf):
                required_length = 4096 if p == 65521 and not kind else length
                extra = f'ntt_convolution = "{mode}"\nntt_length = {required_length}\nntt_window_bits = {window}\nedwards_naf_window_bits = {kind}\n'
                work, cmd = fixture(f"ed_{p}_{b1}_{kind}", extra=extra, model="ed")
                call, data = direct(work, cmd, p, b1, sigma, curves)
                log = run(work, call, text=data)
                if kind and p > 65:
                    assert "stage1_ntt_edwards_naf: window=" in log
                    assert any(record["ntt_bits"] == str(p) and record["ntt_type"] == mode
                               and (not required_length or record["ntt_length"] == str(required_length))
                               and record["naf"] == str(kind) for record in ntt_executions(log)), log
                points = saved_points(work / "out.save", p, b1, True)
                if kind:
                    assert points == expected, (p, points, expected)
                else:
                    expected = points
                if p == 65:
                    assert points == {9: 31} and "factor[0]=31" in log
                elif p == 35:
                    assert points == {16: 0} and "; STATUS=IDENTITY;" in (work / "out.save").read_text()
                    assert "Edwards identity: sigma=16" in log
                    if kind:
                        main, replay = ntt_executions(log)
                        assert main["chain"] == "naf" and main["naf"] == str(kind)
                        assert replay["chain"] == "x-only" and replay["naf"] == "0", log
                        assert log.index("x-only replay sigma=16") < log.rindex("Stage1 execution:")
            if p > 65:
                n = (1 << p) - 1
                oracle = {}
                for seed in range(sigma, sigma + curves):
                    _, x, z = edwards_stage1(n, seed, b1)
                    oracle[seed] = x * pow(z, -1, n) % n
                assert expected == oracle
            print(f"PASS Edwards NTT w{naf} {mode} p={p}, {curves} curves", flush=True)
        work, cmd = fixture("edwards_xonly_reject_short", model="ed",
                            extra='edwards_naf_window_bits = 0\nntt_convolution = "weighted"\nntt_length = 2560\n')
        call, data = direct(work, cmd, 65521, 20, 26, 1)
        assert "unsafe weighted NTT length" in run(work, call, text=data, ok=False)
        assert not (work / "out.save").exists()
        # M63487 and M63493 straddle the 4U < q fusion gate at the same NTT length.
        for p, bpw, length, fused in ((65521, 26, 2560, False), (65521, 25, 4096, True),
                                     (66067, 0, 4096, True), (63487, 0, 2560, True),
                                     (63493, 0, 2560, False)):
            extra = f'edwards_naf_window_bits = 6\nntt_convolution = "weighted"\nntt_bits_per_word = {bpw}\n'
            work, cmd = fixture(f"edwards_weighted_auto_{p}_{bpw}", extra=extra, model="ed")
            call, data = direct(work, cmd, p, 20, 26, 1)
            log = run(work, call, text=data)
            [plan] = ntt_executions(log)
            assert plan["ntt_length"] == str(length) and plan["ntt_bpw"] == str((p + length - 1) // length)
            [dataflow] = re.findall(r"stage1_ntt_plan:.*\bdataflow=(\d+)", log)
            assert bool(int(dataflow) & EDWARDS_2XY_FREQUENCY) == fused, log
            assert bool(int(dataflow) & TIGHT_CARRY_SPANS) == (int(plan["ntt_u"]).bit_length() < 64), log
            [sigma] = saved_points(work / "out.save", p, 20, True)
            assert sigma == 26
        work, cmd = fixture("edwards_force_unsafe_weighted", model="ed",
                            extra='edwards_naf_window_bits = 6\nntt_convolution = "weighted"\nntt_length = 2560\n')
        call, data = direct(work, cmd, 66067, 20, 26, 1)
        assert "unsafe weighted NTT length" in run(work, call, text=data, ok=False)
        assert not (work / "out.save").exists()
        print("PASS Edwards weighted auto length, bpw cap, 2XY fusion bound, and forced unsafe length rejection",
              flush=True)

        points_by_mode = {}
        p, b1, sigma, curves = 8191, 32, 26, 30
        for length, graph_bits, expects_direct in ((1024, 16, False), (512, 16, True), (512, 0, True)):
            work, cmd = fixture(f"edwards_inverse_input_{length}_{graph_bits}", model="ed",
                                extra='edwards_naf_window_bits = 8\nntt_convolution = "weighted"\n'
                                      f'ntt_length = {length}\nntt_window_bits = {graph_bits}\n')
            call, data = direct(work, cmd, p, b1, sigma, curves)
            log = run(work, call, text=data)
            [execution] = ntt_executions(log)
            assert int(execution["ntt_graph_bits"]) == graph_bits, log
            [dataflow] = re.findall(r"stage1_ntt_plan:.*\bdataflow=(\d+)", log)
            assert int(dataflow) & EDWARDS_2XY_FREQUENCY, log
            assert bool(int(dataflow) & EDWARDS_2XY_INVERSE_INPUT) == expects_direct, log
            points = saved_points(work / "out.save", p, b1, True)
            assert set(points) == set(range(sigma, sigma + curves))
            points_by_mode[length, graph_bits] = points
        graph_points = points_by_mode[512, 16]
        assert graph_points == points_by_mode[512, 0] == points_by_mode[1024, 16]
        n = (1 << p) - 1
        _, x, z = edwards_stage1(n, sigma, b1)
        assert graph_points[sigma] == x * pow(z, -1, n) % n
        print("PASS Edwards 2XY inverse input dispatch, Graph/direct and integer endpoint", flush=True)

        for p, expects_direct in ((3433, True), (3457, False)):
            work, cmd = fixture(f"edwards_inverse_input_128_{p}", model="ed",
                                extra='edwards_naf_window_bits = 8\nntt_convolution = "weighted"\nntt_length = 128\n')
            call, data = direct(work, cmd, p, 20, 26, 1)
            log = run(work, call, text=data)
            [execution] = ntt_executions(log)
            assert execution["ntt_length"] == "128", log
            [dataflow] = re.findall(r"stage1_ntt_plan:.*\bdataflow=(\d+)", log)
            frequency_flags = EDWARDS_2XY_FREQUENCY | EDWARDS_2XY_INVERSE_INPUT
            assert int(dataflow) & frequency_flags == (frequency_flags if expects_direct else 0), log
            points = saved_points(work / "out.save", p, 20, True)
            n = (1 << p) - 1
            _, x, z = edwards_stage1(n, 26, 20)
            assert points == {26: x * pow(z, -1, n) % n}, (p, points.keys())
        print("PASS Edwards 128-point inverse input and exact-bound fallback", flush=True)

        p, b1, sigma = 19937, 20, 26
        batch_points = {}
        for length, tile, dataflow_bit in ((8192, 12, FIXED_WIDE_INVERSE), (5120, 10, FIXED_WIDE_FORWARD)):
            for curves, graph_bits in ((23, 16), (24, 16), (24, 0)):
                work, cmd = fixture(f"edwards_wide_tile_{tile}_{curves}_{graph_bits}", model="ed",
                                    extra='edwards_naf_window_bits = 2\nntt_convolution = "weighted"\n'
                                          f'ntt_length = {length}\nntt_tile = {tile}\nntt_window_bits = {graph_bits}\n')
                call, data = direct(work, cmd, p, b1, sigma, curves)
                log = run(work, call, text=data)
                [execution] = ntt_executions(log)
                assert (int(execution["ntt_length"]), int(execution["ntt_tile_bits"]),
                        int(execution["ntt_graph_bits"])) == (length, tile, graph_bits), log
                [dataflow] = re.findall(r"stage1_ntt_plan:.*\bdataflow=(\d+)", log)
                assert bool(int(dataflow) & dataflow_bit) == (curves == 24), log
                points = saved_points(work / "out.save", p, b1, True)
                assert set(points) == set(range(sigma, sigma + curves))
                expected = batch_points.setdefault((curves, graph_bits), points)
                assert points == expected
        assert batch_points[24, 16] == batch_points[24, 0]
        assert batch_points[23, 16] == {s: batch_points[24, 16][s] for s in range(sigma, sigma + 23)}
        n = (1 << p) - 1
        last_sigma = sigma + 23
        _, x, z = edwards_stage1(n, last_sigma, b1)
        assert batch_points[24, 16][last_sigma] == x * pow(z, -1, n) % n
        print("PASS Edwards fixed wide forward/inverse, 23/24-curve dispatch and Graph/direct", flush=True)

        default_points = None
        for mode, flags, length, upper in (
            ("default", "", 2560, 13695445919225353215),
            ("disabled", 'ntt_convolution = "weighted"\nntt_tight_weighted_bound = false\n',
             4096, 2 * 4096 * (2**16 - 1)**2),
        ):
            work = out / f"edwards_tight_bound_{mode}"
            work.mkdir()
            cfg = work / "ecm.toml"
            cfg.write_text('config_version = 4\n[shared]\nlog_file = "screen.log"\n'
                           '[stage1]\nparam = "e"\ntarget = "gpu"\ngpu_algorithm = "ntt"\n'
                           f'stage1_gpu_device = {args.device}\nedwards_naf_window_bits = 6\nhandoff_mode = "off"\n' +
                           flags, encoding="utf-8")
            call, data = direct(work, [str(exe), "--config", str(cfg)], 65521, 20, 26, 1)
            log = run(work, call, text=data)
            [plan] = ntt_executions(log)
            assert plan == ntt_executions((work / "screen.log").read_text(encoding="utf-8"))[0]
            assert plan["ntt_type"] == "weighted" and int(plan["ntt_length"]) == length
            assert plan["ntt_bpw_avg"] == f"{65521 / length:.6f}"
            assert plan["ntt_bpw"] == plan["ntt_bpw_max"] == str((65521 + length - 1) // length)
            assert int(plan["ntt_u"]) == upper and int(plan["ntt_q"]) == 2**64 - 2**32 + 1
            assert ("source=edwards_unsigned_bound" in log) == (mode == "default")
            assert not (work / "tune.toml").exists()
            points = saved_points(work / "out.save", 65521, 20, True)
            if default_points is None:
                default_points = points
            else:
                assert points == default_points
        print("PASS default native auto bound, opt-out, save equality, and mirrored bpw/U/q logging", flush=True)

        work, cmd = fixture("edwards_dictionary_factor", extra="edwards_naf_window_bits = 6\n", model="ed")
        call, data = direct(work, cmd, 55, 64, 5, 1)
        log = run(work, call, text=data)
        assert saved_points(work / "out.save", 55, 64, True) == {5: 881}
        assert "factor[0]=881" in log
        if args.edwards_dictionary == "gpu":
            assert "backend=gpu" in log
        work, cmd = fixture("edwards_dictionary_budget", extra="edwards_naf_window_bits = 12\nnaf_memory_mb = 1\n", model="ed")
        call, data = direct(work, cmd, 4253, 40, 26, 2)
        assert "dictionary exceeds" in run(work, call, text=data, ok=False)
        budget_save = work / "out.save"
        assert not budget_save.exists() or budget_save.stat().st_size == 0

        # At 2560 points, two M65521 w6 curves need 2,621,440 bytes for
        # spectra and 1,048,576 more for the GPU builder. A 3 MB limit
        # admits the spectra, but only with the host dictionary builder.
        automatic = {key: val for key, val in env.items() if key != "ECM_EDWARDS_NTT_DICTIONARY"}
        dictionary_config = 'edwards_naf_window_bits = 6\nntt_convolution = "weighted"\nntt_length = 2560\n'
        expected = None
        for limit, builder, workspace in ((0, "gpu", 1048576), (3, "host", 0)):
            work, cmd = fixture(f"edwards_auto_dictionary_{limit}mb", model="ed",
                                extra=dictionary_config + f"naf_memory_mb = {limit}\n")
            call, data = direct(work, cmd, 65521, 20, 26, 2)
            log = run(work, call, text=data, env_override=automatic)
            assert re.findall(r"stage1_ntt_edwards_dictionary: [^\r\n]*\bbackend=(\w+)", log) == [builder]
            assert f"dictionary_spectra_bytes=2621440 dictionary_workspace_bytes={workspace}" in log
            assert ("GPU dictionary workspace exceeds the VRAM budget, building on the host" in log) == bool(limit)
            points = saved_points(work / "out.save", 65521, 20, True)
            assert set(points) == {26, 27}
            if expected is None:
                expected = points
            else:
                assert points == expected
        work, cmd = fixture("edwards_forced_gpu_dictionary_3mb", model="ed",
                            extra=dictionary_config + "naf_memory_mb = 3\n")
        call, data = direct(work, cmd, 65521, 20, 26, 2)
        log = run(work, call, text=data, ok=False,
                  env_override=dict(automatic, ECM_EDWARDS_NTT_DICTIONARY="gpu"))
        assert "dictionary exceeds its VRAM budget" in log
        budget_save = work / "out.save"
        assert not budget_save.exists() or budget_save.stat().st_size == 0
        print("PASS Edwards NTT dictionary fallback keeps the points inside a 3 MB budget", flush=True)

        class Security(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("descriptor", wintypes.LPVOID),
                        ("inherit", wintypes.BOOL)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateEventW.argtypes = [ctypes.POINTER(Security), wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.CreateEventW.restype = wintypes.HANDLE
        kernel.SetEvent.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]

        p, b1, sigma, curves = 63487, 4096, 26, 1
        ref, ref_cmd = fixture("edwards_naf_resume_reference", model="ed")
        call, data = direct(ref, ref_cmd, p, b1, sigma, curves)
        run(ref, call, text=data)
        reference = saved_points(ref / "out.save", p, b1, True)

        for old, new in ((6, 8), (6, 0), (0, 6)):
            extra = (f"edwards_naf_window_bits = {old}\nsigma = {sigma}\nntt_window_bits = 4\n"
                     'handoff_mode = "ecm-elf"\n')
            work, cmd = fixture(f"edwards_naf_resume_{old}_to_{new}", extra=extra, model="ed")
            config = work / "ecm.toml"
            save = work / f"e_m{p}_{b1}.save"
            todo = work / "worktodo.txt"
            row = f'ECMSTAGE1=1,2,{p},-1,"{save.as_posix()}",{10*b1},0,{curves}\n'
            todo.write_text(row, encoding="ascii")
            checkpoint = work / f".e_s1_m{p}_{b1}.ckpt"
            magic, cursor_offset = (b"ECMNTTE1", 16) if old else (b"ECMNTTS1", 12)
            attributes = Security(ctypes.sizeof(Security), None, True)
            event = kernel.CreateEventW(ctypes.byref(attributes), True, False, None)
            assert event
            with (work / "stop.log").open("wb") as log:
                process = subprocess.Popen(cmd + ["--stage1-stop-handle", str(event)], cwd=work,
                                           stdout=log, stderr=log, close_fds=False, env=env,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    wait_for_ntt_snapshot(process, checkpoint, magic, cursor_offset)
                    assert kernel.SetEvent(event) and process.wait(timeout=30) != 0
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    kernel.CloseHandle(event)
            assert todo.read_text() == row and (not save.exists() or not save.stat().st_size)
            saved_sigmas, payload = ntt_inflight(checkpoint)
            assert saved_sigmas == [sigma] and payload.startswith(magic)
            count, = struct.unpack_from("<I", payload, 8)
            cursor, active = struct.unpack_from("<II", payload, cursor_offset)
            assert 0 < cursor < count and active == (1 << curves) - 1
            if old:
                assert struct.unpack_from("<I", payload, 12)[0] == old

            revised = config.read_text().replace(f"edwards_naf_window_bits = {old}", f"edwards_naf_window_bits = {new}")
            if new == 8:
                # Resume with deferred pack and frequency fusion just below 4U=q.
                revised = revised.replace('ntt_convolution = "linear"', 'ntt_convolution = "weighted"') + "ntt_length = 2560\n"
            config.write_text(revised)
            resumed = run(work, cmd, "resume")
            [record] = ntt_executions(resumed)
            assert record["ntt_resume"] == str(cursor) and record["naf"] == str(old), resumed
            assert record["chain"] == ("naf" if old else "x-only"), resumed
            assert f"stage1_ntt_naf_resume: saved_window={old} configured_window={new} cursor={cursor}" in resumed
            if new == 8:
                assert record["ntt_type"] == "weighted" and record["ntt_length"] == "2560", resumed
                [dataflow] = re.findall(r"stage1_ntt_plan:.*\bdataflow=(\d+)", resumed)
                expected_dataflow = DEFERRED_PACK | EDWARDS_2XY_FREQUENCY
                assert int(dataflow) & expected_dataflow == expected_dataflow, resumed
            assert todo.read_text().lstrip().startswith("ECMSTAGE2=") and not checkpoint.exists()
            assert saved_points(save, p, b1, True) == reference
            print(f"PASS Edwards NTT resume {old} -> {new} at {cursor}/{count} with matching save", flush=True)
        return

    work, cmd = fixture("graph", '"NtT"', "ntt_tile = 8\nntt_outer_radix_limit = 2\n")
    call, data = direct(work, cmd, 4003, 128, 4294967311, 4, "--lcm", "12")
    log = run(work, call, text=data)
    mirror = (work / "screen.log").read_text(encoding="utf-8")
    assert "length=512" in log and "forward_layers=1+8" in log and "inverse_layers=8+1" in log
    assert "convolution=linear curve=mont0 dataflow=517" in log
    assert "inverse_tile_threads=128 inverse_tile_smem_bytes=2048" in log
    assert "stage1_ntt_plan:" in mirror and "stage1_ntt_timing:" in mirror
    [execution] = ntt_executions(log)
    assert execution == ntt_executions(mirror)[0] and log.index("Stage1 execution:") < log.index("stage1_ntt_plan:")
    assert execution["ntt_bits"] == "4003" and execution["ntt_type"] == "linear"
    assert execution["ntt_length"] == "512" and execution["naf"] == "0"
    assert execution["ntt_forward_pass_bits"] == "1+8" and execution["ntt_inverse_pass_bits"] == "8+1"
    assert execution["ntt_shape"] == "0:8:2:16:1:1,8:8,1" and execution["chain"] == "ladder"
    n = (1 << 4003) - 1
    assert saved_points(work / "out.save", 4003, 128) == {
        s: stage1(s, 128, n, torsion=12)["x"] for s in range(4294967311, 4294967315)}
    print("PASS mixed-case algorithm, explicit plan, Graph, log mirror and 64-bit sigma", flush=True)

    work, cmd = fixture("direct", "1", "ntt_window_bits = 0\nntt_bits_per_word = 27\nntt_length = 1024\n")
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

    for name, rule, use_ntt in (
        ("policy_upper_bound", '"ntt[:4003], cgbn[]"', False),
        ("policy_lower_bound", '"NTT[ 4003 : ], cgbn[]"', True),
    ):
        work, cmd = fixture(name, rule)
        call, data = direct(work, cmd, 4003, 32, 26, 2)
        assert bool(ntt_executions(run(work, call, text=data))) == use_ntt
        assert saved_points(work / "out.save", 4003, 32) == expected
    work, cmd = fixture("policy_no_eligible_route", '"ntt[:4003]"')
    call, data = direct(work, cmd, 4003, 32, 26, 2)
    assert "gpu_algorithm policy has no eligible backend for 4003 bits" in run(work, call, text=data, ok=False)
    assert not (work / "out.save").exists()
    work, cmd = fixture("policy_ineligible_ntt", '"ntt[], cgbn[]"')
    call, _ = direct(work, cmd, 127, 32, 26, 2)
    cofactor = (1 << 127) - 3
    assert not ntt_executions(run(work, call, text=str(cofactor) + "\n"))
    assert (work / "out.save").exists()
    print("PASS GPU policy bounds, strict exhaustion and full-Mersenne admission", flush=True)

    work, cmd = fixture("weighted_direct", extra='ntt_convolution = "WeIgHtEd"\nntt_window_bits = 0\n')
    call, data = direct(work, cmd, 4003, 32, 26, 2)
    log = run(work, call, text=data)
    assert "length=256 bpw=16" in log and "convolution=weighted" in log and "graph_bits=0" in log
    [execution] = ntt_executions(log)
    assert execution["ntt_type"] == "weighted" and execution["ntt_length"] == "256" and execution["ntt_bpw"] == "16"
    assert "convolution=weighted curve=mont0 dataflow=543" in log  # One tile, with cached carry.
    assert saved_points(work / "out.save", 4003, 32) == expected
    print("PASS weighted direct launch against independent Python", flush=True)

    for p, length, mode, curves, threads, shared in (
        (127, 8, "weighted", 24, 32, 64),
        (521, 64, "linear", 16, 512, 512),
        (521, 64, "linear", 24, 32, 0),
    ):
        work, cmd = fixture(f"forward_tile_{p}_{curves}", extra=f'ntt_convolution = "{mode}"\nntt_length = {length}\n')
        call, data = direct(work, cmd, p, 20, 26, curves)
        log = run(work, call, text=data)
        assert re.search(rf"\btile_threads={threads} tile_smem_bytes={shared}\b", log)
        modulus = (1 << p) - 1
        assert saved_points(work / "out.save", p, 20) == {
            seed: stage1(seed, 20, modulus)["x"] for seed in range(26, 26 + curves)}
    print("PASS packed forward fusion and block sizing against independent Python", flush=True)

    p, b1, sigma = 131071, 20, 26
    wide_n = (1 << p) - 1
    for model, curves, dataflow in (("mont", 2, 1575), ("ed", 1, 1639)):
        work, cmd = fixture(f"wide_segments_{model}", extra='ntt_convolution = "weighted"\n'
                            'ntt_length = 8192\nntt_tile = 12\n', model=model)
        call, data = direct(work, cmd, p, b1, sigma, curves)
        log = run(work, call, text=data)
        curve = "edwards" if model == "ed" else "mont0"
        assert f"curve={curve} dataflow={dataflow}" in log
        if model == "ed":
            _, x, z = edwards_stage1(wide_n, sigma, b1)
            wide_expected = {sigma: x * pow(z, -1, wide_n) % wide_n}
        else:
            wide_expected = {seed: stage1(seed, b1, wide_n)["x"] for seed in range(sigma, sigma + curves)}
        assert saved_points(work / "out.save", p, b1, edwards=model == "ed") == wide_expected
    print("PASS wide segmented carry and deferred pack against independent Python", flush=True)

    for extended_p, lengths in ((900001, (131072, 40960, 65536)), (1048573, (131072, 65536))):
        extended_points = None
        for length in lengths:
            mode, naf = ("linear", 0) if length == 131072 else ("weighted", 6)
            extra = (f'ntt_convolution = "{mode}"\nntt_length = {length}\n'
                     f'edwards_naf_window_bits = {naf}\n')
            work, cmd = fixture(f"edwards_wide_{extended_p}_{length}", extra=extra, model="ed")
            call, data = direct(work, cmd, extended_p, 7, sigma, 1)
            [plan] = ntt_executions(run(work, call, text=data))
            assert (plan["ntt_bits"], plan["ntt_length"], plan["naf"]) == (str(extended_p), str(length), str(naf))
            points = saved_points(work / "out.save", extended_p, 7, edwards=True)
            assert set(points) == {sigma}
            if extended_points is None:
                extended_points = points
            else:
                assert points == extended_points
        print(f"PASS wide Edwards x-only and NAF save p={extended_p}, lengths={lengths}", flush=True)

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
        ("undersized_transform", "ntt_length = 8\n", "2^4003-1\n", 1, "covering 2*digits+1"),
        ("unsafe_digits", "ntt_bits_per_word = 31\n", "2^4003-1\n", 1, "coefficient bound"),
        ("unsafe_weighted", 'ntt_convolution = "weighted"\nntt_length = 2\n',
         "2^4003-1\n", 1, "unsafe weighted NTT length"),
        ("unsafe_radix5", 'ntt_convolution = "weighted"\nntt_length = 2560\n',
         "2^64001-1\n", 1, "unsafe weighted NTT length"),
        ("radix5_over_limit", 'ntt_convolution = "weighted"\nntt_length = 81920\n',
         "2^4003-1\n", 1, "outside the supported domain"),
        ("exponent_over_limit", "", "2^1048577-1\n", 1, "1048576"),
        ("linear_radix5", 'ntt_length = 2560\n', "2^4003-1\n", 1, "power of two"),
        ("too_many_passes", "ntt_length = 65536\nntt_tile = 1\nntt_outer_radix_limit = 1\n",
         "2^4003-1\n", 1, "more than eight outer passes"),
    ):
        work, cmd = fixture(name, extra=extra)
        call, _ = direct(work, cmd, 65, 20, 26, curves)
        assert reason in run(work, call, text=data, ok=False)
        assert not (work / "out.save").exists()
    print("PASS unsupported route and unsafe-plan rejection", flush=True)

    for length in (5, 10, 20, 40, 80, 160, 320, 640, 1280):
        work, cmd = fixture(f"retired_radix5_{length}", extra=f'ntt_convolution = "weighted"\nntt_length = {length}\n')
        call, data = direct(work, cmd, 4003, 20, 26, 1)
        assert "outside the supported domain" in run(work, call, text=data, ok=False)

    work, cmd = fixture("mont_radix5", extra='ntt_convolution = "weighted"\nntt_length = 2560\n')
    call, data = direct(work, cmd, 4003, 32, 26, 2)
    log = run(work, call, text=data)
    assert "length=2560" in log and "forward_layers=r5+9 inverse_layers=9+r5" in log
    assert "convolution=weighted curve=mont0 dataflow=1943" in log  # PFA with fused forward and inverse tiles.
    assert saved_points(work / "out.save", 4003, 32) == expected

    for p, length, curves, flow in ((58631, 2560, 1, 6119), (106501, 5120, 1, 6119), (106501, 5120, 4, 2023)):
        work, cmd = fixture(f"mixed_segments_{p}_{curves}", extra=f'ntt_convolution = "weighted"\nntt_length = {length}\n')
        call, data = direct(work, cmd, p, 20, 26, curves)
        log = run(work, call, text=data)
        assert f"convolution=weighted curve=mont0 dataflow={flow}" in log
        modulus = (1 << p) - 1
        assert saved_points(work / "out.save", p, 20) == {
            seed: stage1(seed, 20, modulus)["x"] for seed in range(26, 26 + curves)}
    print("PASS mixed fused and early segmented carry against independent Python", flush=True)

    for mode, p, length, threads in (
        ("linear", 61, 8, 8),
        ("weighted", 61, 2, 2),
        ("weighted", 521, 32, 32),
    ):
        curves, b1, sigma = 65, 20, 26
        work, cmd = fixture(f"short_inverse_{mode}_{p}",
                            extra=f'ntt_convolution = "{mode}"\nntt_length = {length}\n', model="ed")
        call, data = direct(work, cmd, p, b1, sigma, curves)
        log = run(work, call, text=data)
        assert f"inverse_tile_threads={threads} inverse_tile_smem_bytes=0" in log
        assert int(re.search(r"\bdataflow=(\d+)", log)[1]) & 8192
        n = (1 << p) - 1
        expected = {}
        for seed in range(sigma, sigma + curves):
            _, x, z = edwards_stage1(n, seed, b1)
            expected[seed] = x * pow(z, -1, n) % n
        assert saved_points(work / "out.save", p, b1, edwards=True) == expected
    print("PASS short inverse tiles with a partial batch at the end of the grid", flush=True)

    p, b1, sigma, curves = 4423, 32, 26, 128
    n = (1 << p) - 1
    expected = {}
    for seed in range(sigma, sigma + curves):
        _, x, z = edwards_stage1(n, seed, b1)
        expected[seed] = x * pow(z, -1, n) % n
    for mode in ("linear", "weighted", "radix5", "auto", "cgbn"):
        if mode == "radix5":
            extra = 'ntt_convolution = "weighted"\nntt_length = 2560\n'
        else:
            extra = f'ntt_convolution = "{mode}"\n' if mode != "cgbn" else ""
        work, cmd = fixture(f"edwards_{mode}", "1" if mode == "cgbn" else "2",
                            extra=extra, model="ed")
        call, data = direct(work, cmd, p, b1, sigma, curves)
        log = run(work, call, text=data)
        if mode == "auto":
            assert "stage1_ntt_selection:" in log and "curve=edwards" in log
        elif mode == "radix5":
            assert "length=2560" in log and "forward_layers=r5+9 inverse_layers=9+r5" in log
        elif mode != "cgbn":
            assert f"convolution={mode} curve=edwards" in log
        assert saved_points(work / "out.save", p, b1, edwards=True) == expected
    print("PASS Edwards linear, weighted, radix-5, auto and CGBN agree with independent 48LCM reference", flush=True)

    work, cmd = fixture("edwards_factor", model="ed")
    call, data = direct(work, cmd, 65, 8, 9, 1)
    log = run(work, call, text=data)
    assert "factor[0]=31" in log and saved_points(work / "out.save", 65, 8, edwards=True) == {9: 31}
    submission = [json.loads(row) for row in (work / "results.json.txt").read_text().splitlines()]
    assert any(row["status"] == "F" and row["curves"] == 1 and "Edwards" in row for row in submission)

    work, cmd = fixture("edwards_identity", model="ed")
    call, data = direct(work, cmd, 35, 128, 16, 1)
    assert "Edwards identity: sigma=16" in run(work, call, text=data)
    assert saved_points(work / "out.save", 35, 128, edwards=True) == {16: 0}
    assert "; STATUS=IDENTITY;" in (work / "out.save").read_text()
    print("PASS Edwards factor publication and identity without a Stage2 point", flush=True)

    p, b1, sigma = 4423, 32, 26
    n = (1 << p) - 1
    expected = {seed: stage1(seed, b1, n)["x"] for seed in range(sigma, sigma + 128)}
    for curves in (16, 32, 128):
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

    for model, curves in (("mont", 16), ("mont", 32), ("ed", 16), ("mont", 65), ("ed", 128)):
        work, cmd = fixture(f"resume_{model}_{curves}", extra="handoff_mode = \"ecm-elf\"\nntt_tile = 8\n", model=model)
        p, b1, sigma = 9689, 4096, 26
        prefix = "e" if model == "ed" else "m0"
        save = work / f"{prefix}_m{p}_{b1}.save"
        row = f'ECMSTAGE1=1,2,{p},-1,"{save.as_posix()}",{10*b1},0,{curves}\n'
        todo = work / "worktodo.txt"
        todo.write_text(row, encoding="ascii")
        config = work / "ecm.toml"
        config.write_text(config.read_text() + f"sigma = {sigma}\n", encoding="utf-8")
        checkpoint = work / f".{prefix}_s1_m{p}_{b1}.ckpt"

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
                wait_for_ntt_snapshot(process, checkpoint, b"ECMNTTS", 12)
                assert kernel.SetEvent(event) and process.wait(timeout=30) != 0
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                kernel.CloseHandle(event)
        assert todo.read_text() == row and (not save.exists() or not save.stat().st_size)
        saved_sigmas, payload = ntt_inflight(checkpoint)
        width = len(saved_sigmas)
        assert 0 < width <= curves and saved_sigmas == list(range(sigma, sigma + width))
        magic, count, cursor = struct.unpack_from("<8sII", payload)
        mask = int.from_bytes(payload[16:16 + 4 * ((width + 31) // 32)], "little")
        assert magic == (b"ECMNTTS1" if width <= 32 else b"ECMNTTS2")
        assert 0 < cursor < count and mask == (1 << width) - 1
        config.write_text(config.read_text().replace("ntt_tile = 8", "ntt_tile = 9")
                          .replace('ntt_convolution = "linear"', 'ntt_convolution = "weighted"')
                          + "ntt_window_bits = 32\n" + ("ntt_length = 2560\n" if model == "ed" else ""), encoding="utf-8")
        log = run(work, cmd, "resume")
        assert f"resume={cursor}" in log and "graph_bits=32" in log and "convolution=weighted" in log
        if model == "ed":
            assert "length=2560" in log and "forward_layers=r5+9" in log
        assert "ECMSTAGE2=" in todo.read_text() and not checkpoint.exists()
        ref, ref_cmd = fixture(f"uninterrupted_{model}_{curves}", model=model)
        call, data = direct(ref, ref_cmd, p, b1, sigma, curves)
        run(ref, call, text=data)
        assert saved_points(save, p, b1, model == "ed") == saved_points(ref / "out.save", p, b1, model == "ed")
        print(f"PASS real stop and resume with {curves} curves at bit {cursor}/{count}, linear to weighted, Stage2 handoff", flush=True)


if __name__ == "__main__":
    main()
