"""Verify that a held Windows phase gate or workspace lease is excluded from curve time."""

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import os
import re
import subprocess
import time


def record_hash(save):
    value = 14695981039346656037
    for byte in save.read_bytes().split(b"\n", 1)[0]:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("exe", "save", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    exe, save, output = (p.resolve() for p in (args.exe, args.save, args.output))
    output.mkdir(parents=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]

    for workspace in (False, True):
        name = "workspace" if workspace else "phase"
        work = output / name
        work.mkdir()
        group = f"timing_{os.getpid()}_{name}"
        mutex = kernel.CreateMutexW(None, True, f"Local\\ECM-ELF.Stage2.Phase.{group}"
                                    + (".workspace" if workspace else ""))
        assert mutex, ctypes.get_last_error()
        report = work / "results.jsonl"
        cmd = [str(exe), "--save", str(save), "--b2", "20000", "--d", "30",
               "--submission", str(work / "results.json.txt"), "--results", str(report),
               "--device", str(args.device), "--log-level", "debug", "--tune-runtime", "0"]
        env = dict(os.environ)
        if workspace:
            cmd += ["--curve-worker", "--record-index", "1", "--record-offset", "0",
                    "--record-hash", record_hash(save), "--fixed-queue-route",
                    "--queue-phase-gate", f"0:{group}", "--queue-workspace-lease", "2"]
        else:
            cmd += ["--curves", "1", "--short-calibration", "0", "--log", str(work / "screen.log")]
            env["ECM_STAGE2_PHASE_GATE"] = f"1:{group}"
        console = work / "console.log"
        started = time.monotonic()
        try:
            with console.open("wb") as log, subprocess.Popen(
                cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT
            ) as child:
                time.sleep(5)
                assert kernel.ReleaseMutex(mutex), ctypes.get_last_error()
                assert child.wait(timeout=30) == 0, console.read_text(errors="replace")
        finally:
            kernel.CloseHandle(mutex)
        wall = time.monotonic() - started
        row, = (json.loads(line) for line in report.read_text(encoding="utf-8").splitlines())
        active = row["stage2-engine"]["seconds"]
        waiting = row["stage2-engine"]["lock_wait_seconds"]
        log = console.read_text(encoding="utf-8", errors="replace")
        assert waiting > 2 and 0 < active < wall - 2, (name, waiting, active, wall)
        assert active + waiting <= wall + .1, (name, waiting, active, wall)
        if not workspace:
            done = re.search(r"curve_done: .* wall=([\d.]+) s active=([\d.]+)", log)
            assert done and abs(float(done[2]) - active) < .01, log
            assert float(done[1]) - float(done[2]) >= waiting - .01, log
        print(f"{name}: active={active:.3f}s lock wait={waiting:.3f}s process wall={wall:.3f}s")


if __name__ == "__main__":
    main()
