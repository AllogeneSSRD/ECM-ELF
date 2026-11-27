"""Check a real CUDA P-1 assignment through the native worker and log views."""
import argparse
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import zlib

from win32_probe import close, window

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_pm1_stage1_extend import lcm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "worker", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--extend", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True)
    cfg, queue, screen, audit = (out / name for name in ("ecm.toml", "worktodo.txt", "worker.log", "gui.log"))
    p, old, target = 400381, 14000, 28000 if args.extend else 25000
    queue.write_text(f"[Worker #1]\nPminus1=1,2,{p},-1,{target},0\n")
    if args.extend:
        exponent = 2 * p * lcm(old)
        bits, cursor = exponent.bit_length(), 1001
        checkpoint = out / "pm1" / f"M{p}-B1-{old}" / f"p1-{p}"
        checkpoint.mkdir(parents=True)
        # Seed the cursor for UI coverage. Separate worker tests verify arithmetic.
        limbs = (3).to_bytes(4 * math.ceil(p / 32), "little")
        (checkpoint / f"{p}-{cursor:09d}.p1").write_bytes(
            f"OWL P1 2 N=1*2^{p}-1 B1={old} B2=0 k={cursor} bits={bits} err=0 time=0\n".encode()
            + struct.pack("<I", zlib.crc32(limbs)) + limbs)
        expected = bits - cursor + (lcm(target) // lcm(old)).bit_length() - 1
    quote = lambda path: json.dumps(str(path))
    cfg.write_text(f'config_version=4\n[shared]\nworktodo_file={quote(queue)}\nlog_file={quote(screen)}\n'
                   f'finished_file={quote(out / "finished.txt")}\ntune_file={quote(out / "tune.toml")}\n[stage2]\nb2_ratio=10\n'
                   '[stage1]\ntarget="gpu"\nhandoff_mode_pm1="off"\nprogress_log_seconds=0\n'
                   f'[gui]\nstage1_exe={quote(args.worker.resolve())}\npage=0\nlogs=true\n'
                   'display_language="en"\nexit_confirm="stop"\n[worker.1]\nautostart=true\n')
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    proc = subprocess.Popen([str(args.exe.resolve()), "--config", str(cfg), "--diagnostics", str(audit),
                             "--audit-app", "--no-nvml", "--no-pet"], cwd=out, startupinfo=startup)
    hwnd = None
    seen = False
    try:
        hwnd = window(proc.pid, "ecm_gui_v2")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            assert proc.poll() is None
            trace = audit.read_text(encoding="utf-8", errors="replace")
            assert "INPUT[config]" not in trace, trace[:2000]
            task = f"P-1 M{p}  B1={old} → {target}" if args.extend else f"P-1 M{p}  B1={target}"
            if not seen and (f"{task}  B2={10 * target}" in trace and
                             re.search(r"Stage1 CUDA/NTT \| iter=\d+ \| length=\d+ \| bpw=\d+ \| shape=\S+", trace) and
                             re.search(r"batch=Pminus1 1  \d+ / \d+ iter", trace) and
                             re.search(r"P-1 progress pct=[\d.]+ iter=\d+/\d+ iter_per_s=[1-9]", trace) and
                             re.search(r"log_detail: worker=1 table_eta=([1-9]\d* s) \| log_eta=\1", trace)):
                seen = True
            log = screen.read_text(encoding="utf-8", errors="replace") if screen.exists() else ""
            if "queue done, 1 task" in log:
                break
            time.sleep(.15)
        else:
            raise AssertionError(log[-2000:])
        assert seen, trace[-3000:]
        assert f"P-1 Stage1 handoff M{p} B1={target} save only" in log
        samples = re.findall(r"pm1_stage1_progress: iter=(\d+)/(\d+) [^\r\n]+", log)
        assert len(samples) == 1 and samples[0][0] == samples[0][1], samples
        if args.extend:
            assert samples == [(str(expected), str(expected))]
            assert "Finishing old B1" in trace and "Extending B1" in trace
            assert "phase=finish_old remaining_iter=" in log and "phase=extend remaining_iter=" in log
        assert "Pminus1=" not in queue.read_text()
        print("P-1 native GUI task, live ETA, structured log and file cadence: PASS")
    finally:
        if hwnd:
            close(proc, hwnd)
        elif proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()
