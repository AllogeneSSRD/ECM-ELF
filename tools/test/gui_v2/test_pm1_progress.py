"""Check a real CUDA P-1 assignment through the native worker and log views."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from win32_probe import close, window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "worker", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True)
    cfg, queue, screen, audit = (out / name for name in ("ecm.toml", "worktodo.txt", "worker.log", "gui.log"))
    queue.write_text("[Worker #1]\nPminus1=1,2,400381,-1,200000,0\n")
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
            if not seen and ("P-1 M400381  B1=200000  B2=2000000" in trace and
                             "Stage1 CUDA/NTT | length=" in trace and
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
        assert "P-1 Stage1 handoff M400381 B1=200000 save only" in log
        samples = re.findall(r"pm1_stage1_progress: iter=(\d+)/(\d+) [^\r\n]+", log)
        assert len(samples) == 1 and samples[0][0] == samples[0][1], samples
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
