"""Exercise resumed NTT Stage1 task/batch progress in the native GUI."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import time

from win32_probe import capture, close, user, window


def main():
    parser = argparse.ArgumentParser()
    for name in ("exe", "worker", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    user.SetThreadDpiAwarenessContext.argtypes=[ctypes.c_void_p]
    user.SetThreadDpiAwarenessContext.restype=ctypes.c_void_p
    assert user.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
    out = args.out.resolve()
    out.mkdir(parents=True)
    queue = out / "worktodo.txt"
    queue.write_text('[Worker #1]\nECM2=1,2,15919,-1,44000000,4400000000,1000,"1782929"\n')
    config = out / "ecm.toml"
    config.write_text(
        '[shared]\nworktodo=' + json.dumps(str(queue)) + '\n[gui]\nNumWorkers=1\n'
        'exit_confirm="stop"\ngraceful_stop_ms=1500\nexe=' + json.dumps(str(args.worker.resolve())) +
        '\n[workers.1]\nautostart=1\nextra_args="--scenario v2-ntt-progress"\n', encoding="utf-8")
    state = out / "ui.toml"
    state.write_text('[ui]\npage=0\nlogs=true\ninspector=false\nlanguage="zh"\n')
    log = out / "ui.log"
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    process = subprocess.Popen([str(args.exe.resolve()), "--config", str(config),
        "--ui-state", str(state), "--diagnostics", str(log), "--audit-app",
        "--no-nvml", "--no-pet", "--language", "zh"], cwd=out, startupinfo=startup)
    hwnd = None
    try:
        hwnd = window(process.pid, "ecm_gui_v2")
        user.SetWindowPos(hwnd, None, 40, 40, 1500, 900, 0x14)
        user.ShowWindow(hwnd, 4)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            text = log.read_text(encoding="utf-8")
            if "batch=curve 129–160  50.0%" in text:
                break
            assert process.poll() is None, text[-2000:]
            time.sleep(.1)
        else:
            raise AssertionError(log.read_text(encoding="utf-8")[-3000:])
        assert "pct=14.4 s_per_curve=1.000 eta=856.0 done=128 total=1000" in text
        rows = [json.loads(line.split(" text=", 1)[1]) for line in text.splitlines()
                if line.startswith("worker_summary: index=1 ")]
        lines = rows[-1].splitlines()
        assert len(lines) == 2 and "M15919 / 已知因子  curves = 1000" in lines[0]
        assert "Stage2 →" in lines[0] and "CUDA/NTT" in lines[1], lines
        time.sleep(.3)
        capture(hwnd, out / "ntt-wide.png")
        user.SetWindowPos(hwnd, None, 40, 40, 780, 900, 0x14)
        time.sleep(.7)
        capture(hwnd, out / "ntt-narrow.png")
        (out / "summary.json").write_text(json.dumps({"task_pct": 14.4,
            "batch_pct": 50, "range": [129, 160], "s_per_curve": 1,
            "eta_seconds": 856, "caption_lines": lines}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("NTT total/batch progress, caption, s/curve and ETA: PASS")
    finally:
        if hwnd:
            close(process, hwnd)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
