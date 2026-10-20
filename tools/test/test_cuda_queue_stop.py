"""Check that a Stage1 stop leaves the current and later worktodo lines intact.

Usage: python tools/test/test_cuda_queue_stop.py PATH_TO_ECM_CUDA PATH_TO_OUTPUT_DIR [--ctrl-break]
"""

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


def main() -> None:
    exe = Path(sys.argv[1]).resolve()
    work = Path(sys.argv[2]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    first = "ECM2=1,2,4999,-1,1000000,0,8,100000"
    second = "ECM2=1,2,4999,-1,10000,0,8,200000"
    todo = work / "worktodo.txt"
    todo.write_text(first + "\n" + second + "\n", encoding="ascii")
    finished = work / "finished.txt"
    ini = work / "ecm.toml"
    ini.write_text(
        f"curve = mont\ntarget = gpu\ndevice = 0\ngpu_param = 0\nckpt_seconds = 0\n"
        f"worktodo = {todo}\nfinished = {finished}\ntmp_dir = {work}\nlog_file =\n",
        encoding="ascii",
    )

    ctrl_break = len(sys.argv) > 3 and sys.argv[3] == "--ctrl-break"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateEventW.argtypes = [ctypes.POINTER(SecurityAttributes), wintypes.BOOL,
                                    wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateEventW.restype = wintypes.HANDLE
    kernel.SetEvent.argtypes = [wintypes.HANDLE]
    kernel.SetEvent.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    event = None
    if not ctrl_break:
        sa = SecurityAttributes(ctypes.sizeof(SecurityAttributes), None, True)
        event = kernel.CreateEventW(ctypes.byref(sa), True, False, None)
        if not event:
            raise OSError(ctypes.get_last_error(), "CreateEventW")
    try:
        with (work / "run.log").open("wb") as log:
            args = [str(exe), "--config", str(ini)]
            if event is not None:
                args += ["--stage1-stop-handle", str(event)]
            proc = subprocess.Popen(
                args,
                cwd=work, stdout=log, stderr=subprocess.STDOUT,
                close_fds=ctrl_break,
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if ctrl_break
                               else subprocess.CREATE_NO_WINDOW),
            )
            try:
                time.sleep(3)
                if proc.poll() is not None:
                    raise AssertionError(f"queue exited before stop request: {proc.returncode}")
                if ctrl_break:
                    os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                elif not kernel.SetEvent(event):
                    raise OSError(ctypes.get_last_error(), "SetEvent")
                code = proc.wait(timeout=90)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
    finally:
        if event is not None:
            kernel.CloseHandle(event)

    output = (work / "run.log").read_text(errors="replace")
    remaining = todo.read_text(encoding="ascii").splitlines()
    assert code != 0, f"interrupted queue returned success: {code}"
    assert remaining == [first, second], f"worktodo changed after stop: {remaining}"
    assert output.count("START:") == 1, "queue started another task after stop"
    assert not finished.exists() or not finished.read_text().strip(), "interrupted task finished"
    assert list(work.glob(".ecm_ckpt_*.dat")), "interrupted task has no checkpoint"
    print(f"CUDA queue {('Ctrl+Break' if ctrl_break else 'GUI event')} stop preserved "
          "both worktodo lines and checkpoint")


if __name__ == "__main__":
    main()
