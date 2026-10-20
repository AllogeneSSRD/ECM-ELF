"""Exercise real Edwards GPU progress and unchanged save points in isolated folders."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

PROGRESS = re.compile(
    r"GPU: \[[=> ]{40}\] ([\d.]+)%  (\d+), \+(\d+) (bits|columns) "
    r"\(~([\d.]+) s/curve\)  elapsed ([\d.]+)s  remaining ([\d.]+)s"
)


def validate(text, mode):
    rows = [m.groups() for m in PROGRESS.finditer(text)]
    assert rows, "no live GPU progress; Edwards still uses the old stage1: bar"
    assert "stage1: [" not in text, "old progress renderer still active"
    first = text.index("GPU: [")
    assert text.index("GPU Edwards initialization:") < first
    assert text.index("GPU Edwards dictionary:") < first
    counts = [int(row[1]) for row in rows]
    assert counts[0] > 0 and all(b > a for a, b in zip(counts, counts[1:])), counts
    assert all(0 < int(row[2]) <= 4096 for row in rows), rows
    assert all(row[3] == ("columns" if mode == "naf" else "bits") for row in rows)
    assert all(float(row[4]) > 0 for row in rows), rows
    assert rows[-1][0] == "100.0" and float(rows[-1][6]) == 0, rows[-1]
    return dict(samples=len(rows), first=rows[0], last=rows[-1])


def resume_check(exe, out, device, exponent, mode, env, reference):
    import ctypes
    from ctypes import wintypes
    from stage1_checkpoint_decode import batch
    class Attributes(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("descriptor", wintypes.LPVOID), ("inherit", wintypes.BOOL)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateEventW.argtypes = [ctypes.POINTER(Attributes), wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateEventW.restype = wintypes.HANDLE
    kernel.SetEvent.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    work = out / (mode + "_resume")
    work.mkdir()
    queue, ini = work / "worktodo.txt", work / "ecm.toml"
    task = f"ECM2=1,2,{exponent},-1,100000,0,8,20260922\n"
    queue.write_text(task, encoding="ascii")
    ini.write_text("[shared]\nworktodo = " + json.dumps(str(queue)) +
                   '\nuser = ""\ncomputer = ""\n[stage1]\ncurve = "ed"\ntarget = "gpu"\n' +
                   f"device = {device}\nckpt_seconds = 0\n" +
                   "finished = " + json.dumps(str(work / 'finished.txt')) +
                   "\ntmp_dir = " + json.dumps(str(work)) + '\nlog_file = ""\n', encoding="utf-8")
    env = env | {"ECM_EDWARDS_STAGE1_ALGO": mode}
    event = kernel.CreateEventW(ctypes.byref(Attributes(ctypes.sizeof(Attributes), None, True)), True, False, None)
    assert event
    log = work / "stopped.log"
    with log.open("wb") as stream:
        child = subprocess.Popen([str(exe), "--config", str(ini), "--stage1-stop-handle", str(event)],
                                 cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 close_fds=False, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic() + 120
            ready = False
            while time.monotonic() < deadline:
                text = log.read_text(encoding="utf-8", errors="replace")
                if len(list(PROGRESS.finditer(text))) >= 2:
                    assert kernel.SetEvent(event)
                    ready = True
                    break
                assert child.poll() is None, text[-4000:]
                time.sleep(.01)
            assert ready, "progress did not reach the stop boundary"
            assert child.wait(timeout=120) != 0, "stopped task reported completion"
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            kernel.CloseHandle(event)
    assert queue.read_text() == task and not (work / "finished.txt").exists()
    ckpt = next(path for path in work.glob(".e_s1_*.ckpt") if ".bak" not in path.name)
    raw = ckpt.read_bytes()
    (work / "frozen_stopped.ckpt").write_bytes(raw)
    _, rows = batch(ckpt)
    assert rows[0][1].startswith(b"ECMEDB3\0")
    before = struct.unpack_from("<I", rows[0][1], 24)[0]
    run = subprocess.run([str(exe), "--config", str(ini)], cwd=work, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", timeout=120)
    (work / "resumed.log").write_text(run.stdout, encoding="utf-8")
    assert run.returncode == 0, run.stdout[-4000:]
    stats = validate(run.stdout, mode)
    assert "GPU Edwards resume:" in run.stdout
    assert int(stats["first"][1]) == before + int(stats["first"][2]), "restored work counted as new work"
    assert float(stats["first"][5]) < 5, "preparation contaminated the live loop clock"
    assert not queue.read_text().strip()
    point = next(work.glob("e_*.save")).read_bytes()
    assert point == (reference / mode / "point.save").read_bytes(), "resumed endpoint changed"
    return dict(mode=mode, restored_done=before, **stats, passed=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--exponent", type=int, default=1279)
    p.add_argument("--reference", type=Path, help="baseline output folder containing mode/point.save")
    p.add_argument("--resume", action="store_true", help="also stop/resume both modes with a real Windows stop event")
    args = p.parse_args()
    exe, out = args.exe.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(exe=str(exe), sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  device=args.device, exponent=args.exponent, B1=100000, curves=8, cases=[])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ECM_", "NTT_"))}
    failures = []
    for mode in ("naf", "binary"):
        work = out / mode
        work.mkdir()
        ini = work / "ecm.toml"
        ini.write_text('[shared]\nuser = ""\ncomputer = ""\n[stage1]\nlog_file = ""\nckpt_seconds = 0\n' +
                       'exp_cache = ' + json.dumps(str(out / 'exp_cache')) +
                       '\ntmp_dir = ' + json.dumps(str(work)) + '\n', encoding="utf-8")
        cmd = [str(exe), "--config", str(ini), "--curve", "ed", "--target", "gpu",
               "-d", str(args.device), "-sigma", "20260922", "-gpucurves", "8",
               "-save", str(work / "point.save"), "100000", "100000"]
        env["ECM_EDWARDS_STAGE1_ALGO"] = mode
        run = subprocess.run(cmd, input=str((1 << args.exponent) - 1) + "\n", text=True,
                             encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=env, cwd=work, timeout=120)
        (work / "run.log").write_text(run.stdout, encoding="utf-8")
        (work / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
        case = dict(mode=mode, exit_code=run.returncode)
        try:
            assert run.returncode == 0, run.stdout[-4000:]
            point = (work / "point.save").read_bytes()
            assert len(point.splitlines()) == 8
            case["save_sha256"] = hashlib.sha256(point).hexdigest()
            if args.reference:
                assert point == (args.reference / mode / "point.save").read_bytes(), "save point changed"
            case.update(validate(run.stdout, mode))
            case["passed"] = True
        except AssertionError as error:
            case.update(passed=False, error=str(error))
            failures.append(f"{mode}: {error}")
        report["cases"].append(case)
    if args.resume and not failures:
        report["resume"] = [resume_check(exe, out, args.device, args.exponent, mode, env, args.reference or out)
                            for mode in ("naf", "binary")]
    report["passed"] = not failures
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
