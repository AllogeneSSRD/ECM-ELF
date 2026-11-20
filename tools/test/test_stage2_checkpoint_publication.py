"""Exercise Stage2 checkpoint publication, including archived output and interrupted acknowledgement."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from test_result_output_consistency import crc_valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("exe", "save", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument("--b2", type=int, default=10000)
    parser.add_argument("--d", type=int)
    parser.add_argument("--saved-factor", action="store_true",
                        help="use a one-curve save whose X is a verified proper factor")
    parser.add_argument("--diagnostics", action="store_true",
                        help="write diagnostic JSONL on the first run and remove it before recovery")
    args = parser.parse_args()
    if args.cpu and args.parallel != 1:
        parser.error("the CPU worker uses the serial queue path")
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    point_count = 1 if args.saved_factor else 4
    points = args.save.read_bytes().splitlines()[:point_count]
    assert len(points) == point_count
    (work / "point.save").write_bytes(b"\n".join(points) + b"\n")
    config = work / "ecm.toml"
    device = "cpu" if args.cpu else "gpu"
    config_text = ('config_version=4\n[shared]\nuser="original"\nfinished_file="finished.txt"\n'
                   f'[stage2]\ndevice="{device}"\nb2_ratio=0\nstage2_save_dir="."\ncheckpoint_dir="progress"\n'
                   'stage2_submission_file="results.json.txt"\n')
    config.write_text(config_text, encoding="utf-8")
    queue = work / "worktodo.txt"
    target = "1,2,521,-1"
    if args.saved_factor:
        fields = dict(field.strip().split(b"=", 1) for field in points[0].split(b";") if b"=" in field)
        modulus = int(fields[b"N"], 0)
        target = f"{modulus // 2},2,1,{modulus % 2}"
    task = f'ECMSTAGE2={target},"point.save",{args.b2},0,{point_count}\n'
    queue.write_text(task, encoding="utf-8")
    submission = work / "results.json.txt"
    cmd = [str(args.exe.resolve()), "--config", str(work / "ecm.toml"),
           "--worktodo", str(queue), "--once", "--log-level", "1"]
    if args.cpu:
        cmd.extend(["--backend", "plain", "--tune-runtime", "0"])
    else:
        cmd.extend(["--device", str(args.device), "--short-calibration", "0",
                    "--tune-runtime", "0", "--auto-carrier", "0", "--parallel", str(args.parallel)])
        if args.saved_factor:
            cmd.extend(["--b2-round-chain", "0"])
    if args.d:
        cmd.extend(["--d", str(args.d)])

    def run(name, success, extra=()):
        process = subprocess.run(cmd + list(extra), cwd=work, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, encoding="utf-8",
                                 errors="replace", timeout=180)
        (work / f"{name}.log").write_text(process.stdout, encoding="utf-8")
        assert (process.returncode == 0) == success, process.stdout
        return process.stdout

    # Block replacement after the task has checkpointed the current batch.
    blocked = submission.with_suffix(".txt.publish.tmp")
    blocked.mkdir()
    diagnostics = work / "curves.jsonl"
    log = run("publication-blocked", False,
              ("--results", str(diagnostics)) if args.diagnostics else ())
    if args.parallel == 2:
        assert "workers=2" in log, log
    assert "cannot create temporary file" in log, log
    if not args.saved_factor:
        assert log.count("curve_done:") == point_count, log
    assert queue.read_text(encoding="utf-8") == task
    checkpoint, = (work / "progress").glob("*.dat")
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    fields = shlex.split(lines[0])
    assert fields[:3] == ["6", str(point_count), str(point_count)]
    assert not submission.exists() and not (work / "stage2_results.jsonl").exists()
    assert not (work / "data").exists()
    if args.diagnostics and not args.saved_factor:
        rows = [json.loads(line) for line in diagnostics.read_bytes().splitlines()]
        assert len(rows) == point_count and all(map(crc_valid, rows))
        diagnostics.unlink()
    blocked.rmdir()

    # The same on-disk state arises if replacement succeeded and the process was
    # killed before acknowledging it. Materialize just that checkpointed row.
    report = shlex.split(lines[-3])[0] if args.saved_factor else fields[13]
    published = (report + "\n").encode("utf-8")
    submission.write_bytes(published)
    config.write_text(config_text.replace('user="original"', 'user="after-restart"'), encoding="utf-8")
    finished = work / "finished.txt"
    finished.mkdir()
    log = run("finished-blocked", False)
    assert "curve_start:" not in log and checkpoint.exists()
    assert submission.read_bytes() == published
    row, = map(json.loads, published.splitlines())
    assert row["status"] == ("F" if args.saved_factor else "NF")
    assert row["user"] == "original" and row["curves"] == point_count and crc_valid(row)

    override = work / "other-results.json.txt"
    log = run("different-destination", False, ("--submission", str(override)))
    assert "different submission path" in log and not override.exists() and checkpoint.exists()
    finished.rmdir()
    # The acknowledgement is already durable. Archiving the output before
    # retrying the worktodo edit cannot re-emit any report.
    archive = work / "archived-results.json.txt"
    submission.rename(archive)
    log = run("resume", True)
    assert "curve_start:" not in log and not checkpoint.exists()
    assert archive.read_bytes() == published and not submission.exists()
    assert not queue.read_text().strip()
    assert not (work / "data").exists()
    if not args.saved_factor:
        prefix = b"manual archive marker\r\n"
        submission.write_bytes(prefix)
        queue.write_text(task, encoding="utf-8")
        run("after-manual-archive", True)
        current = submission.read_bytes()
        assert current.startswith(prefix)
        fresh, = map(json.loads, current[len(prefix):].splitlines())
        assert fresh["status"] == "NF" and fresh["curves"] == point_count and crc_valid(fresh)
    print(f"Stage2 checkpoint publication passed (parallel={args.parallel}, device={args.device})")

    if args.cpu:
        # --save uses the same outcome and publication protocol as worktodo.
        direct = work / "direct.json.txt"
        direct_task = ["--save", str(work / "point.save"), "--b2", str(args.b2),
                       "--curves", str(point_count), "--submission", str(direct)]
        direct_cmd = [value for i, value in enumerate(cmd)
                      if i not in (cmd.index("--worktodo"), cmd.index("--worktodo") + 1)]
        direct_blocked = direct.with_suffix(".txt.publish.tmp")
        direct_blocked.mkdir()
        failed = subprocess.run(direct_cmd + direct_task, cwd=work, capture_output=True, timeout=180)
        assert failed.returncode and b"cannot create temporary file" in failed.stderr
        direct_blocked.rmdir()
        for _ in range(2):
            resumed = subprocess.run(direct_cmd + direct_task, cwd=work, capture_output=True, timeout=180)
            assert resumed.returncode == 0 and b"completed save-line=" not in resumed.stdout, resumed.stdout
            direct_row, = map(json.loads, direct.read_bytes().splitlines())
            assert direct_row["status"] == row["status"] and crc_valid(direct_row)
        assert not (work / "data").exists()


if __name__ == "__main__":
    main()
