"""Exercise Stage2 checkpoint publication, including resuming with a new output path."""
import argparse
import json
from pathlib import Path
import subprocess

from test_result_output_consistency import crc_valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("exe", "save", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument("--b2", type=int, default=10000)
    parser.add_argument("--d", type=int)
    parser.add_argument("--saved-factor", action="store_true",
                        help="use a one-curve save whose X is a verified proper factor")
    parser.add_argument("--diagnostics", action="store_true",
                        help="write diagnostic JSONL on the first run and remove it before recovery")
    args = parser.parse_args()
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    point_count = 1 if args.saved_factor else 4
    points = args.save.read_bytes().splitlines()[:point_count]
    assert len(points) == point_count
    (work / "point.save").write_bytes(b"\n".join(points) + b"\n")
    config = work / "ecm.toml"
    config_text = ('[shared]\nb2_ratio=0\nuser="original"\n[stage2]\nstage2_save_dir="."\n'
                   'stage2_finished="finished.txt"\nstage2_progress_dir="progress"\n'
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
           "--worktodo", str(queue), "--once", "--device", str(args.device),
           "--short-calibration", "0",
           "--tune-runtime", "0", "--auto-carrier", "0", "--log-level", "1",
           "--parallel", str(args.parallel)]
    if args.d:
        cmd.extend(["--d", str(args.d)])

    def run(name, success, extra=()):
        process = subprocess.run(cmd + list(extra), cwd=work, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, encoding="utf-8",
                                 errors="replace", timeout=180)
        (work / f"{name}.log").write_text(process.stdout, encoding="utf-8")
        assert (process.returncode == 0) == success, process.stdout
        return process.stdout

    # Block atomic replacement with a directory at the temp path. This occurs
    # after final NF bytes are saved in the checkpoint and before queue edit.
    blocked = submission.with_suffix(".txt.new")
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
    assert checkpoint.read_text(encoding="utf-8").startswith(f"5 {point_count} {point_count} ")
    assert not submission.exists() and not (work / "stage2_results.jsonl").exists()
    assert not (work / "data").exists()
    if args.diagnostics:
        rows = [json.loads(line) for line in diagnostics.read_bytes().splitlines()]
        assert len(rows) == point_count and all(map(crc_valid, rows))
        diagnostics.unlink()
    blocked.rmdir()
    restarted_config = config_text.replace('user="original"', 'user="after-restart"')
    config.write_text(restarted_config, encoding="utf-8")
    # Simulate another failure after successful publication but before finished
    # bookkeeping. Replay must leave the already published line byte-identical.
    finished = work / "finished.txt"
    finished.mkdir()
    log = run("finished-blocked", False)
    assert "curve_start:" not in log and checkpoint.exists()
    published = submission.read_bytes()
    row, = map(json.loads, published.splitlines())
    assert row["status"] == ("F" if args.saved_factor else "NF")
    assert row["user"] == "original", row
    assert row["curves"] == point_count and crc_valid(row)

    new_submission = work / "new-results.json.txt"
    config.write_text('config_version=3\n[shared]\nuser="after-restart"\nfinished_file="finished.txt"\n'
                      '[stage2]\nb2_ratio=0\nsave_dir="."\ncheckpoint_dir="progress"\n'
                      'submission_file="new-results.json.txt"\n', encoding="utf-8")
    log = run("new-submission-finished-blocked", False)
    assert "curve_start:" not in log and checkpoint.exists()
    assert new_submission.read_bytes() == published
    assert submission.read_bytes() == published

    cli_submission = work / "cli-results.json.txt"
    cli_override = ("--submission", str(cli_submission))
    log = run("cli-submission-finished-blocked", False, cli_override)
    assert "curve_start:" not in log and checkpoint.exists()
    assert cli_submission.read_bytes() == published

    finished.rmdir()
    log = run("resume", True, cli_override)
    assert "curve_start:" not in log and not checkpoint.exists()
    assert new_submission.read_bytes() == published and submission.read_bytes() == published
    assert cli_submission.read_bytes() == published
    assert not queue.read_text().strip()
    assert not (work / "data").exists() and not (work / "stage2_results.jsonl").exists()
    print(f"Stage2 checkpoint publication passed (parallel={args.parallel}, device={args.device})")


if __name__ == "__main__":
    main()
