"""Exercise Stage2 checkpoint publication, including resuming with a new output path."""
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
    assert not (work / "data" / "results.json.txt.ecm-ledger").exists()
    if args.diagnostics:
        rows = [json.loads(line) for line in diagnostics.read_bytes().splitlines()]
        assert len(rows) == point_count and all(map(crc_valid, rows))
        diagnostics.unlink()
    blocked.rmdir()
    if not args.saved_factor:
        # Block the receipt after atomic replacement. Restart the real worker
        # with the same checkpoint and require byte-identical publication.
        run_id = shlex.split(checkpoint.read_text(encoding="utf-8").splitlines()[0])[3]
        key = ("stage2-nf|" + run_id).encode()
        fingerprint = 14695981039346656037
        for byte in key:
            fingerprint = ((fingerprint ^ byte) * 1099511628211) & ((1 << 64) - 1)
        writer = work / "data" / "results.json.txt.ecm-publish"
        blocked_receipt = writer / "receipts" / f"{fingerprint}.done.new"
        blocked_receipt.mkdir(parents=True)
        log = run("receipt-blocked", False)
        assert "cannot create temporary file" in log and "curve_start:" not in log, log
        assert (writer / "pending").is_file()
        published_before_ack = submission.read_bytes()
        blocked_receipt.rmdir()
    restarted_config = config_text.replace('user="original"', 'user="after-restart"')
    config.write_text(restarted_config, encoding="utf-8")
    # Simulate another failure after successful publication but before finished
    # bookkeeping. Replay must leave the already published line byte-identical.
    finished = work / "finished.txt"
    finished.mkdir()
    log = run("finished-blocked", False)
    assert "curve_start:" not in log and checkpoint.exists()
    published = submission.read_bytes()
    if not args.saved_factor:
        assert published == published_before_ack and not (writer / "pending").exists()
    row, = map(json.loads, published.splitlines())
    assert row["status"] == ("F" if args.saved_factor else "NF")
    assert row["user"] == "original", row
    assert row["curves"] == point_count and crc_valid(row)

    new_submission = work / "new-results.json.txt"
    config.write_text('config_version=4\n[shared]\nuser="after-restart"\nfinished_file="finished.txt"\n'
                      f'[stage2]\ndevice="{device}"\nb2_ratio=0\nstage2_save_dir="."\ncheckpoint_dir="progress"\n'
                      'stage2_submission_file="new-results.json.txt"\n', encoding="utf-8")
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
    assert not list((work / "data").glob("*.ecm-publish/pending"))
    assert not (work / "stage2_results.jsonl").exists()
    if not args.saved_factor:
        submission.rename(work / "archived-results.json.txt")
        prefix = b"manual archive marker\r\n"
        submission.write_bytes(prefix)
        queue.write_text(task, encoding="utf-8")
        run("after-manual-archive", True, ("--submission", str(submission)))
        assert (work / "archived-results.json.txt").read_bytes() == published
        current = submission.read_bytes()
        assert current.startswith(prefix)
        fresh, = map(json.loads, current[len(prefix):].splitlines())
        assert fresh["status"] == "NF" and fresh["curves"] == point_count and crc_valid(fresh)
    print(f"Stage2 checkpoint publication passed (parallel={args.parallel}, device={args.device})")


if __name__ == "__main__":
    main()
