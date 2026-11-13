"""Edit the live TOML after Stage2 admits a real task, then check publication."""
import argparse
import json
from pathlib import Path
import queue
import subprocess
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("exe", "save", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    points = args.save.read_bytes().splitlines()[:4]
    assert len(points) == 4, "expected a four-curve M521 save"
    (work / "point.save").write_bytes(b"\n".join(points) + b"\n")
    config = work / "ecm.toml"
    config.write_text(
        '[shared]\nuser="base"\ncomputer="base"\nb2_ratio=0\n'
        '[stage2]\nstage2_save_dir="."\nstage2_finished="finished.txt"\n'
        '[workers.2.shared]\nuser="用户"\ncomputer=""\n', encoding="utf-8"
    )
    (work / "worktodo.txt").write_text(
        '[Worker #2]\nECMSTAGE2=1,2,521,-1,"point.save",20000,0,4\n', encoding="utf-8"
    )
    cmd = [str(args.exe.resolve()), "--config", str(config), "--worker", "2",
           "--device", str(args.device), "--short-calibration", "0",
           "--tune-runtime", "0", "--auto-carrier", "0", "--log-level", "1",
           "--parallel", str(args.parallel), "--d", "30", "--results", "curves.jsonl"]
    if args.direct:
        cmd.extend(["--save", "point.save", "--b2", "20000", "--submission", "direct.txt"])
    else:
        cmd.extend(["--worktodo", "worktodo.txt", "--once"])
    seen = queue.Queue()
    with subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding="utf-8", errors="replace") as child:
        def collect():
            with (work / "run.log").open("w", encoding="utf-8") as log:
                for line in child.stdout:
                    log.write(line)
                    if line.startswith("stage2_plan:"):
                        seen.put(True)
                seen.put(False)

        reader = threading.Thread(target=collect)
        reader.start()
        try:
            assert seen.get(timeout=30), "worker stopped before task admission"
            config.write_text('[shared]\nuser="unterminated\n', encoding="utf-8")
            assert child.wait(timeout=180) == 0, work / "run.log"
        finally:
            if child.poll() is None:
                child.kill()
            reader.join(timeout=10)

    publication = work / ("direct.txt" if args.direct else "results.json_2.txt")
    rows = [json.loads(line) for line in publication.read_text(encoding="utf-8").splitlines()]
    assert rows and sum(row["curves"] for row in rows) == 4, rows
    assert all(row["user"] == "用户" and "computer" not in row for row in rows), rows
    assert not (work / "results.json.txt").exists()
    curves = [json.loads(line) for line in (work / "curves.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(curves) == 4 and all(row["user"] == "用户" for row in curves), curves
    invalid = subprocess.run(cmd, cwd=work, capture_output=True, text=True, errors="replace", timeout=30)
    assert invalid.returncode != 0 and "stage2_plan:" not in invalid.stdout, invalid.stdout + invalid.stderr
    print(f"Stage2 configuration snapshot passed (direct={args.direct}, parallel={args.parallel})")


if __name__ == "__main__":
    main()
