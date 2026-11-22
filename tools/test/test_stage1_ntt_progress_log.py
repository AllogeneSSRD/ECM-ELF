"""Verify that Stage1 NTT progress cadence affects only the file mirror."""

import argparse
from pathlib import Path
import re
import subprocess


PROGRESS = re.compile(r"(stage1_ntt_progress: (?:bits|columns)=(\d+)/(\d+)[^\r\n]*)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    exe = args.exe.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    for model in ("m0", "e"):
        for cadence in (0, 3600, -1):
            work = output / f"{model}-{cadence}"
            work.mkdir()
            (work / "worktodo.txt").write_text("ECM=1,2,4423,-1,1024,4096,4,20260922\n", encoding="ascii")
            config = work / "ecm.toml"
            config.write_text(
                'config_version = 4\n[shared]\nworktodo_file = "worktodo.txt"\n'
                'finished_file = "finished.txt"\nlog_file = "worker.log"\n'
                f'[stage1]\nparam = "{model}"\ntarget = "gpu"\nstage1_gpu_device = {args.device}\n'
                'gpu_algorithm = "ntt"\nntt_convolution = "linear"\nedwards_naf_window_bits = 6\n'
                f'progress_log_seconds = {cadence}\nstage1_save_dir = "{work.as_posix()}"\n'
                'handoff_mode = "off"\n[stage2]\ntune_runtime_enabled = false\n', encoding="utf-8")
            run = subprocess.run([exe, "--config", config], cwd=work, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=120)
            (work / "console.log").write_text(run.stdout + run.stderr, encoding="utf-8")
            assert run.returncode == 0, (work, run.stdout[-3000:], run.stderr)
            mirror = (work / "worker.log").read_text(encoding="utf-8", errors="replace")
            console_rows = PROGRESS.findall(run.stdout)
            file_rows = PROGRESS.findall(mirror)
            finals = [row for row in console_rows if row[1] == row[2]]
            assert finals and len(console_rows) > len(finals), (work, console_rows)
            assert all(" eta=" in row[0] for row in console_rows), (work, console_rows)
            assert all(row[0].endswith(" eta=0.000") for row in finals), (work, finals)
            assert "queue done" in mirror, work
            assert file_rows == (console_rows if cadence < 0 else finals), (work, file_rows, console_rows)
            print(f"PASS {model} cadence={cadence}: pipe={len(console_rows)}, file={len(file_rows)}", flush=True)


if __name__ == "__main__":
    main()
