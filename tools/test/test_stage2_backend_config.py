"""Check OpenCL backend config admission without running curve arithmetic."""

import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True, help="Montgomery M127 save with B1=100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    exe = args.exe.resolve()

    def run(name, settings, flags, code, message, *, task_b2=None, marked=False, worker=1):
        case = output / name
        case.mkdir()
        config = case / "ecm.toml"
        config.write_text(
            'config_version = 4\n[shared]\nworktodo_file = "worktodo.txt"\n'
            'finished_file = "finished.txt"\n[stage2]\ndevice = "gpu"\nstage2_gpu_backend = "opencl"\n'
            'tune_runtime_enabled = false\n' + settings,
            encoding="utf-8",
        )
        queue = case / "worktodo.txt"
        task = f'ECMSTAGE2=1,2,127,-1,"point.save",{task_b2},0,1\n' if task_b2 is not None else ""
        queue.write_text(task, encoding="utf-8")
        original = queue.read_bytes()
        shutil.copy2(args.save, case / "point.save")
        result = subprocess.run(
            [str(exe), "--config", str(config), "--worker", str(worker), "--once",
             "--short-calibration", "0", *flags],
            cwd=case, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace", timeout=30,
        )
        (case / "probe.log").write_text(result.stdout, encoding="utf-8")
        assert result.returncode == code and message in result.stdout, (name, result.returncode, result.stdout)
        finished = case / "finished.txt"
        if marked:
            assert queue.read_text(encoding="utf-8-sig").startswith("# ERROR " + task.strip())
            assert "source=worktodo" in finished.read_text(encoding="utf-8-sig")
        else:
            assert queue.read_bytes() == original and not finished.exists(), name

    run("dormant_cuda_presets", 'weighted_ntt = true\ncurve_concurrency = "dual"\n',
        [], 0, "invalid_tasks=0")
    run("dormant_cuda_point_backends", 'baby_backend = "coop32"\ngiant_backend = "coop32"\nseed_backend = "coop32"\n',
        [], 0, "invalid_tasks=0")
    run("explicit_cuda_point_backend", "", ["--giant-backend", "coop32"], 2,
        "cooperative point backends are CUDA-only")
    run("explicit_weighted", "", ["--weighted-ntt", "1"], 2, "weighted NTT requires")
    run("explicit_parallel", "", ["--parallel", "2"], 2, "does not support --parallel 2")
    run("automatic_parallel_flag", "", ["--parallel", "auto"], 0, "invalid_tasks=0")
    run("serial_parallel_flag", "", ["--parallel", "1"], 0, "invalid_tasks=0")
    run("auto_b2", "auto_b2 = true\n", [], 2, "OpenCL Stage2 requires a nonzero B2", task_b2=0)
    run("ratio_invalid", "b2_ratio = 1\n", [], 2, "source=b2_ratio", task_b2=0)
    run("config_b2_invalid", "b2 = 99\n", [], 2, "source=config", task_b2=0)
    run("cli_b2_invalid", "", ["--b2", "99"], 2, "source=cli", task_b2=200)
    run("task_b2_invalid", "", [], 1, "source=worktodo", task_b2=99, marked=True)
    ratio = "b2_ratio = 3\n"
    run("cli_before_ratio", ratio, ["--b2", "400", "--dry-run"], 0,
        "B2=400 D=0 device=0 worker=1 B2_source=cli", task_b2=200)
    run("task_interval_before_ratio", ratio, ["--dry-run"], 0,
        "B2=220 D=0 device=0 worker=1 B2_source=worktodo", task_b2="150-220")
    run("config_before_ratio", ratio + "b2 = 250\n", ["--dry-run"], 0,
        "B2=250 D=0 device=0 worker=1 B2_source=config", task_b2=0)
    run("ratio_fallback", ratio, ["--dry-run"], 0,
        "B2=300 D=0 device=0 worker=1 B2_source=b2_ratio", task_b2=0)
    inactive = '[worker.2]\nweighted_ntt = "invalid"\n'
    run("worker_skips_unrelated_value", inactive, [], 0, "invalid_tasks=0")
    run("worker_validates_selected_value", inactive, [], 2, "weighted_ntt", worker=2)
    print("PASS: Stage2 backend presets, B2 sources, queue preservation and worker scopes")


if __name__ == "__main__":
    main()
