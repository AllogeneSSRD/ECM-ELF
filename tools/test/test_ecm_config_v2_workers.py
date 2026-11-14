"""Compare v1 and v2 with real Stage1/Stage2 workers in isolated directories."""
import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-stage2", type=Path)
    args = parser.parse_args()
    build, out = args.build.resolve(), args.out.resolve()
    out.mkdir(parents=True)
    for filename in ("ecm.exe", "ecm_opencl_stage2.exe", "ecm_cpu_stage2.exe", "gmp-10.dll"):
        shutil.copy2(build / filename, out / filename)
    if args.cuda_stage2:
        shutil.copy2(args.cuda_stage2, out / "ecm_cuda_stage2.exe")

    def run(exe, directory, *options, error=None):
        config = directory / "ecm.toml"
        original = config.read_bytes()
        process = subprocess.run([str(out / exe), "--config", str(config), "--worker", "2", *options],
                                 cwd=directory, capture_output=True, timeout=30)
        log = (process.stdout + process.stderr).decode("utf-8", errors="replace")
        (directory / (exe + ".log")).write_text(log, encoding="utf-8")
        assert config.read_bytes() == original
        if error:
            assert process.returncode != 0 and error in log, log
        else:
            assert process.returncode == 0, log
        return log

    settings = {
        "legacy": """
[shared]
worktodo = "worktodo.txt"
[stage1]
target = "cpu"
impl = "scalar"
sigma = 26
tmp_dir = "{save_dir}"
finished = "{finished}"
log_file = ""
exp_cache = "off"
auto_stage2 = 1
auto_stage2_workers = "3"
[stage2]
stage2_backend = "opencl"
stage2_device = 999
[workers.2.stage1]
device = 0
[workers.2.stage2]
stage2_device = 1
""",
        "current": """
config_version = 2
[shared]
worktodo_file = "worktodo.txt"
[stage1]
target = "cpu"
cpu_implementation = "scalar"
sigma = 26
save_dir = "{save_dir}"
finished_file = "{finished}"
log_file = ""
exponent_cache_dir = "off"
handoff.enabled = true
handoff.workers = "3"
[stage2]
gpu_backend = "opencl"
gpu_device = 999
[worker.2]
stage1.gpu_device = 0
stage2.gpu_device = 1
""",
    }
    completed = {}
    for label, config in settings.items():
        case = out / label
        case.mkdir()
        (case / "ecm.toml").write_text(config.format(save_dir=(case / "saves").as_posix(),
                                                    finished=(case / "finished.txt").as_posix()),
                                       encoding="utf-8")
        todo = case / "worktodo.txt"
        save = case / "m0_m127_100.save"
        todo.write_text(f'[Worker #2]\nECMSTAGE1=1,2,127,-1,"{save}",2000,0,1\n'
                        '[Worker #3]\n', encoding="utf-8")
        run("ecm.exe", case)
        queue = todo.read_text(encoding="utf-8")
        assert "[Worker #3]\nECMSTAGE2=" in queue, queue
        completed[label] = (queue.replace(str(case), "<case>"),
                            (case / "finished.txt").read_text(encoding="utf-8").replace(str(case), "<case>"),
                            save.read_text(encoding="utf-8"))
        before = todo.read_bytes()
        assert "curves_this_run=0 invalid_tasks=0" in run("ecm_opencl_stage2.exe", case, "--once")
        assert "CPU Edwards Stage2" in run("ecm_cpu_stage2.exe", case, "--help")
        assert todo.read_bytes() == before

    def stable_save(lines):
        return [field.strip() for field in lines.split(";")
                if not field.strip().startswith(("TIME=", "COMMENT="))]

    legacy, current = completed.values()
    assert legacy[:2] == current[:2], completed
    assert stable_save(legacy[2]) == stable_save(current[2]), completed
    invalid = out / "future"
    invalid.mkdir()
    (invalid / "ecm.toml").write_text("config_version = 4\n", encoding="ascii")
    for exe in ("ecm.exe", "ecm_opencl_stage2.exe", "ecm_cpu_stage2.exe"):
        run(exe, invalid, error="unsupported config_version: 4")
    if args.cuda_stage2:
        for label, configuration in (
            ("legacy", '[shared]\nworktodo="worktodo.txt"\n[stage2]\nstage2_backend="cuda"\n'),
            ("current", 'config_version=2\n[shared]\nworktodo_file="worktodo.txt"\n[stage2]\ngpu_backend="cuda"\n'),
        ):
            case = out / ("cuda_" + label)
            case.mkdir()
            (case / "ecm.toml").write_text(configuration, encoding="utf-8")
            (case / "worktodo.txt").write_text("[Worker #2]\n", encoding="utf-8")
            assert "curves_this_run=0 invalid_tasks=0" in run("ecm_cuda_stage2.exe", case, "--once")
        run("ecm_cuda_stage2.exe", invalid, error="unsupported config_version: 4")
    print("Stage1 handoff, OpenCL Stage2 queue, CPU Stage2 config and version rejections passed")


if __name__ == "__main__":
    main()
