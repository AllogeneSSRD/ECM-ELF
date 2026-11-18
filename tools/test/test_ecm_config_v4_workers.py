"""Validate v4 configuration and reject older files through real worker entry points."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-stage2", type=Path)
    parser.add_argument("--cpu-stage2", type=Path, required=True)
    args = parser.parse_args()
    build, out = args.build.resolve(), args.out.resolve()
    out.mkdir(parents=True)
    for filename in ("ecm.exe", "ecm_opencl_stage2.exe", "gmp-10.dll"):
        shutil.copy2(build / filename, out / filename)
    shutil.copy2(args.cpu_stage2, out / "ecm_cpu_stage2.exe")
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

    case = out / "v4"
    case.mkdir()
    save = case / "m0_m127_2000.save"
    config = case / "ecm.toml"
    config.write_text(f'''config_version = 4
[shared]
worker_count = 3
worktodo_file = "worktodo.txt"
log_file = ""
finished_file = "finished.txt"
[stage1]
target = "cpu"
cpu_implementation = "scalar"
sigma = 26
stage1_save_dir = "."
exponent_cache_dir = "off"
handoff_mode = "ecm-elf"
handoff_workers = "3"
[stage2]
device = "gpu"
stage2_gpu_backend = "opencl"
stage2_gpu_device = 999
[worker.2]
stage1_gpu_device = 0
stage2_gpu_device = 1
''', encoding="utf-8")
    todo = case / "worktodo.txt"
    todo.write_text(f'[Worker #2]\nECMSTAGE1=1,2,127,-1,{json.dumps(save.as_posix())},2000,0,1\n'
                    '[Worker #3]\n', encoding="utf-8")
    run("ecm.exe", case)
    assert save.is_file() and "[Worker #3]\nECMSTAGE2=" in todo.read_text(encoding="utf-8")
    before = todo.read_bytes()
    assert "curves_this_run=0 invalid_tasks=0" in run("ecm_opencl_stage2.exe", case, "--once")
    assert "CPU Edwards Stage2" in run("ecm_cpu_stage2.exe", case, "--help")
    if args.cuda_stage2:
        assert "curves_this_run=0 invalid_tasks=0" in run("ecm_cuda_stage2.exe", case, "--gpu-backend", "cuda", "--once")
    assert todo.read_bytes() == before

    workers = [('ecm.exe', ()), ('ecm_opencl_stage2.exe', ('--once',)),
               ('ecm_cpu_stage2.exe', ())]
    if args.cuda_stage2:
        workers.append(('ecm_cuda_stage2.exe', ('--gpu-backend', 'cuda', '--once')))
    for version in (None, 1, 2, 3, 5):
        invalid = out / f"invalid_{version}"
        invalid.mkdir()
        body = "config_version = " + str(version) + "\n" if version else "[shared]\nworktodo_file = 'worktodo.txt'\n"
        (invalid / "ecm.toml").write_text(body, encoding="utf-8")
        expected = f"unsupported config_version: {version}" if version else "config_version = 4"
        for exe, flags in workers:
            run(exe, invalid, *flags, error=expected)
    print("Stage1 handoff, flat worker projection and config version rejection passed")


if __name__ == "__main__":
    main()
