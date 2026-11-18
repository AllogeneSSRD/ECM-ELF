"""Exercise v4 handoff modes with real CPU Stage1 and offline receiver queues."""
import argparse
from pathlib import Path
import shutil
import subprocess


TASKS = {curve: f'ECMSTAGE1=1,2,127,-1,"{curve}_m127_100.save",2000,0,1'
         for curve in ("e", "m0")}


def verify(build, root):
    root.mkdir(parents=True)
    worker = build / "ecm.exe"
    if not worker.is_file():
        worker = build / "ecm_cuda.exe"

    def scenario(name, mode, curve="m0", fallback="ecm-elf", external=False,
                 prime95_paths=True, blocked=False, expected="off", error=None,
                 local_queue="receiver.txt", log_file="", shared_extra="", stage1_extra="",
                 overrides="", extra_queue="", local_receiver=None):
        case = root / name
        bin_dir, config_dir, p95 = case / "bin", case / "config", case / "prime95"
        for directory in (bin_dir, config_dir, p95):
            directory.mkdir(parents=True)
        exe = bin_dir / "ecm.exe"
        shutil.copy2(worker, exe)
        shutil.copy2(build / "gmp-10.dll", bin_dir)
        queue, config = config_dir / "worktodo.txt", config_dir / "ecm.toml"
        finished = config_dir / "finished.txt"
        add = p95 / "worktodo.add"
        (p95 / "worktodo.txt").write_text("", encoding="ascii")
        target = p95 / "saves" / f"{curve}_m127_100.save"
        if blocked:
            target.mkdir(parents=True)
        settings = f"""config_version = 4
[shared]
worktodo_file = "worktodo.txt"
finished_file = "finished.txt"
log_file = "{log_file}"
{shared_extra}
[stage1]
target = "cpu"
cpu_implementation = "scalar"
sigma = 26
exponent_cache_dir = "off"
handoff_unsupported = "{fallback}"
"""
        if mode is not None:
            settings += f'handoff_mode = "{mode}"\n'
        if external:
            settings += f'handoff_worktodo_file = "{local_queue}"\nhandoff_workers = "2"\n'
        if prime95_paths:
            settings += (f'prime95_worktodo_file = "{(p95 / "worktodo.txt").as_posix()}"\n'
                         f'prime95_save_dir = "{(p95 / "saves").as_posix()}"\n')
        config.write_text(settings + stage1_extra + overrides, encoding="ascii")
        queue.write_text(TASKS[curve] + "\n" + extra_queue, encoding="ascii")

        def run(label, error=None):
            process = subprocess.run([str(exe), "--config", str(config), "--worker", "1"],
                                     cwd=case, capture_output=True, timeout=40)
            log = (process.stdout + process.stderr).decode("utf-8", "replace")
            (case / f"{label}.log").write_text(log, encoding="utf-8")
            if error:
                assert process.returncode != 0 and error in log, (process.returncode, log)
            else:
                assert process.returncode == 0, (process.returncode, log)
            return log

        if error:
            run("rejected", error)
            assert queue.read_text(encoding="ascii") == TASKS[curve] + "\n" + extra_queue
            assert not (bin_dir / f"{curve}_m127_100.save").exists()
            return

        run("first")
        remaining = queue.read_text(encoding="utf-8")
        receiver = config_dir / local_queue
        assert "ECMSTAGE1=" not in remaining, remaining
        assert "ECMSTAGE1=" in finished.read_text(encoding="utf-8")
        assert (bin_dir / f"{curve}_m127_100.save").is_file()
        assert (("ECMSTAGE2=" in remaining) == (expected == "in-place" or local_receiver is not None)), remaining
        if local_receiver is not None:
            assert f"[Worker #{local_receiver}]\nECMSTAGE2=" in remaining, remaining
        assert receiver.is_file() == (expected == "external")
        if expected == "external":
            assert "[Worker #2]\nECMSTAGE2=" in receiver.read_text(encoding="utf-8")
        assert add.is_file() == (expected == "p95" and not blocked), remaining
        if expected == "p95":
            if blocked:
                # Retry the frozen intent after changing mode and removing P95 paths.
                assert any((bin_dir / "p95_add_pending.txt.handoff").glob("*.intent"))
                target.rmdir()
                config.write_text(settings.split("prime95_worktodo_file")[0].replace(
                    'handoff_mode = "prime95"', 'handoff_mode = "off"'), encoding="ascii")
                run("restart")
                run("replay")
            assert target.is_file()
            assert add.read_text(encoding="utf-8").count("ECMSTAGE2=") == 1
            assert not any((bin_dir / "p95_add_pending.txt.handoff").glob("*.intent"))

    scenario("default_local", None, expected="in-place")
    scenario("explicit_off", "off")
    scenario("local_in_place", "ecm-elf", expected="in-place")
    scenario("local_external_shared_finished", "ecm-elf", external=True, expected="external")
    scenario("prime95_no_local_duplicate", "prime95", external=True, expected="p95")
    scenario("prime95_fallback", "prime95", "e", prime95_paths=False, expected="in-place")
    scenario("prime95_keep_save", "prime95", "e", fallback="keep-save", external=True)
    scenario("prime95_missing_path", "prime95", prime95_paths=False,
             error="missing Prime95 queue")
    scenario("prime95_pending_after_switch", "prime95", external=True, blocked=True, expected="p95")
    scenario("local_queue_conflicts_with_log", "ecm-elf", external=True, local_queue="ecm.log",
             log_file="ecm.log", error="Stage2 transfer queue/finished paths conflict")
    scenario("local_queue_conflicts_with_config", "ecm-elf", external=True, local_queue="ecm.toml",
             error="Stage2 transfer queue/finished paths conflict")
    scenario("local_auto_eligible_workflow_worker", "ecm-elf", local_receiver=3,
             shared_extra="worker_count = 4", stage1_extra='handoff_workers = "auto"\n',
             overrides='[worker.1]\nqueue_skip = "stage2"\n'
                       '[worker.2]\nworktodo_file = "other.txt"\n'
                       '[worker.4]\nqueue_skip = "invalid"\n',
             extra_queue="[Worker #5]\n# outside the configured workflow\n")
    scenario("local_auto_without_receiver", "ecm-elf",
             shared_extra='worker_count = 1\nqueue_skip = "stage2"',
             stage1_extra='handoff_workers = "auto"\n',
             error="automatic Stage2 handoff found no eligible worker")
    print("Stage1 handoff modes: 13 queue scenarios passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.build.resolve(), args.out.resolve())
