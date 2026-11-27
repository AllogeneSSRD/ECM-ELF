"""Run the CUDA Pminus1 worker and independently check both native save formats."""
import argparse
import math
import struct
import subprocess
import zlib
from pathlib import Path


def saved_residue(root, p, b1):
    prime = (root / f"m{p:07d}").read_bytes()
    magic, version, k, base, exponent, c = struct.unpack_from("<IIdIIi", prime)
    assert (magic, version, k, base, exponent, c) == (0x317A394B, 8, 1, 2, p, -1)
    checksum, state, b_done, c_done, have_x, words = struct.unpack_from("<IIQQII", prime, 48)
    assert (state, b_done, c_done, have_x) == (5, b1, b1, 1)
    assert len(prime) == 80 + words * 4
    assert (sum(n for n, in struct.iter_unpack("<I", prime[52:])) + words) & 0xffffffff == checksum
    residue = int.from_bytes(prime[80:], "little")
    bits = (2 * p * math.lcm(*range(1, b1 + 1))).bit_length()
    files = list((root / f"p1-{p}").glob("*.p1"))
    for path in files:
        header, data = path.read_bytes().split(b"\n", 1)
        if f" k={bits} bits={bits} ".encode() in header:
            assert len(data) == 4 + 4 * ((p + 31) // 32)
            assert zlib.crc32(data[4:]) == int.from_bytes(data[:4], "little")
            assert int.from_bytes(data[4:], "little") == residue
            return residue
    raise AssertionError("missing completed GpuOwl checkpoint")


def run(exe, directory, p, b1, b2, ratio, after, factor, tail="", whole=False,
        stage1_settings="", worker_settings="", publish=None):
    directory.mkdir(parents=True)
    aid = "1234567890abcdef1234567890abcdef"
    prefix = f"Pminus1={aid},1,2,{p},-1,{b1},"
    line = prefix + str(b2) + tail
    (directory / "worktodo.txt").write_text("[Worker #2]\n" + line + "\n")
    config = directory / "ecm.toml"
    config.write_text(
        'config_version=4\n[shared]\nworktodo_file="worktodo.txt"\nfinished_file="finished.txt"\n'
        f'[stage1]\n{stage1_settings}\n[stage2]\nb2_ratio=99\n[worker.2]\n{worker_settings}\n'
        f'b2_ratio={ratio}\nstage2_after_factored={after}\n'
    )
    result = subprocess.run([str(exe), "--config", str(config), "--worker", "2"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=240)
    (directory / "worker.log").write_bytes(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout.decode(errors="replace"))
    root = directory / "pm1" / f"M{p}-B1-{b1}"
    x = saved_residue(root, p, b1)
    if p < 1000:
        assert x == pow(3, 2 * p * math.lcm(*range(1, b1 + 1)), 2**p - 1)
    assert line not in (directory / "worktodo.txt").read_text()
    record = (directory / "finished.txt").read_text().strip()
    assert record.startswith(line + " # P-1 Stage1 ")
    assert record.endswith("gcd=target" if whole else f"factor={factor}" if factor else "no factor")
    add = root / "worktodo.add"
    assert add.exists() == ((not factor or bool(after)) if publish is None else publish)
    assert not (root / "worktodo.add.txt").exists()
    if add.exists():
        assert add.read_text().splitlines() == [prefix + str(b2 or b1 * ratio) + tail]
    print(directory.name, "OK", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--exe", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True, help="fresh directory for evidence")
    args = cli.parse_args()
    exe, output = args.exe.resolve(), args.output.resolve()
    for after in (0, 1):
        run(exe, output / f"found{after}", 37, 3, 0, 2, after, 223)
        run(exe, output / f"none{after}", 37, 2, 0, 0, after, None)
    run(exe, output / "known", 37, 3, 13, 10, 0, None, ',72,7,"223"')
    run(exe, output / "whole", 37, 3, 13, 10, 0, None, ',"616318177"', whole=True)
    run(exe, output / "p95_stage2", 400381, 500, 1000, 0, 0, None)
    run(exe, output / "p95_forced", 400381, 1000, 2000, 0, 1, 671839319)
    run(exe, output / "p95_auto", 400381, 500, 0, 0, 0, None)
    run(exe, output / "default_over_global_off", 37, 2, 0, 0, 0, None,
        stage1_settings='handoff_mode="off"', publish=True)
    run(exe, output / "pm1_off", 37, 2, 0, 0, 0, None,
        stage1_settings='handoff_mode="prime95"\nhandoff_mode_pm1="off"', publish=False)
    run(exe, output / "worker_off_after_factor", 37, 3, 0, 2, 1, 223,
        stage1_settings='handoff_mode_pm1="prime95"',
        worker_settings='handoff_mode_pm1="off"', publish=False)
    run(exe, output / "worker_prime95", 37, 2, 0, 0, 0, None,
        stage1_settings='handoff_mode_pm1="off"',
        worker_settings='handoff_mode_pm1="prime95"', publish=True)
