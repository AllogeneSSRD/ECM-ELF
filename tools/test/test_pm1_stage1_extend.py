"""Verify B1 edits through the CUDA worker against independent integer arithmetic."""
import argparse
import json
import math
import re
import struct
import subprocess
import time
import zlib
from pathlib import Path

from test_stage1_ntt_task_batches import pausable_process


def lcm(bound):
    # Take one maximum power per prime; practical also for the live edit case.
    primes = bytearray([1]) * (bound + 1)
    powers = []
    for p in range(2, bound + 1):
        if primes[p]:
            primes[p * 2::p] = bytes(bound // p - 1)
            q = p
            while q * p <= bound:
                q *= p
            powers.append(q)
    return math.prod(powers)


def prime_file(p, b1, residue, state=5, version=8):
    def giant(x):
        count = (x.bit_length() + 31) // 32
        return struct.pack("<I", count) + x.to_bytes(4 * count, "little"), count
    x, extra = giant(residue)
    body = struct.pack("<IQQI", state, b1, b1, 1) + x
    if state == 2:
        invx, n = giant(pow(residue, -1, (1 << p) - 1))
        body += invx + struct.pack("<I", 0)
        extra += n
    checksum = (sum(w for w, in struct.iter_unpack("<I", body)) + extra) & 0xffffffff
    return struct.pack("<IIdIIi12sdI", 0x317a394b, version, 1., 2, p, -1, b"S5", 1., checksum) + body


def crc32c(data):
    crc = 0xffffffff
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82f63b78 if crc & 1 else 0)
    return crc ^ 0xffffffff


def check(exe, output):
    p = 127
    modulus = (1 << p) - 1
    filename = f"m{p:07d}"

    def row(b1):
        return f"Pminus1=1,2,{p},-1,{b1},0"

    def setup(name, bound, settings="", shared=False):
        directory = output / name
        directory.mkdir(parents=True)
        if shared:
            settings += "\nprime95_save_dir=" + json.dumps(directory.as_posix())
        (directory / "ecm.toml").write_text('config_version=4\n[shared]\nlog_file=""\n'
            'worktodo_file="worktodo.txt"\nfinished_file="finished.txt"\n[stage1]\n' + settings +
            '\n[stage2]\nb2_ratio=0\nstage2_after_factored=1\n')
        (directory / "worktodo.txt").write_text(row(bound) + "\n")
        return directory

    def root(directory, bound):
        return directory / "pm1" / f"M{p}-B1-{bound}"

    def run(directory, success=True, edit=None):
        log = directory / "worker.log"
        with log.open("wb") as stream, subprocess.Popen(
                [exe, "--config", directory / "ecm.toml"], cwd=directory, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT) as process:
            if edit:
                until = time.monotonic() + 60
                while b"pm1_stage1_progress:" not in log.read_bytes():
                    assert time.monotonic() < until and process.poll() is None, log.read_bytes()
                    time.sleep(.01)
                (directory / "edited.txt").write_text(row(edit) + "\n")
                (directory / "edited.txt").replace(directory / "worktodo.txt")
            status = process.wait(timeout=180)
        text = log.read_text(errors="replace")
        assert (status == 0) == success, text
        return text

    def verify(directory, bound, residue, shared=False, prior_finished=()):
        target = directory if shared else root(directory, bound)
        assert (target / filename).read_bytes() == prime_file(p, bound, residue)
        assert (target / "worktodo.add").read_text().strip() == row(bound)
        assert not (directory / "worktodo.txt").read_text().strip()
        records = (directory / "finished.txt").read_text().splitlines()
        assert [record.split(" # P-1 Stage1 ")[0] for record in records] == [
            *map(row, prior_finished), row(bound)
        ]

    for name, old, new, k, state, version in [
        ("powers", 7, 9, 6, 5, 8), ("identity", 13, 14, 1, 5, 8),
        ("midstage", 7, 93, lcm(93) // lcm(7), 2, 7),
        ("u32_boundary", 4294967290, 4294967296, 2 * 4294967291, 5, 8),
    ]:
        directory = setup(name, new, 'ntt_window_bits=0', shared=True)
        base = (1 << 97) + 105
        (directory / filename).write_bytes(prime_file(p, old, base, state, version))
        log = run(directory)
        assert f"B1={old} -> {new} K_bits={k.bit_length()}" in log
        verify(directory, new, pow(base, k, modulus), shared=True)

    # An unfinished OWL exponent takes precedence even over a newer native seed.
    old, new, k = 37, 82, 11
    directory = setup("owl_first", new)
    exponent = 2 * p * lcm(old)
    value = pow(3, exponent >> (exponent.bit_length() - k), modulus)
    owl = root(directory, old) / f"p1-{p}"
    owl.mkdir(parents=True)
    limbs = value.to_bytes(16, "little")
    (owl / f"{p}-{k:09d}.p1").write_bytes(
        f"OWL P1 2 N=1*2^{p}-1 B1={old} B2=915 k={k} bits={exponent.bit_length()} err=0 time=0\n".encode()
        + struct.pack("<I", zlib.crc32(limbs)) + limbs)
    (directory / filename).write_bytes(prime_file(p, old + 1, 7))
    log = run(directory)
    assert f"B1={old} B2=915" in log and f"k={k}/{exponent.bit_length()}" in log
    old_left = exponent.bit_length() - k
    total = old_left + (lcm(new) // lcm(old)).bit_length() - 1
    samples = re.findall(r"pm1_stage1_progress: iter=(\d+)/(\d+).* first_iter=(\d+) "
                         rf"old_B1={old} target_B1={new} phase=(\w+)", log)
    assert samples[0] == ("0", str(total), str(total), "finish_old")
    assert (str(old_left), str(total), str(total), "extend") in samples
    assert samples[-1] == (str(total), str(total), str(total), "extend")
    finished_owl = owl / f"{p}-{exponent.bit_length():09d}.p1"
    assert int.from_bytes(finished_owl.read_bytes().split(b"\n", 1)[1][4:], "little") == pow(3, exponent, modulus)
    verify(directory, new, pow(3, 2 * p * lcm(new), modulus))
    assert not (root(directory, old) / "worktodo.add").exists()

    for reraised in (False, True):
        old, intermediate = 7, 59
        new = 131 if reraised else intermediate
        directory = setup("raise_again" if reraised else "resume_extension", new)
        base = 19
        ratio = lcm(intermediate) // lcm(old)
        bits, pos = ratio.bit_length(), 8
        residue = pow(base, ratio >> (bits - pos), modulus)
        data = struct.pack("<8sIQQQQI", b"ECMP1EX1", p, intermediate, old, bits, pos, 1)
        data += base.to_bytes(16, "little") + residue.to_bytes(16, "little")
        root(directory, intermediate).mkdir(parents=True)
        (root(directory, intermediate) / "extension.pm1").write_bytes(data + struct.pack("<I", crc32c(data)))
        log = run(directory)
        assert f"k={pos}/{bits}" in log
        remaining = bits - pos + ((lcm(new) // lcm(intermediate)).bit_length() - 1 if reraised else 0)
        assert f"pm1_stage1_progress: iter=0/{remaining} " in log
        assert f"pm1_stage1_progress: iter={remaining}/{remaining} " in log
        verify(directory, new, pow(base, lcm(new) // lcm(old), modulus))

    directory = setup("same_bound", 17, shared=True)
    trailer = b"MOREINFOJSONDATA" + struct.pack("<III", 10, 1, 0) + b"{}"
    native = prime_file(p, 17, 11, version=7) + trailer
    (directory / filename).write_bytes(native)
    assert "using completed Prime95 endpoint" in run(directory)
    assert (directory / filename).read_bytes() == native
    assert (directory / "worktodo.add").read_text().strip() == row(17)

    directory = setup("damaged_extension", 59)
    root(directory, 59).mkdir(parents=True)
    damaged = bytearray((root(output / "resume_extension", 59) / "extension.pm1").read_bytes())
    damaged[-1] ^= 1
    (root(directory, 59) / "extension.pm1").write_bytes(damaged)
    assert "invalid P-1 extension checkpoint" in run(directory, success=False)
    assert (directory / "worktodo.txt").read_text().strip() == row(59)

    for name in ("bad_checksum", "wrong_p", "lower_bound"):
        directory = setup(name, 17, shared=True)
        seed = bytearray(prime_file(131 if name == "wrong_p" else p, 19 if name == "lower_bound" else 7, 11))
        if name == "bad_checksum":
            seed[-1] ^= 0x80
        (directory / filename).write_bytes(seed)
        run(directory, success=False)
        assert (directory / filename).read_bytes() == seed
        assert (directory / "worktodo.txt").read_text().strip() == row(17)

    old, new = 12000, 12100
    directory = setup("live_edit", old)
    log = run(directory, edit=new)
    assert "B1 increased in worktodo" in log and f"B1={old} -> {new}" in log, log
    assert not (root(directory, old) / "worktodo.add").exists()
    verify(directory, new, pow(3, 2 * p * lcm(new), modulus))

    directory = setup("stop_extension", old)
    seed = 19
    (directory / filename).write_bytes(prime_file(p, 7, seed))
    stop_log = directory / "stop.log"
    with pausable_process([exe, "--config", directory / "ecm.toml"], directory, None, stop_log) as (process, stop):
        deadline = time.monotonic() + 60
        while b"pm1_stage1_progress:" not in stop_log.read_bytes():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(.01)
        stop()
        assert process.wait(timeout=90) != 0
    data = (root(directory, old) / "extension.pm1").read_bytes()
    bits, pos = struct.unpack_from("<QQ", data, 28)
    assert 1 < pos < bits and crc32c(data[:-4]) == struct.unpack_from("<I", data, len(data) - 4)[0]
    assert (directory / "worktodo.txt").read_text().strip() == row(old)
    (directory / "worktodo.txt").write_text(row(new) + "\n")
    assert f"k={pos}/{bits}" in run(directory)
    verify(directory, new, pow(seed, lcm(new) // lcm(7), modulus))

    # The old assignment is still private if both receiver mailboxes were occupied.
    old, new = 13, 21
    directory = setup("mailbox_edit", old)
    root(directory, old).mkdir(parents=True)
    for mailbox in ("worktodo.add", "worktodo.add.txt"):
        (root(directory, old) / mailbox).write_text("pending\n")
    assert "await receiver pickup" in run(directory, success=False)
    assert (directory / "finished.txt").read_text().splitlines() == [row(old) + " # P-1 Stage1 no factor"]
    (directory / "worktodo.txt").write_text(row(new) + "\n")
    run(directory)
    assert not list(root(directory, old).glob("*.ready"))
    for mailbox in ("worktodo.add", "worktodo.add.txt"):
        assert (root(directory, old) / mailbox).read_text() == "pending\n"
    verify(directory, new, pow(3, 2 * p * lcm(new), modulus), prior_finished=(old,))
    print("P-1 B1 extension, checkpoint priority, repeated increase and live queue edit passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    check(args.exe.resolve(strict=True), args.output.resolve())
