"""数据层：素数集生成/缓存/加载 + 经验命中测量。

整合原 gen_primes.py（primesieve 生成 + manifest 留痕）与 measure.py
（stage-1 命中测量）的库逻辑。CLI 入口见 ecm_sweep.py。
"""

from __future__ import annotations

import array
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import ecmath
import curves

TOOL_DIR = Path(__file__).resolve().parent
PRIME_DIR = TOOL_DIR / "data" / "primes"
PRIMESIEVE = Path(r"D:\code\MPA-OpenCl\.refactor\primesieve-12.15-win-x64\primesieve.exe")
MANIFEST = PRIME_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# 素数集：生成 / 缓存 / 加载
# ---------------------------------------------------------------------------
def sieve_range(lo: int, hi: int) -> list[int]:
    """All primes in [lo, hi] via primesieve.exe."""
    cmd = [str(PRIMESIEVE), str(lo), str(hi), "-p", "--no-status"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [int(x) for x in out.stdout.split() if x.strip()]


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_primes(bit: int) -> dict:
    lo = 1 << (bit - 1)
    hi = (1 << bit) - 1
    primes = sieve_range(lo, hi)
    data = array.array("Q", primes).tobytes()
    fpath = PRIME_DIR / f"bits{bit}.bin"
    fpath.write_bytes(data)
    return {
        "bit": bit, "lo": lo, "hi": hi, "count": len(primes),
        "sha256": _sha256(data), "generator": "primesieve 12.15",
        "cmd": f"primesieve {lo} {hi} -p",
        "format": "uint64 little-endian, one prime per 8 bytes",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "exhaustive (all primes in [2^(b-1), 2^b-1])",
    }


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")


def gen_primes(bits: list[int]) -> None:
    PRIME_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    for b in bits:
        key = str(b)
        fpath = PRIME_DIR / f"bits{b}.bin"
        prev = manifest.get(key)
        if fpath.exists() and prev and prev.get("sha256") == _sha256(fpath.read_bytes()):
            print(f"bits{b}: cached ({prev['count']} primes, sha256 ok)")
            continue
        entry = write_primes(b)
        manifest[key] = entry
        save_manifest(manifest)
        print(f"bits{b}: generated {entry['count']} primes -> {fpath.name} "
              f"(sha256 {entry['sha256'][:12]}...)")
    save_manifest(manifest)


def load_primes(bit: int) -> list[int]:
    data = (PRIME_DIR / f"bits{bit}.bin").read_bytes()
    arr = array.array("Q")
    arr.frombytes(data)
    if arr.itemsize != 8:
        arr.byteswap()
    return list(arr)


# ---------------------------------------------------------------------------
# 经验命中测量
# ---------------------------------------------------------------------------
def measure_primes(primes: list[int], B1: int, roster=None,
                   verbose: bool = True, label: str = "") -> dict:
    """Measure stage-1 hit fraction over an explicit prime list."""
    s = ecmath.batch_s(B1)
    roster = roster if roster is not None else curves.ROSTER
    out = {"B1": B1, "n_primes": len(primes), "label": label, "curves": {}}
    for c in roster:
        t0 = time.time()
        hits = sum(1 for p in primes if ecmath.curve_hits(c, p, s))
        frac = hits / len(primes) if primes else 0.0
        out["curves"][c["name"]] = {
            "torsion": c["torsion"], "hits": hits,
            "fraction": frac, "pct": 100.0 * frac,
        }
        if verbose:
            print(f"  {c['name']:16s} {c['torsion']:10s} "
                  f"{hits:7d}/{len(primes)} = {100.0*frac:.4f}%  ({time.time()-t0:.1f}s)")
    return out


def measure_bit(bit: int, B1: int, roster=None, verbose: bool = True) -> dict:
    out = measure_primes(load_primes(bit), B1, roster, verbose, label=f"bit{bit}")
    out["bit"] = bit
    return out
