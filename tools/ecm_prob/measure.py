"""Empirical stage-1 success measurement over cached prime sets.

For each curve and each B1, compute the fraction of primes p in a bit-width
set for which stage 1 succeeds, i.e. [s]P == identity (mod p), s = lcm(1..B1)
-- equivalently ord(P mod p) is B1-powersmooth.

Caches results to out/measure_{bit}.json so repeated runs are cheap.
"""

from __future__ import annotations

import array
import json
import time
from pathlib import Path

import curves
import ecmath

TOOL_DIR = Path(__file__).resolve().parent
PRIME_DIR = TOOL_DIR / "data" / "primes"
OUT_DIR = TOOL_DIR / "out"


def load_primes(bit: int) -> list[int]:
    data = (PRIME_DIR / f"bits{bit}.bin").read_bytes()
    arr = array.array("Q")
    arr.frombytes(data)
    if arr.itemsize != 8:
        arr.byteswap()
    return list(arr)


def measure_primes(primes: list[int], B1: int, roster=None, verbose=True,
                   label: str = "") -> dict:
    """Measure stage-1 hit fraction over an explicit prime list."""
    s = ecmath.batch_s(B1)
    roster = roster if roster is not None else curves.ROSTER
    out = {"B1": B1, "n_primes": len(primes), "label": label, "curves": {}}
    for c in roster:
        t0 = time.time()
        hits = 0
        for p in primes:
            if ecmath.curve_hits(c, p, s):
                hits += 1
        frac = hits / len(primes) if primes else 0.0
        out["curves"][c["name"]] = {
            "torsion": c["torsion"],
            "hits": hits,
            "fraction": frac,
            "pct": 100.0 * frac,
        }
        if verbose:
            print(f"  {c['name']:16s} {c['torsion']:10s} "
                  f"{hits:7d}/{len(primes)} = {100.0*frac:.4f}%  ({time.time()-t0:.1f}s)")
    return out


def measure_bit(bit: int, B1: int, roster=None, verbose=True) -> dict:
    primes = load_primes(bit)
    out = measure_primes(primes, B1, roster, verbose, label=f"bit{bit}")
    out["bit"] = bit
    return out


def run(bits: list[int], B1_by_bit: dict[int, int], out_name: str = "measure") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for bit in bits:
        B1 = B1_by_bit[bit]
        print(f"bit={bit} (n={len(load_primes(bit))}), B1={B1}:")
        results[str(bit)] = measure_bit(bit, B1)
    out = OUT_DIR / f"{out_name}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    import sys
    # default: reproduce paper Section 9.1 at 20-bit/B1=256
    bits = [int(a) for a in sys.argv[1:]] or [20]
    B1_by_bit = {b: (256 if b == 20 else 256) for b in bits}
    run(bits, B1_by_bit)
