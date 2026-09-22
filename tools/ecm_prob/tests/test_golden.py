"""Golden-number validation against Bernstein-Birkner-Lange-Peters,
"ECM using Edwards curves", Section 9.1 (20-bit primes, B1=256).

Expected (paper Table/9.1):
  Z/12      -> 12467 / 38635 = 32.2687%
  Z/2xZ/8   ->            32.8433%
  Z/2xZ/4   ->            27.4854%
  Z/4       ->            23.4709%
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ecmath
import curves

B1 = 256
LO = 1 << 19
HI = 1 << 20


def main():
    primes = [p for p in ecmath.primes_upto(HI) if p >= LO]
    assert len(primes) == 38635, len(primes)
    s = ecmath.batch_s(B1)
    print(f"primes in [2^19,2^20): {len(primes)}  (expect 38635)")
    print(f"s = lcm(1..{B1}), bits={s.bit_length()}")

    golden = {
        "edwards_Z12": 12467,
        "edwards_Z2xZ8": None,   # paper gives % only
        "edwards_Z2xZ4": None,
        "edwards_Z4": None,
    }
    for c in curves.ROSTER:
        if c["form"] != "edwards":
            continue
        t0 = time.time()
        hits = sum(1 for p in primes if ecmath.curve_hits(c, p, s))
        dt = time.time() - t0
        pct = 100.0 * hits / len(primes)
        exp = golden.get(c["name"])
        mark = ""
        if exp is not None:
            mark = "OK" if hits == exp else f"MISMATCH (expect {exp})"
        print(f"{c['name']:16s} {c['torsion']:8s} {hits:6d}/{len(primes)} "
              f"= {pct:.4f}%  {mark}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
