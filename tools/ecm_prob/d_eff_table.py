"""Print D_eff across bit widths for key curves, to show the normalization.

Compares two calibrations:
  (A) point:  stage1_prob(B1, p_ref, delta) == f,  p_ref = 2^(bit-0.5)
  (B) averaged: (1/N) sum_{p in primes} stage1_prob(B1, p, delta) == f
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"D:\code\MPA-OpenCl\tools\ecm_prob")
import rho
import curves
import measure
import calibrate

OUT = Path(r"D:\code\MPA-OpenCl\tools\ecm_prob\out")
B1 = 256
KEYS = ["suyama_s10", "param3_s10", "edwards_Z12", "edwards_Z2xZ8", "pm1"]


def d_eff_point(f, bit):
    p_ref = 2 ** (bit - 0.5)
    return math.exp(calibrate.fit_delta(f, B1, p_ref))


def d_eff_averaged(f, primes):
    # bisect delta so that average of stage1_prob over primes == f
    lo, hi = -20.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        avg = sum(rho.stage1_prob(B1, p, delta=mid) for p in primes) / len(primes)
        if avg > f:
            hi = mid
        else:
            lo = mid
    return math.exp(0.5 * (lo + hi))


def main():
    t = {c["name"]: c.get("T") for c in curves.ROSTER}
    print(f"{'curve':16s} {'T':>3s} " + "".join(f"{'bit'+str(b):>9s}" for b in range(15, 26)))
    for key in KEYS:
        row = f"{key:16s} {t[key]:>3d} "
        for b in range(15, 26):
            f = OUT / f"measure_{b}_{B1}.json"
            if not f.exists():
                row += f"{'--':>9s}"
                continue
            blk = json.loads(f.read_text(encoding="utf-8"))
            if "curves" not in blk:
                row += f"{'--':>9s}"
                continue
            frac = blk["curves"][key]["fraction"]
            d = d_eff_point(frac, b)
            row += f"{d:9.2f}"
        print(row)

    # averaged version for two representative bits only (suyama_s10)
    print()
    print("averaged-over-prime-set D_eff (suyama_s10, bits 20 & 25 only):")
    row = "                  "
    for b in (20, 25):
        f = OUT / f"measure_{b}_{B1}.json"
        blk = json.loads(f.read_text(encoding="utf-8"))
        frac = blk["curves"]["suyama_s10"]["fraction"]
        primes = measure.load_primes(b)
        if b > 20:
            import random
            primes = random.Random(20260101 + b).sample(primes, 65536)
        row += f"{d_eff_averaged(frac, primes):9.2f}"
    print(row)


if __name__ == "__main__":
    main()
