"""Sweep measurement across bit widths, with caching.

  * bits 15..20: exhaustive (all primes in [2^(b-1), 2^b-1]).
  * bits 21..25: reproducible random sample of SAMPLE_N primes (fixed seed),
                 matching the paper's 30-bit sample size (65536).

B1 is fixed at 256 (the paper's 20-bit anchor), so success fractions decrease
with bit width -- the intended "difficulty vs size" picture.

Results cached to out/measure_{bit}_{B1}.json (same schema as measure.py).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import curves
import measure

TOOL_DIR = Path(__file__).resolve().parent
OUT_DIR = TOOL_DIR / "out"

B1 = 256
SAMPLE_N = 65536
SAMPLE_SEED = 20260101          # fixed, documented, reproducible
EXHAUSTIVE_MAX_BIT = 20


def get_primes(bit: int) -> tuple[list[int], str]:
    allp = measure.load_primes(bit)
    if bit <= EXHAUSTIVE_MAX_BIT:
        return allp, f"exhaustive ({len(allp)})"
    rng = random.Random(SAMPLE_SEED + bit)   # per-bit stable sample
    return rng.sample(allp, SAMPLE_N), f"sample {SAMPLE_N}/{len(allp)} seed={SAMPLE_SEED + bit}"


def sweep(bits: list[int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for bit in bits:
        out = OUT_DIR / f"measure_{bit}_{B1}.json"
        if out.exists():
            print(f"bit={bit}: cached {out.name}")
            continue
        primes, desc = get_primes(bit)
        t0 = time.time()
        print(f"bit={bit}: {desc}, B1={B1} ...")
        block = measure.measure_primes(primes, B1, curves.ROSTER, verbose=True,
                                       label=f"bit{bit} {desc}")
        block["bit"] = bit
        block["sampling"] = desc
        block["sample_seed"] = SAMPLE_SEED + bit if bit > EXHAUSTIVE_MAX_BIT else None
        out.write_text(json.dumps(block, indent=2), encoding="utf-8")
        print(f"  -> {out.name}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    import sys
    bits = [int(a) for a in sys.argv[1:]] or list(range(15, 26))
    sweep(bits)
