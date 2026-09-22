"""Fit the "effective divisor" D per curve from empirical success fractions.

The model is GMP-ECM's own stage-1 probability (rho.py stage1_prob), whose free
parameter `delta` is directly comparable to GMP-ECM's ECM_EXTRA_SMOOTHNESS
(Suyama: delta = 3.134, i.e. effective divisor D = exp(3.134) ~ 22.97).

    D_eff = exp(delta)   where  stage1_prob(B1, p_ref, delta) == f_empirical

with p_ref = 2^(bit - 0.5) (geometric center of the bit range).  `extra = D/T`
is the Galois extra-smoothness factor above the guaranteed torsion T.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import rho as rho_mod

TOOL_DIR = Path(__file__).resolve().parent


def fit_delta(f_target: float, B1: float, N: float,
              lo: float = -20.0, hi: float = 40.0, iters: int = 120) -> float:
    """Bisect delta so that stage1_prob(B1, N, delta) == f_target.

    f(delta) is increasing in delta (larger delta -> smaller effN -> smoother).
    """
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = rho_mod.stage1_prob(B1, N, delta=mid)
        if f > f_target:            # too smooth -> need smaller delta
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def calibrate_block(block: dict, B1: int, roster, p_ref: float = None) -> dict:
    """Fit D_eff per curve for one flat measure block (has 'bit' and 'curves')."""
    bit = block.get("bit")
    if p_ref is None:
        p_ref = 2 ** (bit - 0.5) if bit is not None else 2 ** 19.5
    t_by_name = {c["name"]: c.get("T") for c in roster}
    out = {"B1": B1, "bit": bit, "curves": {}}
    for name, cur in block["curves"].items():
        f = cur["fraction"]
        delta = fit_delta(f, B1, p_ref)
        D = math.exp(delta)
        T = t_by_name.get(name)
        rec = {"fraction": f, "delta": delta, "D_eff": D, "torsion": T}
        if T:
            rec["extra"] = D / T
        out["curves"][name] = rec
    return out


def calibrate(measure_json: Path, B1: int, roster, p_ref_by_bit: dict = None) -> dict:
    """Convenience wrapper: calibrate every bit-block in a measure file."""
    data = json.loads(measure_json.read_text(encoding="utf-8"))
    # flat block?
    if "curves" in data and "bit" in data:
        return calibrate_block(data, B1, roster,
                               (p_ref_by_bit or {}).get(data["bit"]))
    out = {}
    for key, block in data.items():
        if not isinstance(block, dict) or "curves" not in block:
            continue
        out[key] = calibrate_block(block, B1, roster,
                                   (p_ref_by_bit or {}).get(block.get("bit")))
    return out


if __name__ == "__main__":
    import curves
    import sys
    mj = TOOL_DIR / "out" / (sys.argv[1] if len(sys.argv) > 1 else "measure_20_256.json")
    B1 = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    cal = calibrate(mj, B1, curves.ROSTER)
    print(json.dumps(cal, indent=2))
