"""The five naive success-probability estimates from Bernstein et al.,
"ECM using Edwards curves", Section 9.3 (reviewed/invalidated in Section 9.4).

For a prime range [L, R] and a curve guaranteeing t | #E(F_p), the five
estimates for Pr[uniform prime p in [L,R] has B1-powersmooth #E(F_p)] are:

  1. powersmooth probability of a uniform element of  tZ ∩ [L, R]
  2. powersmooth probability of a uniform element of  tZ ∩ [1, R]
  3. powersmooth probability of a uniform element of  Z  ∩ [1, R/t]
  4. rho(u),  u = log(R/t) / log B1   (Dickman rho)
  5. u^-u,   same u

"n is B1-powersmooth" means n divides s = lcm(1..B1); we test s % n == 0
(exact for the small ranges used here).

Golden targets (paper Section 9.4, L=2^19, R=2^20, B1=256):
  t=12: 23.6067, 30.0317, 30.7652, 28.1894, 22.8824  (%)
  t=16: 24.8192, 31.3019, 33.3694, 30.6853, 25.0000
  t=8 : 20.5777, 26.4328, 27.3689, 24.9832, 20.1540
  t=4 : 16.8006, 21.8632, 22.2511, 20.2442, 16.1283
"""

from __future__ import annotations

import math

import rho as rho_mod


def is_powersmooth(n: int, s: int) -> bool:
    """n is B1-powersmooth  <=>  n | s = lcm(1..B1)."""
    return n != 0 and s % n == 0


def powersmooth_tZ_interval(L: int, R: int, t: int, s: int) -> float:
    """Estimate 1: Pr[uniform in tZ ∩ [L,R] is B1-powersmooth]."""
    lo = ((L + t - 1) // t) * t          # first multiple of t >= L
    hits = 0
    total = 0
    for n in range(lo, R + 1, t):
        total += 1
        if is_powersmooth(n, s):
            hits += 1
    return hits / total if total else 0.0


def powersmooth_tZ_1R(R: int, t: int, s: int) -> float:
    """Estimate 2: Pr[uniform in tZ ∩ [1,R] is B1-powersmooth]."""
    hits = 0
    total = 0
    for n in range(t, R + 1, t):
        total += 1
        if is_powersmooth(n, s):
            hits += 1
    return hits / total if total else 0.0


def powersmooth_Z_1R_over_t(R: int, t: int, s: int) -> float:
    """Estimate 3: Pr[uniform in Z ∩ [1, R/t] is B1-powersmooth]."""
    M = R // t
    hits = sum(1 for n in range(1, M + 1) if is_powersmooth(n, s))
    return hits / M if M else 0.0


def rho_estimate(R: int, t: int, B1: int) -> float:
    """Estimate 4: rho(u), u = log(R/t)/log B1."""
    u = math.log(R / t) / math.log(B1)
    return rho_mod.rho(u)


def uu_estimate(R: int, t: int, B1: int) -> float:
    """Estimate 5: u^-u, u = log(R/t)/log B1."""
    u = math.log(R / t) / math.log(B1)
    return u ** -u


def all_estimates(L: int, R: int, t: int, B1: int, s: int) -> list[float]:
    return [
        powersmooth_tZ_interval(L, R, t, s),
        powersmooth_tZ_1R(R, t, s),
        powersmooth_Z_1R_over_t(R, t, s),
        rho_estimate(R, t, B1),
        uu_estimate(R, t, B1),
    ]


GOLDEN = {
    12: [23.6067, 30.0317, 30.7652, 28.1894, 22.8824],
    16: [24.8192, 31.3019, 33.3694, 30.6853, 25.0000],
    8:  [20.5777, 26.4328, 27.3689, 24.9832, 20.1540],
    4:  [16.8006, 21.8632, 22.2511, 20.2442, 16.1283],
}


if __name__ == "__main__":
    from ecmath import batch_s
    B1 = 256
    L, R = 1 << 19, 1 << 20
    s = batch_s(B1)
    names = ["pow tZ[L,R]", "pow tZ[1,R]", "pow Z[1,R/t]", "rho(u)", "u^-u"]
    for t in (12, 16, 8, 4):
        est = all_estimates(L, R, t, B1, s)
        gold = GOLDEN[t]
        print(f"t={t}:")
        for nm, e, g in zip(names, est, gold):
            ok = abs(e * 100 - g) < 5e-4
            print(f"   {nm:14s} {e*100:.4f}%  (paper {g:.4f}%)  {'OK' if ok else 'DIFF'}")
