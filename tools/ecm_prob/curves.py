"""Curve roster for the ECM parameterization probability tool.

Each entry is an exact-rational curve over Q (parameters as (num, den) integer
pairs), reduced mod p at measurement time.  Forms:

  * ``montgomery``  -- b*y^2 = x^3 + A*x^2 + x, base point (x0 : 1) in XZ.
  * ``edwards``     -- x^2 + y^2 = 1 + d*x^2*y^2  (plain Edwards, a=1),
                       base point P = (x1, y1).
  * ``pminus1``     -- Pollard p-1 (multiplicative group), model: (p-1) | s.
  * ``pplus1``      -- Williams p+1 (quadratic twist), model: (p+1) | s.
"""

from __future__ import annotations


def suyama_params(sigma: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """GMP-ECM Suyama parametrization (param 0).

    u = sigma^2 - 5, v = 4*sigma
    A = (v-u)^3*(3u+v)/(4*u^3*v) - 2,   x0 = u^3/v^3
    Returns (A as (num,den), x0 as (num,den)).
    """
    u = sigma * sigma - 5
    v = 4 * sigma
    numA = (v - u) ** 3 * (3 * u + v)
    denA = 4 * u ** 3 * v
    numA = numA - 2 * denA          # A = numA/denA - 2
    return (numA, denA), (u ** 3, v ** 3)


def batch_square_params(sigma: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """GMP-ECM param 1 (batch square): d = sigma^2/2^64, A = 4d-2, x0 = 2."""
    d = (sigma * sigma, 1 << 64)
    numA = 4 * d[0]
    denA = d[1]
    numA = numA - 2 * denA           # A = 4d - 2
    return (numA, denA), (2, 1)


def batch_32bit_params(sigma: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """GMP-ECM param 3 (batch 32-bit d): d = sigma/2^32, A = 4d-2, x0 = 2."""
    d = (sigma, 1 << 32)
    numA = 4 * d[0]
    denA = d[1]
    numA = numA - 2 * denA           # A = 4d - 2
    return (numA, denA), (2, 1)


# Canonical sigma values.  sigma=10 is the paper's Suyama reference curve
# (paper Section 9.2: "GMP-ECM with a typical Suyama curve, sigma = 10").
SIGMA_REF = 10
SIGMA_RANDOM = 1707370477          # from tools/FindGroupOrder3_example.gp


def build_roster() -> list[dict]:
    """Return the full curve roster (Montgomery + Edwards + p-1/p+1)."""
    curves: list[dict] = []

    # ---- Montgomery (GMP-ECM basic parametrizations) ----------------------
    A, x0 = suyama_params(SIGMA_REF)
    curves.append(dict(name="suyama_s10", form="montgomery", torsion="Z/12", T=12,
                       sigma=SIGMA_REF, A=A, x0=x0))

    A, x0 = suyama_params(SIGMA_RANDOM)
    curves.append(dict(name="suyama_srand", form="montgomery", torsion="Z/12", T=12,
                       sigma=SIGMA_RANDOM, A=A, x0=x0))

    A, x0 = batch_square_params(SIGMA_REF)
    curves.append(dict(name="param1_s10", form="montgomery", torsion="Z/4", T=4,
                       sigma=SIGMA_REF, A=A, x0=x0))

    # param 2 (6-torsion): A is derived from sigma*(-3:3:1) on y^2=x^3+36 mod p,
    # so it is computed per-prime in ecmath.curve_hits.
    curves.append(dict(name="param2_s10", form="montgomery_param2",
                       torsion="Z/6", T=6, sigma=SIGMA_REF, A=None, x0=(2, 1)))

    A, x0 = batch_32bit_params(SIGMA_REF)
    curves.append(dict(name="param3_s10", form="montgomery", torsion="Z/4", T=4,
                       sigma=SIGMA_REF, A=A, x0=x0))

    # ---- Edwards (paper Section 9.1, four curves) -------------------------
    # x^2 + y^2 = 1 + d*x^2*y^2, base point (x1, y1).
    curves.append(dict(name="edwards_Z4", form="edwards", torsion="Z/4", T=4,
                       d=(1, 3), x1=(2, 1), y1=(3, 1)))
    curves.append(dict(name="edwards_Z2xZ4", form="edwards", torsion="Z/2xZ/4", T=8,
                       d=(1, 36), x1=(8, 1), y1=(9, 1)))
    curves.append(dict(name="edwards_Z12", form="edwards", torsion="Z/12", T=12,
                       d=(-24167, 25), x1=(5, 23), y1=(-1, 7)))
    curves.append(dict(name="edwards_Z2xZ8", form="edwards", torsion="Z/2xZ/8", T=16,
                       d=(25921, 83521), x1=(13, 7), y1=(289, 49)))

    # ---- Baselines --------------------------------------------------------
    curves.append(dict(name="pm1", form="pminus1", torsion="Z/2", T=2))
    curves.append(dict(name="pp1", form="pplus1", torsion="Z/2", T=2))

    return curves


ROSTER = build_roster()


if __name__ == "__main__":
    for c in ROSTER:
        print(c["name"], c["form"], c["torsion"])
