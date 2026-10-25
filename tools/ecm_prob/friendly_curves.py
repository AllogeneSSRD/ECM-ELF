"""Exact rational ECM constructions for the GKL 2017 family-(iv) experiment.

These are research curve identities, not production Edwards sigma aliases.
The production Atkin-Morain construction remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from math import isqrt

EXPERIMENT_VERSION = "1.0.0"


def auxiliary_add(p, q, a2=0, a4=0):
    """Affine addition over Q on y²=x³+a2*x²+a4*x+a6."""
    if p is None:
        return q
    if q is None:
        return p
    x, y = p
    u, v = q
    if x == u:
        if y + v == 0:
            return None
        slope = (3*x*x + 2*a2*x + a4) / (2*y)
    else:
        slope = (v-y) / (u-x)
    xx = slope*slope - a2 - x - u
    return xx, slope*(x-xx)-y


def auxiliary_mul(m, point, a2=0, a4=0):
    if not isinstance(m, int) or m < 1:
        raise ValueError("seed must be a positive integer")
    result = None
    while m:
        if m & 1:
            result = auxiliary_add(result, point, a2, a4)
        m >>= 1
        if m:
            point = auxiliary_add(point, point, a2, a4)
    return result


def is_square(q):
    q = F(q)
    return q >= 0 and isqrt(q.numerator)**2 == q.numerator and isqrt(q.denominator)**2 == q.denominator


@dataclass(frozen=True)
class Curve:
    family: str
    seed: int
    a: int
    d: F
    x: F
    y: F
    rational_torsion_order: int
    rational_torsion_exponent: int
    e: F | None = None
    t: F | None = None

    def __post_init__(self):
        if self.d in (0, self.a) or self.a*self.x**2+self.y**2 != 1+self.d*self.x**2*self.y**2:
            raise ValueError("singular curve or off-curve starting point")

    def weierstrass(self):
        """Keep the Montgomery B twist: Y²=X³+A*B*X²+B²*X."""
        aa = 2*(self.a+self.d)/(self.a-self.d)
        bb = 4/(self.a-self.d)
        if self.y == 1:
            point = None
        elif self.y == -1:
            point = (F(0), F(0))
        else:
            u = (1+self.y)/(1-self.y)
            point = (bb*u, bb*bb*u/self.x)
        return (F(0), aa*bb, F(0), bb*bb, F(0)), point

    def record(self):
        pair = lambda q: [str(q.numerator), str(q.denominator)]
        return dict(family=self.family, seed=self.seed, a=self.a,
                    d=pair(self.d), x=pair(self.x), y=pair(self.y),
                    e=pair(self.e) if self.e is not None else None,
                    t=pair(self.t) if self.t is not None else None,
                    rational_torsion_order=self.rational_torsion_order,
                    rational_torsion_exponent=self.rational_torsion_exponent)


def from_t(t, seed=0, family="gkl_generic"):
    """GKL Table 1, row 4; reject undefined/singular values algebraically."""
    t = F(t)
    e = (t*t+4*t)/(t*t-4)
    x = (2*t**3+2*t*t-8*t-8)/(t**4+6*t**3+12*t*t+16*t)
    y = ((t**6+6*t**5+10*t**4-16*t**3-48*t*t-32*t-32) /
         (t**6+6*t**5+10*t**4+16*t**3+48*t*t+64*t))
    return Curve(family, seed, -1, -e**4, x, y, 8, 4, e, t)


def family_iv(m):
    """GKL Table 2: [m](5,8) on y²=x³−x²−9x+9; no seed remapping."""
    q = auxiliary_mul(m, (F(5), F(8)), a2=-1, a4=-9)
    if q is None:
        raise ValueError("auxiliary point at infinity")
    s, r = q
    result = from_t((4*s+4)/(r-4), m, "gkl_iv")
    if not is_square(result.e**2+1):
        raise ValueError("family-(iv) square condition failed")
    return result


def atkin_morain(sigma):
    """Same original curve and point as ecm_edwards_curve.h, without ×48."""
    q = auxiliary_mul(sigma, (F(12), F(40)), a4=-8)
    if q is None:
        raise ValueError("auxiliary point at infinity")
    s, t = q
    alpha = (s-9)/(t+s+16)
    beta = 2*alpha*(4*alpha+1)/(8*alpha*alpha-1)
    h = 2*beta-1
    d = (2*h*h-1)/h**4
    x = h*(4*beta-3)/(6*beta-5)
    y = h*(t*t+50*t-2*s**3+27*s*s-104)/((t+3*s-2)*(t+s+16))
    return Curve("atkin_morain", sigma, 1, d, x, y, 16, 8)


def paper_c6():
    x, y = F(825, 2752), F(1521, 1504)
    d = (-x*x+y*y-1)/(x*x*y*y)
    return Curve("paper_c6", 1, -1, d, x, y, 6, 6)


def legacy_z2x8():
    return Curve("legacy_fixed_z2x8", 1, 1, F(25921, 83521), F(13, 7), F(289, 49), 16, 8)
