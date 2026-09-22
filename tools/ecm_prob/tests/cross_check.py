"""Cross-check ecmath's Edwards point order against PARI/GP for a few primes.

Edwards x^2+y^2=1+d*x^2*y^2  ->  Montgomery B v^2 = u^3 + A u^2 + u
  A = 2(1+d)/(1-d), B = 4/(1-d),  u=(1+y)/(1-y), v=u/x
Montgomery -> Weierstrass y^2 = x^3 + a*x + b
  a = (3 - A^2)/(3 B^2), b = (2 A^3 - 9 A)/(27 B^3),  X = u/B + A/(3B), Y = v/B
"""
from __future__ import annotations

import subprocess
import sys

sys.path.insert(0, r"D:\code\MPA-OpenCl\tools\ecm_prob")
import ecmath  # noqa: E402

GP = r"D:\AppData\Pari64-2-17-3\gp.exe"

# Z/12 curve: d = -24167/25, P = (5/23, -1/7)
D = (-24167, 25)
PX = (5, 23)
PY = (-1, 7)


def edwards_to_weierstrass(d, x1, y1, p):
    d = ecmath.modfrac(d, p)
    x1 = ecmath.modfrac(x1, p)
    y1 = ecmath.modfrac(y1, p)
    A = 2 * (1 + d) * pow(1 - d, -1, p) % p
    B = 4 * pow(1 - d, -1, p) % p
    u = (1 + y1) * pow(1 - y1, -1, p) % p
    v = u * pow(x1, -1, p) % p
    a = (3 - A * A) * pow(3 * B * B, -1, p) % p
    b = (2 * A ** 3 - 9 * A) * pow(27 * B ** 3, -1, p) % p
    X = (u * pow(B, -1, p) + A * pow(3 * B, -1, p)) % p
    Y = v * pow(B, -1, p) % p
    return a, b, X, Y


def gp_order(p):
    a, b, X, Y = edwards_to_weierstrass(D, PX, PY, p)
    script = f"E=ellinit([Mod({a},{p}),Mod({b},{p})]);P=[Mod({X},{p}),Mod({Y},{p})];print(ellorder(E,P));print(factor(ellorder(E,P)));quit"
    r = subprocess.run([GP, "-q"], input=script, capture_output=True, text=True)
    return r.stdout.strip()


def main():
    s = ecmath.batch_s(256)
    for p in [524287, 524309, 1048573, 1048583]:
        a, b, X, Y = edwards_to_weierstrass(D, PX, PY, p)
        # my Edwards: does [s]P = identity?
        d = ecmath.modfrac(D, p)
        x1 = ecmath.modfrac(PX, p)
        y1 = ecmath.modfrac(PY, p)
        Xm, Ym, Zm = ecmath.edwards_mul(s, x1, y1, d, p)
        mine_id = ecmath.edwards_is_identity(Xm, Ym, Zm, p)
        gpo = gp_order(p)
        print(f"p={p}: my [s]P identity={mine_id}")
        print(f"    gp: {gpo}")
        print()


if __name__ == "__main__":
    main()
