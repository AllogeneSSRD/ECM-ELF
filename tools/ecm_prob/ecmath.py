"""Pure-Python elliptic-curve arithmetic for the ECM empirical stage.

Implements, over a prime field F_p = Z/pZ:

  * plain Edwards (a=1)  x^2 + y^2 = 1 + d x^2 y^2   -- projective (X:Y:Z),
  * Montgomery           b y^2 = x^3 + A x^2 + x     -- XZ differential ladder,
  * short Weierstrass    y^2 = x^3 + 36              -- affine (for param 2).

The "stage-1 hit" test is exactly: compute [s]P (s = lcm(1..B1)) and check it
equals the group identity, which is equivalent to ord(P mod p) being
B1-powersmooth (the paper Section 9 criterion).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------
def modfrac(f, p: int) -> int:
    """Reduce exact rational (num, den) to F_p:  num * den^-1 mod p."""
    num, den = f
    return (num % p) * pow(den % p, -1, p) % p


def primes_upto(n: int) -> list[int]:
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:n + 1:i] = b"\x00" * len(range(i * i, n + 1, i))
    return [i for i in range(2, n + 1) if sieve[i]]


def batch_s(B1: int) -> int:
    """s = lcm(1..B1) = prod_{p<=B1} p^{floor(log_p B1)}."""
    s = 1
    for p in primes_upto(B1):
        pe = p
        while pe * p <= B1:
            pe *= p
        s *= pe
    return s


# ---------------------------------------------------------------------------
# Edwards (a=1) projective group law  -- EFD dbl-2007-bl / add-2007-bl / madd
# ---------------------------------------------------------------------------
def edwards_dbl(X: int, Y: int, Z: int, d: int, p: int):
    B = (X + Y) ** 2 % p
    C = X * X % p
    D = Y * Y % p
    E = C                       # a = 1
    F = (E + D) % p
    H = Z * Z % p
    J = (F - 2 * H) % p
    X3 = (B - C - D) * J % p
    Y3 = F * ((E - D) % p) % p
    Z3 = F * J % p
    return X3, Y3, Z3


def edwards_madd(X1: int, Y1: int, Z1: int, x2: int, y2: int, d: int, p: int):
    """Mixed addition of (X1:Y1:Z1) and affine (x2,y2) (Z2 = 1)."""
    B = Z1 * Z1 % p
    C = X1 * x2 % p
    D = Y1 * y2 % p
    E = d * C % p * D % p
    F = (B - E) % p
    G = (B + E) % p
    X3 = Z1 * F % p * ((X1 + Y1) * (x2 + y2) - C - D) % p
    Y3 = Z1 * G % p * ((D - C) % p) % p      # a = 1
    Z3 = F * G % p
    return X3, Y3, Z3


def edwards_mul(s: int, x1: int, y1: int, d: int, p: int):
    """[s]P for affine P=(x1,y1); returns projective (X:Y:Z)."""
    X, Y, Z = 0, 1, 1                    # identity (0 : 1 : 1)
    for bit in bin(s)[2:]:
        X, Y, Z = edwards_dbl(X, Y, Z, d, p)
        if bit == "1":
            X, Y, Z = edwards_madd(X, Y, Z, x1, y1, d, p)
    return X, Y, Z


def edwards_is_identity(X: int, Y: int, Z: int, p: int) -> bool:
    return (X % p == 0) and (Y % p == Z % p)


def edwards_hits(d_f, x1_f, y1_f, s: int, p: int) -> bool:
    d = modfrac(d_f, p)
    x1 = modfrac(x1_f, p)
    y1 = modfrac(y1_f, p)
    X, Y, Z = edwards_mul(s, x1, y1, d, p)
    return edwards_is_identity(X, Y, Z, p)


# ---------------------------------------------------------------------------
# Montgomery XZ differential ladder  -- EFD ladd-1987-m-3
# ---------------------------------------------------------------------------
def mont_ladd(X2: int, Z2: int, X3: int, Z3: int, X1: int, Z1: int,
              a24: int, p: int):
    """(P2+P3, 2*P2) given P2, P3 and difference P2-P3 = (X1:Z1)."""
    A = (X2 + Z2) % p
    AA = A * A % p
    B = (X2 - Z2) % p
    BB = B * B % p
    E = (AA - BB) % p
    C = (X3 + Z3) % p
    D = (X3 - Z3) % p
    DA = D * A % p
    CB = C * B % p
    X5 = Z1 * ((DA + CB) ** 2 % p) % p      # addition
    Z5 = X1 * ((DA - CB) ** 2 % p) % p
    X4 = AA * BB % p                        # doubling
    Z4 = E * ((BB + a24 * E) % p) % p
    return (X5, Z5), (X4, Z4)


def montgomery_mul(s: int, x0: int, A: int, p: int):
    """[s]P via XZ ladder; P = (x0:1). Returns (X:Z) of [s]P."""
    a24 = (A + 2) * pow(4, -1, p) % p
    x0 = x0 % p
    R0 = (1, 0)          # identity (infinity): Z = 0
    R1 = (x0, 1)         # P
    diff = (x0, 1)       # R1 - R0 = P  (x-coordinate, sign-agnostic)
    for bit in bin(s)[2:]:
        if bit == "0":
            (R1, R0) = mont_ladd(R0[0], R0[1], R1[0], R1[1],
                                 diff[0], diff[1], a24, p)
        else:
            (R0, R1) = mont_ladd(R1[0], R1[1], R0[0], R0[1],
                                 diff[0], diff[1], a24, p)
    return R0


def montgomery_hits(A_f, x0_f, s: int, p: int) -> bool:
    A = modfrac(A_f, p)
    x0 = modfrac(x0_f, p)
    X, Z = montgomery_mul(s, x0, A, p)
    return Z % p == 0


# ---------------------------------------------------------------------------
# Short Weierstrass y^2 = x^3 + 36 (affine), for GMP-ECM param 2.
# ---------------------------------------------------------------------------
def _w_dbl(x: int, y: int, p: int):
    if y % p == 0:
        return None, None
    lam = (3 * x * x) * pow(2 * y % p, -1, p) % p
    x3 = (lam * lam - 2 * x) % p
    y3 = (lam * (x - x3) - y) % p
    return x3, y3


def _w_add(x1, y1, x2, y2, p):
    if x1 is None:
        return x2, y2
    if x2 is None:
        return x1, y1
    if (x1 - x2) % p == 0:
        if (y1 + y2) % p == 0:
            return None, None
        return _w_dbl(x1, y1, p)
    lam = (y2 - y1) * pow((x2 - x1) % p, -1, p) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return x3, y3


def param2_A_mod_p(sigma: int, p: int) -> int:
    """GMP-ECM param 2: A = -(3*x3^4 + 6*x3^2 - 1) / (4*x3^3), where
    (x3, y3) = sigma*(-3:3:1) on y^2 = x^3 + 36."""
    x, y = -3 % p, 3 % p
    rx, ry = None, None
    k = sigma
    bx, by = x, y
    while k > 0:
        if k & 1:
            rx, ry = _w_add(rx, ry, bx, by, p)
        bx, by = _w_dbl(bx, by, p)
        k >>= 1
    if rx is None:
        raise ValueError("param2: sigma*(-3,3) is the point at infinity")
    # x3 = (3x + y + 6) / (2(y - 3))
    x3 = (3 * rx + ry + 6) * pow(2 * ((ry - 3) % p) % p, -1, p) % p
    # A = -(3*x3^4 + 6*x3^2 - 1) / (4*x3^3)
    x2 = x3 * x3 % p
    x3c = x2 * x3 % p
    num = -(3 * x2 % p * x2 % p + 6 * x2 % p - 1) % p
    den = 4 * x3c % p
    return num * pow(den, -1, p) % p


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def curve_hits(curve: dict, p: int, s: int) -> bool:
    """Stage-1 hit for one curve at prime p.

    A ValueError (non-invertible denominator) means the curve is degenerate or
    its base point is singular for this p (e.g. a Suyama sigma with p | sigma);
    such primes are treated as "not found" under the strict B1-powersmooth
    criterion (GMP-ECM would reject this sigma and pick another).
    """
    form = curve["form"]
    try:
        if form == "edwards":
            return edwards_hits(curve["d"], curve["x1"], curve["y1"], s, p)
        if form == "montgomery":
            return montgomery_hits(curve["A"], curve["x0"], s, p)
        if form == "montgomery_param2":
            A = param2_A_mod_p(curve["sigma"], p)
            return montgomery_hits((A, 1), curve["x0"], s, p)
        if form == "pminus1":
            return s % (p - 1) == 0
        if form == "pplus1":
            return s % (p + 1) == 0
    except ValueError:
        return False
    raise ValueError(f"unknown form: {form}")


if __name__ == "__main__":
    # Sanity: batch_s(256) bit length, and a tiny identity check.
    s = batch_s(256)
    print("s = lcm(1..256), bits =", s.bit_length())
    print("edwards identity:", edwards_is_identity(0, 1, 1, 5))
