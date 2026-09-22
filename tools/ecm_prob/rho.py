"""Port of GMP-ECM's rho.c (Dickman rho / ECM success probability model).

Faithful Python re-implementation of the numerical machinery in
`.refactor/ecm/rho.c`, so that `ecmprob(...)` reproduces the "expected
number of curves" that GMP-ECM prints with `-v`.

Stage-1-only probability is `dickmanlocal(alpha, effN)` with
`alpha = log(effN)/log(B1)`, `effN = N/exp(delta)`; see `stage1_prob`.

The full `ecmprob` (stage 1 + stage 2 + Brent-Suyama) is also ported for
completeness, but note the tool's default comparisons use stage-1 only
(see `stage1_prob`).

Differences from rho.c that are *semantically* equivalent:
  * `isprime` (which rho.c implements via a 667-byte `primemap` table
    valid only up to 20010) is replaced by a proper sieve of Eratosthenes.
  * `gcd` / `eulerphi` are plain Python.

Constants and formulas are verbatim from rho.c (Kruppa's thesis model).
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Constants (from rho.c)
# ---------------------------------------------------------------------------
ECM_EXTRA_SMOOTHNESS = 3.134  # exp(3.134) ~= 22.97 effective divisor (Suyama)

M_PI_SQR = 9.869604401089358619      # pi^2
M_PI_SQR_6 = 1.644934066848226436    # pi^2 / 6
M_EULER = 0.577215664901532861       # Euler-Mascheroni gamma
M_EULER_1 = 0.422784335098467139     # 1 - gamma

# Smoothness corrections for the batch parametrizations (ecm.c:46-56).
# These are the *additional* factors (relative to Suyama) by which the
# effective group-order size is scaled.  See docs/ECM_PARAMETERIZATION_ANALYSIS.md
EXTRA_SMOOTHNESS_SQUARE = 0.416384512396064      # param 1 (d square)
EXTRA_SMOOTHNESS_32BITS_D = 0.330484606500389    # param 3 (d random 32-bit)


# ---------------------------------------------------------------------------
# Dilogarithm (Li_2), via series for |z| <= 0.5 (rho.c dilog_series / dilog)
# ---------------------------------------------------------------------------
def _dilog_series(z: float) -> float:
    r = 0.0
    zk = z
    k2 = 1
    for k in range(1, 45):
        r += zk / k2
        k2 += 2 * k + 1
        zk *= z
    return r


def _dilog(x: float) -> float:
    # rho.c dilog(x), assumes x <= -1.0
    assert x <= -1.0
    if x <= -2.0:
        return (-_dilog_series(1.0 / x) - M_PI_SQR_6
                - 0.5 * math.log(-1.0 / x) * math.log(-1.0 / x))
    log1x = math.log(1.0 - x)
    return (_dilog_series(1.0 / (1.0 - x))
            - M_PI_SQR_6 + log1x * (0.5 * log1x - math.log(-x)))


# ---------------------------------------------------------------------------
# Dickman rho, exact for x <= 3
# ---------------------------------------------------------------------------
def rhoexact(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x <= 1.0:
        return 1.0
    if x <= 2.0:
        return 1.0 - math.log(x)
    # 2 < x <= 3
    return (1.0 - math.log(x) * (1.0 - math.log(x - 1.0))
            + _dilog(1.0 - x) + 0.5 * M_PI_SQR_6)


# ---------------------------------------------------------------------------
# rho table (rhoinit / dickmanrho / dickmanlocal / dickmanlocal_i)
# ---------------------------------------------------------------------------
class _RhoTable:
    def __init__(self, invh: int = 256, tablemax: int = 10):
        self.invh = invh
        self.h = 1.0 / invh
        self.tablemax = tablemax
        n = invh * tablemax
        self.rhotable = [0.0] * n

        lim = (3 if 3 < tablemax else tablemax) * invh
        for i in range(lim):
            self.rhotable[i] = rhoexact(i * self.h)

        for i in range(3 * invh, tablemax * invh):
            # rho(i*h) = rho((i-4)*h) - int_{(i-4)h}^{i*h} rho(x-1)/x dx
            # 4-interval closed Newton-Cotes (Boole) rule; h cancels.
            v = self.rhotable[i - 4] - 2.0 / 45.0 * (
                7.0 * self.rhotable[i - invh - 4] / (i - 4)
                + 32.0 * self.rhotable[i - invh - 3] / (i - 3)
                + 12.0 * self.rhotable[i - invh - 2] / (i - 2)
                + 32.0 * self.rhotable[i - invh - 1] / (i - 1)
                + 7.0 * self.rhotable[i - invh] / i)
            self.rhotable[i] = v if v >= 0.0 else 0.0

    def rho(self, alpha: float) -> float:
        if alpha <= 3.0:
            return rhoexact(alpha)
        a = int(math.floor(alpha * self.invh))
        rho1 = self.rhotable[a]
        rho2 = self.rhotable[a + 1] if (a + 1) < self.tablemax * self.invh else 0.0
        return rho1 + (rho2 - rho1) * (alpha * self.invh - a)

    def local(self, alpha: float, x: float) -> float:
        # rho.c dickmanlocal (Kruppa thesis eq 5.6)
        if alpha <= 1.0:
            return rhoexact(alpha)
        if alpha < self.tablemax:
            return self.rho(alpha) - M_EULER * self.rho(alpha - 1.0) / math.log(x)
        return 0.0

    def local_i(self, ai: int, x: float) -> float:
        # rho.c dickmanlocal_i (integer-indexed version)
        if ai <= 0:
            return 0.0
        if ai <= self.invh:
            return 1.0
        if ai <= 2 * self.invh and ai < self.tablemax * self.invh:
            return self.rhotable[ai] - M_EULER / math.log(x)
        if ai < self.tablemax * self.invh:
            logx = math.log(x)
            return (self.rhotable[ai]
                    - (M_EULER * self.rhotable[ai - self.invh]
                       + M_EULER_1 * self.rhotable[ai - 2 * self.invh] / logx) / logx)
        return 0.0


# Singleton table used by the module-level functions below.
_table = _RhoTable(256, 10)


# ---------------------------------------------------------------------------
# Stage-2 helpers (dickmanmu) -- ported for completeness
# ---------------------------------------------------------------------------
def _primes_upto(n: int) -> list[int]:
    """Sieve of Eratosthenes: primes <= n (replaces rho.c primemap/isprime)."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:n + 1:i] = b"\x00" * len(range(i * i, n + 1, i))
    return [i for i in range(2, n + 1) if sieve[i]]


_PRIME_CACHE: dict[int, list[int]] = {}


def _primes_between(lo: int, hi: int) -> list[int]:
    if hi not in _PRIME_CACHE:
        _PRIME_CACHE[hi] = _primes_upto(hi)
    return [p for p in _PRIME_CACHE[hi] if p > lo]


def _dickmanmu_sum(B1: float, B2: float, x: float) -> float:
    # rho.c dickmanmu_sum (eq 5.10, Kruppa thesis): sum over primes in (B1,B2]
    s = 0.0
    inv_logB1 = 1.0 / math.log(B1)
    logx = math.log(x)
    for p in _primes_between(int(B1), int(B2)):
        s += _table.local((logx - math.log(p)) * inv_logB1, x / p) / p
    return s


def _dickmanmu(alpha: float, beta: float, x: float) -> float:
    # rho.c dickmanmu
    invh = _table.invh
    h = _table.h
    tablemax = _table.tablemax

    ai = int(math.ceil((alpha - beta) * invh))
    if ai > tablemax * invh:
        ai = tablemax * invh
    a = ai * h
    bi = int(math.floor((alpha - 1.0) * invh))
    if bi > tablemax * invh:
        bi = tablemax * invh
    b = bi * h

    total = 0.0
    for i in range(ai + 1, bi):
        total += _table.local_i(i, x) / (alpha - i * h)
    total += 0.5 * _table.local_i(ai, x) / (alpha - a)
    total += 0.5 * _table.local_i(bi, x) / (alpha - b)
    total *= h
    total += (a - alpha + beta) * 0.5 * (
        _table.local_i(ai, x) / (alpha - a) + _table.local(alpha - beta, x) / beta)
    total += (alpha - 1.0 - b) * 0.5 * (
        _table.local(alpha - 1.0, x) + _table.local_i(bi, x) / (alpha - b))
    return total


# ---------------------------------------------------------------------------
# Brent-Suyama (ported for completeness; not used in stage-1 mode)
# ---------------------------------------------------------------------------
def _gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def _eulerphi(n: int) -> int:
    phi = n
    p = 2
    m = n
    while p * p <= m:
        if m % p == 0:
            phi -= phi // p
            while m % p == 0:
                m //= p
        p += 1 if p == 2 else 2
    if m > 1:
        phi -= phi // m
    return phi


def _brentsuyama(B1: float, B2: float, N: float, nr: float) -> float:
    alpha = math.log(N) / math.log(B1)
    beta = math.log(B2) / math.log(B1)
    invh = _table.invh
    h = _table.h
    ai = int(math.floor((alpha - beta) * invh))
    if ai > _table.tablemax * invh:
        ai = _table.tablemax * invh
    a = ai * h
    total = 0.0
    for i in range(1, ai):
        total += (_table.local_i(i, N) / (alpha - i * h)
                  * (1 - math.exp(-nr * B1 ** (-alpha + i * h))))
    total += 0.5 * (1 - math.exp(-nr / B1 ** alpha))
    total += (0.5 * _table.local_i(ai, N) / (alpha - a)
              * (1 - math.exp(-nr * B1 ** (-alpha + a))))
    total *= h
    total += 0.5 * (alpha - beta - a) * (
        _table.local_i(ai, N) / (alpha - a) + _table.local(alpha - beta, N) / beta)
    return total


def _brsudickson(B1, B2, N, nr, S):
    total = 0.0
    f = _eulerphi(S) / 2
    for i in range(1, S // 2 + 1):
        if _gcd(i, S) == 1:
            total += _brentsuyama(B1, B2, N,
                                  nr * (_gcd(i - 1, S) + _gcd(i + 1, S) - 4) / 2)
    return total / f


def _brsupower(B1, B2, N, nr, S):
    total = 0.0
    f = _eulerphi(S)
    for i in range(1, S):
        if _gcd(i, S) == 1:
            total += _brentsuyama(B1, B2, N, nr * (_gcd(i - 1, S) - 2))
    return total / f


# ---------------------------------------------------------------------------
# Top-level probability (rho.c prob / ecmprob / pm1prob)
# ---------------------------------------------------------------------------
SUMHOLD = 20000.0


def prob(B1: float, B2: float, N: float, nr: float, S: int, delta: float) -> float:
    """rho.c prob(): probability that a number near N is B1,B2-smooth,
    with effective size N/exp(delta)."""
    effN = N / math.exp(delta)
    if effN <= B1:
        return 1.0
    if B1 < 2.0 or N <= 1.0:
        return 0.0

    alpha = math.log(effN) / math.log(B1)
    stage1 = _table.local(alpha, effN)

    stage2 = 0.0
    if B2 > B1:
        if B1 < SUMHOLD:
            stage2 += _dickmanmu_sum(B1, min(B2, SUMHOLD), effN)
            beta = math.log(B2) / math.log(min(B2, SUMHOLD))
        else:
            beta = math.log(B2) / math.log(B1)
        if beta > 1.0:
            stage2 += _dickmanmu(alpha, beta, effN)

    brsu = 0.0
    if S < -1:
        brsu = _brsudickson(B1, B2, effN, nr, -S * 2)
    if S > 1:
        brsu = _brsupower(B1, B2, effN, nr, S * 2)

    total = stage1 + stage2 + brsu
    return total if total > 0.0 else 0.0


def ecmprob(B1: float, B2: float, N: float, nr: float, S: int) -> float:
    return prob(B1, B2, N, nr, S, ECM_EXTRA_SMOOTHNESS)


def stage1_prob(B1: float, N: float, delta: float = ECM_EXTRA_SMOOTHNESS) -> float:
    """Stage-1-only success probability: the group order ~ N is as likely
    B1-smooth as a random integer of size N/exp(delta).

    Equivalent to rho.c `prob(B1, B1, N, 0, 0, delta)` (stage2/brsu vanish).
    """
    effN = N / math.exp(delta)
    if effN <= B1:
        return 1.0
    if B1 < 2.0 or N <= 1.0:
        return 0.0
    return _table.local(math.log(effN) / math.log(B1), effN)


def rho(u: float) -> float:
    """Standard Dickman rho function (used by the paper's 'rho(u)' estimate)."""
    return _table.rho(u)


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Sanity checks against known values.
    print("rho(1)      =", rho(1.0), "(expect 1.0)")
    print("rho(2)      =", rho(2.0), "(expect 1 - ln 2 =", 1 - math.log(2), ")")
    u12 = math.log(2 ** 20 / 12) / math.log(256)
    print("u(12)       =", u12)
    print("rho(u12)    =", rho(u12), "(paper: 28.1894%)")
    print("u^-u        =", u12 ** -u12, "(paper: 22.8824%)")
    # GMP-ECM-style expected curves for a 35-digit factor, B1=1e6, stage-1 only.
    N = 10 ** 34.5
    p1 = stage1_prob(1e6, N)
    print("stage1_prob(1e6, 10^34.5) =", p1, "-> curves ~", 1 / p1)
