#!/usr/bin/env python3
"""suyama_mont_ref.py -- reference recipe + oracle harness for Suyama-sigma Montgomery
ECM stage 1 (Prime95 sigma_type=1).  Pure Python (stdlib only), no build needed.

WHY THIS EXISTS
  The C/C++/SIMD implementation must agree *byte for byte* with a reference before it can
  be trusted; this file is that reference, and it also generates test vectors.  Every
  convention below was pinned by experiment during design (see
  docs/ECM_Montgomery_STAGE1.md 3.1), not guessed:

    curve      : u = sigma^2 - 5, v = 4*sigma
                 A = (v-u)^3 (3u+v) / (4 u^3 v) - 2   (mod N)      [= A+2 -> An/Ad]
    start      : (X:Z) = (u^3 : v^3)
    exponent   : s = lcm(1..B1)                 <-- gmp-ecm param0 convention
                 (Prime95's choose12 uses 12*lcm(1..B1); expose it with --torsion 12)
    stage-1    : [s]P by x-only Montgomery ladder; hit <=> gcd(Z, N) > 1
    saved X    : normalised Montgomery x = X/Z (Z=1)
    curve plot : k = (A+2)/4  (a24 in the RFC-7748 sense, used by the doubling)

VALIDATION
    python tools/stat/suyama_mont_ref.py --check-gmp-ecm
  runs gmp-ecm (if present) for several (sigma, B1) and asserts the saved X matches.

EXAMPLES
    python tools/stat/suyama_mont_ref.py --sigma 12345 --B1 1000 --N 2^1277-1 --save-line
    python tools/stat/suyama_mont_ref.py --sigma 20260922 --B1 5000 --N 2^3001-1
    python tools/stat/suyama_mont_ref.py --check-gmp-ecm
"""
import argparse
import math
import os
import subprocess
import sys

DEFAULT_ECM = r"D:\code\GIMPS\gmp-ecm\ecm-7.0.5-znver3\ecm.exe"


def parse_int(txt):
    """accept '2^1277-1' style expressions as well as plain decimal.

    NOTE: '^' must be translated to Python's '**' -- in Python '^' is XOR, so
    eval("2^1277-1") silently yields a tiny number instead of the expression.
    """
    txt = txt.strip().replace(" ", "")
    if any(c in txt for c in "^*-+"):
        return int(eval(txt.replace("^", "**"), {"__builtins__": {}}, {}))
    return int(txt)


def lcm_1_to(B1):
    """lcm(1..B1) via a sieve of prime powers"""
    s = 1
    sieve = bytearray(B1 + 1)
    for p in range(2, B1 + 1):
        if sieve[p]:
            continue
        for q in range(p * p, B1 + 1, p):
            sieve[q] = 1
        e = p
        while e <= B1 // p:
            e *= p
        s *= e
    return s


def suyama_curve(sigma, N):
    """sigma -> (A, a24, X0, Z0) with a24 = (A+2)/4 (doubling constant)"""
    u = sigma * sigma - 5
    v = 4 * sigma
    t = v - u
    A = (t ** 3 * (3 * u + v)) * pow(4 * u ** 3 * v % N, -1, N) % N
    A = (A - 2) % N
    a24 = (A + 2) * pow(4, -1, N) % N
    return A, a24, u ** 3 % N, v ** 3 % N


def xdbl(X, Z, a24, N):
    """Montgomery doubling: (X:Z) -> (X2:Z2), a24 = (A+2)/4"""
    t0 = (X + Z) % N
    t1 = (X - Z) % N
    t0 = t0 * t0 % N
    t1 = t1 * t1 % N
    t2 = (t0 - t1) % N
    return t0 * t1 % N, t2 * (t1 + a24 * t2) % N


def xadd(Xa, Za, Xb, Zb, xdiff, N):
    """Differential addition of (Xa:Za), (Xb:Zb) whose difference has AFFINE x xdiff.

    NOTE (this cost me ~200 wrong verdicts once): xdiff must be the *affine* x of the
    difference point, i.e. Xdiff/Zdiff -- not the raw X of a point with Z != 1.
    """
    t0 = (Xa + Za) % N
    t1 = (Xa - Za) % N
    t2 = (Xb + Zb) % N
    t3 = (Xb - Zb) % N
    t4 = t0 * t3 % N
    t5 = t1 * t2 % N
    return (t4 + t5) ** 2 % N, (t4 - t5) ** 2 % N * xdiff % N


def ladder(s, X0, Z0, a24, N):
    """[s]*(X0:Z0) -> (X:Z); start (X0:Z0) stays the differential difference (affine)."""
    xdiff = X0 * pow(Z0, -1, N) % N if Z0 % N else 0
    R0X, R0Z = X0, Z0
    R1X, R1Z = xdbl(X0, Z0, a24, N)
    for bit in bin(s)[3:]:
        if bit == "1":
            R0X, R0Z = xadd(R1X, R1Z, R0X, R0Z, xdiff, N)
            R1X, R1Z = xdbl(R1X, R1Z, a24, N)
        else:
            R1X, R1Z = xadd(R0X, R0Z, R1X, R1Z, xdiff, N)
            R0X, R0Z = xdbl(R0X, R0Z, a24, N)
    return R0X, R0Z


def stage1(sigma, B1, N, torsion=1):
    """returns dict with A, s, X, Z, gcd(Z,N) and the normalised x"""
    A, a24, X0, Z0 = suyama_curve(sigma, N)
    s = torsion * lcm_1_to(B1)
    X, Z = ladder(s, X0, Z0, a24, N)
    g = math.gcd(Z % N, N)
    x = X * pow(Z, -1, N) % N if Z % N else 0
    return dict(A=A, a24=a24, s=s, X=X, Z=Z, x=x, gcd=g, s_bits=s.bit_length())


def save_line(sigma, B1, N, res, n_expr=None, program="GMP-ECM 7.0.6"):
    """the gmp-ecm/prmers-family text line (X hex, no trailing Z: X is normalised)"""

    def csum(B1v, sigma_v, Nv, f, param):
        MOD = 4294967291          # gmp-ecm's CHKSUMMOD (2^32 - 5)
        v = (B1v % MOD) * (sigma_v % MOD) % MOD
        v = v * (Nv % MOD) % MOD
        v = v * ((f % MOD) or 1) % MOD
        v = v * ((param + 1) % MOD) % MOD
        return v

    return ("METHOD=ECM; PARAM=0; SIGMA=%d; B1=%d; N=%s; X=0x%x; CHECKSUM=%d; "
            "PROGRAM=%s; X0=0x0; Y0=0x0;" %
            (sigma, B1, n_expr if n_expr else str(N), res["x"], csum(B1, sigma, N, 1, 0), program))


def check_against_gmp_ecm(ecm=DEFAULT_ECM, B1s=(2, 3, 5, 7, 11, 97, 1000), sigmas=(12345, 999, 20260922, 31415926)):
    if not os.path.exists(ecm):
        print("gmp-ecm not found at %s -- skipping" % ecm)
        return None
    N = 2 ** 1277 - 1
    ok = bad = 0
    for B1 in B1s:
        for sg in sigmas:
            save = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_run", "ref_%d_%d.save" % (sg, B1))
            os.makedirs(os.path.dirname(save), exist_ok=True)
            if os.path.exists(save):
                os.remove(save)
            subprocess.run([ecm, "-param", "0", "-sigma", str(sg), "-c", "1", "-save", save,
                            str(B1), str(B1)],
                           # B2 = B1 keeps gmp-ecm in stage 1 only; with its default B2 a
                           # step-2 hit would leave a residue that is not the stage-1
                           # point.  (Its stage-1 exponent policy itself is not fully
                           # controllable, hence the hit-comparison rules in the harness.)
                           input=str(N).encode(), capture_output=True)
            if not os.path.exists(save) or os.path.getsize(save) == 0:
                continue
            Xg = None
            for kv in open(save, errors="replace").read().strip().rstrip(";").split(";"):
                if kv.strip().startswith("X="):
                    Xg = int(kv.split("=", 1)[1].strip(), 16) % N
            res = stage1(sg, B1, N, torsion=1)
            same = (res["x"] == Xg)
            ok += same
            bad += (not same)
            print("  B1=%-6d sigma=%-12d  x == gmp-ecm X ? %s" % (B1, sg, same))
    print("gmp-ecm agreement: %d/%d" % (ok, ok + bad))
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description="Suyama-sigma Montgomery stage-1 reference / oracle harness")
    ap.add_argument("--sigma", type=int, default=12345)
    ap.add_argument("--B1", type=int, default=1000)
    ap.add_argument("--N", default="2^1277-1")
    ap.add_argument("--torsion", type=int, default=1,
                    help="1 = gmp-ecm convention (lcm(1..B1)); 12 = Prime95 choose12 convention")
    ap.add_argument("--save-line", action="store_true", help="print the reference-format text save line")
    ap.add_argument("--check-gmp-ecm", action="store_true", help="validate against the gmp-ecm binary")
    ap.add_argument("--ecm", default=DEFAULT_ECM)
    a = ap.parse_args()

    if a.check_gmp_ecm:
        ok = check_against_gmp_ecm(a.ecm)
        sys.exit(0 if ok else 1)

    N = parse_int(a.N)
    res = stage1(a.sigma, a.B1, N, torsion=a.torsion)
    print("sigma   = %d" % a.sigma)
    print("B1      = %d   torsion factor = %d" % (a.B1, a.torsion))
    print("s_bits  = %d" % res["s_bits"])
    print("A       = %d" % res["A"])
    print("gcd(Z,N)= %d   %s" % (res["gcd"], "(FACTOR!)" if 1 < res["gcd"] < N else ""))
    print("x       = 0x%x" % res["x"])
    if a.save_line:
        print(save_line(a.sigma, a.B1, N, res, n_expr=a.N))


if __name__ == "__main__":
    main()
