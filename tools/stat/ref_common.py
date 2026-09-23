"""shared helpers for the reference ladder (exact mpz-equivalent integer math)"""
import re
from pathlib import Path


def load_dict(path):
    raw = Path(path).read_bytes()
    try:
        txt = raw.decode("utf-16")
    except UnicodeDecodeError:
        txt = raw.decode("utf-8", "replace")
    d = int(re.search(r"^d\s*=\s*([0-9a-f]+)", txt, re.M).group(1), 16)
    entries = []
    for m in re.finditer(r"^j=(\d+) x=([0-9a-f]+) y=([0-9a-f]+) dxy=([0-9a-f]+)", txt, re.M):
        entries.append((int(m.group(1)), int(m.group(2), 16), int(m.group(3), 16), int(m.group(4), 16)))
    return d, entries


def naf(k, w):
    """verbatim port of naf_digits / soa_naf_digits"""
    nbits = k.bit_length()
    if nbits == 0:
        return []
    max_val = (1 << (w - 1)) - 1
    out = [0] * (nbits + 1)
    value = 0
    addin = 1
    carry = 0
    start = 0
    for bitnum in range(nbits):
        tb = carry + ((k >> bitnum) & 1)
        carry = tb >> 1
        tb &= 1
        if tb:
            if value == 0:
                start = bitnum
            value += addin
        if value == 0:
            continue
        addin <<= 1
        if bitnum == nbits - 1:
            comp = True
        elif addin < max_val:
            comp = False
        elif value <= max_val and addin - value <= max_val:
            comp = False
        else:
            comp = True
        if not comp:
            continue
        if value <= max_val:
            out[start] = value
        else:
            out[start] = value - addin
            carry = 1
        value = 0
        addin = 1
    if carry:
        out[nbits] = 1
    sz = len(out)
    while sz > 1 and out[sz - 1] == 0:
        sz -= 1
    return out[:sz]


def make_ops(N, d):
    def dbl(p):
        X, Y, Z, T = p
        A = X * X % N
        B = Y * Y % N
        C = 2 * Z * Z % N
        E = ((X + Y) * (X + Y) - A - B) % N
        G = (A + B) % N
        F = (G - C) % N
        H = (A - B) % N
        return (E * F % N, G * H % N, F * G % N, E * H % N)

    def add_aff(p, q):
        X, Y, Z, T = p
        qx, qy, qd = q
        A = X * qx % N
        B = Y * qy % N
        C = T * qd % N
        E = ((X + Y) * (qx + qy) - A - B) % N
        F = (Z - C) % N
        G = (Z + C) % N
        H = (B - A) % N
        return (E * F % N, G * H % N, F * G % N, E * H % N)

    return dbl, add_aff
