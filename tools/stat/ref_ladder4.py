"""Reference implementation of the exact SoA ladder (a=1 Edwards, extended coords),
used to locate the first diverging digit against the SIMD dump."""
import re, sys
sys.set_int_max_str_digits(0)
from pathlib import Path

DUMP = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\code\MPA-OpenCl\.bench_tmp\simd_dump_raw.txt")
DICT = Path(sys.argv[2] if len(sys.argv) > 2 else r"D:\code\MPA-OpenCl\.bench_tmp\dict8.txt")
W = int(sys.argv[3]) if len(sys.argv) > 3 else 8

def load_dict(path):
    raw = path.read_bytes()
    try:
        txt = raw.decode("utf-16")
    except UnicodeDecodeError:
        txt = raw.decode("utf-8", "replace")
    d = int(re.search(r"^d\s*=\s*([0-9a-f]+)", txt, re.M).group(1), 16)
    N = int(re.search(r"^N\s*=\s*([0-9a-f]+)", txt, re.M).group(1), 16)
    entries = []
    for m in re.finditer(r"^j=(\d+) x=([0-9a-f]+) y=([0-9a-f]+) dxy=([0-9a-f]+)", txt, re.M):
        entries.append((int(m.group(1)), int(m.group(2), 16), int(m.group(3), 16), int(m.group(4), 16)))
    return d, N, entries

d, N, entries = load_dict(DICT)
print(f"loaded dict: {len(entries)} entries, N bits={N.bit_length()}")

def naf_digits(k, w):
    """faithful port of soa_naf_digits (Prime95 ecm.cpp:4824)"""
    nbits = k.bit_length()
    if nbits == 0:
        return []
    max_val = (1 << (w - 1)) - 1
    out = [0] * (nbits + 1)
    value = 0; addin = 1; carry = 0; start = 0
    for bitnum in range(nbits):
        this_bit = carry + ((k >> bitnum) & 1)
        carry = this_bit >> 1
        this_bit &= 1
        if this_bit:
            if value == 0:
                start = bitnum
            value += addin
        if value == 0:
            continue
        addin <<= 1
        if bitnum == nbits - 1:
            complete = True
        elif addin < max_val:
            complete = False
        elif value <= max_val and addin - value <= max_val:
            complete = False
        else:
            complete = True
        if not complete:
            continue
        if value <= max_val:
            out[start] = value
        else:
            out[start] = value - addin
            carry = 1
        value = 0; addin = 1
    if carry:
        out[nbits] = 1
    sz = len(out)
    while sz > 1 and out[sz - 1] == 0:
        sz -= 1
    return out[:sz]

def dbl(p):
    X, Y, Z, T = p
    A = X * X % N; B = Y * Y % N; C = 2 * Z * Z % N
    E = ((X + Y) * (X + Y) - A - B) % N
    G = (A + B) % N; F = (G - C) % N; H = (A - B) % N
    return (E * F % N, G * H % N, F * G % N, E * H % N)

def add_affine(p, q):
    X, Y, Z, T = p
    qx, qy, qdxy = q
    A = X * qx % N; B = Y * qy % N; C = T * qdxy % N
    E = ((X + Y) * (qx + qy) - A - B) % N
    F = (Z - C) % N; G = (Z + C) % N; H = (B - A) % N
    return (E * F % N, G * H % N, F * G % N, E * H % N)

# --- read the SIMD dump ------------------------------------------------------
dumps = {}
for line in DUMP.read_text(errors="replace").splitlines():
    m = re.match(r"\[dump\] (\d+) (\d+) (\d+) (\d+) (\d+)", line.strip())
    if m:
        dumps[int(m.group(1))] = tuple(int(m.group(i)) for i in range(2, 6))
print("dump steps:", len(dumps), "first:", min(dumps) if dumps else None, "last:", max(dumps) if dumps else None)

w = W
s_txt = Path(r"D:\code\MPA-OpenCl\.bench_tmp\s_1e5.txt")
s = int(s_txt.read_text().strip())
digits = naf_digits(s, w)
total = len(digits)
print("s bits:", s.bit_length(), "digits:", total, "nonzero:", sum(1 for x in digits if x))

def add(p, q):
    x1,y1 = p; x2,y2 = q
    t = d * x1 % N * x2 % N * y1 % N * y2 % N
    return ((x1*y2 + x2*y1) * pow(1+t, -1, N) % N,
            (y1*y2 - x1*x2) * pow(1-t, -1, N) % N)

# dict entries are (2j+1)P; the ladder uses |digit| = 2j+1 -> j = (|digit|-1)/2
entry = {j: (x, y, dxy) for j, x, y, dxy in entries}

R = (0, 1, 1, 0)
first_bad = None
last_checked = None
targets = sorted(dumps)
for i in range(total):
    R = dbl(R)
    dgt = digits[total - 1 - i]
    if dgt != 0:
        qx, qy, qdxy = entry[(abs(dgt) - 1) // 2]
        if dgt > 0:
            R = add_affine(R, (qx, qy, qdxy))
        else:
            R = add_affine(R, ((N - qx) % N, qy, (N - qdxy) % N))
    if i in dumps:
        last_checked = i
        sx, sy, sz, st = dumps[i]
        if (sx, sy, sz, st) != R:
            first_bad = i
            print(f"FIRST DIVERGENCE at step {i}")
            print("  simd X =", hex(sx))
            print("  ref  X =", hex(R[0]))
            print("  simd Y =", hex(sy))
            print("  ref  Y =", hex(R[1]))
            print("  simd Z =", hex(sz))
            print("  ref  Z =", hex(R[2]))
            print("  simd T =", hex(st))
            print("  ref  T =", hex(R[3]))
            print("  digit now =", dgt, " prev digits =", digits[total - min(i, 6):total - i + 4][::-1])
            break
qx = (R[2] + R[1]) % N; qz = (R[2] - R[1]) % N
print("checked up to step", last_checked, "| first_bad =", first_bad)
print("reference final Qx =", hex(qx))
print("reference final Qz =", hex(qz))
import math; print("reference gcd(Qz,N) =", math.gcd(qz, N))
