"""Third independent check, robust version: projective *binary* double-and-add
mod p (no NAF, no dictionary, no inversions).  Test: (Z - Y) == 0 mod p.
"""
import re, sys
from math import lcm
sys.set_int_max_str_digits(0)

raw = open(r"D:\code\MPA-OpenCl\.bench_tmp\dict_dump.txt", "rb").read()
try:
    txt = raw.decode("utf-16")
except UnicodeDecodeError:
    txt = raw.decode("utf-8", "replace")
d_full = int(re.search(r"^d\s*=\s*([0-9a-f]+)", txt, re.M).group(1), 16)
Px_full = int(re.search(r"^Px\s*=\s*([0-9a-f]+)", txt, re.M).group(1), 16)
Py_full = int(re.search(r"^Py\s*=\s*([0-9a-f]+)", txt, re.M).group(1), 16)

B1 = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
p = int(sys.argv[2]) if len(sys.argv) > 2 else 3217073
m = p
d = d_full % m

s = 1
for i in range(2, B1 + 1):
    s = lcm(s, i)
s *= 48

def dbl(pt):
    X, Y, Z, T = pt
    A = X*X % m; B = Y*Y % m; C = 2*Z*Z % m
    E = ((X+Y)*(X+Y) - A - B) % m
    G = (A + B) % m; F = (G - C) % m; H = (A - B) % m
    return (E*F % m, G*H % m, F*G % m, E*H % m)

def add(p1, p2):
    X1, Y1, Z1, T1 = p1
    X2, Y2, Z2, T2 = p2
    A = X1*X2 % m; B = Y1*Y2 % m; C = d * T1 % m * T2 % m; D = Z1*Z2 % m
    E = ((X1+Y1)*(X2+Y2) - A - B) % m
    F = (D - C) % m; G = (D + C) % m; H = (B - A) % m
    return (E*F % m, G*H % m, F*G % m, E*H % m)

P = (Px_full % m, Py_full % m, 1, Px_full % m * Py_full % m)
R = (0, 1, 1, 0)
bits = bin(s)[2:]
for bit in bits:
    R = dbl(R)
    if bit == '1':
        R = add(R, P)
qz = (R[2] - R[1]) % m
print(f"B1={B1} s_bits={s.bit_length()} p={p}")
print(f"  projective binary double-and-add: (Z-Y) mod p = {qz} -> "
      f"{'IDENTITY mod p => HIT CONFIRMED' if qz == 0 else 'not identity'}")
