"""Quick per-prime timing benchmark at a given bit width."""
import sys
import time

sys.path.insert(0, r"D:\code\MPA-OpenCl\tools\ecm_prob")
import data
import ecmath
import curves

bit = int(sys.argv[1]) if len(sys.argv) > 1 else 25
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
B1 = int(sys.argv[3]) if len(sys.argv) > 3 else 256

primes = data.load_primes(bit)[:n]
s = ecmath.batch_s(B1)
print(f"bit={bit} n={n} B1={B1} s_bits={s.bit_length()}")
for c in curves.ROSTER:
    t0 = time.time()
    hits = sum(1 for p in primes if ecmath.curve_hits(c, p, s))
    dt = time.time() - t0
    print(f"  {c['name']:16s} {dt/n*1000:7.3f} ms/prime   ({hits}/{n} = {100*hits/n:.1f}%)")
