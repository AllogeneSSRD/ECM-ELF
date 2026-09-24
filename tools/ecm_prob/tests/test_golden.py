"""Golden-number validation against Bernstein-Birkner-Lange-Peters,
"ECM using Edwards curves", Section 9.1 (20-bit primes, B1=256).

Expected (paper Table/9.1):
  Z/12      -> 12467 / 38635 = 32.2687%
  Z/2xZ/8   ->            32.8433%
  Z/2xZ/4   ->            27.4854%
  Z/4       ->            23.4709%

已知偏差（pre-existing，与 CUDA/Montgomery 改动无关）：四条曲线都**一致地**比论文低
0.03–0.18 个百分点（Z/12: 12404 vs 12467，Δ=-0.163pp；Z/2xZ/8: -0.179pp；
Z/2xZ/4: -0.028pp；Z/4: -0.036pp）。方向一致 ⇒ 是模型层的系统性小偏差（论文的统计口径
细节），不是某条曲线的独立错误。用 git worktree 在 edb717a（本次改动之前的提交）复核过：
Z/12 同样是 12404，所以不是回归。这里**故意保留精确计数比对**：真有回归时偏差会突然
变大或只剩一条曲线掉队，那种情况不能靠"反正本来就差一点"糊过去。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ecmath
import curves

B1 = 256
LO = 1 << 19
HI = 1 << 20


def main():
    primes = [p for p in ecmath.primes_upto(HI) if p >= LO]
    assert len(primes) == 38635, len(primes)
    s = ecmath.batch_s(B1)
    print(f"primes in [2^19,2^20): {len(primes)}  (expect 38635)")
    print(f"s = lcm(1..{B1}), bits={s.bit_length()}")

    golden = {
        "edwards_Z12": (12467, 32.2687),
        "edwards_Z2xZ8": (None, 32.8433),   # 论文只给了百分比
        "edwards_Z2xZ4": (None, 27.4854),
        "edwards_Z4": (None, 23.4709),
    }
    print(f"{'curve':<16} {'torsion':<8} {'hits':>7} {'pct':>9}  {'paper %':>9} {'delta':>9}  verdict")
    for c in curves.ROSTER:
        if c["form"] != "edwards":
            continue
        t0 = time.time()
        hits = sum(1 for p in primes if ecmath.curve_hits(c, p, s))
        dt = time.time() - t0
        pct = 100.0 * hits / len(primes)
        ref_hits, ref_pct = golden.get(c["name"], (None, None))
        delta = "-" if ref_pct is None else f"{pct - ref_pct:+.4f}pp"
        mark = ""
        if ref_hits is not None:
            mark = "OK" if hits == ref_hits else f"MISMATCH (expect {ref_hits})"
        print(f"{c['name']:16s} {c['torsion']:8s} {hits:7d} {pct:8.4f}%  "
              f"{ref_pct if ref_pct is not None else float('nan'):8.4f}% {delta:>9}  "
              f"{mark}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
