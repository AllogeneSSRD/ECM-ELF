"""check_plot_colors.py — 机器校验 ecm_plot 的配色约定（Montgomery 暖 / Edwards 冷 / 可分辨）。

不靠肉眼，检查两件事：

  1. 族-色系：Montgomery（mont）必须是暖色（H <= 75 或 H >= 330），Edwards（ed）必须是
     冷色（150 <= H <= 280），p-1/p+1（other）不限制。
     **每一族色表里的每一项都要合规**，不只是当前 SERIES 用到的那几项 —— 以后往 SERIES
     里加一条曲线会自动取下一项，越界的颜色要在加之前就被挡住。
  2. 同族可分辨：同一族内任意两色的 RGB 欧氏距离 >= MIN_DIST，否则画在一张图里分不清。

用法：python check_plot_colors.py [--min-dist 60]
退出码：0 = 全部合规；1 = 有违规（CI/脚本可用）。
"""
from __future__ import annotations

import argparse
import colorsys
import sys

import ecm_plot as P

MIN_DIST = 60.0
WARM_RULE = "H<=75 | H>=330"
COOL_RULE = "150<=H<=280"


def hue(hexcolor: str) -> float:
    r, g, b = (int(hexcolor[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360.0


def rgb(hexcolor: str) -> tuple[int, int, int]:
    return tuple(int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def dist(a: str, b: str) -> float:
    ca, cb = rgb(a), rgb(b)
    return sum((x - y) ** 2 for x, y in zip(ca, cb)) ** 0.5


def in_band(fam: str, h: float) -> bool:
    if fam == "mont":
        return h <= 75.0 or h >= 330.0
    if fam == "ed":
        return 150.0 <= h <= 280.0
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="check_plot_colors")
    ap.add_argument("--min-dist", type=float, default=MIN_DIST,
                    help=f"同族两色最小的 RGB 距离（默认 {MIN_DIST:g}）")
    a = ap.parse_args(argv)

    bad = 0

    # --- 1) 色表本身：每一项都要落在本族的色系里 ------------------------------
    print("ramp check (色表全项，含暂未用到的槽位)")
    print(f"{'family':<7} {'slot':>4} {'color':<8} {'hue':>6}  rule                verdict")
    print("-" * 62)
    for fam in ("mont", "ed", "other"):
        rule = WARM_RULE if fam == "mont" else COOL_RULE if fam == "ed" else "-"
        for i, c in enumerate(P.FAMILY_COLORS[fam]):
            h = hue(c)
            ok = in_band(fam, h)
            bad += 0 if ok else 1
            print(f"{fam:<7} {i + 1:>4} {c:<8} {h:6.1f}  {rule:<19} {'OK' if ok else 'BAD'}")

    # --- 2) 同族两两距离：分不清就报出来 --------------------------------------
    print()
    print(f"separation check (同族任意两色 RGB 距离 >= {a.min_dist:g})")
    print(f"{'family':<7} {'pairs':>5} {'min':>7}  closest pair                 verdict")
    print("-" * 62)
    for fam in ("mont", "ed", "other"):
        ramp = P.FAMILY_COLORS[fam]
        worst, pair = None, ("", "")
        for i in range(len(ramp)):
            for j in range(i + 1, len(ramp)):
                d = dist(ramp[i], ramp[j])
                if worst is None or d < worst:
                    worst, pair = d, (ramp[i], ramp[j])
        npairs = len(ramp) * (len(ramp) - 1) // 2
        ok = worst is None or worst >= a.min_dist
        bad += 0 if ok else 1
        print(f"{fam:<7} {npairs:>5} {worst if worst is not None else float('nan'):7.1f}  "
              f"{pair[0]} vs {pair[1]:<20} {'OK' if ok else 'BAD'}")

    # --- 3) 实际分配结果：图上每条曲线拿到什么颜色 ----------------------------
    print()
    print(f"{'key':<16} {'family':<7} {'color':<8} {'hue':>6}  verdict")
    print("-" * 48)
    cols = P.build_colors(list(P.SERIES.keys()))
    for k, c in cols.items():
        fam = P._family_of(k)
        h = hue(c)
        ok = in_band(fam, h)
        bad += 0 if ok else 1
        print(f"{k:<16} {fam:<7} {c:<8} {h:6.1f}  {'OK' if ok else 'BAD'}")

    print()
    print("BAD count =", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
