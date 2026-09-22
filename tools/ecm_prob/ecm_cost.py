"""ecm_cost.py — 用户 CLI：给定 B1 + 曲线，输出 stage-1 成本（点运算层 + 域运算层）。

点运算计数由 C++ 引擎 cost_engine.exe（subprocess）提供；域运算层为静态公式表。

用法：
  python ecm_cost.py --B1 1000000 --curve suyama_s10
  python ecm_cost.py --B1 1000000 --curve edwards_Z12
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
ENGINE = TOOL_DIR / "cost_engine.exe"


def _parse_int(s: str) -> int:
    """Parse an integer that may be given in scientific notation (e.g. '1e6')."""
    try:
        return int(s)
    except ValueError:
        return int(float(s))

# 域运算成本表: (M, S, A, Sub, I, D) 每次倍点 / 差分加法 / 融合加+倍.
# 来源: EFD + GMP-ECM ecm.c(duplicate/add3) + 本项目 ecm_stage1.cl(double_add_v2).
FIELD = {
    "montgomery": {
        "paper_dbl": (2, 2, 0, 0, 0, 1),     # dbl-1987-m-3 (x a24 为 D)
        "paper_add": (4, 2, 0, 0, 0, 0),     # dadd-1987-m-3
        "gmpecm_dbl": (3, 2, 2, 2, 0, 0),    # duplicate (x a24 为满 M)
        "gmpecm_add": (4, 2, 3, 3, 0, 0),    # add3
        "ours_fused": (4, 4, 4, 4, 0, 2),    # double_add_v2 (x d + x 2)
        "p95_dbl_t": 10,                     # FFT 变换
        "p95_add_t": 12,
    },
    "edwards": {
        "paper_dbl": (3, 4, 1, 0, 0, 0),     # a=1 射影倍点
        "paper_add": (8, 0, 0, 0, 0, 0),     # a=-1 扩展统一加法
        "p95_dbl_t": 10,                     # FFT 变换 (近似)
        "p95_add_t": 12,
    },
}

CURVE_INIT = {
    "suyama":    (10, 4, 0, 0, 1, 0, "u=s^2-5,v=4s; A=(v-u)^3(3u+v)/(4u^3v)-2; x0=u^3/v^3"),
    "param1":    (2, 1, 0, 0, 0, 0, "d=s^2/2^64; A=4d-2; x0=2"),
    "param2":    (0, 0, 0, 0, 2, 0, "P=s*(-3:3:1) on y^2=x^3+36, then derive A (2 inversions)"),
    "param3":    (1, 0, 0, 0, 0, 0, "d=s/2^32; A=4d-2; x0=2"),
    "edwards":   (0, 0, 0, 0, 0, 0, "(a,d) small integers, small base point P, no inversion"),
}


def curve_init(curve: str) -> tuple:
    key = curve.split("_")[0]          # suyama_s10 -> suyama, param3_s10 -> param3, edwards_Z12 -> edwards
    if key.startswith("edwards"):
        key = "edwards"
    if key.startswith("param"):
        key = key  # param1/param2/param3
    return CURVE_INIT.get(key, (0, 0, 0, 0, 0, 0, "unknown"))


def run_engine(B1: int) -> dict:
    if not ENGINE.exists():
        raise SystemExit(f"cost_engine.exe not found; compile cost_engine.cpp first")
    out = subprocess.run([str(ENGINE), str(B1)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip())


def fmt_field(t: tuple) -> str:
    M, S, A, Sub, I, D = t
    parts = []
    if M: parts.append(f"{M}M")
    if S: parts.append(f"{S}S")
    if A: parts.append(f"{A}A")
    if Sub: parts.append(f"{Sub}Sub")
    if I: parts.append(f"{I}I")
    if D: parts.append(f"{D}D")
    return "+".join(parts) if parts else "0"


def total_field(dbl: int, add: int, dbl_c, add_c) -> tuple:
    return tuple(dbl * dbl_c[i] + add * add_c[i] for i in range(6))


def report(B1: int, curve: str) -> str:
    e = run_engine(B1)
    init = curve_init(curve)
    form = "edwards" if curve.startswith("edwards") else "montgomery"

    L = [f"ECM stage-1 cost  B1={B1}  curve={curve}  form={form}",
         f"  s = lcm(1..B1) = {e['s_bits']} bits;  primes={e['n_primes']}, "
         f"prime-powers={e['n_powers']}",
         f"  PRAC: {e['total_dbl']} doublings + {e['total_add']} diff-additions "
         f"(2/3 handled separately)", ""]

    L.append("[1] Curve init (once per curve)")
    L.append(f"    {curve}:  {fmt_field(init[:6])}    ({init[6]})")
    L.append("")

    L.append("[2] Stage-1 point operations (addition chain)")
    L.append(f"    {'approach':28s} {'chain':24s} {'doublings':>12s} {'additions':>12s}")
    L.append(f"    {'gmp-ecm param0/1/2/3':28s} {'per-prime PRAC':24s} {e['total_dbl']:>12d} {e['total_add']:>12d}")
    L.append(f"    {'prime95 suyama':28s} {'per-prime PRAC':24s} {e['total_dbl']:>12d} {e['total_add']:>12d}")
    L.append(f"    {'ours param3 (CUDA/OpenCL)':28s} {'single ladder':24s} {e['s_bits']:>12d} {e['s_bits']:>12d} (fused)")
    if form == "edwards":
        L.append(f"    {'prime95 edwards':28s} {'NAF':24s} {e['s_bits']:>12d} {e['s_bits']//3:>12d}")
        L.append(f"    {'python ecmath (edwards)':28s} {'double-and-add':24s} {e['s_bits']:>12d} {e['s_bits']//2:>12d}")
    L.append("")

    L.append("[3] Stage-1 field operations (M/S/A/Sub/I/D)")
    L.append(f"    {'approach':22s} {'per-doubling':16s} {'per-addition':16s} {'total':40s}")
    if form == "montgomery":
        F = FIELD["montgomery"]
        dbl, add = e["total_dbl"], e["total_add"]
        p = total_field(dbl, add, F["paper_dbl"], F["paper_add"])
        g = total_field(dbl, add, F["gmpecm_dbl"], F["gmpecm_add"])
        fused = tuple(e["s_bits"] * x for x in F["ours_fused"])
        L.append(f"    {'paper (EFD)':22s} {fmt_field(F['paper_dbl']):16s} {fmt_field(F['paper_add']):16s} {fmt_field(p):40s}")
        L.append(f"    {'gmp-ecm (param0)':22s} {fmt_field(F['gmpecm_dbl']):16s} {fmt_field(F['gmpecm_add']):16s} {fmt_field(g):40s}")
        L.append(f"    {'ours param3 (fused)':22s} {'4M+4S+4A+4Sub+2D/bit':16s} {'-':16s} {fmt_field(fused):40s}")
        L.append(f"    {'prime95':22s} {'10 transforms':16s} {'12 transforms':16s} {str(dbl*F['p95_dbl_t']+add*F['p95_add_t'])+' transforms':40s}")
    else:
        F = FIELD["edwards"]
        dbl, add = e["s_bits"], e["s_bits"] // 3
        p = total_field(dbl, add, F["paper_dbl"], F["paper_add"])
        L.append(f"    {'paper (Edwards)':22s} {fmt_field(F['paper_dbl']):16s} {fmt_field(F['paper_add']):16s} {fmt_field(p):40s}")
        L.append(f"    {'prime95 (Edwards FFT)':22s} {'~10 transforms':16s} {'~12 transforms':16s} {'~'+str(dbl*F['p95_dbl_t']+add*F['p95_add_t'])+' transforms':40s}")
    L.append("")
    L.append("notes:")
    L.append("  - prime95 uses gwnum FFT (no limb mul); unit is 'FFT transforms', not "
             "directly comparable to M.")
    L.append("  - ours param3 fused ladder saves ~37% M vs gmp-ecm separate "
             "duplicate+add3 (shared squarings AA/BB).")
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ecm_cost")
    ap.add_argument("--B1", type=_parse_int, default=1000000)
    ap.add_argument("--curve", default="suyama_s10")
    args = ap.parse_args(argv)
    print(report(args.B1, args.curve))


if __name__ == "__main__":
    main()
