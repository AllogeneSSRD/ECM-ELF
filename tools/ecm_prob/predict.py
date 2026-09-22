"""ECM 参数化成功率预测程序。

用已标定的有效除子 D 预测任意 (曲线, 位宽, B1) 下的 stage-1 成功率与期望曲线数。

可预测的曲线（D 来源）：
  * 经验 D_eff：15–25 bit 扫掠逐 bit 反解后取几何平均（out/d_eff_representative.json）
  * 理论 D    ：GMP-ECM rho.c 常数（Suyama exp(3.134)、param1/3 的校正常数），仅 Montgomery
  * 扭子群 D  ：朴素下界 D = T（明显低估，仅作参照）

模型：fraction = stage1_prob(B1, p_ref, delta=log D)，p_ref = 2^(bit-0.5)
      expected_curves = 1 / fraction

用法：
  python predict.py                      # 全部曲线，默认 bit=25, B1=256
  python predict.py suyama_s10 30 1024   # 单条曲线 (curve bit B1)
  python predict.py --all 30 1024        # 全部曲线表格
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import rho
import curves
import calibrate

TOOL_DIR = Path(__file__).resolve().parent
OUT_DIR = TOOL_DIR / "out"
REP_FILE = OUT_DIR / "d_eff_representative.json"

# GMP-ECM theoretical effective divisors (report §6.3):
#   Suyama: exp(ECM_EXTRA_SMOOTHNESS) = exp(3.134)
#   param1 (d square):  * EXTRA_SMOOTHNESS_SQUARE
#   param3 (d random):  * EXTRA_SMOOTHNESS_32BITS_D
_THEO = math.exp(rho.ECM_EXTRA_SMOOTHNESS)
THEORETICAL_D = {
    "suyama_s10": _THEO,
    "suyama_srand": _THEO,
    "param1_s10": _THEO * 0.416384512396064,
    "param2_s10": _THEO,
    "param3_s10": _THEO * 0.330484606500389,
}


def compute_representative_d_eff() -> dict[str, float]:
    """Geometric mean of per-bit calibrated D_eff across bits 15..25."""
    acc: dict[str, list[float]] = {}
    for b in range(15, 26):
        f = OUT_DIR / f"measure_{b}_256.json"
        if not f.exists():
            continue
        blk = json.loads(f.read_text(encoding="utf-8"))
        if "curves" not in blk:
            continue
        cal = calibrate.calibrate_block(blk, 256, curves.ROSTER)
        for name, rec in cal["curves"].items():
            acc.setdefault(name, []).append(rec["D_eff"])
    return {n: math.exp(sum(math.log(x) for x in v) / len(v))
            for n, v in acc.items()}


def load_representative_d_eff() -> dict[str, float]:
    if REP_FILE.exists():
        return json.loads(REP_FILE.read_text(encoding="utf-8"))
    d = compute_representative_d_eff()
    REP_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return d


def predict_fraction(name: str, bit: int, B1: int,
                     D: float | None = None) -> float:
    """Stage-1 success fraction for one curve at (bit, B1)."""
    if D is None:
        D = load_representative_d_eff()[name]
    p_ref = 2.0 ** (bit - 0.5)
    return rho.stage1_prob(B1, p_ref, delta=math.log(D))


def expected_curves(name: str, bit: int, B1: int, D: float | None = None) -> float:
    f = predict_fraction(name, bit, B1, D)
    return 1.0 / f if f > 0 else float("inf")


def table(bit: int, B1: int) -> str:
    d_eff = load_representative_d_eff()
    t_by_name = {c["name"]: c.get("T") for c in curves.ROSTER}
    lines = [f"ECM stage-1 prediction  (bit={bit}, B1={B1}, p_ref=2^{bit-0.5})", ""]
    lines.append(f"{'curve':16s} {'T':>3s} {'D_emp':>8s} {'D_theo':>8s} "
                 f"{'f(D_emp)':>9s} {'f(D_theo)':>9s} {'f(T)naive':>9s}")
    lines.append("-" * 70)
    for c in curves.ROSTER:
        name = c["name"]
        d = d_eff.get(name)
        dt = THEORETICAL_D.get(name)
        T = t_by_name[name]
        if d is None:
            continue
        fe = predict_fraction(name, bit, B1, d)
        ft = predict_fraction(name, bit, B1, dt) if dt else None
        fn = predict_fraction(name, bit, B1, T)
        d_s = f"{d:8.2f}" if d else f"{'--':>8s}"
        dt_s = f"{dt:8.2f}" if dt else f"{'--':>8s}"
        ft_s = f"{100*ft:8.3f}%" if ft else f"{'--':>8s}"
        lines.append(f"{name:16s} {T:>3d} {d_s} {dt_s} "
                     f"{100*fe:8.3f}% {ft_s} {100*fn:8.3f}%")
    lines.append("")
    lines.append("f(D_emp): empirical D_eff (geometric mean over bits 15-25)")
    lines.append("f(D_theo): GMP-ECM theoretical constant (Montgomery only)")
    lines.append("f(T)naive: naive torsion-only D=T lower bound (underestimates, paper 9.4)")
    return "\n".join(lines)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--all":
        bit = int(args[1]) if len(args) > 1 else 25
        B1 = int(args[2]) if len(args) > 2 else 256
        print(table(bit, B1))
        return
    if len(args) >= 1 and args[0] not in ("-h", "--help"):
        name = args[0]
        bit = int(args[1]) if len(args) > 1 else 25
        B1 = int(args[2]) if len(args) > 2 else 256
        d_eff = load_representative_d_eff()
        dt = THEORETICAL_D.get(name)
        f = predict_fraction(name, bit, B1, d_eff[name])
        print(f"{name}: bit={bit}, B1={B1}")
        print(f"  D_eff = {d_eff[name]:.2f}" +
              (f",  D_theoretical = {dt:.2f}" if dt else ""))
        print(f"  predicted stage-1 fraction = {100*f:.4f}%")
        print(f"  expected curves (1/f)     = {1.0/f:.1f}")
        return
    print(table(25, 256))


if __name__ == "__main__":
    main()
