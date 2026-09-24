"""模型层：有效除子 D + 成功率预测 + 四元组反解 + GMP 推荐表。

这是预测/反解的唯一模型来源，整合了原 predict.py / params.py / calibrate.py 的
非 CLI 逻辑。`rho.py` 提供底层 Dickman-ρ 数学，本模块在其上叠加：

  * D 的三个来源 —— 经验 D_eff（几何平均）、理论 D（GMP-ECM 常数）、朴素 D=T
  * 正向预测 `predict_fraction(bit, B1, B2)` 与 `expected_curves`
  * 反向求解 `solve_B1` / `solve_bit` 与四元组 `solve`
  * D_eff 拟合 `fit_delta` / `calibrate_block`
  * GMP-ECM 推荐表 `gmp_table`
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import rho
import curves

TOOL_DIR = Path(__file__).resolve().parent
OUT_DIR = TOOL_DIR / "out"
REP_FILE = OUT_DIR / "d_eff_representative.json"

# ---------------------------------------------------------------------------
# 理论有效除子（GMP-ECM rho.c 常数；见 docs/ECM_PARAMETERIZATION_ANALYSIS.md §6.3）
# ---------------------------------------------------------------------------
_THEO = math.exp(rho.ECM_EXTRA_SMOOTHNESS)      # Suyama: exp(3.134) ~= 22.97
THEORETICAL_D = {
    "suyama_s10": _THEO,
    "suyama_srand": _THEO,
    "param1_s10": _THEO * rho.EXTRA_SMOOTHNESS_SQUARE,     # d square
    "param2_s10": _THEO,
    "param3_s10": _THEO * rho.EXTRA_SMOOTHNESS_32BITS_D,   # d random 32-bit
}


# ---------------------------------------------------------------------------
# D_eff 拟合（原 calibrate.py）
# ---------------------------------------------------------------------------
def fit_delta(f_target: float, B1: float, N: float,
              lo: float = -20.0, hi: float = 40.0, iters: int = 120) -> float:
    """Bisect delta so that stage1_prob(B1, N, delta) == f_target (f 随 delta 升)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = rho.stage1_prob(B1, N, delta=mid)
        if f > f_target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def calibrate_block(block: dict, B1: int, roster, p_ref: float = None) -> dict:
    """Fit D_eff per curve for one flat measure block (has 'bit' and 'curves')."""
    bit = block.get("bit")
    if p_ref is None:
        p_ref = 2 ** (bit - 0.5) if bit is not None else 2 ** 19.5
    t_by_name = {c["name"]: c.get("T") for c in roster}
    out = {"B1": B1, "bit": bit, "curves": {}}
    for name, cur in block["curves"].items():
        f = cur["fraction"]
        delta = fit_delta(f, B1, p_ref)
        D = math.exp(delta)
        T = t_by_name.get(name)
        rec = {"fraction": f, "delta": delta, "D_eff": D, "torsion": T}
        if T:
            rec["extra"] = D / T
        out["curves"][name] = rec
    return out


def calibrate(measure_json: Path, B1: int, roster, p_ref_by_bit: dict = None) -> dict:
    """Convenience wrapper: calibrate every bit-block in a measure file."""
    data = json.loads(measure_json.read_text(encoding="utf-8"))
    if "curves" in data and "bit" in data:
        return calibrate_block(data, B1, roster, (p_ref_by_bit or {}).get(data["bit"]))
    out = {}
    for key, block in data.items():
        if not isinstance(block, dict) or "curves" not in block:
            continue
        out[key] = calibrate_block(block, B1, roster, (p_ref_by_bit or {}).get(block.get("bit")))
    return out


# ---------------------------------------------------------------------------
# 经验 D_eff 注册表
# ---------------------------------------------------------------------------
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
        cal = calibrate_block(blk, 256, curves.ROSTER)
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


# ---------------------------------------------------------------------------
# 预测
# ---------------------------------------------------------------------------
def predict_fraction(name: str, bit: float, B1: float,
                     D: float | None = None, B2: float | None = None) -> float:
    """单曲线成功率。B2=None 或 <=B1 -> stage-1 only；B2>B1 -> stage-1+stage-2。"""
    if D is None:
        D = load_representative_d_eff()[name]
    p_ref = 2.0 ** (bit - 0.5)
    if B2 is not None and B2 > B1:
        return rho.stage_prob(B1, B2, p_ref, D)
    return rho.stage1_prob(B1, p_ref, delta=math.log(D))


def expected_curves(name: str, bit: float, B1: float,
                    D: float | None = None, B2: float | None = None) -> float:
    f = predict_fraction(name, bit, B1, D, B2)
    return 1.0 / f if f > 0 else float("inf")


# ---------------------------------------------------------------------------
# 反向求解
# ---------------------------------------------------------------------------
#: Stage-2 bound convention used everywhere when the caller does not supply B2:
#: B2 = B2_FACTOR * B1 (gmp-ecm / Prime95 practice for a 100x stage-2 range).
B2_FACTOR = 100.0


def default_b2(B1: float, factor: float = B2_FACTOR) -> float:
    """B2 的默认取值 = factor * B1（默认 100 倍）。"""
    return float(factor) * float(B1)


def model_p(bit: float, B1: float, D: float, B2: float | None = None) -> float:
    """p = f(bit, B1[, B2])。B2=None -> 默认 100*B1。对 bit 单调降、对 B1 单调升。"""
    p_ref = 2.0 ** (bit - 0.5)
    b2 = default_b2(B1) if B2 is None else B2
    if b2 > B1:
        return rho.stage_prob(B1, b2, p_ref, D)
    return rho.stage1_prob(B1, p_ref, delta=math.log(D))


def solve_B1(bit: float, p: float, D: float, B2: float | None = None) -> float:
    lo, hi = 2.0, 2.0 ** 31
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if model_p(bit, mid, D, B2) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_bit(B1: float, p: float, D: float, B2: float | None = None) -> float:
    lo, hi = 10.0, 250.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if model_p(mid, B1, D, B2) < p:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def resolve_D(curve: str | None, D: float | None) -> float:
    if D is not None:
        return D
    d = load_representative_d_eff()
    if curve in d:
        return d[curve]
    raise SystemExit(f"unknown curve '{curve}'; choose from {sorted(d)}")


# ---------------------------------------------------------------------------
# GMP-ECM 推荐表（params.html，Pierrick 2019，stage1+stage2）
# ---------------------------------------------------------------------------
GMP_BIT_TABLE = {
    30: (1358, 2), 35: (1270, 5), 40: (1629, 10), 45: (4537, 10),
    50: (12322, 9), 55: (12820, 18), 60: (21905, 21), 65: (24433, 41),
    70: (32918, 66), 75: (64703, 71), 80: (76620, 119), 85: (155247, 123),
    90: (183849, 219), 95: (245335, 321), 100: (445657, 339),
    105: (643986, 468), 110: (1305195, 439), 115: (1305195, 818),
    120: (3071166, 649), 125: (3784867, 949), 130: (4572523, 1507),
    135: (7982718, 1497), 140: (9267681, 2399), 145: (22025673, 1826),
    150: (22025673, 3159), 155: (26345943, 4532), 160: (35158748, 6076),
    165: (46919468, 8177), 170: (47862548, 14038), 175: (153319098, 7166),
    180: (153319098, 12017), 185: (188949210, 16238), 190: (410593604, 13174),
    195: (496041799, 17798), 200: (491130495, 29584), 205: (1067244762, 23626),
    210: (1056677983, 38609), 215: (1328416470, 49784),
    220: (1315263832, 81950), 225: (2858117139, 63461),
}


def gmp_table(bits: list[int]) -> str:
    """GMP-ECM 位宽推荐表（硬编码实验值）+ 回归式对照。"""
    lines = ["GMP-ECM params.html (stage1+stage2, experimental):", ""]
    lines.append(f"{'bits':>5s} {'optimal B1':>12s} {'curves N':>9s} "
                 f"{'regB1':>12s} {'regN':>9s}")
    for b in bits:
        B1, N = GMP_BIT_TABLE.get(b, (None, None))
        rB1 = math.exp(0.075 * b + 5.332)
        rN = (rB1 / 150.0) ** (2.0 / 3.0)
        if B1 is None:
            lines.append(f"{b:5d} {'--':>12s} {'--':>9s} {rB1:12.0f} {rN:9.0f}")
        else:
            lines.append(f"{b:5d} {B1:12d} {N:9d} {rB1:12.0f} {rN:9.0f}")
    lines.append("")
    lines.append("regB1/regN: page regression (approximate fit)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 四元组 {bit, B1, N, p} 求解
# ---------------------------------------------------------------------------
def _close(a: float, b: float, rtol: float = 1e-3) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-12)


def solve(bit, B1, curves, prob, D, B2=None, b2_label=None) -> str:
    """正向：p = f(bit,B1[,B2])，curves 算实际 miss；反向：p/N 反解 B1/bit。

    B2=None 表示"用默认约定 B2 = B2_FACTOR*B1"（见 model_p）；b2_label 只影响
    第一行怎么描述 B2（CLI 用它打印 "--b2-factor 10000 -> B2=10000*B1=..."）。
    """
    p_target = prob if prob is not None else (1.0 / curves if curves is not None else None)
    if b2_label is None:
        b2_label = f"B2={B2:g}" if B2 is not None else "B2=100*B1"
    lines = [f"D = {D:.2f}, {b2_label}"]

    if bit is not None and B1 is not None:          # 正向
        p = model_p(bit, B1, D, B2)
        lines.append(f"p = f(bit={bit:g}, B1={B1:g}) = {100*p:.4f}%")
        if curves is not None:
            miss = (1.0 - p) ** curves
            lines.append(f"N = {curves:g} curves -> miss = (1-p)^N = {100*miss:.4f}%")
            lines.append(f"   (ref: miss = e^-1 = 36.8%  needs N = 1/p = {1.0/p:.1f} curves)")
        else:
            lines.append(f"N = 1/p = {1.0 / p:.1f} curves -> miss = e^-1 = 36.8%")
        if prob is not None and not _close(prob, p):
            lines.append(f"   note: given prob = {100*prob:.4f}% differs from model {100*p:.4f}%")
        return "\n".join(lines)

    if bit is not None and p_target is not None:    # 反解 B1
        B1_s = solve_B1(bit, p_target, D, B2)
        lines.append(f"B1 = {B1_s:.1f}")
        lines.append(f"   (standard miss = e^-1 = 36.8%,  p = {100*p_target:.4f}%,  "
                     f"N = {1.0 / p_target:.1f})")
        return "\n".join(lines)

    if B1 is not None and p_target is not None:     # 反解 bit
        bit_s = solve_bit(B1, p_target, D, B2)
        lines.append(f"bit = {bit_s:.2f}")
        lines.append(f"   (standard miss = e^-1 = 36.8%,  p = {100*p_target:.4f}%,  "
                     f"N = {1.0 / p_target:.1f})")
        return "\n".join(lines)

    raise SystemExit("insufficient inputs: need (bit,B1), or (bit, p/N), or (B1, p/N)")
