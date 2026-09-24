"""ecm_plot.py — 用户 CLI：绘图（经验扫掠 + 预测 + 两者叠加）。

整合原 plot.py（经验扫掠图）与 plot_prob.py（预测图）。

三件事在这个文件里是"表驱动"的，改表即改图：

  1. SERIES  —— 每条可绘曲线的唯一登记处（标签 / 族 / 数据来源）。绘制哪些曲线、
     用实测数据还是模型预测，都在这张表里选；命令行 --series 再按 key 过滤。
  2. FAMILY_COLORS —— 配色约定：**Montgomery 族（mont）用暖色，Edwards 族（ed）用冷色**，
     p-1/p+1 等用中性灰。同一条曲线在所有图里颜色一致（按表中顺序取色）。
  3. bit 范围不再硬编码：经验图扫描 out/measure_<bit>_<B1>.json 自动发现可用 bit
     （有多少算多少，不写死在代码里），预测图默认 15–130。

用法：
  python ecm_plot.py list                                   # 打印 SERIES 表
  python ecm_plot.py empirical                              # 逐 bit 柱状 + 成功率 + D_eff
  python ecm_plot.py empirical --b1 256 --bits 15-30
  python ecm_plot.py empirical --series suyama_s10,edwards_Z2xZ8
  python ecm_plot.py predict --B1 256 --bit 40              # 预测（stage1 vs stage1+stage2）
  python ecm_plot.py predict --B2 1e6 --mix-empirical       # 叠加实测点（B1 需与 measure 一致）
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model
import curves

OUT_DIR = Path(__file__).resolve().parent / "out"
PLOT_DIR = OUT_DIR / "plots"
DEFAULT_B1 = 256

# Stage-2 bound: **B2 = 100 * B1 by default**, the same convention model.model_p() and
# ecm_prob.py's prediction table use when no B2 is given.  --b2-factor changes the
# ratio, --B2 overrides it with an absolute value.
B2_FACTOR_DEFAULT = 100.0


def effective_b2(B1: float, B2: float | None, factor: float = B2_FACTOR_DEFAULT) -> float:
    """B2 的实际取值：显式给了就用它，否则 factor * B1（默认 100 倍）。"""
    return float(B2) if B2 is not None else factor * float(B1)


def b2_note(B1: float, B2: float | None, factor: float = B2_FACTOR_DEFAULT) -> str:
    """标题里怎么描述本次用的 B2。"""
    if B2 is not None:
        return f"B2={B2:g} (fixed)"
    return f"B2={factor:g}*B1={effective_b2(B1, None, factor):g}"


def save_fig(fig, name: str, dpi: int = 150, tries: int = 3) -> Path:
    """保存图：先渲染到内存，再一次性写盘，失败重试几次。

    Windows 上偶发 `OSError: [Errno 22]`（杀软/索引器/文件句柄竞争）不该让整张图白画；
    写盘用的是 Path.write_bytes，比交给 PIL 自己开句柄更可控。多次失败才报错退出。
    """
    path = PLOT_DIR / name
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    data = buf.getvalue()
    last: OSError | None = None
    for i in range(max(1, tries)):
        try:
            path.write_bytes(data)
            return path
        except OSError as e:                       # 瞬时占用/共享冲突：等一会儿再来
            last = e
            time.sleep(0.3 * (i + 1))
    raise SystemExit(f"cannot write {path}: {last}")

# ---------------------------------------------------------------------------
# 1) 曲线登记表（内置映射表）
#
#   label  : 图例文字
#   family : "mont" | "ed" | "other" —— 只用来选色系（暖/冷/灰）
#   source : "measure" 只画实测（缺该点就断开）
#            "model"   只画模型预测
#            "auto"    有实测就用实测，否则用预测（默认，适合"实测+外推"的图）
#   style  : 预测图的默认线型（"-" 实线 / "--" 虚线 / ":" 点线），None = 由调用处决定
# ---------------------------------------------------------------------------
SERIES: dict[str, dict] = {
    # --- Montgomery 族（暖色）------------------------------------------------
    "suyama_s10":    {"label": "param0 Z12 (Suyama)", "family": "mont", "source": "auto"},
    "suyama_srand":  {"label": "param0 Z12 (random σ)", "family": "mont", "source": "auto"},
    "param1_s10":    {"label": "param1 Z4", "family": "mont", "source": "auto"},
    "param2_s10":    {"label": "param2 Z6", "family": "mont", "source": "auto"},
    "param3_s10":    {"label": "param3 Z4", "family": "mont", "source": "auto"},
    # --- Edwards 族（冷色）---------------------------------------------------
    "edwards_Z12":   {"label": "Edwards Z12", "family": "ed", "source": "auto"},
    "edwards_Z2xZ8": {"label": "Edwards Z2xZ8 (Prime95)", "family": "ed", "source": "auto"},
    "edwards_Z2xZ4": {"label": "Edwards Z2xZ4", "family": "ed", "source": "auto"},
    "edwards_Z4":    {"label": "Edwards Z4", "family": "ed", "source": "auto"},
    # --- 其它（中性灰）-------------------------------------------------------
    "pm1":           {"label": "p-1", "family": "other", "source": "auto"},
    "pp1":           {"label": "p+1", "family": "other", "source": "auto"},
}

# 2) 配色：Montgomery 暖色系，Edwards 冷色系，其它中性灰
#
#    约束（由 check_plot_colors.py 机器校验，不靠肉眼）：暖色 H<=75 或 H>=330，
#    冷色 150<=H<=280，且同一族内任意两色距离不能太近（否则图例分不出来）。
#    表里**每一项**都要合规，不只是当前用到的前几项 —— 以后往 SERIES 里加曲线时
#    会自动取下一项，越界的颜色会在那一刻变成 BAD。
FAMILY_COLORS = {
    "mont":  ["#b2182b", "#d6604d", "#e08214", "#e9e92a", "#8c510a", "#f4a582"],
    "ed":    ["#172dd6", "#4393c3", "#097D73", "#5e3c99", "#0c2c84", "#92c5de"],
    "other": ["#7f8c8d", "#b3b3b3", "#4d4d4d"],
}
_FAMILY_FALLBACK = {"montgomery": "mont", "montgomery_param2": "mont", "montgomery_param3": "mont",
                    "edwards": "ed", "pminus1": "other", "pplus1": "other"}


def _family_of(key: str) -> str:
    """族 = 表中的显式声明；表里没有的 key 就按 curves.ROSTER 的 form 推断。"""
    if key in SERIES:
        return SERIES[key]["family"]
    for c in curves.ROSTER:
        if c["name"] == key:
            return _FAMILY_FALLBACK.get(c["form"], "other")
    return "other"


def build_colors(keys: list[str]) -> dict[str, str]:
    """按表顺序给每条曲线分配颜色：先满足显式声明的族，再按每族的序号取色。"""
    out: dict[str, str] = {}
    used = {"mont": 0, "ed": 0, "other": 0}
    for k in keys:
        fam = _family_of(k)
        ramp = FAMILY_COLORS[fam]
        out[k] = ramp[used[fam] % len(ramp)]
        used[fam] += 1
    return out


def label_of(key: str) -> str:
    return SERIES.get(key, {}).get("label", key)


def source_of(key: str) -> str:
    return SERIES.get(key, {}).get("source", "auto")


# ---------------------------------------------------------------------------
# 3) measure 数据发现与读取（bit 范围不再硬编码）
# ---------------------------------------------------------------------------
_MEASURE_RE = re.compile(r"^measure_(\d+)_(\d+)\.json$")


def available_bits(b1: int = DEFAULT_B1) -> list[int]:
    """扫描 out/measure_<bit>_<B1>.json，返回该 B1 下所有可用的 bit（升序）。"""
    bits = []
    for f in OUT_DIR.glob(f"measure_*_{b1}.json"):
        m = _MEASURE_RE.match(f.name)
        if m and int(m.group(2)) == b1:
            bits.append(int(m.group(1)))
    return sorted(set(bits))


def load_measure(bit: int, b1: int = DEFAULT_B1) -> dict | None:
    f = OUT_DIR / f"measure_{bit}_{b1}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    if "curves" in d:
        return d
    for v in d.values():                     # 兼容按曲线分组的旧格式
        if isinstance(v, dict) and "curves" in v:
            return v
    return None


def load_blocks(bits: list[int], b1: int) -> dict[int, dict]:
    blocks: dict[int, dict] = {}
    for b in bits:
        blk = load_measure(b, b1)
        if blk is not None:
            blocks[b] = blk
    return blocks


def parse_bits(spec: str | None, b1: int) -> list[int]:
    """--bits 15-30 / 15,17,20 / all（默认 all = 自动发现）。"""
    if spec is None or spec.strip().lower() in ("", "all", "auto"):
        return available_bits(b1)
    bits: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            bits.extend(range(int(lo), int(hi) + 1))
        else:
            bits.append(int(part))
    return sorted(set(bits))


# ---------------------------------------------------------------------------
# 数据取值：按 source 决定用实测还是模型
# ---------------------------------------------------------------------------
def series_values(key: str, bits: list[int], b1: int, b2: float | None,
                  blocks: dict[int, dict], d_eff: dict[str, float]) -> list[float | None]:
    src = source_of(key)
    D = d_eff.get(key)
    ys: list[float | None] = []
    for b in bits:
        blk = blocks.get(b)
        measured = None
        if blk is not None and key in (blk.get("curves") or {}):
            measured = blk["curves"][key].get("pct")
        if src == "measure" or (src == "auto" and measured is not None):
            ys.append(measured)
        elif D is None:
            ys.append(None)
        else:
            ys.append(100.0 * model.predict_fraction(key, b, b1, D, b2))
    return ys


def selected_keys(spec: str | None) -> list[str]:
    if spec is None or not spec.strip():
        return list(SERIES.keys())
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in SERIES]
    if unknown:
        raise SystemExit(f"unknown --series key(s): {', '.join(unknown)} (see `ecm_plot.py list`)")
    return keys


# ---------------------------------------------------------------------------
# empirical
# ---------------------------------------------------------------------------
def _plot_bars(blocks: dict[int, dict], keys: list[str], colors: dict[str, str], b1: int,
               name: str = "bars_per_bit.png") -> None:
    ncols = 4
    nrows = math.ceil(len(blocks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    for idx, bit in enumerate(sorted(blocks)):
        ax = axes[idx // ncols][idx % ncols]
        blk = blocks[bit]
        fracs = [blk["curves"].get(k, {}).get("pct", 0.0) for k in keys]
        ax.bar(range(len(keys)), fracs, width=0.8,
               color=[colors[k] for k in keys])
        ax.set_title(f"bit {bit} (n={blk.get('n_primes', '?')})", fontsize=9)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([label_of(k) for k in keys], rotation=90, fontsize=6)
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.25)
    for idx in range(len(blocks), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(f"ECM stage-1 success fraction by curve, per bit (B1={b1})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, name)
    plt.close(fig)


def _plot_emp_success(blocks: dict[int, dict], keys: list[str], colors: dict[str, str], b1: int,
                      name: str = "emp_success_vs_bit.png") -> None:
    bits = sorted(blocks)
    fig, ax = plt.subplots(figsize=(9, 6))
    for k in keys:
        y = [blocks[b]["curves"].get(k, {}).get("pct") for b in bits]
        ax.plot(bits, y, "-o", label=label_of(k), color=colors[k],
                markersize=4, linewidth=1.6)
    ax.set_xlabel("factor bit width")
    ax.set_ylabel("success fraction (%)")
    ax.set_title(f"ECM stage-1 success vs factor size (B1={b1}, measured)")
    ax.set_xticks(bits)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, name)
    plt.close(fig)


def _plot_emp_d_eff(blocks: dict[int, dict], keys: list[str], colors: dict[str, str], b1: int,
                    name: str = "emp_d_eff_vs_bit.png") -> None:
    bits = sorted(blocks)
    fig, ax = plt.subplots(figsize=(9, 6))
    for k in keys:
        d = []
        for b in bits:
            cal = model.calibrate_block(blocks[b], b1, curves.ROSTER)
            d.append(cal["curves"].get(k, {}).get("D_eff"))
        ax.plot(bits, d, "-o", label=label_of(k), color=colors[k], markersize=4, linewidth=1.6)
    ax.set_xlabel("factor bit width")
    ax.set_ylabel("effective divisor D_eff")
    ax.set_title(f"D_eff vs factor size (B1={b1}, measured)")
    ax.set_xticks(bits)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, name)
    plt.close(fig)


def cmd_empirical(a: argparse.Namespace) -> None:
    keys = selected_keys(a.series)
    colors = build_colors(keys)
    bits = parse_bits(a.bits, a.b1)
    blocks = load_blocks(bits, a.b1)
    if not blocks:
        raise SystemExit(f"no measure_{{[bit]}}_{a.b1}.json under {OUT_DIR}; "
                         f"have: {available_bits(a.b1) or 'nothing for this B1'}")
    missing = [b for b in bits if b not in blocks]
    if missing:
        print(f"note: no data for bit(s) {missing} at B1={a.b1} (skipped)")
    _plot_bars(blocks, keys, colors, a.b1)
    _plot_emp_success(blocks, keys, colors, a.b1)
    _plot_emp_d_eff(blocks, keys, colors, a.b1)
    print(f"wrote empirical plots for bits {sorted(blocks)} to {PLOT_DIR}")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
def _plot_pred_vs_bit(b1: int, b2_arg: float | None, factor: float, bits: list[int],
                      keys: list[str], colors: dict[str, str], blocks: dict[int, dict],
                      mix: bool) -> None:
    b2_eff = effective_b2(b1, b2_arg, factor)          # default: 100 * B1
    d_eff = model.load_representative_d_eff()
    fig, ax = plt.subplots(figsize=(9, 6))
    for k in keys:
        y1 = series_values(k, bits, b1, None, blocks if mix else {}, d_eff)
        y12 = series_values(k, bits, b1, b2_eff, {}, d_eff)
        ax.plot(bits, y1, "--", color=colors[k], alpha=0.8, linewidth=1.3)
        ax.plot(bits, y12, "-", color=colors[k], linewidth=1.8, label=label_of(k))
        if mix:
            mb = [b for b, v in zip(bits, y1) if v is not None and b in blocks]
            mv = [v for b, v in zip(bits, y1) if v is not None and b in blocks]
            if mb:
                ax.plot(mb, mv, "o", color=colors[k], markersize=5,
                        markerfacecolor="none", markeredgewidth=1.2,
                        label=None if k != keys[-1] else "measured (stage 1)")
    ax.set_xlabel("factor bit width")
    ax.set_ylabel("success fraction (%)")
    ax.set_title(f"ECM success vs factor size (B1={b1:g}, {b2_note(b1, b2_arg, factor)}); "
                 "dashed = stage1, solid = stage1+stage2"
                 + (f"; circles = measured at B1={b1}" if mix else ""))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, "success_vs_bit.png")
    plt.close(fig)


def _plot_pred_vs_B1(bit: int, b2: float | None, factor: float, keys: list[str],
                     colors: dict[str, str]) -> None:
    b1s = np.logspace(2, 9, 48)
    d_eff = model.load_representative_d_eff()
    fig, ax = plt.subplots(figsize=(9, 6))
    for k in keys:
        D = d_eff.get(k)
        if D is None:
            continue
        y1 = [100.0 * model.predict_fraction(k, bit, x, D) for x in b1s]
        # B2 follows the same default: 100*B1 per point (or one fixed B2 for all points)
        y12 = [100.0 * model.predict_fraction(k, bit, x, D, effective_b2(x, b2, factor))
               for x in b1s]
        ax.plot(b1s, y1, "--", color=colors[k], alpha=0.8, linewidth=1.3)
        ax.plot(b1s, y12, "-", color=colors[k], linewidth=1.8, label=label_of(k))
    ax.set_xscale("log")
    ax.set_xlabel("B1 (log scale)")
    ax.set_ylabel("success fraction (%)")
    ax.set_title(f"ECM success vs B1 (bit={bit}, "
                 + (f"B2={b2:g} (fixed)" if b2 is not None else f"B2={factor:g}*B1 (per point)")
                 + "); dashed = stage1, solid = stage1+stage2")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, "success_vs_B1.png")
    plt.close(fig)


def cmd_predict(a: argparse.Namespace) -> None:
    keys = selected_keys(a.series)
    colors = build_colors(keys)
    if a.bits_from > a.bits_to:
        raise SystemExit("--bits-from must be <= --bits-to")
    bits = list(range(a.bits_from, a.bits_to + 1, a.bits_step))
    b1_int = int(a.B1)   # measure files are indexed by integer B1
    blocks = load_blocks(available_bits(b1_int), b1_int) if a.mix_empirical else {}
    if a.mix_empirical and not blocks:
        print(f"note: --mix-empirical requested but no measure_*_{b1_int}.json exists; "
              f"available B1 values: {sorted({int(m.group(2)) for f in OUT_DIR.glob('measure_*.json') if (m := _MEASURE_RE.match(f.name))})}")
    print(f"stage 2 bound: {b2_note(a.B1, a.B2, a.b2_factor)}"
          + ("" if a.B2 is not None else "  (use --B2 to pin an absolute value, --b2-factor to change the ratio)"))
    _plot_pred_vs_bit(a.B1, a.B2, a.b2_factor, bits, keys, colors, blocks, a.mix_empirical)
    _plot_pred_vs_B1(a.bit, a.B2, a.b2_factor, keys, colors)
    print(f"wrote {PLOT_DIR / 'success_vs_bit.png'} and {PLOT_DIR / 'success_vs_B1.png'}")


# ---------------------------------------------------------------------------
def cmd_list(_a: argparse.Namespace) -> None:
    print(f"{'key':<16} {'family':<7} {'source':<8} label")
    print("-" * 64)
    for k, v in SERIES.items():
        print(f"{k:<16} {v['family']:<7} {v['source']:<8} {v['label']}")
    bits = available_bits(DEFAULT_B1)
    print()
    print(f"measure files at B1={DEFAULT_B1}: "
          f"{('bits ' + str(bits[0]) + '-' + str(bits[-1]) + f' ({len(bits)} sizes)') if bits else 'none'}")
    for b1 in sorted({int(m.group(2)) for f in OUT_DIR.glob('measure_*.json')
                      if (m := _MEASURE_RE.match(f.name))}):
        if b1 != DEFAULT_B1:
            print(f"measure files at B1={b1}: bits {available_bits(b1)}")


def main(argv=None) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser(prog="ecm_plot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("empirical", help="plot empirical sweep data (auto-discovers available bits)")
    e.add_argument("--b1", type=int, default=DEFAULT_B1, help="B1 of the measure files (default 256)")
    e.add_argument("--bits", default=None, help="15-30 | 15,17,20 | all (default: auto-discover)")
    e.add_argument("--series", default=None, help="comma separated keys, see `list` (default: all)")
    e.set_defaults(func=cmd_empirical)

    p = sub.add_parser("predict", help="plot prediction (stage1 vs stage1+stage2)")
    p.add_argument("--B1", type=float, default=float(DEFAULT_B1))
    p.add_argument("--B2", type=float, default=None,
                   help="stage-2 bound; **default: 100 * B1** (see --b2-factor)")
    p.add_argument("--b2-factor", type=float, default=B2_FACTOR_DEFAULT,
                   help=f"B2/B1 ratio used when --B2 is not given (default {B2_FACTOR_DEFAULT:g})")
    p.add_argument("--bit", type=int, default=40, help="bit width for the vs-B1 plot")
    p.add_argument("--bits-from", type=int, default=15)
    p.add_argument("--bits-to", type=int, default=130)
    p.add_argument("--bits-step", type=int, default=5)
    p.add_argument("--series", default=None, help="comma separated keys, see `list` (default: all)")
    p.add_argument("--mix-empirical", action="store_true",
                   help="overlay measured stage-1 points from measure_<bit>_<B1>.json")
    p.set_defaults(func=cmd_predict)

    sub.add_parser("list", help="print the SERIES table and available data").set_defaults(func=cmd_list)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
