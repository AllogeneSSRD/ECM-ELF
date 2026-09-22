"""ecm_plot.py — 用户 CLI：绘图（经验扫掠 + 预测）。

整合原 plot.py（经验扫掠图）与 plot_prob.py（预测图）。

用法：
  python ecm_plot.py empirical                     # 逐 bit 柱状图 + 成功率/bit + D_eff/bit
  python ecm_plot.py predict --B1 256 --bit 40     # 预测图（stage1 vs stage1+stage2）
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model
import curves

OUT_DIR = Path(__file__).resolve().parent / "out"
PLOT_DIR = OUT_DIR / "plots"
B1_SWEEP = 256

KEY = ["suyama_s10", "param2_s10", "param3_s10",
       "edwards_Z12", "edwards_Z2xZ8", "pm1"]
LABEL = {"suyama_s10": "Suyama (Z12)", "param2_s10": "param2 (Z6)",
         "param3_s10": "param3 (Z4)", "edwards_Z12": "Edwards Z12",
         "edwards_Z2xZ8": "Edwards Z2xZ8", "pm1": "p-1"}
COLOR = {"suyama_s10": "#c0392b", "param2_s10": "#2c3e50", "param3_s10": "#16a085",
         "edwards_Z12": "#27ae60", "edwards_Z2xZ8": "#d35400", "pm1": "#7f8c8d"}
SHORT = {"suyama_s10": "Suyama s10", "suyama_srand": "Suyama srand",
         "param1_s10": "param1", "param2_s10": "param2", "param3_s10": "param3",
         "edwards_Z4": "Edw Z4", "edwards_Z2xZ4": "Edw Z2xZ4",
         "edwards_Z12": "Edw Z12", "edwards_Z2xZ8": "Edw Z2xZ8",
         "pm1": "p-1", "pp1": "p+1"}


# ---------------------------------------------------------------------------
# empirical (原 plot.py)
# ---------------------------------------------------------------------------
def _load_measure(bit: int) -> dict | None:
    f = OUT_DIR / f"measure_{bit}_{B1_SWEEP}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    if "curves" in d:
        return d
    for v in d.values():
        if isinstance(v, dict) and "curves" in v:
            return v
    return None


def _plot_bars(blocks: dict[int, dict]) -> None:
    names = [c["name"] for c in curves.ROSTER]
    ncols = 4
    nrows = math.ceil(len(blocks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                             squeeze=False)
    for idx, bit in enumerate(sorted(blocks)):
        ax = axes[idx // ncols][idx % ncols]
        blk = blocks[bit]
        fracs = [blk["curves"][n]["pct"] for n in names]
        ax.bar(range(len(names)), fracs, width=0.8)
        ax.set_title(f"bit {bit} (n={blk['n_primes']})", fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([SHORT[n] for n in names], rotation=90, fontsize=6)
        ax.set_ylim(0, 100)
    for idx in range(len(blocks), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(f"ECM stage-1 success fraction by curve, per bit (B1={B1_SWEEP})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PLOT_DIR / "bars_per_bit.png", dpi=150)
    plt.close(fig)


def _plot_emp_success(blocks: dict[int, dict]) -> None:
    names = [c["name"] for c in curves.ROSTER]
    bits = sorted(blocks)
    fig, ax = plt.subplots(figsize=(9, 6))
    for n in names:
        y = [blocks[b]["curves"][n]["pct"] for b in bits]
        ax.plot(bits, y, "-o", label=SHORT[n], markersize=4, linewidth=1.6)
    ax.set_xlabel("factor bit width")
    ax.set_ylabel("success fraction (%)")
    ax.set_title(f"ECM stage-1 success vs factor size (B1={B1_SWEEP})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "emp_success_vs_bit.png", dpi=150)
    plt.close(fig)


def _plot_emp_d_eff(blocks: dict[int, dict]) -> None:
    names = [c["name"] for c in curves.ROSTER]
    bits = sorted(blocks)
    fig, ax = plt.subplots(figsize=(9, 6))
    for n in names:
        d = [model.calibrate_block(blocks[b], B1_SWEEP, curves.ROSTER)["curves"][n]["D_eff"]
             for b in bits]
        ax.plot(bits, d, "-o", label=SHORT[n], markersize=4, linewidth=1.6)
    ax.set_xlabel("factor bit width")
    ax.set_ylabel("effective divisor D_eff")
    ax.set_title(f"D_eff vs factor size (B1={B1_SWEEP})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "emp_d_eff_vs_bit.png", dpi=150)
    plt.close(fig)


def cmd_empirical(_a: argparse.Namespace) -> None:
    blocks = {b: blk for b in range(15, 26) if (blk := _load_measure(b))}
    if not blocks:
        raise SystemExit("no measure files; run `ecm_sweep.py sweep` first")
    _plot_bars(blocks)
    _plot_emp_success(blocks)
    _plot_emp_d_eff(blocks)
    print(f"wrote empirical plots to {PLOT_DIR}")


# ---------------------------------------------------------------------------
# predict (原 plot_prob.py)
# ---------------------------------------------------------------------------
def _ys(name: str, bits, B1: int, B2: float | None):
    D = model.load_representative_d_eff()[name]
    y1 = [100.0 * model.predict_fraction(name, b, B1, D) for b in bits]
    y12 = [100.0 * model.predict_fraction(name, b, B1, D, B2) for b in bits]
    return y1, y12


def _plot_pred_vs_bit(B1: int, B2: float | None) -> None:
    if B2 is None:
        B2 = 100.0 * B1
    bits = list(range(15, 131, 5))
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in KEY:
        y1, y12 = _ys(name, bits, B1, B2)
        ax.plot(bits, y1, "--", color=COLOR[name], alpha=0.75, linewidth=1.3)
        ax.plot(bits, y12, "-", color=COLOR[name], linewidth=1.8, label=LABEL[name])
    ax.set_xlabel("factor bit width")
    ax.set_ylabel("success fraction (%)")
    ax.set_title(f"ECM success vs factor size (B1={B1}, B2={B2:g}); "
                 "dashed = stage1, solid = stage1+stage2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "success_vs_bit.png", dpi=150)
    plt.close(fig)


def _plot_pred_vs_B1(bit: int, B2: float | None) -> None:
    B1s = np.logspace(2, 9, 48)
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in KEY:
        D = model.load_representative_d_eff()[name]
        y1 = [100.0 * model.predict_fraction(name, bit, B1, D) for B1 in B1s]
        y12 = [100.0 * model.predict_fraction(name, bit, B1, D,
                                              (B2 if B2 is not None else 100.0 * B1))
               for B1 in B1s]
        ax.plot(B1s, y1, "--", color=COLOR[name], alpha=0.75, linewidth=1.3)
        ax.plot(B1s, y12, "-", color=COLOR[name], linewidth=1.8, label=LABEL[name])
    ax.set_xscale("log")
    ax.set_xlabel("B1 (log scale)")
    ax.set_ylabel("success fraction (%)")
    ax.set_title(f"ECM success vs B1 (bit={bit}); dashed = stage1, solid = stage1+stage2")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "success_vs_B1.png", dpi=150)
    plt.close(fig)


def cmd_predict(a: argparse.Namespace) -> None:
    _plot_pred_vs_bit(a.B1, a.B2)
    _plot_pred_vs_B1(a.bit, a.B2)
    print(f"wrote {PLOT_DIR / 'success_vs_bit.png'} and {PLOT_DIR / 'success_vs_B1.png'}")


def main(argv=None) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser(prog="ecm_plot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("empirical", help="plot empirical sweep data").set_defaults(func=cmd_empirical)

    p = sub.add_parser("predict", help="plot prediction (stage1 vs stage1+stage2)")
    p.add_argument("--B1", type=int, default=256)
    p.add_argument("--B2", type=float, default=None)
    p.add_argument("--bit", type=int, default=40)
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
