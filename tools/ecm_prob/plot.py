"""Visualization for the bit-width sweep: per-bit bar charts + aggregate lines.

Reads out/measure_{bit}_{B1}.json (flat blocks) and produces, under out/plots/:

  * bars_per_bit.png      -- success fraction per curve, one subplot per bit
  * success_vs_bit.png    -- success fraction vs bit width, one line per curve
  * d_eff_vs_bit.png      -- effective divisor D_eff vs bit width (line per curve)

Run after sweep.py has produced all measure files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import calibrate  # noqa: E402
import curves  # noqa: E402

TOOL_DIR = Path(__file__).resolve().parent
OUT_DIR = TOOL_DIR / "out"
PLOT_DIR = OUT_DIR / "plots"
B1 = 256
BITS = list(range(15, 26))

SHORT = {
    "suyama_s10": "Suyama σ=10",
    "suyama_srand": "Suyama σrand",
    "param1_s10": "param1 (Z4, d sq)",
    "param2_s10": "param2 (Z6)",
    "param3_s10": "param3 (Z4)",
    "edwards_Z4": "Edw Z4",
    "edwards_Z2xZ4": "Edw Z2×Z4",
    "edwards_Z12": "Edw Z12",
    "edwards_Z2xZ8": "Edw Z2×Z8",
    "pm1": "p−1",
    "pp1": "p+1",
}
# color per curve (stable across plots)
COLORS = {
    "suyama_s10": "#c0392b",
    "suyama_srand": "#e67e22",
    "param1_s10": "#8e44ad",
    "param2_s10": "#2c3e50",
    "param3_s10": "#16a085",
    "edwards_Z4": "#2980b9",
    "edwards_Z2xZ4": "#1abc9c",
    "edwards_Z12": "#27ae60",
    "edwards_Z2xZ8": "#d35400",
    "pm1": "#7f8c8d",
    "pp1": "#95a5a6",
}


def load_measure(bit: int) -> dict | None:
    f = OUT_DIR / f"measure_{bit}_{B1}.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text(encoding="utf-8"))
    if "curves" in data:
        return data                     # flat block
    # wrapped {str(bit): block} (older report.py format) -> unwrap
    for v in data.values():
        if isinstance(v, dict) and "curves" in v:
            return v
    return None


def plot_bars(blocks: dict[int, dict]) -> None:
    names = [c["name"] for c in curves.ROSTER]
    ncols = 4
    nrows = math.ceil(len(blocks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                             squeeze=False)
    for idx, bit in enumerate(sorted(blocks)):
        ax = axes[idx // ncols][idx % ncols]
        block = blocks[bit]
        fracs = [block["curves"][n]["pct"] for n in names]
        cols = [COLORS[n] for n in names]
        ax.bar(range(len(names)), fracs, color=cols, width=0.8)
        ax.set_title(f"bit {bit}  (n={block['n_primes']})", fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([SHORT[n] for n in names], rotation=90, fontsize=6)
        ax.set_ylim(0, 100)
        ax.set_ylabel("%", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    for idx in range(len(blocks), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(f"ECM stage-1 success fraction by curve, per bit (B1={B1})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PLOT_DIR / "bars_per_bit.png", dpi=150)
    plt.close(fig)


def plot_success_vs_bit(blocks: dict[int, dict]) -> None:
    names = [c["name"] for c in curves.ROSTER]
    bits = sorted(blocks)
    fig, ax = plt.subplots(figsize=(9, 6))
    for n in names:
        y = [blocks[b]["curves"][n]["pct"] for b in bits]
        ax.plot(bits, y, "-o", label=SHORT[n], color=COLORS[n], markersize=4,
                linewidth=1.6)
    ax.set_xlabel("factor bit width")
    ax.set_ylabel("stage-1 success fraction (%)")
    ax.set_title(f"ECM stage-1 success vs factor size (B1={B1}, 15-20 exhaustive, 21-25 sampled 65536)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "success_vs_bit.png", dpi=150)
    plt.close(fig)


def plot_d_eff_vs_bit(blocks: dict[int, dict]) -> None:
    names = [c["name"] for c in curves.ROSTER]
    bits = sorted(blocks)
    fig, ax = plt.subplots(figsize=(9, 6))
    for n in names:
        d = []
        for b in bits:
            block = blocks[b]
            cal = calibrate.calibrate_block(block, B1, curves.ROSTER)
            d.append(cal["curves"][n]["D_eff"])
        ax.plot(bits, d, "-o", label=SHORT[n], color=COLORS[n], markersize=4,
                linewidth=1.6)
    ax.set_xlabel("factor bit width")
    ax.set_ylabel("effective divisor D_eff")
    ax.set_title(f"Effective divisor D_eff vs factor size (B1={B1})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "d_eff_vs_bit.png", dpi=150)
    plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = {}
    for b in BITS:
        blk = load_measure(b)
        if blk:
            blocks[b] = blk
    if not blocks:
        raise SystemExit("no measure files found; run sweep.py first")
    print(f"plotting {len(blocks)} bits: {sorted(blocks)}")
    plot_bars(blocks)
    plot_success_vs_bit(blocks)
    plot_d_eff_vs_bit(blocks)
    print(f"wrote plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
