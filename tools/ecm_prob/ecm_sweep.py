"""ecm_sweep.py — 用户 CLI：素数生成 / 经验扫掠 / 汇总报告。

整合原 gen_primes.py、sweep.py、report.py 的用户功能。库逻辑在 data.py / model.py。

用法：
  python ecm_sweep.py primes [bits...]           # 生成/缓存穷举素数集（默认 15-25）
  python ecm_sweep.py sweep [bits...]            # 跨位宽测量（15-20 穷举，21-25 采样）
  python ecm_sweep.py report <bit> <B1>          # 生成 out/report.md + summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import data
import model
import curves
import estimates
import ecmath

OUT_DIR = Path(__file__).resolve().parent / "out"

B1_DEFAULT = 256
SAMPLE_N = 65536
SAMPLE_SEED = 20260101
EXHAUSTIVE_MAX_BIT = 20


# ---------------------------------------------------------------------------
# primes
# ---------------------------------------------------------------------------
def cmd_primes(a: argparse.Namespace) -> None:
    bits = a.bits or list(range(15, 26))
    data.gen_primes(bits)


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def _get_primes(bit: int) -> tuple[list[int], str]:
    allp = data.load_primes(bit)
    if bit <= EXHAUSTIVE_MAX_BIT:
        return allp, f"exhaustive ({len(allp)})"
    rng = random.Random(SAMPLE_SEED + bit)
    return rng.sample(allp, SAMPLE_N), f"sample {SAMPLE_N}/{len(allp)} seed={SAMPLE_SEED + bit}"


def cmd_sweep(a: argparse.Namespace) -> None:
    bits = a.bits or list(range(15, 26))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for bit in bits:
        out = OUT_DIR / f"measure_{bit}_{B1_DEFAULT}.json"
        if out.exists():
            print(f"bit={bit}: cached {out.name}")
            continue
        primes, desc = _get_primes(bit)
        t0 = time.time()
        print(f"bit={bit}: {desc}, B1={B1_DEFAULT} ...")
        block = data.measure_primes(primes, B1_DEFAULT, curves.ROSTER,
                                    verbose=True, label=f"bit{bit} {desc}")
        block["bit"] = bit
        block["sampling"] = desc
        block["sample_seed"] = SAMPLE_SEED + bit if bit > EXHAUSTIVE_MAX_BIT else None
        out.write_text(json.dumps(block, indent=2), encoding="utf-8")
        print(f"  -> {out.name}  ({time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def cmd_report(a: argparse.Namespace) -> None:
    bit, B1 = a.bit, a.B1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mj = OUT_DIR / f"measure_{bit}_{B1}.json"
    if not mj.exists():
        block = data.measure_bit(bit, B1)
        mj.write_text(json.dumps(block, indent=2), encoding="utf-8")

    blk = json.loads(mj.read_text(encoding="utf-8"))
    cal = model.calibrate_block(blk, B1, curves.ROSTER)
    s = ecmath.batch_s(B1)
    L, R = 1 << (bit - 1), (1 << bit) - 1

    lines = [f"# ECM 参数化概率分析报告  (bit={bit}, B1={B1})", "",
             f"- 素数范围: [{L}, {R}]  (共 {blk['n_primes']} 个素数)",
             f"- 阶段: 仅 stage 1  (s = lcm(1..{B1}))",
             "", "## 1. 经验成功率 + 有效除子 D_eff", "",
             "| 曲线 | T | 命中数 | 成功率 | D_eff | extra=D/T |",
             "|---|---|---|---|---|---|"]
    for c in curves.ROSTER:
        name = c["name"]
        cur = blk["curves"][name]
        cc = cal["curves"][name]
        ex = cc.get("extra")
        lines.append(f"| {name} | {c['torsion']} | {cur['hits']} | "
                     f"{cur['pct']:.4f}% | {cc['D_eff']:.2f} | "
                     f"{ex:.3f}" if ex is not None else
                     f"| {name} | {c['torsion']} | {cur['hits']} | {cur['pct']:.4f}% | {cc['D_eff']:.2f} | -- |")
    lines += ["", "## 2. 预测估计（论文 §9.3 五类）", "",
              "| t | pow tZ[L,R] | pow tZ[1,R] | pow Z[1,R/t] | rho(u) | u^-u |",
              "|---|---|---|---|---|---|"]
    for t in (16, 12, 8, 4):
        est = estimates.all_estimates(L, R, t, B1, s)
        lines.append("| " + " | ".join([str(t)] + [f"{e*100:.4f}%" for e in est]) + " |")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["curve", "form", "torsion", "T", "hits", "n", "fraction",
                    "D_eff", "extra"])
        for c in curves.ROSTER:
            name = c["name"]
            cur = blk["curves"][name]
            cc = cal["curves"][name]
            w.writerow([name, c["form"], c["torsion"], c.get("T", ""), cur["hits"],
                        blk["n_primes"], cur["fraction"], f"{cc['D_eff']:.4f}",
                        f"{cc['extra']:.4f}" if cc.get("extra") else ""])
    print(f"wrote {OUT_DIR / 'report.md'} and {OUT_DIR / 'summary.csv'}")


def _parse_int(s):
    try:
        return int(s)
    except ValueError:
        return int(float(s))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ecm_sweep")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("primes", help="generate/cache exhaustive prime sets")
    p.add_argument("bits", type=int, nargs="*")
    p.set_defaults(func=cmd_primes)

    s = sub.add_parser("sweep", help="cross-bit empirical measurement")
    s.add_argument("bits", type=int, nargs="*")
    s.set_defaults(func=cmd_sweep)

    r = sub.add_parser("report", help="generate out/report.md + summary.csv")
    r.add_argument("bit", type=_parse_int, default=20)
    r.add_argument("B1", type=_parse_int, default=256)
    r.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
