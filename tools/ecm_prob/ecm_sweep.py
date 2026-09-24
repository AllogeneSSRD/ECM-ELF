"""ecm_sweep.py — 用户 CLI：素数生成 / 经验扫掠 / 汇总报告。

整合原 gen_primes.py、sweep.py、report.py 的用户功能。库逻辑在 data.py / model.py。

要点（2026-09-24 更新）：
  * 素数集有规模上限：bit <= 30 穷举；**bit >= 31 只生成 65536 个**（区间内均匀窗口采样，
    见 data.py 顶部注释）——穷举 bit31 已经是几百 MB，再往上不可行。
  * B1 可由命令行给出（--B1，支持 1e5 这类写法），结果按 <bit>_<B1> 分文件存放。
  * 运算前先查缓存：`out/measure_<bit>_<B1>.json` 存在且内容自洽就跳过（--force 可强制重算），
    素数集同样按 manifest 的 sha256 判断是否可复用。

用法：
  python ecm_sweep.py primes                            # 生成 15-30（31+ 自动改为采样 65536）
  python ecm_sweep.py primes 31 32 --count 65536
  python ecm_sweep.py sweep                             # 跨位宽测量（默认 bit 15-30, B1=256）
  python ecm_sweep.py sweep --B1 1e5                    # 换 B1（与 B1=256 的结果并存）
  python ecm_sweep.py sweep 20 22 --B1 1e4 --force
  python ecm_sweep.py report 20 256
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import data
import model
import curves
import estimates
import ecmath

OUT_DIR = Path(__file__).resolve().parent / "out"

B1_DEFAULT = 256
SAMPLE_N = data.SAMPLE_COUNT
SAMPLE_SEED = 20260101
# 两个不同的上限，别混：
#   EXHAUSTIVE_MAX_BIT        —— **生成**上限：bit > 30 只生成 65536 个素数（data.py 顶部）
#   MEASURE_EXHAUSTIVE_MAX_BIT —— **测量**口径：bit <= 20 量全部素数，21+ 从素数表里采 65536 个
#     （历史行为：bits21-30 的素数表有 120 万~2620 万个，但 measure_*_256.json 一直是
#      n_primes=65536；把这两个阈值合成一个会让 21-30 的缓存判断失效并触发数小时的重算。）
EXHAUSTIVE_MAX_BIT = data.EXHAUSTIVE_MAX_BIT
MEASURE_EXHAUSTIVE_MAX_BIT = 20
DEFAULT_BITS = list(range(15, EXHAUSTIVE_MAX_BIT + 1))


def _parse_bit_spec(s: str) -> list[int]:
    """位宽写法：`31`、`31-40`（闭区间）、`31..40`（同上，顺手支持）。

    位置参数收的是**列表**（`sweep 15 16 21` = 三个 bit），历史上 `sweep 31 40` 被误读成
    "区间 31..40" 从而只算了 31 和 40 两个 bit（2026-09-24 实际踩到），所以显式支持区间写法。
    """
    t = s.strip().replace("..", "-")
    if "-" in t:
        lo_s, _, hi_s = t.partition("-")
        try:
            lo, hi = int(lo_s), int(hi_s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"bad bit spec: {s!r} (use 31, 31-40 or 31..40)")
        if hi < lo:
            raise argparse.ArgumentTypeError(f"bad bit range: {s!r} (hi < lo)")
        return list(range(lo, hi + 1))
    try:
        return [int(t)]
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad bit spec: {s!r} (use 31, 31-40 or 31..40)")


def _flat_bits(specs: list[list[int]] | None) -> list[int]:
    """把 `_parse_bit_spec` 的结果展平（保序去重）。"""
    out: list[int] = []
    for grp in specs or []:
        for b in grp:
            if b not in out:
                out.append(b)
    return out


def _fmt_bits(bits: list[int]) -> str:
    """回显用的位宽写法：连续区间压成 15-30，其余逗号分隔。"""
    if not bits:
        return "-"
    parts, i = [], 0
    while i < len(bits):
        j = i
        while j + 1 < len(bits) and bits[j + 1] == bits[j] + 1:
            j += 1
        parts.append(str(bits[i]) if j == i else f"{bits[i]}-{bits[j]}")
        i = j + 1
    return ",".join(parts)


# ---------------------------------------------------------------------------
# primes
# ---------------------------------------------------------------------------
def cmd_primes(a: argparse.Namespace) -> None:
    bits = _flat_bits(a.bits) or DEFAULT_BITS
    data.gen_primes(bits, count=a.count, exhaustive_max_bit=a.exhaustive_max_bit,
                    windows=a.windows, force=a.force)


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def _get_primes(bit: int, count: int) -> tuple[list[int], str, int | None]:
    """返回 (素数列表, 描述, 种子)。

    bit <= MEASURE_EXHAUSTIVE_MAX_BIT 且未指定 --count ⇒ 用整张表；
    否则从表里确定性随机取 count（默认 65536）个。表本身可能已经是采样集
    （bit >= 31 时只存 65536 个），那时"取 65536"就等于全表。
    """
    path = data.PRIME_DIR / f"bits{bit}.bin"
    if not path.exists():
        # 缺素数集就地生成（bit > EXHAUSTIVE_MAX_BIT 自动走 65536 采样，见 data.py 顶部），
        # 不再因为缺文件就退出。
        print(f"bit={bit}: no {path.name}, generating it first "
              f"(exhaustive <= {data.EXHAUSTIVE_MAX_BIT}, else sample {SAMPLE_N})")
        data.gen_primes([bit])
    allp = data.load_primes(bit)
    if count <= 0 and bit <= MEASURE_EXHAUSTIVE_MAX_BIT:
        return allp, f"exhaustive ({len(allp)})", None
    n = count if count > 0 else SAMPLE_N
    if n >= len(allp):
        return allp, f"all cached primes ({len(allp)})", None
    seed = SAMPLE_SEED + bit
    return data.sample_from_file(allp, n, seed), f"sample {n}/{len(allp)} seed={seed}", seed


def _measure_is_complete(path: Path, bit: int, B1: int, n_primes: int) -> tuple[bool, str]:
    """缓存判断：文件存在、能解析、B1/bit 对得上、素数个数与本次将要用的集合一致。"""
    if not path.exists():
        return False, "no cache"
    try:
        blk = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:                                   # 半个文件（上次被中断）
        return False, f"unreadable cache ({e.__class__.__name__})"
    if int(blk.get("B1", -1)) != int(B1):
        return False, f"cache B1={blk.get('B1')} != {B1}"
    if "bit" in blk and int(blk["bit"]) != int(bit):
        return False, f"cache bit={blk.get('bit')} != {bit}"
    if int(blk.get("n_primes", -1)) != int(n_primes):
        return False, f"cache n_primes={blk.get('n_primes')} != {n_primes}"
    names = set((blk.get("curves") or {}).keys())
    want = {c["name"] for c in curves.ROSTER}
    if not want.issubset(names):
        missing = ", ".join(sorted(want - names))
        return False, f"cache misses curve(s): {missing}"
    return True, "cached"


def cmd_sweep(a: argparse.Namespace) -> None:
    bits = _flat_bits(a.bits) or DEFAULT_BITS
    B1 = a.B1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    print(f"sweep bits {_fmt_bits(bits)} (B1={B1}, {len(bits)} size(s))")
    for bit in bits:
        primes, desc, seed = _get_primes(bit, a.count)
        out = OUT_DIR / f"measure_{bit}_{B1}.json"
        ok, why = _measure_is_complete(out, bit, B1, len(primes))
        if ok and not a.force:
            print(f"bit={bit}: {why} {out.name}  (skip; use --force to recompute)")
            skipped += 1
            continue
        if out.exists() and not a.force:
            print(f"bit={bit}: {why}; recomputing")
        t0 = time.time()
        print(f"bit={bit}: {desc}, B1={B1} ...")
        block = data.measure_primes(primes, B1, curves.ROSTER, verbose=not a.quiet,
                                    label=f"bit{bit} {desc}")
        block["bit"] = bit
        block["sampling"] = desc
        block["sample_seed"] = seed
        out.write_text(json.dumps(block, indent=2), encoding="utf-8")
        print(f"  -> {out.name}  ({time.time()-t0:.1f}s)")
        done += 1
    print(f"sweep done: {done} computed, {skipped} cached (B1={B1}, bits {_fmt_bits(bits)})"
          if bits else "sweep done: nothing to do")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def cmd_report(a: argparse.Namespace) -> None:
    bit, B1 = a.bit, a.B1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mj = OUT_DIR / f"measure_{bit}_{B1}.json"
    if not mj.exists():
        primes, desc, seed = _get_primes(bit, a.count)
        print(f"bit={bit}: no {mj.name}, measuring ({desc}, B1={B1})")
        block = data.measure_primes(primes, B1, curves.ROSTER, verbose=not a.quiet,
                                    label=f"bit{bit} {desc}")
        block["bit"] = bit
        block["sampling"] = desc
        block["sample_seed"] = seed
        mj.write_text(json.dumps(block, indent=2), encoding="utf-8")

    blk = json.loads(mj.read_text(encoding="utf-8"))
    cal = model.calibrate_block(blk, B1, curves.ROSTER)
    s = ecmath.batch_s(B1)
    L, R = 1 << (bit - 1), (1 << bit) - 1

    lines = [f"# ECM 参数化概率分析报告  (bit={bit}, B1={B1})", "",
             f"- 素数范围: [{L}, {R}]  (共 {blk['n_primes']} 个素数, {blk.get('sampling', '?')})",
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

    p = sub.add_parser("primes", help="generate/cache prime sets (exhaustive <= 30, sampled above)")
    p.add_argument("bits", type=_parse_bit_spec, nargs="*",
                   help="bit widths: 31, or a range 31-40 (default: 15-30)")
    p.add_argument("--count", type=int, default=SAMPLE_N,
                   help=f"primes to generate for bit > {EXHAUSTIVE_MAX_BIT} (default {SAMPLE_N})")
    p.add_argument("--exhaustive-max-bit", type=int, default=EXHAUSTIVE_MAX_BIT,
                   help=f"highest bit generated exhaustively (default {EXHAUSTIVE_MAX_BIT})")
    p.add_argument("--windows", type=int, default=data.SAMPLE_WINDOWS,
                   help="number of evenly spread sieve windows for the sampled sets")
    p.add_argument("--force", action="store_true", help="regenerate even if the cache is valid")
    p.set_defaults(func=cmd_primes)

    s = sub.add_parser("sweep", help="cross-bit empirical measurement")
    s.add_argument("bits", type=_parse_bit_spec, nargs="*",
                   help="bit widths: 20 22 (list) or 31-40 (range; default: 15-30)")
    s.add_argument("--B1", type=_parse_int, default=B1_DEFAULT,
                   help=f"stage-1 bound (accepts 1e5 style; default {B1_DEFAULT}); "
                        f"results go to out/measure_<bit>_<B1>.json")
    s.add_argument("--count", type=int, default=0,
                   help="primes per bit (0 = exhaustive below the limit, else all cached)")
    s.add_argument("--exhaustive-max-bit", type=int, default=EXHAUSTIVE_MAX_BIT)
    s.add_argument("--force", action="store_true", help="recompute even if a valid result exists")
    s.add_argument("--quiet", action="store_true", help="do not print the per-curve lines")
    s.set_defaults(func=cmd_sweep)

    r = sub.add_parser("report", help="generate out/report.md + summary.csv")
    r.add_argument("bit", type=_parse_int, default=20)
    r.add_argument("B1", type=_parse_int, default=B1_DEFAULT)
    r.add_argument("--count", type=int, default=0)
    r.add_argument("--exhaustive-max-bit", type=int, default=EXHAUSTIVE_MAX_BIT)
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
