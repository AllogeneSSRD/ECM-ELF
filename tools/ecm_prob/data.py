"""数据层：素数集生成/缓存/加载 + 经验命中测量。

整合原 gen_primes.py（primesieve 生成 + manifest 留痕）与 measure.py
（stage-1 命中测量）的库逻辑。CLI 入口见 ecm_sweep.py。
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import ecmath
import curves

TOOL_DIR = Path(__file__).resolve().parent
PRIME_DIR = TOOL_DIR / "data" / "primes"
PRIMESIEVE = Path(r"D:\code\MPA-OpenCl\.refactor\primesieve-12.15-win-x64\primesieve.exe")
MANIFEST = PRIME_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# 素数集规模上限（2026-09-24 用户要求）
#
# 穷举一个 bit 段的素数开销随 bit 指数增长：bits30.bin 已经是 209 MB，
# bit 31 大约翻倍（~420 MB），再往上不可接受。所以 bit >= EXHAUSTIVE_MAX_BIT+1
# 只生成 SAMPLE_COUNT 个素数，用"区间内均匀分布的 K 个窗口、每窗口取 M 个"的方式
# 采样（primesieve 没有"前 N 个素数"选项，而且取区间头部会带来采样偏差）。
# 采样总代价 ≈ K 次小窗口筛（bit=40 时每次仅几十 KB 数字），比穷举快几个数量级。
# ---------------------------------------------------------------------------
EXHAUSTIVE_MAX_BIT = 30
SAMPLE_COUNT = 65536
SAMPLE_WINDOWS = 64


# ---------------------------------------------------------------------------
# 素数集：生成 / 缓存 / 加载
# ---------------------------------------------------------------------------
def sieve_range(lo: int, hi: int) -> list[int]:
    """All primes in [lo, hi] via primesieve.exe."""
    cmd = [str(PRIMESIEVE), str(lo), str(hi), "-p", "--no-status"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [int(x) for x in out.stdout.split() if x.strip()]


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_primes(bit: int) -> dict:
    """穷举写盘（只对小 bit 用；生成策略见 gen_primes）。"""
    lo = 1 << (bit - 1)
    hi = (1 << bit) - 1
    primes = sieve_range(lo, hi)
    return _write_prime_file(bit, primes, {
        "bit": bit, "lo": lo, "hi": hi, "count": len(primes),
        "generator": "primesieve 12.15",
        "cmd": f"primesieve {lo} {hi} -p",
        "format": "uint64 little-endian, one prime per 8 bytes",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sampling": "exhaustive",
        "note": "exhaustive (all primes in [2^(b-1), 2^b-1])",
    })


def sample_primes_range(bit: int, count: int = SAMPLE_COUNT,
                        windows: int = SAMPLE_WINDOWS) -> list[int]:
    """[2^(b-1), 2^b-1] 内均匀分布的 count 个素数：分成 windows 个窗口各取一批。

    每个窗口从 (lo + i*span/windows) 开始筛一小段，取前 per 个素数。窗口宽度按
    "per 个素数 × 平均间隔 ln(hi) × 4 倍余量" 估，够取满即可 —— 因此总筛量是
    O(count·ln hi)，与 bit 段大小无关（bit=40 时总共约 5 MB 数字）。
    """
    lo = 1 << (bit - 1)
    hi = (1 << bit) - 1
    span = hi - lo
    per = max(1, count // windows)
    width = max(4096, per * max(4, int(math.log(hi))) * 4)
    got: list[int] = []
    for i in range(windows):
        if len(got) >= count:
            break
        start = lo + (span * i) // windows
        end = min(hi, start + width)
        got.extend(sieve_range(start, end)[:per])
    return got[:count]


def write_primes_sampled(bit: int, count: int = SAMPLE_COUNT,
                         windows: int = SAMPLE_WINDOWS) -> dict:
    primes = sample_primes_range(bit, count, windows)
    return _write_prime_file(bit, primes, {
        "bit": bit, "lo": 1 << (bit - 1), "hi": (1 << bit) - 1, "count": len(primes),
        "generator": "primesieve 12.15",
        "cmd": f"{windows} windows x {max(1, count // windows)} primes (evenly spread)",
        "format": "uint64 little-endian, one prime per 8 bytes",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sampling": f"window sample {len(primes)}/{windows} windows",
        "note": (f"sampled {len(primes)} primes (bit >= {EXHAUSTIVE_MAX_BIT + 1} is not "
                 f"generated exhaustively: that would be ~{2 ** (bit - 1) // max(1, bit)} primes "
                 f"and hundreds of MB)"),
    })


def _write_prime_file(bit: int, primes: list[int], entry: dict) -> dict:
    PRIME_DIR.mkdir(parents=True, exist_ok=True)
    data = array.array("Q", primes).tobytes()
    (PRIME_DIR / f"bits{bit}.bin").write_bytes(data)
    entry["sha256"] = _sha256(data)
    return entry


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")


def gen_primes(bits: list[int], count: int = SAMPLE_COUNT,
               exhaustive_max_bit: int = EXHAUSTIVE_MAX_BIT,
               windows: int = SAMPLE_WINDOWS, force: bool = False) -> None:
    """生成/缓存素数集。

    bit <= exhaustive_max_bit : 穷举（幂等，sha256 校验）
    bit >  exhaustive_max_bit : 只生成 `count` 个（窗口均匀采样）
    """
    PRIME_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    for b in bits:
        key = str(b)
        fpath = PRIME_DIR / f"bits{b}.bin"
        prev = manifest.get(key)
        sampled = b > exhaustive_max_bit
        want = count if sampled else None
        if fpath.exists() and prev and prev.get("sha256") == _sha256(fpath.read_bytes()):
            have = prev.get("count", 0)
            ok = (not force) and (want is None or have <= want)
            if ok:
                if want is not None and have < want:
                    print(f"bits{b}: cached ({have} primes, sampled; wanted {want})")
                else:
                    print(f"bits{b}: cached ({have} primes, sha256 ok)")
                continue
            if force:
                print(f"bits{b}: --force, regenerating")
        entry = write_primes_sampled(b, count, windows) if sampled else write_primes(b)
        manifest[key] = entry
        save_manifest(manifest)
        print(f"bits{b}: generated {entry['count']} primes "
              f"({entry['sampling']}) -> {fpath.name} (sha256 {entry['sha256'][:12]}...)")
    save_manifest(manifest)


def load_primes(bit: int) -> list[int]:
    data = (PRIME_DIR / f"bits{bit}.bin").read_bytes()
    arr = array.array("Q")
    arr.frombytes(data)
    if arr.itemsize != 8:
        arr.byteswap()
    return list(arr)


def sample_from_file(primes: list[int], n: int, seed: int) -> list[int]:
    """从已加载的素数表里随机取 n 个（< n 时全取），确定性种子。"""
    if n <= 0 or n >= len(primes):
        return list(primes)
    import random
    return random.Random(seed).sample(primes, n)


# ---------------------------------------------------------------------------
# 经验命中测量
# ---------------------------------------------------------------------------
def measure_primes(primes: list[int], B1: int, roster=None,
                   verbose: bool = True, label: str = "") -> dict:
    """Measure stage-1 hit fraction over an explicit prime list."""
    s = ecmath.batch_s(B1)
    roster = roster if roster is not None else curves.ROSTER
    out = {"B1": B1, "n_primes": len(primes), "label": label, "curves": {}}
    for c in roster:
        t0 = time.time()
        hits = sum(1 for p in primes if ecmath.curve_hits(c, p, s))
        frac = hits / len(primes) if primes else 0.0
        out["curves"][c["name"]] = {
            "torsion": c["torsion"], "hits": hits,
            "fraction": frac, "pct": 100.0 * frac,
        }
        if verbose:
            print(f"  {c['name']:16s} {c['torsion']:10s} "
                  f"{hits:7d}/{len(primes)} = {100.0*frac:.4f}%  ({time.time()-t0:.1f}s)")
    return out


def measure_bit(bit: int, B1: int, roster=None, verbose: bool = True) -> dict:
    out = measure_primes(load_primes(bit), B1, roster, verbose, label=f"bit{bit}")
    out["bit"] = bit
    return out
