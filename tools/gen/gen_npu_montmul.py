#!/usr/bin/env python3
"""Generate RyzenAI/npu_montmul.py (UTF-8). Run: python tools/gen_npu_montmul.py"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "RyzenAI" / "npu_montmul.py"

CONTENT = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU Montgomery mul/sqr microbench - mirrors OpenCL mont.cl / mont_priv.cl (CIOS).

Run environment:
  conda activate ryzen-ai-1.7.1
  python RyzenAI/npu_montmul.py --bits 512 10000 128 2
  python RyzenAI/npu_montmul.py --self-test

Operations (cgbn_mont_mul / cgbn_mont_sqr):
  mont_mul  r = a * b * R^{-1} mod n   (R = 2^bits)
  mont_sqr  r = a * a * R^{-1} mod n
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

SELF_TEST_WIDTHS = (512, 1024, 2048, 4096)
DEFAULT_LIMB_SWEEP = (8, 16, 32, 64)
SUPPORTED_LIMB_BITS = (8, 16, 32, 64)
DEFAULT_LIMB_BITS_SWEEP = SUPPORTED_LIMB_BITS


class MpWidth:
    """N-bit integer with configurable limb width (8/16/32/64 bits per limb)."""

    MIN_BITS = 512
    MAX_BITS = 4096
    DEFAULT_LIMB_BITS = 32
    MIN_LIMBS_32 = MIN_BITS // DEFAULT_LIMB_BITS
    MAX_LIMBS_32 = MAX_BITS // DEFAULT_LIMB_BITS
    SWEEP_MIN_LIMBS = 8

    __slots__ = ("bits", "limb_bits")

    def __init__(self, bits: int, limb_bits: int = DEFAULT_LIMB_BITS) -> None:
        err = MpWidth.validate(bits, limb_bits)
        if err:
            raise ValueError(err)
        self.bits = bits
        self.limb_bits = limb_bits

    @classmethod
    def from_limb_count(
        cls,
        limb_count: int,
        limb_bits: int = DEFAULT_LIMB_BITS,
        *,
        allow_sweep: bool = False,
    ) -> "MpWidth":
        err = cls.validate_limb_count(limb_count, limb_bits, allow_sweep=allow_sweep)
        if err:
            raise ValueError(err)
        bits = limb_count * limb_bits
        if not allow_sweep:
            err = cls.validate(bits, limb_bits)
            if err:
                raise ValueError(err)
        obj = object.__new__(cls)
        obj.bits = bits
        obj.limb_bits = limb_bits
        return obj

    from_limbs = from_limb_count

    @staticmethod
    def validate(bits: int, limb_bits: int = DEFAULT_LIMB_BITS) -> Optional[str]:
        if limb_bits not in SUPPORTED_LIMB_BITS:
            return f"limb_bits must be one of {SUPPORTED_LIMB_BITS}"
        if bits < MpWidth.MIN_BITS or bits > MpWidth.MAX_BITS:
            return f"bits must be in [{MpWidth.MIN_BITS}, {MpWidth.MAX_BITS}]"
        if bits % limb_bits != 0:
            return f"bits ({bits}) must be a multiple of limb_bits ({limb_bits})"
        return None

    @staticmethod
    def validate_limb_count(
        limb_count: int,
        limb_bits: int = DEFAULT_LIMB_BITS,
        *,
        allow_sweep: bool = False,
    ) -> Optional[str]:
        if limb_bits not in SUPPORTED_LIMB_BITS:
            return f"limb_bits must be one of {SUPPORTED_LIMB_BITS}"
        min_limbs = MpWidth.SWEEP_MIN_LIMBS if (allow_sweep and limb_bits == 32) else 1
        max_limbs = MpWidth.MAX_BITS // limb_bits
        if limb_count < min_limbs or limb_count > max_limbs:
            return f"limb count must be in [{min_limbs}, {max_limbs}] for {limb_bits}b/limb"
        return None

    @staticmethod
    def validate_limbs(limbs: int, *, allow_sweep: bool = False) -> Optional[str]:
        return MpWidth.validate_limb_count(limbs, MpWidth.DEFAULT_LIMB_BITS, allow_sweep=allow_sweep)

    @property
    def limbs(self) -> int:
        return self.bits // self.limb_bits

    @property
    def limb_mask(self) -> int:
        return (1 << self.limb_bits) - 1


def limbs_from_int(value: int, width: MpWidth) -> np.ndarray:
    mask = width.limb_mask
    lb = width.limb_bits
    return np.array([(value >> (lb * i)) & mask for i in range(width.limbs)], dtype=np.uint64)


def int_from_limbs(arr: np.ndarray, width: MpWidth) -> int:
    result = 0
    mask = np.uint64(width.limb_mask)
    lb = width.limb_bits
    flat = np.asarray(arr, dtype=np.uint64).reshape(-1)
    for i in range(flat.shape[0]):
        result |= int(flat[i] & mask) << (lb * i)
    return result


def opencl_test_vectors(width: MpWidth) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same fixed values as opencl_ecm_addsub_bench.cpp."""
    bits = width.bits
    mask = (1 << bits) - 1
    n_val = (1 << (bits - 1)) - 109
    a_val = ((1 << (bits - 1)) - 991) % n_val
    b_val = ((1 << (bits - 1)) - 8218291649) % n_val
    return (
        limbs_from_int(n_val & mask, width),
        limbs_from_int(a_val & mask, width),
        limbs_from_int(b_val & mask, width),
    )


def tile_vectors(a: np.ndarray, b: np.ndarray, n: np.ndarray, instances: int):
    a_batch = np.tile(a.astype(np.uint64), (instances, 1))
    b_batch = np.tile(b.astype(np.uint64), (instances, 1))
    n_batch = np.tile(n.astype(np.uint64), (instances, 1))
    return a_batch, b_batch, n_batch


def inv_limb_odd(x: int, limb_bits: int) -> int:
    """Newton iteration for inverse modulo 2^limb_bits; x must be odd."""
    mod = 1 << limb_bits
    x &= mod - 1
    y = 1
    # Each step doubles precision; need log2(limb_bits) steps (5 for 32b, 6 for 64b).
    for _ in range(max(1, (limb_bits - 1).bit_length())):
        y = (y * (2 - x * y)) & (mod - 1)
    return y


def find_np0(n_limb0: int, limb_bits: int) -> int:
    """np0 = -n0^{-1} mod 2^limb_bits (matches OpenCL inv32_odd + negate)."""
    n0 = int(n_limb0) & ((1 << limb_bits) - 1)
    if (n0 & 1) == 0:
        raise ValueError("modulus low limb must be odd for Montgomery")
    inv = inv_limb_odd(n0, limb_bits)
    return (-inv) & ((1 << limb_bits) - 1)


def np0_from_modulus(n_vec: np.ndarray, width: MpWidth) -> int:
    return find_np0(int(n_vec[0]), width.limb_bits)


def ref_mont_mul(a: int, b: int, n: int, bits: int) -> int:
    rinv = pow(1 << bits, -1, n)
    return (a * b * rinv) % n


def ref_mont_sqr(a: int, n: int, bits: int) -> int:
    rinv = pow(1 << bits, -1, n)
    return (a * a * rinv) % n


def _cmp_ge_limbs(
    t: np.ndarray,
    t_hi: np.ndarray,
    n_arr: np.ndarray,
    limbs: int,
) -> np.ndarray:
    """Per-row lexicographic compare: t >= n (with extension limbs t[limbs], t_hi)."""
    batch = t.shape[0]
    ge = (t_hi != 0) | (t[:, limbs] != 0)
    pending = ~ge
    for i in range(limbs - 1, -1, -1):
        if not np.any(pending):
            break
        tv = t[:, i]
        nv = n_arr[:, i]
        gt = pending & (tv > nv)
        lt = pending & (tv < nv)
        ge = ge | gt
        pending = pending & ~gt & ~lt
    return ge


def _conditional_sub_limbs(
    t: np.ndarray,
    n_arr: np.ndarray,
    ge: np.ndarray,
    width: MpWidth,
) -> None:
    lb = width.limb_bits
    mask = np.uint64(width.limb_mask)
    limbs = width.limbs
    borrow = np.zeros(t.shape[0], dtype=np.uint64)
    for j in range(limbs):
        tv = t[:, j]
        nv = n_arr[:, j]
        w = tv - nv - borrow
        new_t = w & mask
        new_borrow = (tv < nv + borrow).astype(np.uint64)
        t[:, j] = np.where(ge, new_t, tv)
        borrow = np.where(ge, new_borrow, np.uint64(0))


def mont_mul_cios_row(
    a_row: np.ndarray,
    b_row: np.ndarray,
    n_row: np.ndarray,
    np0: int,
    width: MpWidth,
) -> np.ndarray:
    """CIOS Montgomery mul for one instance (mont.cl reference)."""
    lb = width.limb_bits
    mask = width.limb_mask
    limbs = width.limbs
    t = [0] * (limbs + 1)
    t_hi = 0
    B = [int(x) for x in b_row]
    N = [int(x) for x in n_row]
    A = [int(x) for x in a_row]

    for i in range(limbs):
        ai = A[i]
        carry = 0
        for j in range(limbs):
            uv = t[j] + ai * B[j] + carry
            t[j] = uv & mask
            carry = uv >> lb
        uvh = t[limbs] + carry
        t[limbs] = uvh & mask
        t_hi += uvh >> lb

        m = (t[0] * np0) & mask
        carry = 0
        for j in range(limbs):
            uv = t[j] + m * N[j] + carry
            if j > 0:
                t[j - 1] = uv & mask
            carry = uv >> lb
        top = t[limbs] + carry
        t[limbs - 1] = top & mask
        top2 = t_hi + (top >> lb)
        t[limbs] = top2 & mask
        t_hi = top2 >> lb

    ge = t_hi != 0 or t[limbs] != 0
    if not ge:
        for i in range(limbs - 1, -1, -1):
            if t[i] > N[i]:
                ge = True
                break
            if t[i] < N[i]:
                ge = False
                break

    if ge:
        borrow = 0
        for i in range(limbs):
            tv = t[i]
            nv = N[i]
            w = tv - nv - borrow
            t[i] = w & mask
            borrow = 1 if tv < nv + borrow else 0

    return np.array(t[:limbs], dtype=np.uint64)


def mont_sqr_cios_row(
    a_row: np.ndarray,
    n_row: np.ndarray,
    np0: int,
    width: MpWidth,
) -> np.ndarray:
    return mont_mul_cios_row(a_row, a_row, n_row, np0, width)


def numpy_mont_mul_cios_vec(
    width: MpWidth,
    a: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    np0: int,
) -> np.ndarray:
    """Batched CIOS Montgomery mul (vectorized over batch dimension)."""
    if width.limb_bits == 64:
        return numpy_mont_mul_cios_legacy(width, a, b, n, np0)

    lb = width.limb_bits
    mask = np.uint64(width.limb_mask)
    limbs = width.limbs
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    batch = a.shape[0]

    t = np.zeros((batch, limbs + 1), dtype=np.uint64)
    t_hi = np.zeros(batch, dtype=np.uint64)
    np0_u = np.uint64(np0)

    for i in range(limbs):
        ai = a[:, i]
        carry = np.zeros(batch, dtype=np.uint64)
        for j in range(limbs):
            uv = t[:, j] + ai * b[:, j] + carry
            t[:, j] = uv & mask
            carry = uv >> lb
        uvh = t[:, limbs] + carry
        t[:, limbs] = uvh & mask
        t_hi += uvh >> lb

        m = (t[:, 0] * np0_u) & mask
        carry = np.zeros(batch, dtype=np.uint64)
        for j in range(limbs):
            uv = t[:, j] + m * n[:, j] + carry
            if j > 0:
                t[:, j - 1] = uv & mask
            carry = uv >> lb
        top = t[:, limbs] + carry
        t[:, limbs - 1] = top & mask
        top2 = t_hi + (top >> lb)
        t[:, limbs] = top2 & mask
        t_hi = top2 >> lb

    ge = _cmp_ge_limbs(t, t_hi, n, limbs)
    _conditional_sub_limbs(t, n, ge, width)
    return t[:, :limbs].copy()


def numpy_mont_sqr_cios_vec(
    width: MpWidth,
    a: np.ndarray,
    n: np.ndarray,
    np0: int,
) -> np.ndarray:
    return numpy_mont_mul_cios_vec(width, a, a, n, np0)


def numpy_mont_mul_cios_legacy(
    width: MpWidth,
    a: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    np0: int,
) -> np.ndarray:
    batch = a.shape[0]
    out = np.zeros((batch, width.limbs), dtype=np.uint64)
    for i in range(batch):
        out[i] = mont_mul_cios_row(a[i], b[i], n[i], np0, width)
    return out


def numpy_mont_sqr_cios_legacy(
    width: MpWidth,
    a: np.ndarray,
    n: np.ndarray,
    np0: int,
) -> np.ndarray:
    return numpy_mont_mul_cios_legacy(width, a, a, n, np0)


numpy_mont_mul = numpy_mont_mul_cios_vec
numpy_mont_sqr = numpy_mont_sqr_cios_vec


class NumpyMontBackend:
    def __init__(self, width: MpWidth):
        self.width = width

    def mont_mul(self, a: np.ndarray, b: np.ndarray, n: np.ndarray, np0: int) -> np.ndarray:
        return numpy_mont_mul(self.width, a, b, n, np0)

    def mont_sqr(self, a: np.ndarray, n: np.ndarray, np0: int) -> np.ndarray:
        return numpy_mont_sqr(self.width, a, n, np0)


def verify_mont_ops(
    width: MpWidth,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    n_vec: np.ndarray,
    backends: List[Tuple[str, NumpyMontBackend]],
) -> bool:
    bits = width.bits
    limbs = width.limbs
    mask = (1 << bits) - 1
    a0 = int_from_limbs(a_vec, width) & mask
    b0 = int_from_limbs(b_vec, width) & mask
    n0 = int_from_limbs(n_vec, width) & mask
    np0 = np0_from_modulus(n_vec, width)

    expects = {
        "mont_mul": ref_mont_mul(a0, b0, n0, bits),
        "mont_sqr": ref_mont_sqr(a0, n0, bits),
    }
    a_batch = a_vec.reshape(1, limbs).astype(np.uint64)
    b_batch = b_vec.reshape(1, limbs).astype(np.uint64)
    n_batch = n_vec.reshape(1, limbs).astype(np.uint64)
    runners = {
        "mont_mul": lambda be: be.mont_mul(a_batch, b_batch, n_batch, np0),
        "mont_sqr": lambda be: be.mont_sqr(a_batch, n_batch, np0),
    }

    all_ok = True
    for be_name, backend in backends:
        for op_name, expected in expects.items():
            got = int_from_limbs(runners[op_name](backend)[0], width)
            ok = got == expected
            if ok:
                print(f"  [{be_name}:{op_name}] verify: PASS")
            else:
                print(f"  [{be_name}:{op_name}] verify: FAIL (got={got}, expect={expected})")
                all_ok = False
    return all_ok


def verify_vec_matches_legacy(
    width: MpWidth,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    n_vec: np.ndarray,
) -> bool:
    limbs = width.limbs
    np0 = np0_from_modulus(n_vec, width)
    a_batch = a_vec.reshape(1, limbs).astype(np.uint64)
    b_batch = b_vec.reshape(1, limbs).astype(np.uint64)
    n_batch = n_vec.reshape(1, limbs).astype(np.uint64)
    pairs = (
        ("mont_mul", numpy_mont_mul_cios_vec, numpy_mont_mul_cios_legacy),
        ("mont_sqr", numpy_mont_sqr_cios_vec, numpy_mont_sqr_cios_legacy),
    )
    all_ok = True
    for name, fast, slow in pairs:
        if name == "mont_mul":
            got = fast(width, a_batch, b_batch, n_batch, np0)
            ref = slow(width, a_batch, b_batch, n_batch, np0)
        else:
            got = fast(width, a_batch, n_batch, np0)
            ref = slow(width, a_batch, n_batch, np0)
        if np.array_equal(got, ref):
            print(f"  [vec_vs_legacy:{name}] PASS")
        else:
            print(f"  [vec_vs_legacy:{name}] FAIL")
            all_ok = False
    return all_ok


def run_self_test(
    widths: Tuple[int, ...] = SELF_TEST_WIDTHS,
    limb_bits_list: Tuple[int, ...] = SUPPORTED_LIMB_BITS,
) -> bool:
    print("npu_montmul self-test")
    all_ok = True
    for bits in widths:
        for lb in limb_bits_list:
            err = MpWidth.validate(bits, lb)
            if err:
                print(f"  [{bits}-bit, {lb}b/limb] width: FAIL ({err})")
                all_ok = False
                continue
            width = MpWidth(bits, lb)
            n_vec, a_vec, b_vec = opencl_test_vectors(width)
            backend = NumpyMontBackend(width)
            print(f"\n--- {bits}-bit, {lb}b/limb (limbs={width.limbs}), np0=0x{np0_from_modulus(n_vec, width):x} ---")
            if not verify_mont_ops(width, a_vec, b_vec, n_vec, [("numpy", backend)]):
                all_ok = False
            if not verify_vec_matches_legacy(width, a_vec, b_vec, n_vec):
                all_ok = False
    print()
    if all_ok:
        print("self-test: ALL PASS")
    else:
        print("self-test: FAILED")
    return all_ok


def bench_op(fn: Callable[[], None], warmup: int, iters: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        for _ in range(iters):
            fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def mont_mul_bench_chain(
    backend: NumpyMontBackend,
    a: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    np0: int,
    kernel_iterations: int,
) -> np.ndarray:
    """Mirror ecm_mont_mul_priv_bench: first mul(a,b), then mul(out,b)."""
    out = backend.mont_mul(a, b, n, np0)
    for _ in range(kernel_iterations - 1):
        out = backend.mont_mul(out, b, n, np0)
    return out


def mont_sqr_bench_chain(
    backend: NumpyMontBackend,
    a: np.ndarray,
    n: np.ndarray,
    np0: int,
    kernel_iterations: int,
) -> np.ndarray:
    """Mirror ecm_mont_sqr_priv_bench: first sqr(a), then mul(out,out)."""
    out = backend.mont_sqr(a, n, np0)
    for _ in range(kernel_iterations - 1):
        out = backend.mont_mul(out, out, n, np0)
    return out


def collect_timings(
    width: MpWidth,
    instances: int,
    kernel_iterations: int,
    launch_repeats: int,
    warmup: int,
) -> Tuple[float, List[Tuple[str, str, float]]]:
    n_vec, a_vec, b_vec = opencl_test_vectors(width)
    a_batch, b_batch, n_batch = tile_vectors(a_vec, b_vec, n_vec, instances)
    np0 = np0_from_modulus(n_vec, width)
    backend = NumpyMontBackend(width)
    op_count = float(instances * kernel_iterations * launch_repeats)
    timings: List[Tuple[str, str, float]] = []

    timings.append(
        (
            "mont_mul",
            "numpy_cios",
            bench_op(
                lambda: mont_mul_bench_chain(backend, a_batch, b_batch, n_batch, np0, kernel_iterations),
                warmup,
                1,
                launch_repeats,
            ),
        )
    )
    timings.append(
        (
            "mont_sqr",
            "numpy_cios",
            bench_op(
                lambda: mont_sqr_bench_chain(backend, a_batch, n_batch, np0, kernel_iterations),
                warmup,
                1,
                launch_repeats,
            ),
        )
    )
    return op_count, timings


def print_timings(timings: List[Tuple[str, str, float]], op_count: float) -> None:
    by_op: dict = {}
    for label, path, ms in timings:
        by_op.setdefault(label, []).append((path, ms))

    print()
    for op in ("mont_mul", "mont_sqr"):
        rows = by_op.get(op, [])
        for path, ms in rows:
            ops_s = op_count / (ms / 1000.0)
            print(f"  {op} ({path}): {ms:.4f} ms, {ops_s:.6g} mont-ops/s")
    print()
    for op in ("mont_mul", "mont_sqr"):
        rows = by_op.get(op, [])
        if rows:
            path, ms = rows[0]
            ops_s = op_count / (ms / 1000.0)
            print(f"{op}: {ms:.4f} ms, {ops_s:.6g} ops/s [{path}]")


def run_benchmark(
    width: MpWidth,
    instances: int,
    kernel_iterations: int,
    launch_repeats: int,
    warmup: int,
) -> None:
    op_count, timings = collect_timings(
        width, instances, kernel_iterations, launch_repeats, warmup
    )
    print_timings(timings, op_count)


def parse_limb_list(text: str) -> Tuple[int, ...]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("empty limb list")
    return tuple(out)


def parse_limb_bits_list(text: str) -> Tuple[int, ...]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lb = int(part)
        if lb not in SUPPORTED_LIMB_BITS:
            raise ValueError(f"unsupported limb_bits {lb}, want {SUPPORTED_LIMB_BITS}")
        out.append(lb)
    if not out:
        raise ValueError("empty limb_bits list")
    return tuple(out)


def run_limb_bits_sweep(
    bits: int,
    limb_bits_list: Tuple[int, ...],
    instances: int,
    kernel_iterations: int,
    launch_repeats: int,
    warmup: int,
) -> bool:
    print(
        f"Mont limb-bits sweep: {bits}-bit, limb_bits={limb_bits_list}, "
        f"kernel_iterations={kernel_iterations}, instances={instances}, launch_repeats={launch_repeats}"
    )
    rows: List[dict] = []
    all_ok = True
    for lb in limb_bits_list:
        err = MpWidth.validate(bits, lb)
        if err:
            print(f"  [{bits}b, {lb}b/limb] SKIP: {err}")
            all_ok = False
            continue
        width = MpWidth(bits, lb)
        n_vec, a_vec, b_vec = opencl_test_vectors(width)
        backend = NumpyMontBackend(width)
        if not verify_mont_ops(width, a_vec, b_vec, n_vec, [("numpy", backend)]):
            all_ok = False
            continue
        op_count, timings = collect_timings(
            width, instances, kernel_iterations, launch_repeats, warmup
        )
        mul_ms = next(ms for op, _, ms in timings if op == "mont_mul")
        sqr_ms = next(ms for op, _, ms in timings if op == "mont_sqr")
        rows.append(
            {
                "limb_bits": lb,
                "limbs": width.limbs,
                "mul_ms": mul_ms,
                "sqr_ms": sqr_ms,
                "op_count": op_count,
            }
        )

    if rows:
        print()
        print(f"{'lb':>4} {'limbs':>5} | {'mont_mul':>10} {'mont_sqr':>10} | {'us/limb':>10}")
        print("-" * 50)
        for row in rows:
            us = row["mul_ms"] / row["limbs"] * 1000.0
            print(
                f"{row['limb_bits']:4d} {row['limbs']:5d} | "
                f"{row['mul_ms']:10.4f} {row['sqr_ms']:10.4f} | {us:10.2f}"
            )
    return all_ok


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Montgomery mul/sqr microbench (OpenCL mont.cl CIOS)",
        epilog=(
            "Environment: conda activate ryzen-ai-1.7.1\n"
            "Example: python RyzenAI/npu_montmul.py --bits 512 10000 128 2\n"
            "Limb bits sweep: python RyzenAI/npu_montmul.py --limb-bits-sweep --bits 512 500 32 1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bits", type=int, default=None, help=f"Total bit width, {MpWidth.MIN_BITS}..{MpWidth.MAX_BITS}")
    p.add_argument(
        "--limb-bits",
        type=int,
        default=MpWidth.DEFAULT_LIMB_BITS,
        choices=list(SUPPORTED_LIMB_BITS),
        help="Bits per limb (8/16/32/64)",
    )
    p.add_argument(
        "--limb-bits-sweep",
        nargs="?",
        const=",".join(str(x) for x in DEFAULT_LIMB_BITS_SWEEP),
        default=None,
        help=f"Sweep limb bit-widths at fixed --bits (default {DEFAULT_LIMB_BITS_SWEEP})",
    )
    p.add_argument("--limbs", type=int, default=None, help="Limb count at 32b/limb")
    p.add_argument(
        "--limbs-sweep",
        nargs="?",
        const=",".join(str(x) for x in DEFAULT_LIMB_SWEEP),
        default=None,
        help=f"Sweep limb counts (default {DEFAULT_LIMB_SWEEP})",
    )
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations (mont is slower than addsub)")
    p.add_argument("--self-test", action="store_true", help="Correctness tests only")
    p.add_argument("positional", nargs="*", help="kernel_iterations [instances] [launch_repeats]")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return 0 if run_self_test() else 1

    kernel_iterations = int(args.positional[0]) if len(args.positional) >= 1 else 10000
    instances = int(args.positional[1]) if len(args.positional) >= 2 else 128
    launch_repeats = int(args.positional[2]) if len(args.positional) >= 3 else 2

    if args.limb_bits_sweep is not None:
        bits = args.bits if args.bits is not None else 512
        try:
            lb_list = parse_limb_bits_list(args.limb_bits_sweep)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        ok = run_limb_bits_sweep(
            bits, lb_list, instances, kernel_iterations, launch_repeats, args.warmup
        )
        return 0 if ok else 1

    if args.limbs is not None and args.bits is not None:
        print("Use only one of --bits or --limbs", file=sys.stderr)
        return 1

    if args.limbs is not None:
        err = MpWidth.validate_limbs(args.limbs, allow_sweep=True)
        if err:
            print(err, file=sys.stderr)
            return 1
        width = MpWidth.from_limb_count(args.limbs, allow_sweep=True)
    else:
        bits = args.bits if args.bits is not None else 512
        err = MpWidth.validate(bits, args.limb_bits)
        if err:
            print(err, file=sys.stderr)
            return 1
        width = MpWidth(bits, args.limb_bits)

    print(
        f"Montgomery microbench: {width.bits}-bit ({width.limb_bits}b/limb x {width.limbs} limbs), "
        f"kernel_iterations={kernel_iterations}, instances={instances}, launch_repeats={launch_repeats}"
    )

    n_vec, a_vec, b_vec = opencl_test_vectors(width)
    backend = NumpyMontBackend(width)
    np0 = np0_from_modulus(n_vec, width)
    print(f"  np0=0x{np0:0{width.limb_bits // 4}x} (low limb of N)")

    print()
    if not verify_mont_ops(width, a_vec, b_vec, n_vec, [("numpy", backend)]):
        print("\nVerify FAILED")
        return 1

    run_benchmark(width, instances, kernel_iterations, launch_repeats, args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(CONTENT, encoding="utf-8", newline="\n")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
