#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU add/sub/mod microbench - mirrors opencl_ecm_addsub core ops.

Run environment:
  conda activate ryzen-ai-1.7.1
  python RyzenAI/npu_addsub.py --bits 512 10000 128 2
  python RyzenAI/npu_addsub.py --bits 2048 500 32 1

Operations (ecm_addsub_bench.cl):
  mp_add_n    r = a + b
  mp_sub_n    r = a - N
  mp_add_mod  r = (a + b) mod N   (fused speculative subtract)
  mp_sub_mod  r = (a - b) mod N

Carry paths (benchmark path labels):
  npu_*_serial      ONNX/NPU limb ops + host serial/fused mod reduce
  npu_*_kogge       ONNX/NPU limb ops + host Kogge-Stone mod reduce
  npu_*_branchfree  ONNX/NPU limb ops + host branch-free mod reduce
  cpu_serial        NumPy serial ripple carry
  cpu_kogge         NumPy Kogge-Stone parallel prefix (add/mod)
  cpu_fused         NumPy fused speculative mod subtract (legacy)
  cpu_branchfree    NumPy branch-free vector mod add (SIMD-friendly)

Edit this file directly (RyzenAI/npu_addsub.py); tools/gen_npu_addsub.py is legacy.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False

DEFAULT_ORT_EPS = [
    "VitisAIExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]

SELF_TEST_WIDTHS = (512, 1024, 2048, 4096)
DEFAULT_LIMB_SWEEP = (8, 16, 32, 64)  # limb count at 32-bit/limb
SUPPORTED_LIMB_BITS = (8, 16, 32, 64)
DEFAULT_LIMB_BITS_SWEEP = SUPPORTED_LIMB_BITS


def create_ort_session(model, preferred_eps: Optional[List[str]] = None):
    """Create ONNX Runtime session; returns (session, ep_name) or (None, None)."""
    if not HAS_ORT or model is None:
        return None, None
    eps = preferred_eps if preferred_eps is not None else DEFAULT_ORT_EPS
    available = ort.get_available_providers()
    for ep in eps:
        if ep not in available:
            continue
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if ep == "VitisAIExecutionProvider":
                providers = [(ep, {"target_engine": "NPU"})]
            else:
                providers = [ep]
            session = ort.InferenceSession(
                model.SerializeToString(),
                sess_options=opts,
                providers=providers,
            )
            return session, session.get_providers()[0]
        except Exception:
            continue
    try:
        session = ort.InferenceSession(model.SerializeToString())
        return session, session.get_providers()[0]
    except Exception:
        return None, None


def create_limb_binary_model(op: str, limbs: int, graph_name: str):
    """Element-wise int64 Add/Sub over [batch, limbs]."""
    if not HAS_ONNX:
        return None
    X = helper.make_tensor_value_info("X", TensorProto.INT64, ["batch", limbs])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, ["batch", limbs])
    Z = helper.make_tensor_value_info("Z", TensorProto.INT64, ["batch", limbs])
    node = helper.make_node(op, ["X", "Y"], ["Z"])
    graph = helper.make_graph([node], graph_name, [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    return model


class MpWidth:
    """N-bit integer with configurable limb width (8/16/32/64 bits per limb)."""

    MIN_BITS = 512
    MAX_BITS = 4096
    DEFAULT_LIMB_BITS = 32
    MIN_LIMBS_32 = MIN_BITS // DEFAULT_LIMB_BITS
    MAX_LIMBS_32 = MAX_BITS // DEFAULT_LIMB_BITS
    SWEEP_MIN_LIMBS = 8  # 256-bit at 32b/limb

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
    """Same fixed integer values as opencl_ecm_addsub_bench.cpp, any limb layout."""
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
    a_batch = np.tile(a.astype(np.int64), (instances, 1))
    b_batch = np.tile(b.astype(np.int64), (instances, 1))
    n_batch = np.tile(n.astype(np.int64), (instances, 1))
    return a_batch, b_batch, n_batch


def limbwise_add(width: MpWidth, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise limb add with serial carry propagation (O(limbs) steps)."""
    lb = width.limb_bits
    mask = width.limb_mask
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    batch, limbs = a.shape
    result = np.zeros((batch, limbs), dtype=np.uint64)
    if lb == 64:
        for i in range(batch):
            c = 0
            for j in range(limbs):
                s = int(a[i, j]) + int(b[i, j]) + c
                result[i, j] = s & mask
                c = s >> lb
        return result
    mask_np = np.uint64(mask)
    carry = np.zeros(batch, dtype=np.uint64)
    for j in range(limbs):
        s = a[:, j] + b[:, j] + carry
        result[:, j] = s & mask_np
        carry = s >> lb
    return result


def kogge_stone_prefix(g: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Parallel prefix on (generate, propagate) pairs; O(log2(limbs)) rounds."""
    limbs = g.shape[1]
    shift = 1
    while shift < limbs:
        g_shifted = np.zeros_like(g)
        p_shifted = np.zeros_like(p)
        g_shifted[:, shift:] = g[:, :-shift]
        p_shifted[:, shift:] = p[:, :-shift]
        g = g | (p & g_shifted)
        p = p & p_shifted
        shift <<= 1
    return g, p


def kogge_stone_add_with_carry(
    a_arr: np.ndarray, b_arr: np.ndarray, limb_bits: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kogge-Stone parallel carry add over [batch, limbs].
    Returns (result_limbs, carry_out) where carry_out is per-row MS carry.
    """
    if limb_bits == 64:
        batch, limbs = a_arr.shape
        out = np.zeros((batch, limbs), dtype=np.uint64)
        carry_out = np.zeros(batch, dtype=np.uint64)
        for i in range(batch):
            row, c = kogge_stone_add_row(
                np.asarray(a_arr[i], dtype=np.uint64),
                np.asarray(b_arr[i], dtype=np.uint64),
                limb_bits,
            )
            out[i] = row
            carry_out[i] = c
        return out, carry_out

    mask = np.uint64((1 << limb_bits) - 1)
    a_arr = np.asarray(a_arr, dtype=np.uint64)
    b_arr = np.asarray(b_arr, dtype=np.uint64)
    limbs = a_arr.shape[1]

    raw_sum = a_arr.astype(np.int64) + b_arr.astype(np.int64)
    g = (raw_sum >> limb_bits).astype(np.int64)
    p = ((raw_sum & int(mask)) == int(mask)).astype(np.int64)
    g, _p = kogge_stone_prefix(g, p)

    carry_in = np.zeros_like(g)
    carry_in[:, 1:] = g[:, :-1]
    result = (raw_sum + carry_in) & int(mask)
    carry_out = ((raw_sum[:, -1] + carry_in[:, -1]) >> limb_bits).astype(np.uint64)
    return result.astype(np.uint64), carry_out


def kogge_stone_add(a_arr: np.ndarray, b_arr: np.ndarray, limb_bits: int) -> np.ndarray:
    """Kogge-Stone add; carry depth O(log2(limbs)) instead of O(limbs) serial ripple."""
    result, _carry = kogge_stone_add_with_carry(a_arr, b_arr, limb_bits)
    return result


def kogge_stone_sub_borrow(
    a_arr: np.ndarray, b_arr: np.ndarray, limb_bits: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel-prefix subtract a - b over limbs (OpenCL sub_wg borrow propagation).
    Returns (result_limbs, borrow_out) where borrow_out=1 iff a < b.
    """
    if limb_bits == 64:
        batch, limbs = a_arr.shape
        out = np.zeros((batch, limbs), dtype=np.uint64)
        borrow_out = np.zeros(batch, dtype=np.uint64)
        mask = (1 << limb_bits) - 1
        for i in range(batch):
            ai = np.asarray(a_arr[i], dtype=np.uint64)
            bi = np.asarray(b_arr[i], dtype=np.uint64)
            d = [(int(ai[j]) - int(bi[j])) & mask for j in range(limbs)]
            g = [1 if int(ai[j]) < int(bi[j]) else 0 for j in range(limbs)]
            p = [1 if d[j] == 0 else 0 for j in range(limbs)]
            shift = 1
            while shift < limbs:
                new_g = list(g)
                new_p = list(p)
                for j in range(shift, limbs):
                    new_g[j] = g[j] | (p[j] & g[j - shift])
                    new_p[j] = p[j] & p[j - shift]
                g = new_g
                p = new_p
                shift <<= 1
            borrow_in = [0] * limbs
            for j in range(1, limbs):
                borrow_in[j] = g[j - 1]
            out[i] = np.array([(d[j] - borrow_in[j]) & mask for j in range(limbs)], dtype=np.uint64)
            borrow_out[i] = g[limbs - 1]
        return out, borrow_out

    mask = np.uint64((1 << limb_bits) - 1)
    a_arr = np.asarray(a_arr, dtype=np.uint64)
    b_arr = np.asarray(b_arr, dtype=np.uint64)
    d = (a_arr - b_arr) & mask
    g = (a_arr < b_arr).astype(np.int64)
    p = (d == 0).astype(np.int64)
    g, _p = kogge_stone_prefix(g, p)

    borrow_in = np.zeros_like(g)
    borrow_in[:, 1:] = g[:, :-1]
    result = (d.astype(np.int64) - borrow_in) & int(mask)
    borrow_out = g[:, -1].astype(np.uint64)
    return result.astype(np.uint64), borrow_out


def kogge_stone_add_row(a_row: np.ndarray, b_row: np.ndarray, limb_bits: int) -> Tuple[np.ndarray, int]:
    """Single-row Kogge-Stone add (64b/limb wide arithmetic)."""
    mask = (1 << limb_bits) - 1
    limbs = a_row.shape[0]
    raw_sum = [int(a_row[j]) + int(b_row[j]) for j in range(limbs)]
    g = [(s >> limb_bits) for s in raw_sum]
    p = [1 if (s & mask) == mask else 0 for s in raw_sum]

    shift = 1
    while shift < limbs:
        new_g = list(g)
        new_p = list(p)
        for j in range(shift, limbs):
            new_g[j] = g[j] | (p[j] & g[j - shift])
            new_p[j] = p[j] & p[j - shift]
        g = new_g
        p = new_p
        shift <<= 1

    carry_in = [0] * limbs
    for j in range(1, limbs):
        carry_in[j] = g[j - 1]

    result = np.array([(raw_sum[j] + carry_in[j]) & mask for j in range(limbs)], dtype=np.uint64)
    carry_out = (raw_sum[-1] + carry_in[-1]) >> limb_bits
    return result, carry_out


def limbwise_add_kogge(width: MpWidth, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return kogge_stone_add(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64), width.limb_bits)


def propagate_borrow_sub(width: MpWidth, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiprecision subtract a - b with per-limb borrow (unsigned limbs)."""
    mask = np.uint64(width.limb_mask)
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    borrow = np.zeros(batch, dtype=np.uint64)
    for j in range(limbs):
        av = a[:, j]
        bv = b[:, j]
        r[:, j] = (av - bv - borrow) & mask
        borrow = ((av < bv) | ((av == bv) & (borrow > 0))).astype(np.uint64)
    return r


def ref_mp_add_n(a: int, b: int, bits: int) -> int:
    return (a + b) & ((1 << bits) - 1)


def ref_mp_sub_n(a: int, n: int, bits: int) -> int:
    return (a - n) & ((1 << bits) - 1)


def ref_mp_add_mod(a: int, b: int, n: int) -> int:
    return (a + b) % n


def ref_mp_sub_mod(a: int, b: int, n: int) -> int:
    return (a - b) % n


def numpy_mp_add_n(width: MpWidth, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return limbwise_add(width, a, b)


def numpy_mp_add_n_kogge(width: MpWidth, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return limbwise_add_kogge(width, a, b)


def numpy_mp_sub_n(width: MpWidth, a: np.ndarray, n: np.ndarray) -> np.ndarray:
    return propagate_borrow_sub(width, a, n)


def numpy_mp_add_mod_legacy(width: MpWidth, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    lb = width.limb_bits
    mask = width.limb_mask
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    for i in range(batch):
        carry_add = 0
        carry_sub = 1
        ri = np.zeros(limbs, dtype=np.uint64)
        ai = a[i].astype(np.uint64)
        bi = b[i].astype(np.uint64)
        ni = n[i].astype(np.uint64)
        for j in range(limbs):
            s = int(ai[j]) + int(bi[j]) + carry_add
            carry_add = s >> lb
            t = (s & mask) + ((~int(ni[j])) & mask) + carry_sub
            carry_sub = t >> lb
            ri[j] = t & mask
        if (carry_add | carry_sub) == 0:
            c = 0
            for j in range(limbs):
                s = int(ri[j]) + int(ni[j]) + c
                ri[j] = s & mask
                c = s >> lb
        r[i] = ri
    return r


def numpy_mp_sub_mod_legacy(width: MpWidth, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    lb = width.limb_bits
    mask = width.limb_mask
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    for i in range(batch):
        borrow = 0
        ri = np.zeros(limbs, dtype=np.uint64)
        ai = a[i].astype(np.uint64)
        bi = b[i].astype(np.uint64)
        ni = n[i].astype(np.uint64)
        for j in range(limbs):
            av = int(ai[j])
            bv = int(bi[j])
            w = av - bv - borrow
            ri[j] = w & mask
            borrow = 1 if av < bv + borrow else 0
        if borrow:
            c = 0
            for j in range(limbs):
                s = int(ri[j]) + int(ni[j]) + c
                ri[j] = s & mask
                c = s >> lb
        r[i] = ri
    return r


def numpy_mp_add_mod_vec(width: MpWidth, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    if width.limb_bits == 64:
        return numpy_mp_add_mod_legacy(width, a, b, n)
    lb = width.limb_bits
    mask = np.uint64(width.limb_mask)
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    carry_add = np.zeros(batch, dtype=np.uint64)
    carry_sub = np.ones(batch, dtype=np.uint64)
    for j in range(limbs):
        s = a[:, j] + b[:, j] + carry_add
        carry_add = s >> lb
        t = (s & mask) + ((~n[:, j]) & mask) + carry_sub
        carry_sub = t >> lb
        r[:, j] = t & mask
    need_fix = (carry_add | carry_sub) == 0
    if np.any(need_fix):
        c = np.zeros(batch, dtype=np.uint64)
        for j in range(limbs):
            s = r[:, j] + n[:, j] + c
            new_r = s & mask
            new_c = s >> lb
            r[:, j] = np.where(need_fix, new_r, r[:, j])
            c = np.where(need_fix, new_c, np.uint64(0))
    return r


def numpy_mp_sub_mod_vec(width: MpWidth, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    if width.limb_bits == 64:
        return numpy_mp_sub_mod_legacy(width, a, b, n)
    lb = width.limb_bits
    mask = np.uint64(width.limb_mask)
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    borrow = np.zeros(batch, dtype=np.uint64)
    for j in range(limbs):
        av = a[:, j]
        bv = b[:, j]
        w = av - bv - borrow
        r[:, j] = w & mask
        borrow = (av < bv + borrow).astype(np.uint64)
    need_fix = borrow.astype(bool)
    if np.any(need_fix):
        c = np.zeros(batch, dtype=np.uint64)
        for j in range(limbs):
            s = r[:, j] + n[:, j] + c
            new_r = s & mask
            new_c = s >> lb
            r[:, j] = np.where(need_fix, new_r, r[:, j])
            c = np.where(need_fix, new_c, np.uint64(0))
    return r


def numpy_mp_add_mod_kogge(width: MpWidth, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    (a + b) mod N via Kogge-Stone carry/borrow (OpenCL mp_add_mod_mask structure).

    S = a + b; D = S - N; pick D when S >= N (carry or non-borrowing subtract).
    """
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    lb = width.limb_bits
    s, carry_add = kogge_stone_add_with_carry(a, b, lb)
    d, borrow = kogge_stone_sub_borrow(s, n, lb)
    need_sub = (carry_add != 0) | (borrow == 0)
    return np.where(need_sub[:, None], d, s)


def numpy_mp_sub_mod_kogge(width: MpWidth, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """(a - b) mod N: Kogge subtract, then Kogge add N when underflow."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    lb = width.limb_bits
    r, borrow = kogge_stone_sub_borrow(a, b, lb)
    need_fix = borrow.astype(bool)
    if not np.any(need_fix):
        return r
    fixed = kogge_stone_add(r, n, lb)
    return np.where(need_fix[:, None], fixed, r)


def _branchfree_blend(
    need_sub: np.ndarray, s: np.ndarray, d: np.ndarray, limb_mask: int
) -> np.ndarray:
    """CMOV-style row select: pick full d limb-vector when need_sub else s."""
    mask = np.uint64(limb_mask)
    full_sel = np.where(need_sub[:, None], mask, np.uint64(0))
    return (d & full_sel) | (s & (~full_sel & mask))


def numpy_mp_add_mod_branchfree(
    width: MpWidth, a: np.ndarray, b: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """
    Branch-free (a + b) mod N for SIMD-friendly host reduce.

    Always computes s = a + b and d = s - N, then blends with a vector mask
    (mp_add_mod_mask semantics: subtract when carry_out or no borrow on s - N).
    """
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    lb = width.limb_bits
    mask = width.limb_mask
    batch, limbs = a.shape

    if lb == 64:
        s, carry_add = kogge_stone_add_with_carry(a, b, lb)
        d, borrow = kogge_stone_sub_borrow(s, n, lb)
        need_sub = (carry_add != 0) | (borrow == 0)
        return _branchfree_blend(need_sub, s, d, mask)

    carry_add = np.zeros(batch, dtype=np.uint64)
    s = np.zeros((batch, limbs), dtype=np.uint64)
    for j in range(limbs):
        t = a[:, j] + b[:, j] + carry_add
        s[:, j] = t & mask
        carry_add = t >> lb

    borrow = np.zeros(batch, dtype=np.uint64)
    d = np.zeros((batch, limbs), dtype=np.uint64)
    for j in range(limbs):
        av = s[:, j]
        nv = n[:, j]
        w = av - nv - borrow
        d[:, j] = w & mask
        borrow = (av < nv + borrow).astype(np.uint64)

    need_sub = (carry_add != 0) | (borrow == 0)
    return _branchfree_blend(need_sub, s, d, mask)


def npu_mp_add_mod_serial(
    width: MpWidth,
    a: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    *,
    add_session=None,
) -> np.ndarray:
    """NPU mod add: ONNX/NPU limb add + host serial/fused mod reduce."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    if add_session is not None:
        add_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})
    return numpy_mp_add_mod_vec(width, a, b, n)


def npu_mp_sub_mod_serial(
    width: MpWidth,
    a: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    *,
    sub_session=None,
    add_session=None,
) -> np.ndarray:
    """NPU mod sub: ONNX/NPU limb sub + host serial/fused mod reduce."""
    del add_session  # host serial path fixes underflow without a second NPU add
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    if sub_session is not None:
        sub_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})
    return numpy_mp_sub_mod_vec(width, a, b, n)


def npu_mp_add_mod_kogge(
    width: MpWidth,
    a: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    *,
    add_session=None,
) -> np.ndarray:
    """
    NPU mod add: ONNX/NPU limb add + host Kogge-Stone mod reduce (mp_add_mod_mask).

    The ORT add session is invoked when present; carry/mod reduction uses Kogge on CPU.
    """
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    if add_session is not None:
        add_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})
    lb = width.limb_bits
    s, carry_add = kogge_stone_add_with_carry(a, b, lb)
    d, borrow = kogge_stone_sub_borrow(s, n, lb)
    need_sub = (carry_add != 0) | (borrow == 0)
    return np.where(need_sub[:, None], d, s)


def npu_mp_sub_mod_kogge(
    width: MpWidth,
    a: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    *,
    sub_session=None,
    add_session=None,
) -> np.ndarray:
    """
    NPU mod sub: ONNX/NPU limb sub + host Kogge-Stone mod reduce.

    Underflow fix uses NPU add session (when present) plus Kogge carry for r + N.
    """
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    if sub_session is not None:
        sub_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})
    lb = width.limb_bits
    r, borrow = kogge_stone_sub_borrow(a, b, lb)
    need_fix = borrow.astype(bool)
    if not np.any(need_fix):
        return r
    if add_session is not None:
        add_session.run(
            None,
            {
                "X": r.astype(np.int64),
                "Y": n.astype(np.int64),
            },
        )
    fixed = kogge_stone_add(r, n, lb)
    return np.where(need_fix[:, None], fixed, r)


def npu_mp_add_mod_branchfree(
    width: MpWidth,
    a: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    *,
    add_session=None,
) -> np.ndarray:
    """NPU mod add: ONNX/NPU limb add + host branch-free mod reduce."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    if add_session is not None:
        add_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})
    return numpy_mp_add_mod_branchfree(width, a, b, n)


class NPUAddSubBackend:
    def __init__(self, width: MpWidth, preferred_eps: Optional[List[str]] = None):
        self.width = width
        self.limbs = width.limbs
        self.preferred_eps = preferred_eps
        self.add_session = None
        self.sub_session = None
        self.add_ep: Optional[str] = None
        self.sub_ep: Optional[str] = None
        self._init_sessions()

    def _init_sessions(self) -> None:
        if not HAS_ONNX or not HAS_ORT:
            return
        tag = f"{self.width.limb_bits}b_x{self.limbs}"
        add_model = create_limb_binary_model("Add", self.limbs, f"mp_add_n_{tag}")
        sub_model = create_limb_binary_model("Sub", self.limbs, f"mp_sub_n_{tag}")
        self.add_session, self.add_ep = create_ort_session(add_model, self.preferred_eps)
        self.sub_session, self.sub_ep = create_ort_session(sub_model, self.preferred_eps)

    @property
    def active(self) -> bool:
        return self.add_session is not None and self.sub_session is not None

    @property
    def is_npu(self) -> bool:
        return ("VitisAI" in (self.add_ep or "")) or ("VitisAI" in (self.sub_ep or ""))

    def _onnx_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.add_session is None:
            return numpy_mp_add_n(self.width, a, b)
        self.add_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})
        return limbwise_add(self.width, a, b)

    def _onnx_sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.sub_session is None:
            return numpy_mp_sub_n(self.width, a, b)
        self.sub_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})
        return propagate_borrow_sub(self.width, a, b)

    def mp_add_n(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._onnx_add(a, b)

    def mp_sub_n(self, a: np.ndarray, n: np.ndarray) -> np.ndarray:
        return self._onnx_sub(a, n)

    def mp_add_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        return npu_mp_add_mod_kogge(self.width, a, b, n, add_session=self.add_session)

    def mp_add_mod_serial(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        return npu_mp_add_mod_serial(self.width, a, b, n, add_session=self.add_session)

    def mp_add_mod_branchfree(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        return npu_mp_add_mod_branchfree(self.width, a, b, n, add_session=self.add_session)

    def mp_sub_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        return npu_mp_sub_mod_kogge(
            self.width, a, b, n, sub_session=self.sub_session, add_session=self.add_session
        )

    def mp_sub_mod_serial(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        return npu_mp_sub_mod_serial(
            self.width, a, b, n, sub_session=self.sub_session, add_session=self.add_session
        )


class NumpyBackend:
    """NumPy limb backend; optional Kogge-Stone paths for add/mod carry chains."""

    def __init__(self, width: MpWidth, *, use_kogge: bool = False):
        self.width = width
        self.use_kogge = use_kogge

    def mp_add_n(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.use_kogge:
            return numpy_mp_add_n_kogge(self.width, a, b)
        return numpy_mp_add_n(self.width, a, b)

    def mp_sub_n(self, a: np.ndarray, n: np.ndarray) -> np.ndarray:
        return numpy_mp_sub_n(self.width, a, n)

    def mp_add_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        if self.use_kogge:
            return numpy_mp_add_mod_kogge(self.width, a, b, n)
        return numpy_mp_add_mod_vec(self.width, a, b, n)

    def mp_sub_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
        if self.use_kogge:
            return numpy_mp_sub_mod_kogge(self.width, a, b, n)
        return numpy_mp_sub_mod_vec(self.width, a, b, n)


def verify_ops(
    width: MpWidth,
    a_vec,
    b_vec,
    n_vec,
    backends,
    *,
    verbose: bool = False,
) -> bool:
    bits = width.bits
    limbs = width.limbs
    mask = (1 << bits) - 1
    a0 = int_from_limbs(a_vec, width) & mask
    b0 = int_from_limbs(b_vec, width) & mask
    n0 = int_from_limbs(n_vec, width) & mask
    expects = {
        "mp_add_n": ref_mp_add_n(a0, b0, bits),
        "mp_sub_n": ref_mp_sub_n(a0, n0, bits),
        "mp_add_mod": ref_mp_add_mod(a0, b0, n0),
        "mp_sub_mod": ref_mp_sub_mod(a0, b0, n0),
    }
    a_batch = a_vec.reshape(1, limbs).astype(np.int64)
    b_batch = b_vec.reshape(1, limbs).astype(np.int64)
    n_batch = n_vec.reshape(1, limbs).astype(np.int64)
    runners = {
        "mp_add_n": lambda be: be.mp_add_n(a_batch, b_batch),
        "mp_sub_n": lambda be: be.mp_sub_n(a_batch, n_batch),
        "mp_add_mod": lambda be: be.mp_add_mod(a_batch, b_batch, n_batch),
        "mp_sub_mod": lambda be: be.mp_sub_mod(a_batch, b_batch, n_batch),
    }
    all_ok = True
    for be_name, backend in backends:
        for op_name, expected in expects.items():
            got_arr = runners[op_name](backend)
            got = int_from_limbs(got_arr[0], width)
            if op_name in ("mp_add_mod", "mp_sub_mod"):
                ok = (got % n0) == expected
            else:
                ok = (got & mask) == (expected & mask)
            if ok:
                if verbose:
                    print(f"  [{be_name}:{op_name}] Python verify: PASS")
            else:
                print(f"  [{be_name}:{op_name}] verify: FAIL (got={got}, expect={expected})")
                all_ok = False
    return all_ok


def verify_fused_vec_matches_legacy(
    width: MpWidth, a_vec, b_vec, n_vec, *, verbose: bool = False
) -> bool:
    limbs = width.limbs
    a_batch = a_vec.reshape(1, limbs).astype(np.int64)
    b_batch = b_vec.reshape(1, limbs).astype(np.int64)
    n_batch = n_vec.reshape(1, limbs).astype(np.int64)
    pairs = (
        ("mp_add_mod", numpy_mp_add_mod_vec, numpy_mp_add_mod_legacy),
        ("mp_sub_mod", numpy_mp_sub_mod_vec, numpy_mp_sub_mod_legacy),
    )
    all_ok = True
    for name, fast, slow in pairs:
        got = fast(width, a_batch, b_batch, n_batch)
        ref = slow(width, a_batch, b_batch, n_batch)
        if np.array_equal(got, ref):
            if verbose:
                print(f"  [cpu_fused:{name}] matches legacy: PASS")
        else:
            print(f"  [cpu_fused:{name}] matches legacy: FAIL")
            all_ok = False
    return all_ok


def verify_kogge_paths(
    width: MpWidth, a_vec, b_vec, n_vec, instances: int = 4, *, verbose: bool = False
) -> bool:
    """Kogge-Stone add/mod paths must match serial fused baselines."""
    limbs = width.limbs
    a_batch, b_batch, n_batch = tile_vectors(a_vec, b_vec, n_vec, instances)
    all_ok = True

    serial_add = limbwise_add(width, a_batch, b_batch)
    kogge_add = limbwise_add_kogge(width, a_batch, b_batch)
    if np.array_equal(serial_add, kogge_add):
        if verbose:
            print(f"  [cpu_kogge:mp_add_n] matches cpu_serial: PASS ({instances} instances)")
    else:
        print("  [cpu_kogge:mp_add_n] matches cpu_serial: FAIL")
        all_ok = False

    fused_add_mod = numpy_mp_add_mod_vec(width, a_batch, b_batch, n_batch)
    kogge_add_mod = numpy_mp_add_mod_kogge(width, a_batch, b_batch, n_batch)
    if np.array_equal(fused_add_mod, kogge_add_mod):
        if verbose:
            print(f"  [cpu_kogge:mp_add_mod] matches cpu_fused: PASS ({instances} instances)")
    else:
        print("  [cpu_kogge:mp_add_mod] matches cpu_fused: FAIL")
        all_ok = False

    fused_sub_mod = numpy_mp_sub_mod_vec(width, a_batch, b_batch, n_batch)
    kogge_sub_mod = numpy_mp_sub_mod_kogge(width, a_batch, b_batch, n_batch)
    if np.array_equal(fused_sub_mod, kogge_sub_mod):
        if verbose:
            print(f"  [cpu_kogge:mp_sub_mod] matches cpu_fused: PASS ({instances} instances)")
    else:
        print("  [cpu_kogge:mp_sub_mod] matches cpu_fused: FAIL")
        all_ok = False

    return all_ok


def verify_branchfree_paths(
    width: MpWidth, a_vec, b_vec, n_vec, instances: int = 4, *, verbose: bool = False
) -> bool:
    """Branch-free mod add must match cpu_fused and cpu_kogge baselines."""
    a_batch, b_batch, n_batch = tile_vectors(a_vec, b_vec, n_vec, instances)
    ref_fused = numpy_mp_add_mod_vec(width, a_batch, b_batch, n_batch)
    ref_kogge = numpy_mp_add_mod_kogge(width, a_batch, b_batch, n_batch)
    got = numpy_mp_add_mod_branchfree(width, a_batch, b_batch, n_batch)
    all_ok = True
    for label, ref in (("cpu_fused", ref_fused), ("cpu_kogge", ref_kogge)):
        if np.array_equal(got, ref):
            if verbose:
                print(f"  [cpu_branchfree:mp_add_mod] matches {label}: PASS ({instances} instances)")
        else:
            print(f"  [cpu_branchfree:mp_add_mod] matches {label}: FAIL")
            all_ok = False
    return all_ok


def verify_npu_mod_paths(
    width: MpWidth,
    a_vec,
    b_vec,
    n_vec,
    npu_be: NPUAddSubBackend,
    instances: int = 4,
    *,
    verbose: bool = False,
) -> bool:
    """Compare NPU serial vs fused, NPU Kogge vs cpu_kogge, and NPU serial vs Kogge."""
    a_batch, b_batch, n_batch = tile_vectors(a_vec, b_vec, n_vec, instances)
    ref_fused_add = numpy_mp_add_mod_vec(width, a_batch, b_batch, n_batch)
    ref_fused_sub = numpy_mp_sub_mod_vec(width, a_batch, b_batch, n_batch)
    ref_kogge_add = numpy_mp_add_mod_kogge(width, a_batch, b_batch, n_batch)
    ref_kogge_sub = numpy_mp_sub_mod_kogge(width, a_batch, b_batch, n_batch)
    ref_branchfree_add = numpy_mp_add_mod_branchfree(width, a_batch, b_batch, n_batch)
    npu_serial_add = npu_mp_add_mod_serial(
        width, a_batch, b_batch, n_batch, add_session=npu_be.add_session
    )
    npu_serial_sub = npu_mp_sub_mod_serial(
        width,
        a_batch,
        b_batch,
        n_batch,
        sub_session=npu_be.sub_session,
        add_session=npu_be.add_session,
    )
    npu_kogge_add = npu_mp_add_mod_kogge(
        width, a_batch, b_batch, n_batch, add_session=npu_be.add_session
    )
    npu_kogge_sub = npu_mp_sub_mod_kogge(
        width,
        a_batch,
        b_batch,
        n_batch,
        sub_session=npu_be.sub_session,
        add_session=npu_be.add_session,
    )
    npu_branchfree_add = npu_mp_add_mod_branchfree(
        width, a_batch, b_batch, n_batch, add_session=npu_be.add_session
    )
    checks = (
        ("npu_add_mod_serial", "cpu_fused", npu_serial_add, ref_fused_add),
        ("npu_sub_mod_serial", "cpu_fused", npu_serial_sub, ref_fused_sub),
        ("npu_add_mod_kogge", "cpu_kogge", npu_kogge_add, ref_kogge_add),
        ("npu_sub_mod_kogge", "cpu_kogge", npu_kogge_sub, ref_kogge_sub),
        ("npu_add_mod_branchfree", "cpu_branchfree", npu_branchfree_add, ref_branchfree_add),
        ("npu_add_mod_serial", "npu_add_mod_kogge", npu_serial_add, npu_kogge_add),
        ("npu_sub_mod_serial", "npu_sub_mod_kogge", npu_serial_sub, npu_kogge_sub),
        ("npu_add_mod_branchfree", "npu_add_mod_kogge", npu_branchfree_add, npu_kogge_add),
        ("cpu_branchfree", "cpu_fused", ref_branchfree_add, ref_fused_add),
    )
    all_ok = True
    for left, right, got, ref in checks:
        if np.array_equal(got, ref):
            if verbose:
                print(f"  [{left}] matches {right}: PASS ({instances} instances)")
        else:
            print(f"  [{left}] matches {right}: FAIL")
            all_ok = False
    return all_ok


def verify_npu_mod_kogge(
    width: MpWidth,
    a_vec,
    b_vec,
    n_vec,
    npu_be: NPUAddSubBackend,
    instances: int = 4,
    *,
    verbose: bool = False,
) -> bool:
    """Backward-compatible alias: full NPU mod path comparison."""
    return verify_npu_mod_paths(
        width, a_vec, b_vec, n_vec, npu_be, instances=instances, verbose=verbose
    )


def verify_kogge_matches_serial(
    width: MpWidth, a_vec, b_vec, n_vec, instances: int = 4, *, verbose: bool = False
) -> bool:
    return verify_kogge_paths(width, a_vec, b_vec, n_vec, instances=instances, verbose=verbose)


def run_self_test(
    widths: Tuple[int, ...] = SELF_TEST_WIDTHS,
    limb_bits_list: Tuple[int, ...] = SUPPORTED_LIMB_BITS,
    preferred_eps: Optional[List[str]] = None,
    include_onnx: bool = True,
    *,
    verbose: bool = False,
) -> bool:
    print("npu_addsub self-test")
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
            backends: List[Tuple[str, object]] = [
                ("cpu", NumpyBackend(width)),
                ("cpu_kogge", NumpyBackend(width, use_kogge=True)),
            ]
            npu_be: Optional[NPUAddSubBackend] = None
            if include_onnx and HAS_ORT:
                npu_be = NPUAddSubBackend(width, preferred_eps=preferred_eps)
                if npu_be.active:
                    backends.insert(0, ("npu", npu_be))
            else:
                npu_be = NPUAddSubBackend(width)
            print(f"\n--- {bits}-bit, {lb}b/limb (limbs={width.limbs}) ---")
            if not verify_ops(width, a_vec, b_vec, n_vec, backends, verbose=verbose):
                all_ok = False
            if not verify_fused_vec_matches_legacy(width, a_vec, b_vec, n_vec, verbose=verbose):
                all_ok = False
            if not verify_kogge_paths(width, a_vec, b_vec, n_vec, verbose=verbose):
                all_ok = False
            if not verify_branchfree_paths(width, a_vec, b_vec, n_vec, verbose=verbose):
                all_ok = False
            if npu_be is not None:
                if not verify_npu_mod_paths(width, a_vec, b_vec, n_vec, npu_be, verbose=verbose):
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


def collect_timings(
    width: MpWidth,
    instances: int,
    kernel_iterations: int,
    launch_repeats: int,
    warmup: int,
    npu_backend: NPUAddSubBackend,
    *,
    include_legacy: bool = False,
    include_kogge: bool = True,
) -> Tuple[float, List[Tuple[str, str, float]]]:
    n_vec, a_vec, b_vec = opencl_test_vectors(width)
    a_batch, b_batch, n_batch = tile_vectors(a_vec, b_vec, n_vec, instances)
    numpy_be = NumpyBackend(width)
    op_count = float(instances * kernel_iterations * launch_repeats)
    timings: List[Tuple[str, str, float]] = []

    if npu_backend.active:
        timings.append(
            (
                "mp_add_n",
                "npu_add_n",
                bench_op(lambda: npu_backend.mp_add_n(a_batch, b_batch), warmup, kernel_iterations, launch_repeats),
            )
        )
        timings.append(
            (
                "mp_sub_n",
                "npu_sub_n",
                bench_op(lambda: npu_backend.mp_sub_n(a_batch, n_batch), warmup, kernel_iterations, launch_repeats),
            )
        )
        timings.append(
            (
                "mp_add_mod",
                "npu_add_mod_serial",
                bench_op(
                    lambda: npu_mp_add_mod_serial(
                        width, a_batch, b_batch, n_batch, add_session=npu_backend.add_session
                    ),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_add_mod",
                "npu_add_mod_kogge",
                bench_op(
                    lambda: npu_backend.mp_add_mod(a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_add_mod",
                "npu_add_mod_branchfree",
                bench_op(
                    lambda: npu_mp_add_mod_branchfree(
                        width, a_batch, b_batch, n_batch, add_session=npu_backend.add_session
                    ),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_sub_mod",
                "npu_sub_mod_serial",
                bench_op(
                    lambda: npu_mp_sub_mod_serial(
                        width,
                        a_batch,
                        b_batch,
                        n_batch,
                        sub_session=npu_backend.sub_session,
                        add_session=npu_backend.add_session,
                    ),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_sub_mod",
                "npu_sub_mod_kogge",
                bench_op(
                    lambda: npu_backend.mp_sub_mod(a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )

    timings.append(
        ("mp_add_n", "cpu_serial", bench_op(lambda: numpy_be.mp_add_n(a_batch, b_batch), warmup, kernel_iterations, launch_repeats))
    )
    if include_kogge:
        timings.append(
            (
                "mp_add_n",
                "cpu_kogge",
                bench_op(
                    lambda: numpy_mp_add_n_kogge(width, a_batch, b_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_add_mod",
                "cpu_kogge",
                bench_op(
                    lambda: numpy_mp_add_mod_kogge(width, a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_sub_mod",
                "cpu_kogge",
                bench_op(
                    lambda: numpy_mp_sub_mod_kogge(width, a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
    timings.append(
        ("mp_sub_n", "cpu_serial", bench_op(lambda: numpy_be.mp_sub_n(a_batch, n_batch), warmup, kernel_iterations, launch_repeats))
    )
    timings.append(
        (
            "mp_add_mod",
            "cpu_branchfree",
            bench_op(
                lambda: numpy_mp_add_mod_branchfree(width, a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )
    timings.append(
        (
            "mp_add_mod",
            "cpu_fused",
            bench_op(
                lambda: numpy_be.mp_add_mod(a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )
    timings.append(
        (
            "mp_sub_mod",
            "cpu_fused",
            bench_op(
                lambda: numpy_be.mp_sub_mod(a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )
    if include_legacy:
        timings.append(
            (
                "mp_add_mod",
                "cpu_fused_legacy",
                bench_op(
                    lambda: numpy_mp_add_mod_legacy(width, a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_sub_mod",
                "cpu_fused_legacy",
                bench_op(
                    lambda: numpy_mp_sub_mod_legacy(width, a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )

    return op_count, timings


def timing_lookup(timings: List[Tuple[str, str, float]], op: str, path: str) -> Optional[float]:
    for label, p, ms in timings:
        if label == op and p == path:
            return ms
    return None


def print_timings(timings: List[Tuple[str, str, float]], op_count: float) -> None:
    by_op: dict = {}
    for label, path, ms in timings:
        by_op.setdefault(label, []).append((path, ms))

    print()
    print("--- mp_add_n / mp_sub_n ---")
    for op in ("mp_add_n", "mp_sub_n"):
        rows = by_op.get(op, [])
        for i, (path, ms) in enumerate(rows):
            ops_s = op_count / (ms / 1000.0)
            line = f"  {op} ({path}): {ms:.4f} ms, {ops_s:.6g} ops/s"
            if i + 1 < len(rows):
                line += f" ({rows[i + 1][1] / ms:.6g}x vs next)"
            print(line)

    print("--- mp_add_mod (priority high -> low) ---")
    add_mod_rows = by_op.get("mp_add_mod", [])
    for i, (path, ms) in enumerate(add_mod_rows):
        ops_s = op_count / (ms / 1000.0)
        line = f"  [{i + 1}] {path}: {ms:.4f} ms, {ops_s:.6g} ops/s"
        if i + 1 < len(add_mod_rows):
            line += f" ({add_mod_rows[i + 1][1] / ms:.6g}x vs next tier)"
        print(line)

    print("--- mp_sub_mod (priority high -> low) ---")
    sub_mod_rows = by_op.get("mp_sub_mod", [])
    for i, (path, ms) in enumerate(sub_mod_rows):
        ops_s = op_count / (ms / 1000.0)
        line = f"  [{i + 1}] {path}: {ms:.4f} ms, {ops_s:.6g} ops/s"
        if i + 1 < len(sub_mod_rows):
            line += f" ({sub_mod_rows[i + 1][1] / ms:.6g}x vs next tier)"
        print(line)

    print()
    print("--- summary (fastest per op) ---")
    for op in ("mp_add_n", "mp_sub_n", "mp_add_mod", "mp_sub_mod"):
        rows = by_op.get(op, [])
        if not rows:
            continue
        sorted_rows = sorted(rows, key=lambda item: item[1])
        path, ms = sorted_rows[0]
        ops_s = op_count / (ms / 1000.0)
        suffix = ""
        if len(sorted_rows) > 1:
            _second_path, second_ms = sorted_rows[1]
            suffix = f" (next: {_second_path} {second_ms / ms:.6g}x slower)"
        print(f"{op}: {ms:.4f} ms, {ops_s:.6g} ops/s [{path}]{suffix}")


def run_benchmark(
    width: MpWidth,
    instances,
    kernel_iterations,
    launch_repeats,
    warmup,
    npu_backend,
    *,
    include_legacy: bool = False,
    include_kogge: bool = True,
):
    op_count, timings = collect_timings(
        width,
        instances,
        kernel_iterations,
        launch_repeats,
        warmup,
        npu_backend,
        include_legacy=include_legacy,
        include_kogge=include_kogge,
    )
    print_timings(timings, op_count)


def run_limb_sweep(
    limb_counts: Tuple[int, ...],
    instances: int,
    kernel_iterations: int,
    launch_repeats: int,
    warmup: int,
    preferred_eps: Optional[List[str]] = None,
    numpy_only: bool = False,
    *,
    include_legacy: bool = False,
    include_kogge: bool = True,
    verbose: bool = False,
) -> bool:
    print(
        f"Limb sweep: counts={limb_counts} (32-bit limbs), "
        f"kernel_iterations={kernel_iterations}, instances={instances}, launch_repeats={launch_repeats}"
    )
    if HAS_ORT:
        print(f"  onnxruntime={ort.__version__}, EPs={ort.get_available_providers()}")
    rows: List[dict] = []
    all_ok = True

    for limbs in limb_counts:
        err = MpWidth.validate_limbs(limbs, allow_sweep=True)
        if err:
            print(f"  [limbs={limbs}] SKIP: {err}")
            all_ok = False
            continue
        width = MpWidth.from_limbs(limbs, allow_sweep=True)
        n_vec, a_vec, b_vec = opencl_test_vectors(width)
        npu = NPUAddSubBackend(width, preferred_eps=preferred_eps)
        if numpy_only:
            npu.add_session = None
            npu.sub_session = None
        backends: List[Tuple[str, object]] = [("cpu", NumpyBackend(width))]
        if npu.active and not numpy_only:
            backends.insert(0, ("npu", npu))
        if not verify_ops(width, a_vec, b_vec, n_vec, backends, verbose=verbose):
            all_ok = False
            continue
        if not verify_kogge_paths(width, a_vec, b_vec, n_vec, verbose=verbose):
            all_ok = False
            continue
        if not verify_branchfree_paths(width, a_vec, b_vec, n_vec, verbose=verbose):
            all_ok = False
            continue
        if npu.active and not numpy_only:
            if not verify_npu_mod_paths(width, a_vec, b_vec, n_vec, npu, verbose=verbose):
                all_ok = False
                continue
        op_count, timings = collect_timings(
            width,
            instances,
            kernel_iterations,
            launch_repeats,
            warmup,
            npu,
            include_legacy=include_legacy,
            include_kogge=include_kogge,
        )
        row = {
            "limbs": limbs,
            "limb_bits": width.limb_bits,
            "bits": width.bits,
            "npu_add_n": timing_lookup(timings, "mp_add_n", "npu_add_n"),
            "cpu_add_n": timing_lookup(timings, "mp_add_n", "cpu_serial"),
            "cpu_kogge_add_n": timing_lookup(timings, "mp_add_n", "cpu_kogge"),
            "cpu_kogge_add_mod": timing_lookup(timings, "mp_add_mod", "cpu_kogge"),
            "cpu_kogge_sub_mod": timing_lookup(timings, "mp_sub_mod", "cpu_kogge"),
            "npu_sub_n": timing_lookup(timings, "mp_sub_n", "npu_sub_n"),
            "cpu_sub_n": timing_lookup(timings, "mp_sub_n", "cpu_serial"),
            "npu_add_mod_serial": timing_lookup(timings, "mp_add_mod", "npu_add_mod_serial"),
            "npu_add_mod_kogge": timing_lookup(timings, "mp_add_mod", "npu_add_mod_kogge"),
            "npu_add_mod_branchfree": timing_lookup(timings, "mp_add_mod", "npu_add_mod_branchfree"),
            "cpu_branchfree_add_mod": timing_lookup(timings, "mp_add_mod", "cpu_branchfree"),
            "npu_sub_mod_serial": timing_lookup(timings, "mp_sub_mod", "npu_sub_mod_serial"),
            "npu_sub_mod_kogge": timing_lookup(timings, "mp_sub_mod", "npu_sub_mod_kogge"),
            "cpu_add_mod": timing_lookup(timings, "mp_add_mod", "cpu_fused"),
            "cpu_sub_mod": timing_lookup(timings, "mp_sub_mod", "cpu_fused"),
            "op_count": op_count,
        }
        rows.append(row)

    if not rows:
        print("No sweep results.")
        return False

    hdr = (
        f"{'limbs':>5} {'bits':>5} | "
        f"{'add_n NPU':>10} {'serial':>10} {'kogge':>10} | "
        f"{'sub_n NPU':>10} {'sub_n cpu':>10} | "
        f"{'add ser':>10} {'add kog':>10} {'sub ser':>10} {'sub kog':>10} | "
        f"{'us/limb add':>11}"
    )
    print()
    print("--- limb count sweep (ms per batch-run, lower is better) ---")
    print(hdr)
    print("-" * len(hdr))
    def _fmt_ms(v: Optional[float]) -> str:
        return f"{v:10.4f}" if v is not None else f"{'n/a':>10}"

    base_row = next((r for r in rows if r["limbs"] == DEFAULT_LIMB_SWEEP[0]), rows[0])
    base_limbs = base_row["limbs"]
    base_add = base_row.get("npu_add_n") or base_row.get("cpu_add_n")
    for row in rows:
        add_npu = row.get("npu_add_n")
        add_cpu = row.get("cpu_add_n")
        kogge = row.get("cpu_kogge_add_n")
        sub_npu = row.get("npu_sub_n")
        sub_cpu = row.get("cpu_sub_n")
        add_mod_serial = row.get("npu_add_mod_serial")
        add_mod_kogge = row.get("npu_add_mod_kogge")
        sub_mod_serial = row.get("npu_sub_mod_serial")
        sub_mod_kogge = row.get("npu_sub_mod_kogge")
        add_ref = add_npu if add_npu is not None else (kogge if kogge is not None else add_cpu)
        us_per_limb = (add_ref / row["limbs"] * 1000.0) if add_ref is not None else None
        print(
            f"{row['limbs']:5d} {row['bits']:5d} | "
            f"{_fmt_ms(add_npu)} {_fmt_ms(add_cpu)} {_fmt_ms(kogge)} | "
            f"{_fmt_ms(sub_npu)} {_fmt_ms(sub_cpu)} | "
            f"{_fmt_ms(add_mod_serial)} {_fmt_ms(add_mod_kogge)} "
            f"{_fmt_ms(sub_mod_serial)} {_fmt_ms(sub_mod_kogge)} | "
            f"{us_per_limb if us_per_limb is not None else float('nan'):11.2f}"
        )

    print()
    print("--- scaling vs limbs=8 (NPU mp_add_n, linear=1.0x per doubling) ---")
    if base_add is not None and base_limbs > 0:
        for row in rows:
            add_npu = row.get("npu_add_n")
            add_cpu = row.get("cpu_add_n")
            ref = add_npu if add_npu is not None else add_cpu
            if ref is None:
                continue
            limb_ratio = row["limbs"] / base_limbs
            time_ratio = ref / base_add
            linear_ratio = time_ratio / limb_ratio if limb_ratio > 0 else float("nan")
            backend = "npu" if add_npu is not None else "cpu"
            print(
                f"  limbs={row['limbs']:3d} ({row['bits']:4d}b) {backend}: "
                f"{time_ratio:.3f}x wall time vs {base_limbs} limbs, "
                f"{linear_ratio:.3f}x vs linear limb scaling"
            )

    print()
    print("Notes:")
    print("  - limbs = 32-bit word count; bits = limbs * 32")
    print("  - us/limb add uses NPU mp_add_n when available, else numpy")
    print("  - linear scaling ~1.0x => overhead dominates; >>1.0x => per-limb work dominates")
    return all_ok


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
    preferred_eps: Optional[List[str]] = None,
    numpy_only: bool = False,
    *,
    include_legacy: bool = False,
    include_kogge: bool = True,
    verbose: bool = False,
) -> bool:
    print(
        f"Limb-bits sweep: {bits}-bit fixed, limb_bits={limb_bits_list}, "
        f"kernel_iterations={kernel_iterations}, instances={instances}, launch_repeats={launch_repeats}"
    )
    if HAS_ORT:
        print(f"  onnxruntime={ort.__version__}, EPs={ort.get_available_providers()}")
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
        npu = NPUAddSubBackend(width, preferred_eps=preferred_eps)
        if numpy_only:
            npu.add_session = None
            npu.sub_session = None
        backends: List[Tuple[str, object]] = [("cpu", NumpyBackend(width))]
        if npu.active and not numpy_only:
            backends.insert(0, ("npu", npu))
        if not verify_ops(width, a_vec, b_vec, n_vec, backends, verbose=verbose):
            all_ok = False
            continue
        if not verify_kogge_paths(width, a_vec, b_vec, n_vec, verbose=verbose):
            all_ok = False
            continue
        if not verify_branchfree_paths(width, a_vec, b_vec, n_vec, verbose=verbose):
            all_ok = False
            continue
        if npu.active and not numpy_only:
            if not verify_npu_mod_paths(width, a_vec, b_vec, n_vec, npu, verbose=verbose):
                all_ok = False
                continue
        op_count, timings = collect_timings(
            width,
            instances,
            kernel_iterations,
            launch_repeats,
            warmup,
            npu,
            include_legacy=include_legacy,
            include_kogge=include_kogge,
        )
        rows.append(
            {
                "limb_bits": lb,
                "limbs": width.limbs,
                "bits": bits,
                "npu_add_n": timing_lookup(timings, "mp_add_n", "npu_add_n"),
                "cpu_add_n": timing_lookup(timings, "mp_add_n", "cpu_serial"),
                "cpu_kogge_add_n": timing_lookup(timings, "mp_add_n", "cpu_kogge"),
                "cpu_kogge_add_mod": timing_lookup(timings, "mp_add_mod", "cpu_kogge"),
                "cpu_kogge_sub_mod": timing_lookup(timings, "mp_sub_mod", "cpu_kogge"),
                "npu_sub_n": timing_lookup(timings, "mp_sub_n", "npu_sub_n"),
                "cpu_sub_n": timing_lookup(timings, "mp_sub_n", "cpu_serial"),
                "npu_add_mod_serial": timing_lookup(timings, "mp_add_mod", "npu_add_mod_serial"),
                "npu_add_mod_kogge": timing_lookup(timings, "mp_add_mod", "npu_add_mod_kogge"),
                "npu_add_mod_branchfree": timing_lookup(timings, "mp_add_mod", "npu_add_mod_branchfree"),
                "cpu_branchfree_add_mod": timing_lookup(timings, "mp_add_mod", "cpu_branchfree"),
                "npu_sub_mod_serial": timing_lookup(timings, "mp_sub_mod", "npu_sub_mod_serial"),
                "npu_sub_mod_kogge": timing_lookup(timings, "mp_sub_mod", "npu_sub_mod_kogge"),
                "op_count": op_count,
            }
        )

    if not rows:
        print("No sweep results.")
        return False

    def _fmt_ms(v: Optional[float]) -> str:
        return f"{v:10.4f}" if v is not None else f"{'n/a':>10}"

    hdr = (
        f"{'lb':>3} {'limbs':>5} {'bits':>5} | "
        f"{'add_n NPU':>10} {'serial':>10} {'kogge':>10} | "
        f"{'sub_n NPU':>10} {'sub_n cpu':>10} | "
        f"{'add ser':>10} {'add kog':>10} {'sub ser':>10} {'sub kog':>10} | "
        f"{'us/limb':>9}"
    )
    print()
    print("--- limb bit-width sweep, same N-bit (ms per batch-run) ---")
    print(hdr)
    print("-" * len(hdr))

    base_row = next((r for r in rows if r["limb_bits"] == DEFAULT_LIMB_BITS_SWEEP[0]), rows[0])
    base_add = base_row.get("npu_add_n") or base_row.get("cpu_add_n")
    for row in rows:
        add_npu = row.get("npu_add_n")
        add_cpu = row.get("cpu_add_n")
        kogge = row.get("cpu_kogge_add_n")
        sub_npu = row.get("npu_sub_n")
        sub_cpu = row.get("cpu_sub_n")
        add_mod_serial = row.get("npu_add_mod_serial")
        add_mod_kogge = row.get("npu_add_mod_kogge")
        sub_mod_serial = row.get("npu_sub_mod_serial")
        sub_mod_kogge = row.get("npu_sub_mod_kogge")
        add_ref = add_npu if add_npu is not None else (kogge if kogge is not None else add_cpu)
        us_per_limb = (add_ref / row["limbs"] * 1000.0) if add_ref is not None else None
        print(
            f"{row['limb_bits']:3d} {row['limbs']:5d} {row['bits']:5d} | "
            f"{_fmt_ms(add_npu)} {_fmt_ms(add_cpu)} {_fmt_ms(kogge)} | "
            f"{_fmt_ms(sub_npu)} {_fmt_ms(sub_cpu)} | "
            f"{_fmt_ms(add_mod_serial)} {_fmt_ms(add_mod_kogge)} "
            f"{_fmt_ms(sub_mod_serial)} {_fmt_ms(sub_mod_kogge)} | "
            f"{us_per_limb if us_per_limb is not None else float('nan'):9.2f}"
        )

    print()
    print(f"--- scaling vs {base_row['limb_bits']}b/limb (mp_add_n wall time) ---")
    if base_add is not None:
        for row in rows:
            add_npu = row.get("npu_add_n")
            add_cpu = row.get("cpu_add_n")
            ref = add_npu if add_npu is not None else add_cpu
            if ref is None:
                continue
            limb_ratio = row["limbs"] / base_row["limbs"]
            time_ratio = ref / base_add
            backend = "npu" if add_npu is not None else "cpu"
            print(
                f"  {row['limb_bits']:2d}b/limb, {row['limbs']:3d} limbs: {backend} add_n "
                f"{time_ratio:.3f}x vs {base_row['limb_bits']}b/limb "
                f"(limb count {limb_ratio:.2f}x)"
            )

    print()
    print("Notes:")
    print("  - Same total bits; wider limbs => fewer limbs, less carry-loop iterations")
    print("  - ONNX still does element-wise add/sub over [batch, limbs] int64 slots")
    return all_ok


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NPU add/sub/mod microbench (OpenCL ecm_addsub core ops)",
        epilog=(
            "Environment: conda activate ryzen-ai-1.7.1\n"
            "Example: python RyzenAI/npu_addsub.py --bits 512 10000 128 2\n"
            "Limb count sweep (32b/limb): --limbs-sweep 8,16,32,64 500 32 1\n"
            "Limb bits sweep (same N): --limb-bits-sweep --bits 512 500 32 1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bits",
        type=int,
        default=None,
        help=f"Total bit width N, {MpWidth.MIN_BITS}..{MpWidth.MAX_BITS}",
    )
    p.add_argument(
        "--limb-bits",
        type=int,
        default=MpWidth.DEFAULT_LIMB_BITS,
        choices=list(SUPPORTED_LIMB_BITS),
        help="Bits per limb (8/16/32/64); limbs = bits / limb_bits",
    )
    p.add_argument(
        "--limb-bits-sweep",
        nargs="?",
        const=",".join(str(x) for x in DEFAULT_LIMB_BITS_SWEEP),
        default=None,
        help=f"Benchmark limb bit-widths at fixed --bits (default {DEFAULT_LIMB_BITS_SWEEP})",
    )
    p.add_argument(
        "--limbs",
        type=int,
        default=None,
        help=f"Limb count at 32b/limb; bits=limbs*32",
    )
    p.add_argument(
        "--limbs-sweep",
        nargs="?",
        const=",".join(str(x) for x in DEFAULT_LIMB_SWEEP),
        default=None,
        help=f"Benchmark limb counts (default {DEFAULT_LIMB_SWEEP}), e.g. --limbs-sweep 8,16,32,64",
    )
    p.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    p.add_argument("--numpy-only", action="store_true", help="Skip ONNX/NPU backend")
    p.add_argument(
        "--self-test",
        action="store_true",
        help=f"Run correctness tests for {SELF_TEST_WIDTHS} and exit (no benchmark)",
    )
    p.add_argument(
        "--onnx-ep",
        type=str,
        default=None,
        help="Comma-separated ONNX EP priority (e.g. CPUExecutionProvider)",
    )
    p.add_argument(
        "--fused-legacy",
        action="store_true",
        help="Include fused_legacy mod tier in benchmark (slow per-row Python loops)",
    )
    p.add_argument(
        "--kogge",
        action="store_true",
        help="Use Kogge-Stone parallel carry for numpy mp_add_n / mp_add_mod / mp_sub_mod",
    )
    p.add_argument(
        "--kogge-add",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-backend verify PASS lines (default: failures only)",
    )
    p.add_argument(
        "--no-kogge-bench",
        action="store_true",
        help="Skip cpu_kogge tiers in benchmark output",
    )
    p.add_argument("positional", nargs="*", help="kernel_iterations [instances] [launch_repeats]")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    preferred_eps = None
    if args.onnx_ep:
        preferred_eps = [s.strip() for s in args.onnx_ep.split(",") if s.strip()]

    if args.self_test:
        return (
            0
            if run_self_test(
                preferred_eps=preferred_eps,
                include_onnx=not args.numpy_only,
                verbose=args.verbose,
            )
            else 1
        )

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
            bits,
            lb_list,
            instances,
            kernel_iterations,
            launch_repeats,
            args.warmup,
            preferred_eps=preferred_eps,
            numpy_only=args.numpy_only,
            include_legacy=args.fused_legacy,
            include_kogge=not args.no_kogge_bench,
            verbose=args.verbose,
        )
        return 0 if ok else 1

    if args.limbs_sweep is not None:
        try:
            limb_counts = parse_limb_list(args.limbs_sweep)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        ok = run_limb_sweep(
            limb_counts,
            instances,
            kernel_iterations,
            launch_repeats,
            args.warmup,
            preferred_eps=preferred_eps,
            numpy_only=args.numpy_only,
            include_legacy=args.fused_legacy,
            include_kogge=not args.no_kogge_bench,
            verbose=args.verbose,
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
        f"NPU add/sub microbench: {width.bits}-bit ({width.limb_bits}b/limb x {width.limbs} limbs), "
        f"kernel_iterations={kernel_iterations}, instances={instances}, launch_repeats={launch_repeats}"
    )
    if HAS_ORT:
        print(f"  onnxruntime={ort.__version__}, EPs={ort.get_available_providers()}")
    else:
        print("  onnxruntime: NOT INSTALLED (NumPy fallback only)")

    npu = NPUAddSubBackend(width, preferred_eps=preferred_eps)
    if args.numpy_only:
        npu.add_session = None
        npu.sub_session = None
    elif npu.active:
        print(f"  NPU backend: add_ep={npu.add_ep}, sub_ep={npu.sub_ep}")
        if npu.is_npu:
            print("  >>> VitisAI NPU acceleration ACTIVE")
        else:
            print(f"  >>> ONNX active on {npu.add_ep}")
    else:
        print("  >>> ONNX sessions unavailable; NumPy baseline only")

    use_kogge = args.kogge or args.kogge_add
    n_vec, a_vec, b_vec = opencl_test_vectors(width)
    backends: List[Tuple[str, object]] = [
        ("cpu", NumpyBackend(width, use_kogge=use_kogge)),
        ("cpu_kogge", NumpyBackend(width, use_kogge=True)),
    ]
    if npu.active and not args.numpy_only:
        backends.insert(0, ("npu", npu))

    print()
    if not verify_ops(width, a_vec, b_vec, n_vec, backends, verbose=args.verbose):
        print("\nVerify FAILED")
        return 1
    if not verify_kogge_paths(width, a_vec, b_vec, n_vec, verbose=args.verbose):
        print("\nKogge-Stone verify FAILED")
        return 1

    run_benchmark(
        width,
        instances,
        kernel_iterations,
        launch_repeats,
        args.warmup,
        npu,
        include_legacy=args.fused_legacy,
        include_kogge=not args.no_kogge_bench,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
