#!/usr/bin/env python3
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

Benchmark path labels:
  npu_mont_*  schoolbook 32b inner mul (lb=32) or limb mul/add + host CIOS
  cpu_cios           scalar inner mul ai*b (NumPy)
  cpu_cios_schoolbook 32b schoolbook inner mul on CPU (lb=32 only)
  npu_mont_schoolbook 32b schoolbook MatMul in CIOS inner mul (lb=32)
  npu_mont_limb       legacy limb int64 Mul/Add ONNX + scalar CIOS inner
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, List, Optional, Protocol, Tuple, runtime_checkable

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
    """Element-wise int64 Add/Mul over [batch, limbs]."""
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

SELF_TEST_WIDTHS = (512, 1024, 2048, 4096)
DEFAULT_LIMB_SWEEP = (8, 16, 32, 64)
SUPPORTED_LIMB_BITS = (8, 16, 32, 64)
DEFAULT_LIMB_BITS_SWEEP = SUPPORTED_LIMB_BITS
CIOS_INNER_BLOCK_BITS = 32


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



def _cios_inner_products(
    ai: np.ndarray,
    b: np.ndarray,
    width: MpWidth,
    inner_mul_be=None,
) -> np.ndarray:
    if (
        inner_mul_be is not None
        and inner_mul_be.active
        and width.limb_bits == CIOS_INNER_BLOCK_BITS
    ):
        from npu_bignum_mul import cios_limb_row_products

        return cios_limb_row_products(ai, b, inner_mul_be, CIOS_INNER_BLOCK_BITS)
    return ai[:, None] * np.asarray(b, dtype=np.uint64)

def numpy_mont_mul_cios_vec(
    width: MpWidth,
    a: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    np0: int,
    *,
    inner_mul_be=None,
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
        products = _cios_inner_products(ai, b, width, inner_mul_be)
        carry = np.zeros(batch, dtype=np.uint64)
        for j in range(limbs):
            uv = t[:, j] + products[:, j] + carry
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
    *,
    inner_mul_be=None,
) -> np.ndarray:
    return numpy_mont_mul_cios_vec(width, a, a, n, np0, inner_mul_be=inner_mul_be)


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


class CiosMontBackend:
    """CIOS Montgomery with optional schoolbook inner-mul backend."""

    def __init__(self, width: MpWidth, inner_mul_be=None):
        self.width = width
        self.inner_mul_be = inner_mul_be

    def mont_mul(self, a: np.ndarray, b: np.ndarray, n: np.ndarray, np0: int) -> np.ndarray:
        return numpy_mont_mul_cios_vec(self.width, a, b, n, np0, inner_mul_be=self.inner_mul_be)

    def mont_sqr(self, a: np.ndarray, n: np.ndarray, np0: int) -> np.ndarray:
        return numpy_mont_sqr_cios_vec(self.width, a, n, np0, inner_mul_be=self.inner_mul_be)


class NumpyMontBackend(CiosMontBackend):
    """Scalar inner-mul CIOS (NumPy ai*b)."""

    def __init__(self, width: MpWidth):
        super().__init__(width, inner_mul_be=None)


@runtime_checkable
class MontBackend(Protocol):
    width: MpWidth

    def mont_mul(self, a: np.ndarray, b: np.ndarray, n: np.ndarray, np0: int) -> np.ndarray: ...

    def mont_sqr(self, a: np.ndarray, n: np.ndarray, np0: int) -> np.ndarray: ...


def npu_mont_mul(
    width: MpWidth,
    a: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    np0: int,
    *,
    inner_mul_be=None,
    mul_session=None,
    add_session=None,
) -> np.ndarray:
    """NPU Montgomery mul: schoolbook inner mul (32b blocks) + host CIOS reduce."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    if inner_mul_be is None:
        if mul_session is not None:
            mul_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})
        if add_session is not None:
            add_session.run(None, {"X": a.astype(np.int64), "Y": b.astype(np.int64)})
    return numpy_mont_mul_cios_vec(width, a, b, n, np0, inner_mul_be=inner_mul_be)


def npu_mont_sqr(
    width: MpWidth,
    a: np.ndarray,
    n: np.ndarray,
    np0: int,
    *,
    inner_mul_be=None,
    mul_session=None,
    add_session=None,
) -> np.ndarray:
    """NPU Montgomery sqr: schoolbook inner mul (32b blocks) + host CIOS reduce."""
    return npu_mont_mul(
        width,
        a,
        a,
        n,
        np0,
        inner_mul_be=inner_mul_be,
        mul_session=mul_session,
        add_session=add_session,
    )


class NPUMontModeBackend:
    """One NPU mont path: either schoolbook inner MatMul or legacy limb Mul/Add."""

    def __init__(
        self,
        width: MpWidth,
        *,
        inner_mul_be=None,
        mul_session=None,
        add_session=None,
    ) -> None:
        self.width = width
        self.inner_mul_be = inner_mul_be
        self.mul_session = mul_session
        self.add_session = add_session

    def mont_mul(self, a: np.ndarray, b: np.ndarray, n: np.ndarray, np0: int) -> np.ndarray:
        return npu_mont_mul(
            self.width,
            a,
            b,
            n,
            np0,
            inner_mul_be=self.inner_mul_be,
            mul_session=self.mul_session,
            add_session=self.add_session,
        )

    def mont_sqr(self, a: np.ndarray, n: np.ndarray, np0: int) -> np.ndarray:
        return npu_mont_sqr(
            self.width,
            a,
            n,
            np0,
            inner_mul_be=self.inner_mul_be,
            mul_session=self.mul_session,
            add_session=self.add_session,
        )


class NPUMontBackend:
    def __init__(self, width: MpWidth, preferred_eps: Optional[List[str]] = None):
        self.width = width
        self.limbs = width.limbs
        self.preferred_eps = preferred_eps
        self.mul_session = None
        self.add_session = None
        self.mul_ep: Optional[str] = None
        self.add_ep: Optional[str] = None
        self.inner_mul_be = None
        self._init_sessions()

    def _init_limb_sessions(self) -> None:
        tag = f"{self.width.limb_bits}b_x{self.limbs}"
        mul_model = create_limb_binary_model("Mul", self.limbs, f"mont_mul_{tag}")
        add_model = create_limb_binary_model("Add", self.limbs, f"mont_add_{tag}")
        self.mul_session, self.mul_ep = create_ort_session(mul_model, self.preferred_eps)
        self.add_session, self.add_ep = create_ort_session(add_model, self.preferred_eps)

    def _init_sessions(self) -> None:
        if not HAS_ONNX or not HAS_ORT:
            return
        self._init_limb_sessions()
        if self.width.limb_bits == CIOS_INNER_BLOCK_BITS:
            from npu_bignum_mul import NPUBigIntMulBackend

            self.inner_mul_be = NPUBigIntMulBackend(
                self.limbs, CIOS_INNER_BLOCK_BITS, preferred_eps=self.preferred_eps
            )

    @property
    def limb_active(self) -> bool:
        return self.mul_session is not None and self.add_session is not None

    @property
    def schoolbook_active(self) -> bool:
        return self.inner_mul_be is not None and self.inner_mul_be.active

    @property
    def active(self) -> bool:
        return self.schoolbook_active or self.limb_active

    @property
    def is_npu(self) -> bool:
        if self.schoolbook_active and self.inner_mul_be.is_npu:
            return True
        return ("VitisAI" in (self.mul_ep or "")) or ("VitisAI" in (self.add_ep or ""))

    def schoolbook_backend(self) -> Optional[NPUMontModeBackend]:
        if not self.schoolbook_active:
            return None
        return NPUMontModeBackend(self.width, inner_mul_be=self.inner_mul_be)

    def limb_backend(self) -> Optional[NPUMontModeBackend]:
        if not self.limb_active:
            return None
        return NPUMontModeBackend(
            self.width,
            mul_session=self.mul_session,
            add_session=self.add_session,
        )

    def mont_mul(self, a: np.ndarray, b: np.ndarray, n: np.ndarray, np0: int) -> np.ndarray:
        be = self.schoolbook_backend() or self.limb_backend()
        assert be is not None
        return be.mont_mul(a, b, n, np0)

    def mont_sqr(self, a: np.ndarray, n: np.ndarray, np0: int) -> np.ndarray:
        be = self.schoolbook_backend() or self.limb_backend()
        assert be is not None
        return be.mont_sqr(a, n, np0)


def verify_mont_ops(
    width: MpWidth,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    n_vec: np.ndarray,
    backends: List[Tuple[str, MontBackend]],
    *,
    verbose: bool = False,
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
                if verbose:
                    print(f"  [{be_name}:{op_name}] verify: PASS")
            else:
                print(f"  [{be_name}:{op_name}] verify: FAIL (got={got}, expect={expected})")
                all_ok = False
    return all_ok


def verify_npu_mont_paths(
    width: MpWidth,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    n_vec: np.ndarray,
    npu_be: NPUMontBackend,
    instances: int = 4,
    *,
    verbose: bool = False,
) -> bool:
    """NPU mont paths must match cpu_cios reference."""
    np0 = np0_from_modulus(n_vec, width)
    a_batch, b_batch, n_batch = tile_vectors(a_vec, b_vec, n_vec, instances)
    ref_mul = numpy_mont_mul_cios_vec(width, a_batch, b_batch, n_batch, np0)
    ref_sqr = numpy_mont_sqr_cios_vec(width, a_batch, n_batch, np0)
    all_ok = True
    checks: List[Tuple[str, np.ndarray, np.ndarray]] = []
    schoolbook = npu_be.schoolbook_backend()
    limb = npu_be.limb_backend()
    if schoolbook is not None:
        checks.append(("npu_mont_schoolbook", schoolbook.mont_mul(a_batch, b_batch, n_batch, np0), ref_mul))
        checks.append(("npu_mont_schoolbook_sqr", schoolbook.mont_sqr(a_batch, n_batch, np0), ref_sqr))
    if limb is not None:
        checks.append(("npu_mont_limb", limb.mont_mul(a_batch, b_batch, n_batch, np0), ref_mul))
        checks.append(("npu_mont_limb_sqr", limb.mont_sqr(a_batch, n_batch, np0), ref_sqr))
    if not checks:
        got_mul = npu_be.mont_mul(a_batch, b_batch, n_batch, np0)
        got_sqr = npu_be.mont_sqr(a_batch, n_batch, np0)
        checks = (
            ("npu_mont_mul", got_mul, ref_mul),
            ("npu_mont_sqr", got_sqr, ref_sqr),
        )
    for left, got, ref in checks:
        if np.array_equal(got, ref):
            if verbose:
                print(f"  [{left}] matches cpu_cios: PASS ({instances} instances)")
        else:
            print(f"  [{left}] matches cpu_cios: FAIL")
            all_ok = False
    return all_ok


def verify_vec_matches_legacy(
    width: MpWidth,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    n_vec: np.ndarray,
    *,
    verbose: bool = False,
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
            if verbose:
                print(f"  [cpu_cios:{name}] matches legacy: PASS")
        else:
            print(f"  [cpu_cios:{name}] matches legacy: FAIL")
            all_ok = False
    return all_ok


def run_self_test(
    widths: Tuple[int, ...] = SELF_TEST_WIDTHS,
    limb_bits_list: Tuple[int, ...] = SUPPORTED_LIMB_BITS,
    preferred_eps: Optional[List[str]] = None,
    include_onnx: bool = True,
    *,
    verbose: bool = False,
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
            cpu_be = NumpyMontBackend(width)
            backends: List[Tuple[str, MontBackend]] = [("cpu", cpu_be)]
            npu_be: Optional[NPUMontBackend] = None
            if include_onnx and HAS_ORT:
                npu_be = NPUMontBackend(width, preferred_eps=preferred_eps)
                if npu_be.active:
                    backends.insert(0, ("npu", npu_be))
            else:
                npu_be = NPUMontBackend(width)
                if npu_be.inner_mul_be is not None:
                    npu_be.inner_mul_be.session = None
            print(f"\n--- {bits}-bit, {lb}b/limb (limbs={width.limbs}), np0=0x{np0_from_modulus(n_vec, width):x} ---")
            if not verify_mont_ops(width, a_vec, b_vec, n_vec, backends, verbose=verbose):
                all_ok = False
            if not verify_vec_matches_legacy(width, a_vec, b_vec, n_vec, verbose=verbose):
                all_ok = False
            if npu_be is not None:
                if not verify_npu_mont_paths(width, a_vec, b_vec, n_vec, npu_be, verbose=verbose):
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
    backend: MontBackend,
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
    backend: MontBackend,
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



def _make_cpu_schoolbook_inner_be(width: MpWidth):
    if width.limb_bits != CIOS_INNER_BLOCK_BITS:
        return None
    from npu_bignum_mul import NPUBigIntMulBackend

    be = NPUBigIntMulBackend(width.limbs, CIOS_INNER_BLOCK_BITS)
    be.session = None
    return be


def print_bench_plan(width: MpWidth, npu_backend: "NPUMontBackend", numpy_only: bool) -> None:
    mul_paths = ["cpu_cios"]
    if width.limb_bits == CIOS_INNER_BLOCK_BITS:
        mul_paths.append("cpu_cios_schoolbook")
    if not numpy_only:
        if npu_backend.schoolbook_active:
            mul_paths.insert(0, "npu_mont_schoolbook")
        if npu_backend.limb_active:
            insert_at = 1 if npu_backend.schoolbook_active else 0
            mul_paths.insert(insert_at, "npu_mont_limb")
    print(f"  bench mont_mul: {', '.join(mul_paths)}")
    sqr_paths = ["cpu_cios"]
    if width.limb_bits == CIOS_INNER_BLOCK_BITS:
        sqr_paths.append("cpu_cios_schoolbook")
    if not numpy_only:
        if npu_backend.schoolbook_active:
            sqr_paths.insert(0, "npu_mont_schoolbook")
        if npu_backend.limb_active:
            insert_at = 1 if npu_backend.schoolbook_active else 0
            sqr_paths.insert(insert_at, "npu_mont_limb")
    print(f"  bench mont_sqr: {', '.join(sqr_paths)}")
    if width.limb_bits == CIOS_INNER_BLOCK_BITS and not numpy_only:
        if npu_backend.schoolbook_active and npu_backend.limb_active:
            print(
                "  compare: npu_mont_schoolbook = CIOS ai*b[j] via 32b schoolbook MatMul; "
                "npu_mont_limb = legacy limb int64 Mul/Add ONNX + CPU scalar ai*b"
            )

def collect_timings(
    width: MpWidth,
    instances: int,
    kernel_iterations: int,
    launch_repeats: int,
    warmup: int,
    npu_backend: NPUMontBackend,
) -> Tuple[float, List[Tuple[str, str, float]]]:
    n_vec, a_vec, b_vec = opencl_test_vectors(width)
    a_batch, b_batch, n_batch = tile_vectors(a_vec, b_vec, n_vec, instances)
    np0 = np0_from_modulus(n_vec, width)
    cpu_backend = NumpyMontBackend(width)
    cpu_schoolbook_inner = _make_cpu_schoolbook_inner_be(width)
    cpu_schoolbook_backend = (
        CiosMontBackend(width, cpu_schoolbook_inner) if cpu_schoolbook_inner is not None else None
    )
    op_count = float(instances * kernel_iterations * launch_repeats)
    timings: List[Tuple[str, str, float]] = []

    npu_schoolbook = npu_backend.schoolbook_backend()
    npu_limb = npu_backend.limb_backend()
    if npu_schoolbook is not None:
        timings.append(
            (
                "mont_mul",
                "npu_mont_schoolbook",
                bench_op(
                    lambda: mont_mul_bench_chain(
                        npu_schoolbook, a_batch, b_batch, n_batch, np0, kernel_iterations
                    ),
                    warmup,
                    1,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mont_sqr",
                "npu_mont_schoolbook",
                bench_op(
                    lambda: mont_sqr_bench_chain(
                        npu_schoolbook, a_batch, n_batch, np0, kernel_iterations
                    ),
                    warmup,
                    1,
                    launch_repeats,
                ),
            )
        )
    if npu_limb is not None:
        timings.append(
            (
                "mont_mul",
                "npu_mont_limb",
                bench_op(
                    lambda: mont_mul_bench_chain(
                        npu_limb, a_batch, b_batch, n_batch, np0, kernel_iterations
                    ),
                    warmup,
                    1,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mont_sqr",
                "npu_mont_limb",
                bench_op(
                    lambda: mont_sqr_bench_chain(
                        npu_limb, a_batch, n_batch, np0, kernel_iterations
                    ),
                    warmup,
                    1,
                    launch_repeats,
                ),
            )
        )

    if cpu_schoolbook_backend is not None:
        timings.append(
            (
                "mont_mul",
                "cpu_cios_schoolbook",
                bench_op(
                    lambda: mont_mul_bench_chain(
                        cpu_schoolbook_backend, a_batch, b_batch, n_batch, np0, kernel_iterations
                    ),
                    warmup,
                    1,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mont_sqr",
                "cpu_cios_schoolbook",
                bench_op(
                    lambda: mont_sqr_bench_chain(
                        cpu_schoolbook_backend, a_batch, n_batch, np0, kernel_iterations
                    ),
                    warmup,
                    1,
                    launch_repeats,
                ),
            )
        )

    timings.append(
        (
            "mont_mul",
            "cpu_cios",
            bench_op(
                lambda: mont_mul_bench_chain(cpu_backend, a_batch, b_batch, n_batch, np0, kernel_iterations),
                warmup,
                1,
                launch_repeats,
            ),
        )
    )
    timings.append(
        (
            "mont_sqr",
            "cpu_cios",
            bench_op(
                lambda: mont_sqr_bench_chain(cpu_backend, a_batch, n_batch, np0, kernel_iterations),
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
        print(f"--- {op} ---")
        for i, (path, ms) in enumerate(rows):
            ops_s = op_count / (ms / 1000.0)
            line = f"  {op} ({path}): {ms:.4f} ms, {ops_s:.6g} ops/s"
            if i + 1 < len(rows):
                line += f" ({rows[i + 1][1] / ms:.6g}x vs next)"
            print(line)

    print()
    print("--- summary (fastest per op) ---")
    for op in ("mont_mul", "mont_sqr"):
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
    instances: int,
    kernel_iterations: int,
    launch_repeats: int,
    warmup: int,
    npu_backend: NPUMontBackend,
) -> None:
    op_count, timings = collect_timings(
        width, instances, kernel_iterations, launch_repeats, warmup, npu_backend
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
    preferred_eps: Optional[List[str]] = None,
    numpy_only: bool = False,
    *,
    verbose: bool = False,
) -> bool:
    print(
        f"Mont limb-bits sweep: {bits}-bit, limb_bits={limb_bits_list}, "
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
        npu = NPUMontBackend(width, preferred_eps=preferred_eps)
        if numpy_only:
            npu.mul_session = None
            npu.add_session = None
        cpu_be = NumpyMontBackend(width)
        backends: List[Tuple[str, MontBackend]] = [("cpu", cpu_be)]
        if npu.active and not numpy_only:
            backends.insert(0, ("npu", npu))
        if not verify_mont_ops(width, a_vec, b_vec, n_vec, backends, verbose=verbose):
            all_ok = False
            continue
        if npu.active and not numpy_only:
            if not verify_npu_mont_paths(width, a_vec, b_vec, n_vec, npu, verbose=verbose):
                all_ok = False
                continue
        op_count, timings = collect_timings(
            width, instances, kernel_iterations, launch_repeats, warmup, npu
        )
        mul_npu = next((ms for op, path, ms in timings if op == "mont_mul" and path == "npu_mont_schoolbook"), None)
        mul_cpu = next((ms for op, path, ms in timings if op == "mont_mul" and path == "cpu_cios"), None)
        sqr_npu = next((ms for op, path, ms in timings if op == "mont_sqr" and path == "npu_mont_schoolbook"), None)
        sqr_cpu = next((ms for op, path, ms in timings if op == "mont_sqr" and path == "cpu_cios"), None)
        rows.append(
            {
                "limb_bits": lb,
                "limbs": width.limbs,
                "mul_npu": mul_npu,
                "mul_cpu": mul_cpu,
                "sqr_npu": sqr_npu,
                "sqr_cpu": sqr_cpu,
                "op_count": op_count,
            }
        )

    if rows:
        def _fmt_ms(v: Optional[float]) -> str:
            return f"{v:10.4f}" if v is not None else f"{'n/a':>10}"

        print()
        print(f"{'lb':>4} {'limbs':>5} | {'mul NPU':>10} {'mul cpu':>10} | {'sqr NPU':>10} {'sqr cpu':>10} | {'us/limb':>10}")
        print("-" * 72)
        for row in rows:
            mul_ref = row["mul_npu"] if row["mul_npu"] is not None else row["mul_cpu"]
            us = (mul_ref / row["limbs"] * 1000.0) if mul_ref is not None else float("nan")
            print(
                f"{row['limb_bits']:4d} {row['limbs']:5d} | "
                f"{_fmt_ms(row['mul_npu'])} {_fmt_ms(row['mul_cpu'])} | "
                f"{_fmt_ms(row['sqr_npu'])} {_fmt_ms(row['sqr_cpu'])} | {us:10.2f}"
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
    p.add_argument(
        "--numpy-only",
        action="store_true",
        help="Skip ONNX/NPU sessions; cpu_cios baseline only",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print PASS lines during verify/self-test",
    )
    p.add_argument(
        "--preferred-eps",
        nargs="+",
        default=None,
        help="Preferred ONNX Runtime execution providers (in order)",
    )
    p.add_argument("positional", nargs="*", help="kernel_iterations [instances] [launch_repeats]")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    preferred_eps = args.preferred_eps

    if args.self_test:
        ok = run_self_test(
            preferred_eps=preferred_eps,
            include_onnx=not args.numpy_only,
            verbose=args.verbose,
        )
        return 0 if ok else 1

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
        f"Montgomery microbench: {width.bits}-bit ({width.limb_bits}b/limb x {width.limbs} limbs), "
        f"kernel_iterations={kernel_iterations}, instances={instances}, launch_repeats={launch_repeats}"
    )
    if HAS_ORT:
        print(f"  onnxruntime={ort.__version__}, EPs={ort.get_available_providers()}")
    else:
        print("  onnxruntime: NOT INSTALLED (NumPy fallback only)")

    npu = NPUMontBackend(width, preferred_eps=preferred_eps)
    if args.numpy_only:
        npu.mul_session = None
        npu.add_session = None
        if npu.inner_mul_be is not None:
            npu.inner_mul_be.session = None
    elif npu.active:
        if npu.schoolbook_active:
            print(
                f"  NPU schoolbook: {CIOS_INNER_BLOCK_BITS}b MatMul in CIOS inner mul, "
                f"ep={npu.inner_mul_be.ep}, grid={width.limbs}x{width.limbs}"
            )
        if npu.limb_active:
            print(f"  NPU limb mul/add: mul_ep={npu.mul_ep}, add_ep={npu.add_ep} (scalar CIOS inner)")
        if npu.is_npu:
            print("  >>> VitisAI NPU acceleration ACTIVE")
        else:
            ep = npu.inner_mul_be.ep if npu.schoolbook_active else npu.mul_ep
            print(f"  >>> ONNX active on {ep}")
    else:
        print("  >>> ONNX sessions unavailable; NumPy baseline only")
    print_bench_plan(width, npu, args.numpy_only)

    n_vec, a_vec, b_vec = opencl_test_vectors(width)
    np0 = np0_from_modulus(n_vec, width)
    print(f"  np0=0x{np0:0{width.limb_bits // 4}x} (low limb of N)")

    backends: List[Tuple[str, MontBackend]] = [("cpu", NumpyMontBackend(width))]
    if npu.active and not args.numpy_only:
        backends.insert(0, ("npu", npu))

    print()
    if not verify_mont_ops(width, a_vec, b_vec, n_vec, backends, verbose=args.verbose):
        print("\nVerify FAILED")
        return 1
    if npu.active and not args.numpy_only:
        if not verify_npu_mont_paths(width, a_vec, b_vec, n_vec, npu, verbose=args.verbose):
            print("\nNPU mont verify FAILED")
            return 1

    run_benchmark(width, instances, kernel_iterations, launch_repeats, args.warmup, npu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
