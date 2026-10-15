#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NPU big-int multiply: schoolbook partial grid via MatMul on NPU."""

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

from npu_montmul import MpWidth, SELF_TEST_WIDTHS, create_ort_session, opencl_test_vectors, tile_vectors

LIMB_BITS = 8
BLOCK_BITS = 8
SUPPORTED_BLOCK_BITS = (8, 16, 32)
DEFAULT_BLOCK_BITS_SWEEP = SUPPORTED_BLOCK_BITS


def parse_block_bits_list(text: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("block-bits list is empty")
    out: List[int] = []
    for p in parts:
        bb = int(p)
        if bb not in SUPPORTED_BLOCK_BITS:
            raise ValueError(f"unsupported block_bits {bb}; choose from {SUPPORTED_BLOCK_BITS}")
        out.append(bb)
    return tuple(out)


def block_mask(block_bits: int) -> int:
    return (1 << block_bits) - 1


def blocks_per_width(width: MpWidth, block_bits: int = BLOCK_BITS) -> int:
    return (width.bits + block_bits - 1) // block_bits


def bytes_per_width(width: MpWidth) -> int:
    return blocks_per_width(width, BLOCK_BITS)


def block_storage_dtype(block_bits: int):
    if block_bits <= 8:
        return np.uint8
    if block_bits <= 16:
        return np.uint16
    return np.uint32


def partial_dtype(block_bits: int):
    if block_bits <= 16:
        return np.uint32
    return np.uint64


def matmul_onnx_dtype(block_bits: int) -> int:
    if block_bits <= 8:
        return TensorProto.FLOAT
    if block_bits <= 16:
        return TensorProto.INT32
    return TensorProto.INT64


def matmul_np_dtype(block_bits: int):
    if block_bits <= 8:
        return np.float32
    if block_bits <= 16:
        return np.int32
    return np.int64


def limbs_to_blocks(arr: np.ndarray, width: MpWidth, block_bits: int = BLOCK_BITS) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.uint64)
    batch, _limbs = arr.shape
    lb = width.limb_bits
    mask = block_mask(block_bits)
    num_blocks = blocks_per_width(width, block_bits)
    out = np.zeros((batch, num_blocks), dtype=block_storage_dtype(block_bits))
    for blk in range(num_blocks):
        bit_start = blk * block_bits
        remaining = block_bits
        shift = 0
        val = np.zeros(batch, dtype=np.uint64)
        pos = bit_start
        while remaining > 0:
            limb_idx = pos // lb
            bit_off = pos % lb
            take = min(remaining, lb - bit_off)
            chunk = (arr[:, limb_idx] >> bit_off) & ((1 << take) - 1)
            val |= chunk << shift
            shift += take
            remaining -= take
            pos += take
        out[:, blk] = (val & mask).astype(out.dtype)
    return out


def limbs_to_u8_blocks(arr: np.ndarray, width: MpWidth) -> np.ndarray:
    return limbs_to_blocks(arr, width, 8)


def blocks_to_int(blocks: np.ndarray, block_bits: int = BLOCK_BITS) -> np.ndarray:
    batch = blocks.shape[0]
    out = np.empty(batch, dtype=object)
    for i in range(batch):
        val = 0
        for k in range(blocks.shape[1]):
            val |= int(blocks[i, k]) << (block_bits * k)
        out[i] = val
    return out


def u8_blocks_to_int(blocks: np.ndarray) -> np.ndarray:
    return blocks_to_int(blocks, 8)


def cpu_schoolbook_partials(
    a_blocks: np.ndarray, b_blocks: np.ndarray, block_bits: int = BLOCK_BITS
) -> np.ndarray:
    dt = partial_dtype(block_bits)
    a_blocks = np.asarray(a_blocks, dtype=dt)
    b_blocks = np.asarray(b_blocks, dtype=dt)
    return a_blocks[:, :, None] * b_blocks[:, None, :]


def collapse_partials(partials: np.ndarray) -> np.ndarray:
    partials = np.asarray(partials, dtype=np.uint64)
    batch, m, n = partials.shape
    acc = np.zeros((batch, m + n), dtype=object)
    for i in range(m):
        acc[:, i : i + n] = acc[:, i : i + n] + partials[:, i, :].astype(object)
    return acc


def propagate_block_carry(acc: np.ndarray, block_bits: int = BLOCK_BITS) -> np.ndarray:
    acc = np.asarray(acc, dtype=object)
    batch, length = acc.shape
    mask = block_mask(block_bits)
    store = block_storage_dtype(block_bits)
    cols: List[np.ndarray] = []
    carry = np.zeros(batch, dtype=object)
    for k in range(length):
        s = acc[:, k] + carry
        cols.append(np.array([int(v) & mask for v in s], dtype=store))
        carry = np.array([int(v) >> block_bits for v in s], dtype=object)
    while np.any(carry):
        cols.append(np.array([int(v) & mask for v in carry], dtype=store))
        carry = np.array([int(v) >> block_bits for v in carry], dtype=object)
    out = np.column_stack(cols) if cols else np.zeros((batch, 1), dtype=store)
    while out.shape[1] > 1 and np.all(out[:, -1] == 0):
        out = out[:, :-1]
    return out if out.shape[1] else np.zeros((batch, 1), dtype=store)


def propagate_byte_carry(acc: np.ndarray) -> np.ndarray:
    return propagate_block_carry(acc, 8)


def partials_to_product(partials: np.ndarray, block_bits: int = BLOCK_BITS) -> np.ndarray:
    return propagate_block_carry(collapse_partials(partials), block_bits)


def cpu_bigint_mul(a_blocks: np.ndarray, b_blocks: np.ndarray, block_bits: int = BLOCK_BITS) -> np.ndarray:
    return partials_to_product(cpu_schoolbook_partials(a_blocks, b_blocks, block_bits), block_bits)


def cpu_bigint_mul_u8(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    return cpu_bigint_mul(a_u8, b_u8, 8)


def create_schoolbook_matmul_model(max_blocks: int, block_bits: int = BLOCK_BITS):
    if not HAS_ONNX:
        return None
    onnx_dt = matmul_onnx_dtype(block_bits)
    a = helper.make_tensor_value_info("A", onnx_dt, ["batch", max_blocks, 1])
    b = helper.make_tensor_value_info("B", onnx_dt, ["batch", 1, max_blocks])
    z = helper.make_tensor_value_info("Partial", onnx_dt, ["batch", max_blocks, max_blocks])
    node = helper.make_node("MatMul", ["A", "B"], ["Partial"], name="schoolbook_outer")
    tag = {TensorProto.FLOAT: "f32", TensorProto.INT32: "i32", TensorProto.INT64: "i64"}.get(onnx_dt, "x")
    graph = helper.make_graph([node], f"schoolbook_{tag}_x{max_blocks}", [a, b], [z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    return model


def npu_schoolbook_partials(
    a_blocks: np.ndarray,
    b_blocks: np.ndarray,
    session,
    max_blocks: int,
    block_bits: int = BLOCK_BITS,
) -> np.ndarray:
    batch, m = a_blocks.shape
    n = b_blocks.shape[1]
    dt = matmul_np_dtype(block_bits)
    a_pad = np.zeros((batch, max_blocks, 1), dtype=dt)
    b_pad = np.zeros((batch, 1, max_blocks), dtype=dt)
    a_pad[:, :m, 0] = a_blocks.astype(dt)
    b_pad[:, 0, :n] = b_blocks.astype(dt)
    partials = session.run(None, {"A": a_pad, "B": b_pad})[0]
    if block_bits <= 8:
        partials = np.rint(partials[:, :m, :n]).astype(partial_dtype(block_bits))
    else:
        partials = partials[:, :m, :n].astype(partial_dtype(block_bits))
    return partials


def npu_bigint_mul(
    a_blocks: np.ndarray,
    b_blocks: np.ndarray,
    session,
    max_blocks: int,
    block_bits: int = BLOCK_BITS,
) -> np.ndarray:
    return partials_to_product(
        npu_schoolbook_partials(a_blocks, b_blocks, session, max_blocks, block_bits),
        block_bits,
    )


def npu_bigint_mul_u8(a_u8: np.ndarray, b_u8: np.ndarray, session, max_bytes: int) -> np.ndarray:
    return npu_bigint_mul(a_u8, b_u8, session, max_bytes, 8)




CIOS_INNER_BLOCK_BITS = 32


def cios_limb_row_products(
    ai: np.ndarray,
    b: np.ndarray,
    mul_be: Optional["NPUBigIntMulBackend"] = None,
    block_bits: int = CIOS_INNER_BLOCK_BITS,
) -> np.ndarray:
    """CIOS inner step: ai * b[j] for all j via schoolbook outer partials (batch, limbs)."""
    dt = partial_dtype(block_bits)
    store = block_storage_dtype(block_bits)
    a_col = np.asarray(ai, dtype=dt).reshape(-1, 1)
    b_blocks = np.asarray(b, dtype=store)
    if mul_be is not None and mul_be.active:
        partials = npu_schoolbook_partials(
            a_col, b_blocks, mul_be.session, mul_be.max_blocks, block_bits
        )
    else:
        partials = cpu_schoolbook_partials(a_col, b_blocks, block_bits)
    return partials[:, 0, :].astype(np.uint64)
class NPUBigIntMulBackend:
    def __init__(
        self,
        max_blocks: int,
        block_bits: int = BLOCK_BITS,
        preferred_eps: Optional[List[str]] = None,
    ):
        self.max_blocks = max_blocks
        self.block_bits = block_bits
        self.session = None
        self.ep: Optional[str] = None
        if HAS_ONNX and HAS_ORT:
            model = create_schoolbook_matmul_model(max_blocks, block_bits)
            self.session, self.ep = create_ort_session(model, preferred_eps)

    @property
    def active(self) -> bool:
        return self.session is not None

    @property
    def is_npu(self) -> bool:
        return "VitisAI" in (self.ep or "")

    def mul(self, a_blocks: np.ndarray, b_blocks: np.ndarray) -> np.ndarray:
        if self.session is None:
            return cpu_bigint_mul(a_blocks, b_blocks, self.block_bits)
        return npu_bigint_mul(a_blocks, b_blocks, self.session, self.max_blocks, self.block_bits)

    def mul_u8(self, a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
        return self.mul(a_u8, b_u8)


def verify_bigint_mul(
    a_blocks,
    b_blocks,
    backends,
    block_bits: int = BLOCK_BITS,
    *,
    verbose=False,
) -> bool:
    a_ints = blocks_to_int(a_blocks, block_bits)
    b_ints = blocks_to_int(b_blocks, block_bits)
    expect = np.array([int(a) * int(b) for a, b in zip(a_ints, b_ints)], dtype=object)
    ref = blocks_to_int(cpu_bigint_mul(a_blocks, b_blocks, block_bits), block_bits)
    all_ok = bool(np.array_equal(ref, expect))
    if not all_ok:
        print("  [cpu_schoolbook] internal reference: FAIL")
        return False
    if verbose:
        print(f"  [cpu_schoolbook] internal reference: PASS ({len(expect)} instances)")
    for name, backend in backends:
        got = blocks_to_int(backend.mul(a_blocks, b_blocks), block_bits)
        if np.array_equal(got, expect):
            if verbose:
                print(f"  [{name}] matches cpu_schoolbook: PASS ({len(expect)} instances)")
        else:
            print(f"  [{name}] matches cpu_schoolbook: FAIL")
            all_ok = False
    return all_ok


def verify_bigint_mul_u8(a_u8, b_u8, backends, *, verbose=False) -> bool:
    return verify_bigint_mul(a_u8, b_u8, backends, 8, verbose=verbose)


def verify_width_mul(
    width,
    a_vec,
    b_vec,
    npu_be,
    block_bits: int = BLOCK_BITS,
    instances=4,
    *,
    verbose=False,
) -> bool:
    a_batch = np.tile(a_vec.astype(np.uint64), (instances, 1))
    b_batch = np.tile(b_vec.astype(np.uint64), (instances, 1))
    a_blocks = limbs_to_blocks(a_batch, width, block_bits)
    b_blocks = limbs_to_blocks(b_batch, width, block_bits)
    cpu = NPUBigIntMulBackend(0, block_bits)
    cpu.session = None
    backends = [("cpu", cpu)]
    if npu_be.active:
        backends.insert(0, ("npu", npu_be))
    return verify_bigint_mul(a_blocks, b_blocks, backends, block_bits, verbose=verbose)


def run_self_test(
    widths=SELF_TEST_WIDTHS,
    block_bits_list=DEFAULT_BLOCK_BITS_SWEEP,
    preferred_eps=None,
    include_onnx=True,
    *,
    verbose=False,
) -> bool:
    print("npu_bignum_mul self-test (schoolbook MatMul + block carry)")
    all_ok = True
    for bits in widths:
        width = MpWidth(bits, LIMB_BITS)
        n_vec, a_vec, b_vec = opencl_test_vectors(width)
        for block_bits in block_bits_list:
            max_blocks = blocks_per_width(width, block_bits)
            npu = NPUBigIntMulBackend(max_blocks, block_bits, preferred_eps=preferred_eps)
            if not include_onnx:
                npu.session = None
            grid = max_blocks * max_blocks
            print(f"\n--- {bits}-bit, {block_bits}b blocks ({max_blocks} blocks, {grid} partials) ---")
            if not verify_width_mul(
                width, a_vec, b_vec, npu, block_bits, instances=8, verbose=verbose
            ):
                all_ok = False
    print()
    print("self-test: ALL PASS" if all_ok else "self-test: FAILED")
    return all_ok


def bench_op(fn, warmup, iters, repeats) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        for _ in range(iters):
            fn()
    return (time.perf_counter() - t0) * 1000.0


def collect_timings(
    width,
    instances,
    kernel_iterations,
    launch_repeats,
    warmup,
    npu_backend,
    block_bits: int = BLOCK_BITS,
):
    n_vec, a_vec, b_vec = opencl_test_vectors(width)
    a_batch, b_batch, _ = tile_vectors(a_vec, b_vec, n_vec, instances)
    a_blocks = limbs_to_blocks(a_batch, width, block_bits)
    b_blocks = limbs_to_blocks(b_batch, width, block_bits)
    cpu = NPUBigIntMulBackend(0, block_bits)
    cpu.session = None
    op_count = float(instances * kernel_iterations * launch_repeats)
    timings = []
    if npu_backend.active:
        timings.append(
            (
                "bigint_mul",
                "npu_schoolbook",
                bench_op(
                    lambda: npu_backend.mul(a_blocks, b_blocks),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
    timings.append(
        (
            "bigint_mul",
            "cpu_schoolbook",
            bench_op(
                lambda: cpu.mul(a_blocks, b_blocks),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )
    return op_count, timings


def print_timings(timings, op_count) -> None:
    by_op = {}
    for label, path, ms in timings:
        by_op.setdefault(label, []).append((path, ms))
    print()
    rows = by_op.get("bigint_mul", [])
    print("--- bigint_mul ---")
    for i, (path, ms) in enumerate(rows):
        ops_s = op_count / (ms / 1000.0)
        line = f"  bigint_mul ({path}): {ms:.4f} ms, {ops_s:.6g} ops/s"
        if i + 1 < len(rows):
            line += f" ({rows[i + 1][1] / ms:.6g}x vs next)"
        print(line)
    print()
    print("--- summary (fastest per op) ---")
    sorted_rows = sorted(rows, key=lambda item: item[1])
    if sorted_rows:
        path, ms = sorted_rows[0]
        ops_s = op_count / (ms / 1000.0)
        suffix = ""
        if len(sorted_rows) > 1:
            p2, ms2 = sorted_rows[1]
            suffix = f" (next: {p2} {ms2 / ms:.6g}x slower)"
        print(f"bigint_mul: {ms:.4f} ms, {ops_s:.6g} ops/s [{path}]{suffix}")



def run_block_bits_sweep(
    bits: int,
    block_bits_list: Tuple[int, ...],
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
        f"Block-bits sweep: {bits}-bit fixed, block_bits={block_bits_list}, "
        f"kernel_iterations={kernel_iterations}, instances={instances}, launch_repeats={launch_repeats}"
    )
    if HAS_ORT:
        print(f"  onnxruntime={ort.__version__}, EPs={ort.get_available_providers()}")
    rows: List[dict] = []
    all_ok = True
    width_base = MpWidth(bits, LIMB_BITS)

    for bb in block_bits_list:
        if bits % bb != 0:
            print(f"  [{bits}b, {bb}b/block] SKIP: bits must be divisible by block_bits")
            all_ok = False
            continue
        width = width_base
        max_blocks = blocks_per_width(width, bb)
        grid = max_blocks * max_blocks
        n_vec, a_vec, b_vec = opencl_test_vectors(width)
        npu = NPUBigIntMulBackend(max_blocks, bb, preferred_eps=preferred_eps)
        if numpy_only:
            npu.session = None
        if not verify_width_mul(width, a_vec, b_vec, npu, bb, instances=4, verbose=verbose):
            all_ok = False
            continue
        op_count, timings = collect_timings(
            width,
            instances,
            kernel_iterations,
            launch_repeats,
            warmup,
            npu,
            bb,
        )
        npu_ms = next((ms for _op, path, ms in timings if path == "npu_schoolbook"), None)
        cpu_ms = next((ms for _op, path, ms in timings if path == "cpu_schoolbook"), None)
        rows.append(
            {
                "block_bits": bb,
                "num_blocks": max_blocks,
                "grid": grid,
                "npu_ms": npu_ms,
                "cpu_ms": cpu_ms,
                "op_count": op_count,
            }
        )

    if rows:
        print()
        print(
            f"{'block_bits':>10} {'blocks':>7} {'grid':>8} {'npu_ms':>10} {'cpu_ms':>10} "
            f"{'npu_ops/s':>12} {'cpu_ops/s':>12} {'npu/cpu':>8}"
        )
        for r in rows:
            npu_ops = r["op_count"] / (r["npu_ms"] / 1000.0) if r["npu_ms"] else float("nan")
            cpu_ops = r["op_count"] / (r["cpu_ms"] / 1000.0) if r["cpu_ms"] else float("nan")
            ratio = r["cpu_ms"] / r["npu_ms"] if r["npu_ms"] and r["cpu_ms"] else float("nan")
            npu_s = f"{r['npu_ms']:.4f}" if r["npu_ms"] is not None else "n/a"
            npu_ops_s = f"{npu_ops:.6g}" if r["npu_ms"] else "n/a"
            ratio_s = f"{ratio:.3f}x" if r["npu_ms"] and r["cpu_ms"] else "n/a"
            print(
                f"{r['block_bits']:>10} {r['num_blocks']:>7} {r['grid']:>8} "
                f"{npu_s:>10} {r['cpu_ms']:>10.4f} {npu_ops_s:>12} {cpu_ops:>12.6g} {ratio_s:>8}"
            )
        best_npu = min((r for r in rows if r["npu_ms"]), key=lambda r: r["npu_ms"], default=None)
        best_cpu = min(rows, key=lambda r: r["cpu_ms"])
        print()
        if best_npu:
            print(
                f"fastest npu: {best_npu['block_bits']}b blocks "
                f"({best_npu['npu_ms']:.4f} ms, grid {best_npu['grid']})"
            )
        print(
            f"fastest cpu: {best_cpu['block_bits']}b blocks "
            f"({best_cpu['cpu_ms']:.4f} ms, grid {best_cpu['grid']})"
        )
    return all_ok

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="NPU schoolbook bigint multiply",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Block-bits sweep: python RyzenAI/npu_bignum_mul.py --block-bits-sweep --bits 512 500 32 1"
        ),
    )
    p.add_argument("--bits", type=int, default=512)
    p.add_argument(
        "--block-bits",
        type=int,
        default=BLOCK_BITS,
        choices=list(SUPPORTED_BLOCK_BITS),
        help="Schoolbook block width in bits (8/16/32)",
    )
    p.add_argument(
        "--block-bits-sweep",
        nargs="?",
        const=",".join(str(x) for x in DEFAULT_BLOCK_BITS_SWEEP),
        default=None,
        help=f"Benchmark block bit-widths at fixed --bits (default {DEFAULT_BLOCK_BITS_SWEEP})",
    )
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--numpy-only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--preferred-eps", nargs="+", default=None)
    p.add_argument("positional", nargs="*")
    args = p.parse_args(argv)

    if args.self_test:
        return (
            0
            if run_self_test(
                include_onnx=not args.numpy_only,
                preferred_eps=args.preferred_eps,
                verbose=args.verbose,
            )
            else 1
        )

    err = MpWidth.validate(args.bits, LIMB_BITS)
    if err:
        print(err, file=sys.stderr)
        return 1

    iters = int(args.positional[0]) if args.positional else 100
    instances = int(args.positional[1]) if len(args.positional) > 1 else 32
    repeats = int(args.positional[2]) if len(args.positional) > 2 else 1

    if args.block_bits_sweep is not None:
        try:
            bb_list = parse_block_bits_list(args.block_bits_sweep)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        ok = run_block_bits_sweep(
            args.bits,
            bb_list,
            instances,
            iters,
            repeats,
            args.warmup,
            preferred_eps=args.preferred_eps,
            numpy_only=args.numpy_only,
            verbose=args.verbose,
        )
        return 0 if ok else 1

    if args.bits % args.block_bits != 0:
        print(f"bits ({args.bits}) must be divisible by block_bits ({args.block_bits})", file=sys.stderr)
        return 1

    width = MpWidth(args.bits, LIMB_BITS)
    max_blocks = blocks_per_width(width, args.block_bits)
    grid = max_blocks * max_blocks
    print(
        f"NPU bigint mul: {width.bits}-bit, {args.block_bits}b blocks "
        f"({max_blocks} blocks, grid {grid}), iters={iters}, batch={instances}"
    )
    if HAS_ORT:
        print(f"  onnxruntime={ort.__version__}, EPs={ort.get_available_providers()}")
    npu = NPUBigIntMulBackend(max_blocks, args.block_bits, preferred_eps=args.preferred_eps)
    if args.numpy_only:
        npu.session = None
    elif npu.active:
        dt_name = {8: "f32", 16: "i32", 32: "i64"}.get(args.block_bits, "?")
        print(f"  MatMul EP: {npu.ep} ({dt_name})" + (" (NPU)" if npu.is_npu else ""))
    n_vec, a_vec, b_vec = opencl_test_vectors(width)
    if not verify_width_mul(width, a_vec, b_vec, npu, args.block_bits, verbose=args.verbose):
        print("Verify FAILED")
        return 1
    op_count, timings = collect_timings(
        width, instances, iters, repeats, args.warmup, npu, args.block_bits
    )
    print_timings(timings, op_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
