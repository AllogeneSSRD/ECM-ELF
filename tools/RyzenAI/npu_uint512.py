#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
512-bit Integer Addition & Subtraction using AMD Ryzen AI NPU
==============================================================

Approach:
  1. Represent 512-bit integers as int64[16] limb arrays (little-endian)
  2. Element-wise Add/Sub offloaded to ONNX Runtime (NPU if available)
  3. Carry/borrow propagation in Python (inherently sequential)
  4. Verify against Python's native arbitrary-precision integers
  5. Benchmark: NPU vs CPU vs Pure Python
"""

import numpy as np
import time
import os
import sys

# ── Check dependencies ──
try:
    import onnx
    from onnx import helper, TensorProto
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("[!] 'onnx' package not found. Install: pip install onnx")

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("[!] 'onnxruntime' not found. Install: pip install onnxruntime")


# ============================================================
#  Configuration
# ============================================================
LIMBS       = 16       # 16 x 32-bit limbs = 512 bits
BATCH_SIZE  = 512     # Number of 512-bit operations per batch
WARMUP      = 1000        # Warmup iterations
BENCH_ITERS = 10000       # Benchmark iterations


# ============================================================
#  1. Uint512 - 512-bit Unsigned Integer
# ============================================================
class Uint512:
    """512-bit unsigned integer: 16 x uint32 limbs, little-endian"""

    def __init__(self, value=0):
        if isinstance(value, int):
            self.limbs = np.array(
                [(value >> (32 * i)) & 0xFFFFFFFF for i in range(LIMBS)],
                dtype=np.uint32
            )
        elif isinstance(value, np.ndarray):
            self.limbs = np.array(value, dtype=np.uint32).flatten()[:LIMBS]
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    def to_int(self):
        result = 0
        for i in range(LIMBS):
            result |= int(self.limbs[i]) << (32 * i)
        return result

    def to_int64_array(self):
        """Convert to int64[16] for ONNX model input"""
        return self.limbs.astype(np.int64).reshape(1, LIMBS)

    @staticmethod
    def random(max_bits=511):
        """Random Uint512 (max_bits < 512 to avoid addition overflow)"""
        val = int.from_bytes(os.urandom(64), byteorder='little')
        val &= (1 << min(max_bits, 512)) - 1
        return Uint512(val)

    def __repr__(self):
        h = hex(self.to_int())
        if len(h) > 24:
            return f"Uint512({h[:14]}...{h[-10:]})"
        return f"Uint512({h})"


def batch_random(batch_size, max_bits=511):
    """Generate batch of random 512-bit integers as int64 [batch, 16]"""
    arr = np.zeros((batch_size, LIMBS), dtype=np.int64)
    for i in range(batch_size):
        val = int.from_bytes(os.urandom(64), byteorder='little')
        val &= (1 << min(max_bits, 512)) - 1
        for j in range(LIMBS):
            arr[i, j] = (val >> (32 * j)) & 0xFFFFFFFF
    return arr


def limbs_to_ints(arr):
    """Convert int64[batch, 16] to Python int list"""
    results = []
    for i in range(arr.shape[0]):
        val = 0
        for j in range(LIMBS):
            val |= int(arr[i, j] & 0xFFFFFFFF) << (32 * j)
        results.append(val)
    return results


# ============================================================
#  2. Carry / Borrow Propagation
# ============================================================
def propagate_carry(sum_arr):
    """
    Carry propagation for addition results.
    Input:  int64 array [batch, 16], each element = a[j] + b[j]
    Output: int64 array [batch, 16], properly reduced with carry
    """
    result = sum_arr.copy()
    for j in range(LIMBS):
        carry = result[:, j] >> 32                    # carry = upper bits
        result[:, j] &= 0xFFFFFFFF                    # keep lower 32 bits
        if j + 1 < LIMBS:
            result[:, j + 1] += carry                 # propagate carry
    return result


def propagate_borrow(diff_arr):
    """
    Borrow propagation for subtraction results.
    Input:  int64 array [batch, 16], each element = a[j] - b[j]
    Output: int64 array [batch, 16], properly reduced with borrow
    """
    result = diff_arr.copy()
    borrow = np.zeros(result.shape[0], dtype=np.int64)
    for j in range(LIMBS):
        result[:, j] -= borrow
        neg_mask = result[:, j] < 0
        result[:, j] = np.where(neg_mask, result[:, j] + (1 << 32), result[:, j])
        borrow = np.where(neg_mask, 1, 0).astype(np.int64)
    # Final result: ensure non-negative
    result = result & 0xFFFFFFFF
    return result


# ============================================================
#  3. Pure Python / Numpy Implementation (Baseline)
# ============================================================
def py_add_512(a_arr, b_arr):
    """Pure numpy 512-bit addition: a + b"""
    sums = a_arr.astype(np.int64) + b_arr.astype(np.int64)
    return propagate_carry(sums)


def py_sub_512(a_arr, b_arr):
    """Pure numpy 512-bit subtraction: a - b (assumes a >= b)"""
    diffs = a_arr.astype(np.int64) - b_arr.astype(np.int64)
    return propagate_borrow(diffs)


# ============================================================
#  4. ONNX Model Creation
# ============================================================
def create_add_model():
    """ONNX model: Z = X + Y (element-wise int64)"""
    if not HAS_ONNX:
        return None
    X = helper.make_tensor_value_info('X', TensorProto.INT64, ['batch', LIMBS])
    Y = helper.make_tensor_value_info('Y', TensorProto.INT64, ['batch', LIMBS])
    Z = helper.make_tensor_value_info('Z', TensorProto.INT64, ['batch', LIMBS])

    add_node = helper.make_node('Add', ['X', 'Y'], ['Z'])
    graph = helper.make_graph([add_node], 'uint512_add_graph', [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)])
    model.ir_version = 8
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"  [!] Model check warning: {e}")
    return model


def create_sub_model():
    """ONNX model: Z = X - Y (element-wise int64)"""
    if not HAS_ONNX:
        return None
    X = helper.make_tensor_value_info('X', TensorProto.INT64, ['batch', LIMBS])
    Y = helper.make_tensor_value_info('Y', TensorProto.INT64, ['batch', LIMBS])
    Z = helper.make_tensor_value_info('Z', TensorProto.INT64, ['batch', LIMBS])

    sub_node = helper.make_node('Sub', ['X', 'Y'], ['Z'])
    graph = helper.make_graph([sub_node], 'uint512_sub_graph', [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)])
    model.ir_version = 8
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"  [!] Model check warning: {e}")
    return model


def create_matmul_model(M=256, K=256, N=256):
    """ONNX model: Z = X @ Y (float32 matmul, NPU-friendly)"""
    if not HAS_ONNX:
        return None
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, ['batch', M, K])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [K, N])
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, ['batch', M, N])

    matmul_node = helper.make_node('MatMul', ['X', 'Y'], ['Z'])
    graph = helper.make_graph([matmul_node], 'matmul_graph', [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)])
    model.ir_version = 8
    return model


# ============================================================
#  5. ONNX Runtime Session Management
# ============================================================
def get_available_eps():
    """List available ONNX Runtime Execution Providers"""
    if not HAS_ORT:
        return []
    eps = ort.get_available_providers()
    return eps


def create_session(model, preferred_eps=None):
    """
    Create ONNX Runtime session, trying preferred EPs first.
    Returns (session, actual_ep_used).
    """
    if not HAS_ORT or model is None:
        return None, None

    if preferred_eps is None:
        preferred_eps = ['VitisAIExecutionProvider',
                         'DmlExecutionProvider',
                         'CPUExecutionProvider']

    available = get_available_eps()

    for ep in preferred_eps:
        if ep in available:
            try:
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                if ep == 'VitisAIExecutionProvider':
                    # VitisAI EP needs a config for NPU
                    provider_options = {
                        'target_engine': 'NPU',
                    }
                    session = ort.InferenceSession(
                        model.SerializeToString(),
                        sess_options=sess_options,
                        providers=[(ep, provider_options)]
                    )
                else:
                    session = ort.InferenceSession(
                        model.SerializeToString(),
                        sess_options=sess_options,
                        providers=[ep]
                    )

                actual_providers = session.get_providers()
                return session, actual_providers[0]
            except Exception as e:
                print(f"  [!] Failed to create session with {ep}: {e}")
                continue

    # Fallback: just use whatever is available
    try:
        session = ort.InferenceSession(model.SerializeToString())
        return session, session.get_providers()[0]
    except Exception as e:
        print(f"  [!] All session creation failed: {e}")
        return None, None


# ============================================================
#  6. NPU-accelerated 512-bit Operations
# ============================================================
class NPUUint512:
    """512-bit arithmetic with NPU offloading via ONNX Runtime"""

    def __init__(self):
        self.add_session = None
        self.sub_session = None
        self.add_ep = None
        self.sub_ep = None
        self._init_sessions()

    def _init_sessions(self):
        print("\n" + "=" * 58)
        print("  Initializing NPU Uint512 Sessions")
        print("=" * 58)

        # Show available EPs
        eps = get_available_eps()
        print(f"  Available EPs: {eps}")

        # Create Add model & session
        print("\n  [1] Creating Addition model...")
        add_model = create_add_model()
        if add_model is not None:
            self.add_session, self.add_ep = create_session(add_model)
            if self.add_session:
                print(f"  [OK] Add session created -> EP: {self.add_ep}")
            else:
                print("  [!!] Add session creation failed")
        else:
            print("  [!!] Could not create Add model (onnx not installed)")

        # Create Sub model & session
        print("\n  [2] Creating Subtraction model...")
        sub_model = create_sub_model()
        if sub_model is not None:
            self.sub_session, self.sub_ep = create_session(sub_model)
            if self.sub_session:
                print(f"  [OK] Sub session created -> EP: {self.sub_ep}")
            else:
                print("  [!!] Sub session creation failed")
        else:
            print("  [!!] Could not create Sub model (onnx not installed)")

    @property
    def is_npu_active(self):
        """Check if any session is using NPU (VitisAI)"""
        return ('VitisAI' in (self.add_ep or '')) or ('VitisAI' in (self.sub_ep or ''))

    def add(self, a_arr, b_arr):
        """
        512-bit addition: a + b
        Input:  int64 arrays [batch, 16]
        Output: int64 array  [batch, 16]
        """
        if self.add_session is not None:
            try:
                result = self.add_session.run(
                    None,
                    {'X': a_arr.astype(np.int64),
                     'Y': b_arr.astype(np.int64)}
                )[0]
                return propagate_carry(result)
            except Exception as e:
                print(f"  [!] ONNX add failed, falling back to numpy: {e}")

        # Fallback
        return py_add_512(a_arr, b_arr)

    def sub(self, a_arr, b_arr):
        """
        512-bit subtraction: a - b  (assumes a >= b per element)
        Input:  int64 arrays [batch, 16]
        Output: int64 array  [batch, 16]
        """
        if self.sub_session is not None:
            try:
                result = self.sub_session.run(
                    None,
                    {'X': a_arr.astype(np.int64),
                     'Y': b_arr.astype(np.int64)}
                )[0]
                return propagate_borrow(result)
            except Exception as e:
                print(f"  [!] ONNX sub failed, falling back to numpy: {e}")

        # Fallback
        return py_sub_512(a_arr, b_arr)


# ============================================================
#  7. Verification
# ============================================================
def verify_operations(npu, a_arr, b_arr):
    """Verify NPU results against Python's native big integers"""
    print("\n" + "=" * 58)
    print("  Verification: NPU vs Python Native int")
    print("=" * 58)

    # Convert to Python ints for verification
    a_ints = limbs_to_ints(a_arr)
    b_ints = limbs_to_ints(b_arr)

    # --- Addition ---
    npu_add = npu.add(a_arr, b_arr)
    npu_add_ints = limbs_to_ints(npu_add)
    py_add_ints = [a + b for a, b in zip(a_ints, b_ints)]

    add_errors = sum(1 for x, y in zip(npu_add_ints, py_add_ints) if x != y)
    add_ok = add_errors == 0

    if add_ok:
        print(f"  [OK] Addition:  ALL {len(py_add_ints)} results CORRECT")
    else:
        print(f"  [XX] Addition:  {add_errors}/{len(py_add_ints)} MISMATCHES!")
        # Show first mismatch
        for i, (x, y) in enumerate(zip(npu_add_ints, py_add_ints)):
            if x != y:
                print(f"       Example mismatch at index {i}:")
                print(f"         NPU:   {hex(x)}")
                print(f"         Python: {hex(y)}")
                break

    # Show one example
    print(f"\n  Example Addition:")
    print(f"    A = {hex(a_ints[0])}")
    print(f"    B = {hex(b_ints[0])}")
    print(f"    A + B = {hex(npu_add_ints[0])}")

    # --- Subtraction ---
    # For subtraction, ensure a >= b (swap if needed)
    a_sub = np.maximum(a_arr, b_arr)
    b_sub = np.minimum(a_arr, b_arr)
    a_sub_ints = limbs_to_ints(a_sub)
    b_sub_ints = limbs_to_ints(b_sub)

    npu_sub = npu.sub(a_sub, b_sub)
    npu_sub_ints = limbs_to_ints(npu_sub)
    py_sub_ints = [a - b for a, b in zip(a_sub_ints, b_sub_ints)]

    sub_errors = sum(1 for x, y in zip(npu_sub_ints, py_sub_ints) if x != y)
    sub_ok = sub_errors == 0

    if sub_ok:
        print(f"\n  [OK] Subtraction: ALL {len(py_sub_ints)} results CORRECT")
    else:
        print(f"\n  [XX] Subtraction: {sub_errors}/{len(py_sub_ints)} MISMATCHES!")
        for i, (x, y) in enumerate(zip(npu_sub_ints, py_sub_ints)):
            if x != y:
                print(f"       Example mismatch at index {i}:")
                print(f"         NPU:   {hex(x)}")
                print(f"         Python: {hex(y)}")
                break

    print(f"\n  Example Subtraction:")
    print(f"    A = {hex(a_sub_ints[0])}")
    print(f"    B = {hex(b_sub_ints[0])}")
    print(f"    A - B = {hex(npu_sub_ints[0])}")

    return add_ok and sub_ok


# ============================================================
#  8. Benchmarking
# ============================================================
def benchmark(npu, a_arr, b_arr):
    """Benchmark: NPU/ONNX vs Pure Numpy vs Pure Python"""
    print("\n" + "=" * 58)
    print("  Performance Benchmark")
    print("=" * 58)
    print(f"  Batch size: {BATCH_SIZE} x 512-bit operations")
    print(f"  Iterations: {WARMUP} warmup + {BENCH_ITERS} measured")

    # Ensure a >= b for subtraction
    a_sub = np.maximum(a_arr, b_arr)
    b_sub = np.minimum(a_arr, b_arr)

    results = {}

    # --- NPU / ONNX RT Addition ---
    print("\n  [1] NPU/ONNX Addition ...")
    for _ in range(WARMUP):
        npu.add(a_arr, b_arr)
    t0 = time.perf_counter()
    for _ in range(BENCH_ITERS):
        npu.add(a_arr, b_arr)
    t1 = time.perf_counter()
    results['NPU Add'] = (t1 - t0) / BENCH_ITERS * 1000

    # --- NPU / ONNX RT Subtraction ---
    print("  [2] NPU/ONNX Subtraction ...")
    for _ in range(WARMUP):
        npu.sub(a_sub, b_sub)
    t0 = time.perf_counter()
    for _ in range(BENCH_ITERS):
        npu.sub(a_sub, b_sub)
    t1 = time.perf_counter()
    results['NPU Sub'] = (t1 - t0) / BENCH_ITERS * 1000

    # --- Pure Numpy Addition ---
    print("  [3] Pure Numpy Addition ...")
    for _ in range(WARMUP):
        py_add_512(a_arr, b_arr)
    t0 = time.perf_counter()
    for _ in range(BENCH_ITERS):
        py_add_512(a_arr, b_arr)
    t1 = time.perf_counter()
    results['Numpy Add'] = (t1 - t0) / BENCH_ITERS * 1000

    # --- Pure Numpy Subtraction ---
    print("  [4] Pure Numpy Subtraction ...")
    for _ in range(WARMUP):
        py_sub_512(a_sub, b_sub)
    t0 = time.perf_counter()
    for _ in range(BENCH_ITERS):
        py_sub_512(a_sub, b_sub)
    t1 = time.perf_counter()
    results['Numpy Sub'] = (t1 - t0) / BENCH_ITERS * 1000

    # --- Pure Python int Addition (single pair, for reference) ---
    a_ints = limbs_to_ints(a_arr)
    b_ints = limbs_to_ints(b_arr)
    print("  [5] Pure Python int Addition (single pair) ...")
    t0 = time.perf_counter()
    for _ in range(BENCH_ITERS * 1000):
        _ = a_ints[0] + b_ints[0]
    t1 = time.perf_counter()
    results['Python int Add (1)'] = (t1 - t0) / (BENCH_ITERS * 1000) * 1e6  # microseconds

    # Print results
    print("\n" + "-" * 58)
    print(f"  {'Operation':<28} {'Time':>12}  {'Notes'}")
    print("-" * 58)
    for name, t in results.items():
        if 'Python int' in name:
            print(f"  {name:<28} {t:>10.2f} us  (single op)")
        else:
            print(f"  {name:<28} {t:>10.2f} ms  (batch {BATCH_SIZE})")
    print("-" * 58)

    npu_ep = npu.add_ep or 'N/A'
    print(f"\n  EP used: {npu_ep}")
    if 'VitisAI' in npu_ep:
        print("  >>> NPU acceleration ACTIVE!")
    else:
        print("  >>> NPU not available, using CPU fallback")
        print("      (VitisAI EP required for NPU acceleration)")

    return results


# ============================================================
#  9. Bonus: NPU MatMul Benchmark
# ============================================================
def benchmark_matmul():
    """Benchmark matrix multiplication on NPU (NPU-friendly workload)"""
    if not HAS_ONNX or not HAS_ORT:
        print("\n  [!] Skipping MatMul benchmark (onnx/onnxruntime not available)")
        return

    print("\n" + "=" * 58)
    print("  Bonus: NPU Matrix Multiplication Benchmark")
    print("=" * 58)

    M, K, N = 256, 256, 256

    model = create_matmul_model(M, K, N)
    if model is None:
        print("  [!] Could not create MatMul model")
        return

    session, ep = create_session(model, preferred_eps=[
        'VitisAIExecutionProvider',
        'DmlExecutionProvider',
        'CPUExecutionProvider'
    ])

    if session is None:
        print("  [!] Could not create MatMul session")
        return

    print(f"  EP: {ep}")
    print(f"  Matrix: [{M}x{K}] @ [{K}x{N}] = [{M}x{N}]")

    X = np.random.randn(1, M, K).astype(np.float32)
    Y = np.random.randn(K, N).astype(np.float32)

    # Warmup
    for _ in range(3):
        session.run(None, {'X': X, 'Y': Y})

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(50):
        session.run(None, {'X': X, 'Y': Y})
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / 50 * 1000

    flops = 2 * M * K * N
    gflops = flops / (avg_ms * 1e-3) / 1e9

    print(f"  Average time: {avg_ms:.2f} ms")
    print(f"  Throughput:   {gflops:.2f} GFLOPS")

    if 'VitisAI' in ep:
        print("  >>> NPU acceleration ACTIVE for MatMul!")
    elif 'Dml' in ep:
        print("  >>> GPU (DirectML) acceleration active for MatMul!")
    else:
        print("  >>> Running on CPU (NPU/GPU not available)")


# ============================================================
#  10. Main
# ============================================================
def main():
    print()
    print("  " + "=" * 56)
    print("  |  512-bit Integer Arithmetic on AMD Ryzen AI NPU     |")
    print("  |  Addition & Subtraction via ONNX Runtime           |")
    print("  " + "=" * 56)
    print()

    # Step 1: Show environment
    print("  Environment:")
    if HAS_ORT:
        print(f"    ONNX Runtime version: {ort.__version__}")
        print(f"    Available EPs: {ort.get_available_providers()}")
    else:
        print("    ONNX Runtime: NOT INSTALLED")
    if HAS_ONNX:
        print(f"    ONNX version: {onnx.__version__}")
    else:
        print("    ONNX: NOT INSTALLED")
    print(f"    NumPy version: {np.__version__}")

    # Step 2: Initialize NPU session
    npu = NPUUint512()

    # Step 3: Generate test data
    print("\n  Generating random 512-bit test data...")
    a_arr = batch_random(BATCH_SIZE, max_bits=511)  # max 511 bits to avoid overflow
    b_arr = batch_random(BATCH_SIZE, max_bits=511)
    print(f"    Generated {BATCH_SIZE} pairs of random 512-bit integers")

    # Step 4: Verify correctness
    all_ok = verify_operations(npu, a_arr, b_arr)

    # Step 5: Benchmark
    bench_results = benchmark(npu, a_arr, b_arr)

    # Step 6: Bonus MatMul benchmark
    benchmark_matmul()

    # Final summary
    print("\n" + "=" * 58)
    print("  Summary")
    print("=" * 58)
    if all_ok:
        print("  [OK] All 512-bit arithmetic results VERIFIED CORRECT")
    else:
        print("  [XX] Some results INCORRECT - check implementation!")

    if npu.is_npu_active:
        print("  [OK] NPU (VitisAI EP) is ACTIVE")
    else:
        print("  [!!] NPU not active - using CPU fallback")
        print("       To enable NPU:")
        print("       1. Install Ryzen AI SDK: https://ryzenai.docs.amd.com")
        print("       2. Install NPU MCDM driver")
        print("       3. pip install onnxruntime-vitisai")

    print()


if __name__ == '__main__':
    main()
