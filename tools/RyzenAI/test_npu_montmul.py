#!/usr/bin/env python3
"""Correctness tests for npu_montmul (standalone, no pytest required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import npu_montmul as m  # noqa: E402


class TestNp0(unittest.TestCase):
    def test_np0_matches_pow_inverse(self):
        width = m.MpWidth(512)
        n_vec, _, _ = m.opencl_test_vectors(width)
        n0 = int(n_vec[0])
        np0 = m.find_np0(n0, 32)
        self.assertEqual((np0 * n0) & 0xFFFFFFFF, 0xFFFFFFFF)


class TestMontOps(unittest.TestCase):
    def test_all_widths(self):
        for bits in m.SELF_TEST_WIDTHS:
            width = m.MpWidth(bits)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
            ok = m.verify_mont_ops(
                width, a_vec, b_vec, n_vec, [("cpu", m.NumpyMontBackend(width))], verbose=False
            )
            self.assertTrue(ok, f"mont verify failed at {bits}-bit")

    def test_vec_matches_legacy(self):
        for bits in m.SELF_TEST_WIDTHS:
            width = m.MpWidth(bits)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
            ok = m.verify_vec_matches_legacy(width, a_vec, b_vec, n_vec, verbose=False)
            self.assertTrue(ok, f"vec vs legacy failed at {bits}-bit")

    def test_limb_bits_layouts(self):
        for lb in m.SUPPORTED_LIMB_BITS:
            width = m.MpWidth(512, lb)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
            ok = m.verify_mont_ops(
                width, a_vec, b_vec, n_vec, [("cpu", m.NumpyMontBackend(width))], verbose=False
            )
            self.assertTrue(ok, f"mont verify failed lb={lb}")

    def test_cios_inner_mul_32b(self):
        import numpy as np
        import npu_bignum_mul as bm
        width = m.MpWidth(512, 32)
        n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
        a_batch = np.tile(a_vec, (4, 1))
        b_batch = np.tile(b_vec, (4, 1))
        npu = m.NPUMontBackend(width)
        npu.inner_mul_be.session = None
        for i in range(width.limbs):
            ai = a_batch[:, i]
            ref = ai[:, None] * b_batch
            got = bm.cios_limb_row_products(ai, b_batch, npu.inner_mul_be)
            np.testing.assert_array_equal(got, ref, err_msg=f'limb index {i}')

    def test_npu_mont_paths(self):
        for lb in m.SUPPORTED_LIMB_BITS:
            width = m.MpWidth(512, lb)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
            npu = m.NPUMontBackend(width)
            self.assertTrue(
                m.verify_npu_mont_paths(width, a_vec, b_vec, n_vec, npu, instances=4, verbose=False),
                f"npu mont paths failed lb={lb}",
            )


class TestSelfTestRunner(unittest.TestCase):
    def test_run_self_test_numpy_only(self):
        ok = m.run_self_test(widths=(512,), limb_bits_list=(32,), include_onnx=False, verbose=False)
        self.assertTrue(ok)


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
