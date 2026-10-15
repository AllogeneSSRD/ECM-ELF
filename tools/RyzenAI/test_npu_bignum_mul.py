#!/usr/bin/env python3
"""Correctness tests for npu_bignum_mul (standalone, no pytest required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import npu_bignum_mul as m  # noqa: E402


class TestSchoolbookMul(unittest.TestCase):
    def test_cpu_schoolbook_small(self):
        a = np.array([[1, 2, 3]], dtype=np.uint8)
        b = np.array([[4, 5]], dtype=np.uint8)
        got = m.u8_blocks_to_int(m.cpu_bigint_mul_u8(a, b))[0]
        self.assertEqual(got, (1 + 2 * 256 + 3 * 65536) * (4 + 5 * 256))

    def test_partials_outer_product(self):
        a = np.array([[3, 7]], dtype=np.uint8)
        b = np.array([[5, 11]], dtype=np.uint8)
        p = m.cpu_schoolbook_partials(a, b)
        self.assertEqual(int(p[0, 0, 0]), 15)
        self.assertEqual(int(p[0, 0, 1]), 33)
        self.assertEqual(int(p[0, 1, 0]), 35)
        self.assertEqual(int(p[0, 1, 1]), 77)

    def test_blocks_16bit_cpu(self):
        import numpy as np
        width = m.MpWidth(512, m.LIMB_BITS)
        n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
        a = m.limbs_to_blocks(np.tile(a_vec, (2, 1)), width, 16)
        b = m.limbs_to_blocks(np.tile(b_vec, (2, 1)), width, 16)
        got = m.blocks_to_int(m.cpu_bigint_mul(a, b, 16), 16)
        expect = m.blocks_to_int(a, 16) * m.blocks_to_int(b, 16)
        np.testing.assert_array_equal(got, expect)

    def test_npu_matches_cpu_all_widths(self):
        import numpy as np
        for bits in m.SELF_TEST_WIDTHS:
            width = m.MpWidth(bits, m.LIMB_BITS)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
            max_blocks = m.blocks_per_width(width, m.BLOCK_BITS)
            npu = m.NPUBigIntMulBackend(max_blocks, m.BLOCK_BITS)
            self.assertTrue(
                m.verify_width_mul(width, a_vec, b_vec, npu, m.BLOCK_BITS, instances=4, verbose=False),
                f"mul verify failed at {bits}-bit",
            )

    def test_run_self_test_numpy_only(self):
        ok = m.run_self_test(widths=(512,), include_onnx=False, verbose=False)
        self.assertTrue(ok)


if __name__ == "__main__":
    import numpy as np  # noqa: E402

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
