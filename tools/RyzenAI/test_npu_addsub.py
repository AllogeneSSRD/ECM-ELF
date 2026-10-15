#!/usr/bin/env python3
"""Correctness tests for npu_addsub (standalone, no pytest required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import npu_addsub as m  # noqa: E402


class TestKoggeStone(unittest.TestCase):
    def test_kogge_paths(self):
        for lb in m.SUPPORTED_LIMB_BITS:
            w = m.MpWidth(512, lb)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(w)
            self.assertTrue(
                m.verify_kogge_paths(w, a_vec, b_vec, n_vec, instances=8, verbose=False),
                f"kogge paths failed lb={lb}",
            )

    def test_branchfree_paths(self):
        for lb in m.SUPPORTED_LIMB_BITS:
            w = m.MpWidth(512, lb)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(w)
            self.assertTrue(
                m.verify_branchfree_paths(w, a_vec, b_vec, n_vec, instances=8, verbose=False),
                f"branchfree paths failed lb={lb}",
            )

    def test_npu_mod_serial_and_kogge(self):
        for lb in m.SUPPORTED_LIMB_BITS:
            w = m.MpWidth(512, lb)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(w)
            npu = m.NPUAddSubBackend(w)
            self.assertTrue(
                m.verify_npu_mod_paths(w, a_vec, b_vec, n_vec, npu, instances=8, verbose=False),
                f"npu mod paths failed lb={lb}",
            )

    def test_kogge_backend_verify_ops(self):
        w = m.MpWidth(512)
        n_vec, a_vec, b_vec = m.opencl_test_vectors(w)
        ok = m.verify_ops(
            w, a_vec, b_vec, n_vec, [("cpu_kogge", m.NumpyBackend(w, use_kogge=True))], verbose=False
        )
        self.assertTrue(ok)


class TestMpWidth(unittest.TestCase):
    def test_valid_widths(self):
        for bits in (512, 1024, 2048, 4096):
            self.assertIsNone(m.MpWidth.validate(bits))
            self.assertEqual(m.MpWidth(bits).limbs, bits // 32)


class TestNumpyOps(unittest.TestCase):
    def test_all_widths_cpu_backend(self):
        for bits in m.SELF_TEST_WIDTHS:
            width = m.MpWidth(bits)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
            ok = m.verify_ops(width, a_vec, b_vec, n_vec, [("cpu", m.NumpyBackend(width))], verbose=False)
            self.assertTrue(ok, f"cpu verify failed at {bits}-bit")

    def test_fused_vec_matches_legacy(self):
        for bits in m.SELF_TEST_WIDTHS:
            width = m.MpWidth(bits)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
            ok = m.verify_fused_vec_matches_legacy(width, a_vec, b_vec, n_vec, verbose=False)
            self.assertTrue(ok, f"fused_vec vs legacy failed at {bits}-bit")


class TestSelfTestRunner(unittest.TestCase):
    def test_run_self_test_numpy_only(self):
        ok = m.run_self_test(widths=(512,), limb_bits_list=(32,), include_onnx=False, verbose=False)
        self.assertTrue(ok)


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
