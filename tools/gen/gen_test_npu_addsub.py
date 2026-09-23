#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "RyzenAI" / "test_npu_addsub.py"

CONTENT = '''#!/usr/bin/env python3
"""Correctness tests for npu_addsub (standalone, no pytest required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import npu_addsub as m  # noqa: E402


class TestMpWidth(unittest.TestCase):
    def test_valid_widths(self):
        for bits in (512, 1024, 2048, 4096):
            self.assertIsNone(m.MpWidth.validate(bits))
            self.assertEqual(m.MpWidth(bits).limbs, bits // 32)

    def test_reject_out_of_range(self):
        self.assertIsNotNone(m.MpWidth.validate(256))
        self.assertIsNotNone(m.MpWidth.validate(8192))

    def test_reject_misaligned(self):
        self.assertIsNotNone(m.MpWidth.validate(520, 64))

    def test_from_limbs_sweep(self):
        w = m.MpWidth.from_limb_count(8, allow_sweep=True)
        self.assertEqual(w.limbs, 8)
        self.assertEqual(w.bits, 256)
        self.assertEqual(w.limb_bits, 32)

    def test_limb_bits_layout(self):
        for lb in m.SUPPORTED_LIMB_BITS:
            w = m.MpWidth(512, lb)
            self.assertEqual(w.limbs, 512 // lb)
            self.assertEqual(w.limb_bits, lb)

    def test_int_limb_roundtrip(self):
        for lb in m.SUPPORTED_LIMB_BITS:
            w = m.MpWidth(512, lb)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(w)
            for vec in (n_vec, a_vec, b_vec):
                v = m.int_from_limbs(vec, w)
                back = m.limbs_from_int(v, w)
                self.assertTrue((back == vec).all(), f"roundtrip failed lb={lb}")


class TestNoNpuUint512Import(unittest.TestCase):
    def test_module_has_no_npu_uint512_dependency(self):
        src = (_ROOT / "npu_addsub.py").read_text(encoding="utf-8")
        self.assertNotIn("from npu_uint512", src)
        self.assertNotIn("import npu_uint512", src)


class TestKoggeStoneAdd(unittest.TestCase):
    def test_kogge_matches_serial_all_limb_bits(self):
        for lb in m.SUPPORTED_LIMB_BITS:
            w = m.MpWidth(512, lb)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(w)
            ok = m.verify_kogge_matches_serial(w, a_vec, b_vec, n_vec, instances=8)
            self.assertTrue(ok, f"kogge vs serial failed lb={lb}")

    def test_kogge_matches_ref_mp_add_n(self):
        w = m.MpWidth(512)
        n_vec, a_vec, b_vec = m.opencl_test_vectors(w)
        a_batch, b_batch, _ = m.tile_vectors(a_vec, b_vec, n_vec, 4)
        got = m.int_from_limbs(m.limbwise_add_kogge(w, a_batch, b_batch)[0], w)
        a0 = m.int_from_limbs(a_vec, w)
        b0 = m.int_from_limbs(b_vec, w)
        expect = m.ref_mp_add_n(a0, b0, w.bits)
        self.assertEqual(got & ((1 << w.bits) - 1), expect & ((1 << w.bits) - 1))


class TestNumpyOps(unittest.TestCase):
    def test_all_widths_numpy_backend(self):
        for bits in m.SELF_TEST_WIDTHS:
            width = m.MpWidth(bits)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
            ok = m.verify_ops(width, a_vec, b_vec, n_vec, [("numpy", m.NumpyBackend(width))])
            self.assertTrue(ok, f"numpy verify failed at {bits}-bit")

    def test_fused_vec_matches_legacy(self):
        for bits in m.SELF_TEST_WIDTHS:
            width = m.MpWidth(bits)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
            ok = m.verify_fused_vec_matches_legacy(width, a_vec, b_vec, n_vec)
            self.assertTrue(ok, f"fused_vec vs legacy failed at {bits}-bit")


class TestOnnxBackend(unittest.TestCase):
    def test_cpu_onnx_matches_numpy(self):
        if not m.HAS_ORT:
            self.skipTest("onnxruntime not installed")
        width = m.MpWidth(512)
        n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
        onnx_be = m.NPUAddSubBackend(width, preferred_eps=["CPUExecutionProvider"])
        if not onnx_be.active:
            self.skipTest("ONNX session unavailable")
        ok = m.verify_ops(
            width,
            a_vec,
            b_vec,
            n_vec,
            [("onnx", onnx_be), ("numpy", m.NumpyBackend(width))],
        )
        self.assertTrue(ok)


class TestSelfTestRunner(unittest.TestCase):
    def test_run_self_test_numpy_only(self):
        ok = m.run_self_test(widths=(512,), limb_bits_list=(32,), include_onnx=False)
        self.assertTrue(ok)


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(CONTENT, encoding="utf-8", newline="\n")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
