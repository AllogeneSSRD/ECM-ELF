#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "RyzenAI" / "test_npu_montmul.py"

CONTENT = '''#!/usr/bin/env python3
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
                width, a_vec, b_vec, n_vec, [("numpy", m.NumpyMontBackend(width))]
            )
            self.assertTrue(ok, f"mont verify failed at {bits}-bit")

    def test_vec_matches_legacy(self):
        for bits in m.SELF_TEST_WIDTHS:
            width = m.MpWidth(bits)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
            ok = m.verify_vec_matches_legacy(width, a_vec, b_vec, n_vec)
            self.assertTrue(ok, f"vec vs legacy failed at {bits}-bit")

    def test_limb_bits_layouts(self):
        for lb in m.SUPPORTED_LIMB_BITS:
            width = m.MpWidth(512, lb)
            n_vec, a_vec, b_vec = m.opencl_test_vectors(width)
            ok = m.verify_mont_ops(
                width, a_vec, b_vec, n_vec, [("numpy", m.NumpyMontBackend(width))]
            )
            self.assertTrue(ok, f"mont verify failed lb={lb}")


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
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
