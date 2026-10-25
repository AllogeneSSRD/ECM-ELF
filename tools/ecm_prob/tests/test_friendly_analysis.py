"""Guard the success criterion against smoothness/prime-power conflation."""
import math
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"bench"))
from analyze_ecm_friendly import hits, estimate


class FriendlyAnalysisTests(unittest.TestCase):
    def test_exact_order_divisibility_and_single_large_prime(self):
        orders = np.array([[0, 1, 2**9, 2**12, 4*257, 257*257, 257*263]])
        scalar = math.lcm(*range(1, 257))
        a, b = hits(orders, scalar, 256, 25600)
        self.assertEqual(a.tolist(), [[False, True, False, False, False, False, False]])
        self.assertEqual(b.tolist(), [[False, True, False, False, True, False, False]])
        a, b = hits(orders, 48*scalar, 256, 25600)
        self.assertEqual(a.tolist(), [[False, True, True, True, False, False, False]])
        self.assertEqual(b.tolist(), [[False, True, True, True, True, False, False]])

    def test_complete_prime_population_has_no_sampling_error(self):
        result, _ = estimate([1, 0, 2], [2, 1, 3], 3)
        self.assertEqual(result["value"], .5)
        self.assertEqual(result["se"], 0)


if __name__ == "__main__":
    unittest.main()
