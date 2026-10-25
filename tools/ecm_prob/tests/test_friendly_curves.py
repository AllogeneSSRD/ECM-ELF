"""Exact identities and domain checks for the GKL experimental constructors."""
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from friendly_curves import family_iv, atkin_morain, from_t, is_square, paper_c6, legacy_z2x8


class FriendlyCurveTests(unittest.TestCase):
    def test_paper_table2_golden(self):
        c = family_iv(1)
        self.assertEqual((c.t, c.e, c.x, c.y), (F(6), F(15, 8), F(28, 195), F(3152, 3495)))
        self.assertEqual((c.a, c.d), (-1, -F(15, 8)**4))
        self.assertTrue(is_square(c.e**2+1))
        self.assertEqual(c.e, (F(4)-1/F(4))/2)

    def test_overview_mistyped_condition(self):
        e = (F(2)-1)/F(2)**2
        self.assertFalse(is_square(e*e+1))

    def test_families_and_birational_maps(self):
        for builder in (family_iv, atkin_morain):
            for m in range(1, 9):
                with self.subTest(family=builder.__name__, m=m):
                    c = builder(m)
                    coefficients, point = c.weierstrass()
                    x, y = point
                    self.assertEqual(y*y, x**3+coefficients[1]*x*x+coefficients[3]*x)
                    if builder is family_iv:
                        self.assertTrue(is_square(c.e*c.e+1))

    def test_reject_invalid_domains(self):
        for m in (0, -1):
            with self.assertRaises(ValueError):
                family_iv(m)
        for t in (0, 2, -2, -1, -4):
            with self.assertRaises((ValueError, ZeroDivisionError)):
                from_t(t)

    def test_fixed_comparison_points_and_valid_t4(self):
        for c in (paper_c6(), legacy_z2x8(), from_t(4)):
            coefficients, point = c.weierstrass()
            x, y = point
            self.assertEqual(y*y, x**3+coefficients[1]*x*x+coefficients[3]*x)


if __name__ == "__main__":
    unittest.main()
