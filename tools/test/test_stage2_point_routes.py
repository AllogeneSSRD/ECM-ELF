"""Keep diagnostic fixtures separate from clean performance receipts."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from bench_stage2_point_routes import validate_clock


class PointRouteClocks(unittest.TestCase):
    def test_explicit_leaf_diagnostics_use_the_engine_fixture_contract(self):
        clock=dict(total='3',init='1',main='2',clean='0')
        validate_clock(clock,leaf_checks=True)
        with self.assertRaises(ValueError):validate_clock(clock)

    def test_perf_requires_clean_and_diag_requires_actual_extra_checks(self):
        clock=dict(total='3',init='1',main='2',clean='1')
        validate_clock(clock)
        with self.assertRaises(ValueError):validate_clock(clock,leaf_checks=True)

    def test_nonfinite_or_unclosed_clocks_are_rejected(self):
        for total in ('nan','inf','-3','4'):
            with self.assertRaises(ValueError):validate_clock(dict(total=total,init='1',main='2',clean='1'))


if __name__=='__main__':unittest.main()
