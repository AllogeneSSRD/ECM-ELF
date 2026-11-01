"""Keep experimental treatment distinct from effective point policy in receipts."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from bench_stage2_point_matrix import point_policy_values


class PointMetadata(unittest.TestCase):
    def test_old_permission_does_not_prove_a_changed_schedule(self):
        self.assertEqual(point_policy_values({'giant_tail_balance':True}),(True,None))
        self.assertEqual(point_policy_values({'giant_tail_balance':False}),(False,None))

    def test_noop_and_active_new_receipts_keep_their_treatment(self):
        self.assertEqual(point_policy_values({'giant_tail_balance_requested':True,'giant_tail_balance':False}),(True,False))
        self.assertEqual(point_policy_values({'giant_tail_balance_requested':True,'giant_tail_balance':True}),(True,True))

    def test_impossible_effective_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            point_policy_values({'giant_tail_balance_requested':False,'giant_tail_balance':True})


if __name__=='__main__':unittest.main()
