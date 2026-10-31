"""Exercise real matrix aggregation/import identity without running GPU work."""
from pathlib import Path
import sys
import tomllib
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from stage2_tune_matrix import aggregate, sample_groot_layout, write_profile


class LayoutIdentity(unittest.TestCase):
    def row(self, identity, seconds, **changes):
        row = dict(condition=0, target_bits=5700, arithmetic_bits=5987,
                   carrier_exponent=5987, modulus_kind='mersenne', b1=260000000,
                   b2=1300000000000, d=1141140, fold_resident=1, frontier_resident=1,
                   execution_path='resident', giant_chunk_points=103680,
                   giant_chain_min=32768, giant_force_ladder=0, source='measured',
                   benchmark_kind='runtime_save', repeats=1, mean_seconds=seconds,
                   stddev_seconds=0, runtime_sample_id=identity)
        return row | changes

    def test_legacy_and_new_populations_stay_separate(self):
        rows = aggregate([self.row('old', 70), self.row('new-a', 50, groot_balanced=1),
                          self.row('new-b', 54, groot_balanced=1)])
        self.assertEqual(len(rows), 2)
        by_layout = {r['groot_balanced']: r for r in rows}
        self.assertEqual(by_layout[0]['mean_seconds'], 70)
        self.assertEqual(by_layout[0]['repeats'], 1)
        self.assertEqual(by_layout[1]['mean_seconds'], 52)
        self.assertEqual(by_layout[1]['repeats'], 2)

    def test_historical_wide_and_explicit_dense(self):
        self.assertEqual(sample_groot_layout(self.row('wide', 1, arithmetic_bits=9000)), 1)
        self.assertEqual(sample_groot_layout(self.row('wide', 1, arithmetic_bits=9000, groot_balanced=0)), 0)
        with self.assertRaises(ValueError):
            aggregate([self.row('invalid', 1, groot_balanced=2)])

    def test_writer_requires_layout_aware_reader(self):
        document = tomllib.loads(write_profile([dict(uuid_hex='fixture')],
            [self.row('old', 70), self.row('new', 50, groot_balanced=1)]))
        self.assertEqual(document['profile']['format'], 10)
        columns, *rows = document['width']['5700']['matrix']
        self.assertEqual({row[columns.index('groot_balanced')] for row in rows}, {0, 1})

    def test_point_policies_are_independent(self):
        rows=aggregate([self.row('legacy',70),self.row('balanced',60,giant_tail_balance=1),
                        self.row('shorter',50,giant_chain_block=32)])
        self.assertEqual(len(rows),3)
        self.assertEqual({(r['giant_tail_balance'],r['giant_chain_block'],r['mean_seconds']) for r in rows},
                         {(0,64,70),(1,64,60),(0,32,50)})
        for changes in ({'giant_tail_balance':2},{'giant_chain_block':128}):
            with self.assertRaises(ValueError):aggregate([self.row('bad',1,**changes)])


if __name__ == '__main__':
    unittest.main()
