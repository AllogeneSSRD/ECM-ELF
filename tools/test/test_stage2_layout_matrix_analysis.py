"""Geometry/log partition and held-out stopping seams; no GPU calls."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from bench_stage2_layout_matrix import phases, stable_repeats
from bench_stage2_version import shape_records
from stage2_layout_matrix_plot import audit,current_detail


class MatrixAnalysis(unittest.TestCase):
    def test_both_geometry_log_generations(self):
        old='real_shape: B1=20 B2=1000 S_bits=127 D=30 P=4 giant_points=35 num_poly_g=9\n'
        new='real_shape: B1=20 B2=1000 S_bits=127\nreal_shape: D=30 P=4 giant=35 poly_g=9\n'
        self.assertEqual(shape_records(old),shape_records(new))
        self.assertEqual(len(shape_records(old+new)),2)

    def test_disjoint_partition(self):
        split=dict(inv='1',giant='1',gtrees='2',fold='1',descent='1',accum='.5')
        p=phases(10,2,dict(ladder='1',affine='.2'),split)
        self.assertAlmostEqual(sum(p.values()),10)
        self.assertAlmostEqual(p['ftree_init'],.8)
        self.assertAlmostEqual(p['other'],1.5)
        with self.assertRaises(ValueError):
            phases(5,2,dict(ladder='1',affine='.2'),split)

    def test_only_new_calls_count_and_layouts_stay_separate(self):
        def record(layout,probe=False):
            return {'stage2-engine':{'tune_plan':dict(D=30,carrier_exponent=127,
                groot_balanced=layout,runtime_probe=probe,runtime_samples=10)}}
        self.assertFalse(stable_repeats([record(0)]))
        self.assertFalse(stable_repeats([record(0),record(0),record(1)]))
        self.assertTrue(stable_repeats([record(0)]*3))
        self.assertFalse(stable_repeats([record(0,True)]*3))

    def test_incomplete_or_wrong_scope_cannot_publish(self):
        with self.assertRaises(ValueError):
            audit(dict(complete=False),Path('.'))
        with self.assertRaises(ValueError):
            audit(dict(complete=True,failures=[],schema='layout-matrix-1',
                       cases=[{}]*12,cells=[{}]*13,validation_repeats=3),Path('.'))

    def test_leaf_preparation_is_removed_from_residual(self):
        row=dict(stats={'real_batched_wall':{'gleaves':'.4'}},
                 phases=dict(ftree_init=.5,other=.6,giant=1))
        detail=current_detail(row)
        self.assertAlmostEqual(sum(detail.values()),sum(row['phases'].values()))
        self.assertAlmostEqual(detail['other'],.2)
        self.assertAlmostEqual(detail['gleaves'],.4)
        self.assertAlmostEqual(row['phases']['other'],.6)
        row['stats']['real_batched_wall']['gleaves']='.7'
        with self.assertRaises(ValueError):current_detail(row)


if __name__=='__main__':unittest.main()
