"""Integer coverage fixtures, not CUDA arithmetic or time-model gates."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from ecm_cost_cases import low_cases,regime_intervals,point_b2,validation_points
from ecm_cost_model import features


class CaseTests(unittest.TestCase):
    def test_adjacent_regimes_cover_every_integer_b2(self):
        for d in (30030,60060,120120):
            bounds=regime_intervals(d,3000000000)
            self.assertEqual(bounds['g1'][1]+1,bounds['g2'][0])
            self.assertEqual(bounds['g2'][1]+1,bounds['bridge'][0])
            self.assertEqual(bounds['bridge'][1]+1,3000000000)
            rows=low_cases(d,3000000000,8192,True,True,True)
            for regime,interval in bounds.items():
                train=[b for r,b,k in rows if r==regime and k=='train']
                self.assertEqual((min(train),max(train)),interval)
                held=[b for r,b,k in rows if r==regime and k=='holdout']
                self.assertEqual(len(held),1)
                self.assertNotIn(held[0],train)
    def test_dispatch_and_unique_samples(self):
        for d in (30030,60060,120120):
            rows=low_cases(d,3000000000,8192,True,True,True)
            self.assertEqual(len(rows),len(set(rows)))
            for regime,b2,kind in rows:
                g=features(d,b2,8191,8192)['G']
                self.assertTrue(g==1 if regime=='g1' else g==2 if regime=='g2' else g>=3)
    def test_legacy_g1_schedule_preserved(self):
        rows=low_cases(30030,3000000000,8192,g1=True)
        self.assertEqual(len(rows),5)
        self.assertTrue(all(r=='g1' for r,_,_ in rows))
    def test_validation_covers_every_regime_and_chunk_boundary(self):
        for bits in (2203,4423,8191):
            for d in (30030,60060,120120):
                bounds=regime_intervals(d,3000000000)|{'multiple':(3000000000,6000000000)}
                for regime,(lo,hi) in bounds.items():
                    scope=dict(bits=bits,regime=regime,b2_min=lo,b2_max=hi)
                    samples=validation_points(scope,d,8192,5250000000)
                    self.assertTrue(samples)
                    self.assertTrue(all(lo<=b<=hi for b,_ in samples))
        scope=dict(bits=8191,regime='multiple',b2_min=3000000000,b2_max=6000000000)
        samples=dict(validation_points(scope,30030,8192,5250000000))
        self.assertEqual(samples[4224320100],'chunk_boundary_blind')


if __name__=='__main__':unittest.main()
