"""Regression fixtures for repeated-feature phase medians (no CUDA required)."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from calibrate_stage2_d import features
from ecm_cost_model import PHASE_FEATURES,fit,giant_work
import fit_ecm_costs


def fixture():
    rows=[]
    for d in (30030,60060,120120):
        for b2 in (3000000000,6000000000):
            for rep in range(2):
                f=features(d,b2,4423);phases={phase:f[key]*1e-7 for phase,key in PHASE_FEATURES.items() if phase!='glue'}
                w=giant_work(f);phases.update(giant=w['chain']*1e-6+w['chain_chunks']*0.1+w['ladder']*1e-5,
                    residual=f['G']*0.01,name=0)
                rows.append(dict(features=f,phases=phases))
    return rows


class CostFitTests(unittest.TestCase):
    def test_constant_work_reuses_samples_across_b2(self):
        rows=fixture();bad=copy.deepcopy(rows)
        bad[8]['phases']['inv']*=100
        self.assertGreater(fit(bad)['inv'],1e-7*2)
        robust=fit_ecm_costs.fit_phase_medians(bad)
        self.assertTrue(math.isclose(robust['inv'],1e-7,rel_tol=1e-12))
        self.assertEqual(bad[8]['phases']['inv'],rows[8]['phases']['inv']*100)

    def test_uncontaminated_rates_are_preserved(self):
        rows=fixture();expected=fit(rows);actual=fit_ecm_costs.fit_phase_medians(rows)
        for key,value in expected.items():
            if key=='giant':
                for route,rate in value.items():self.assertTrue(math.isclose(actual[key][route],rate,rel_tol=1e-12,abs_tol=1e-12))
            else:self.assertTrue(math.isclose(actual[key],value,rel_tol=1e-12,abs_tol=1e-12))


if __name__=='__main__':unittest.main()
