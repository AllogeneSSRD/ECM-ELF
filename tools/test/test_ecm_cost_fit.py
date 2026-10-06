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
import ecm_cost_model as model


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
    def test_per_d_groups_do_not_share_rates(self):
        ds={120120,30030,60060}
        self.assertEqual(fit_ecm_costs.fit_groups(ds,'multiple','per_d'),[[30030],[60060],[120120]])
        self.assertEqual(fit_ecm_costs.fit_groups(ds,'multiple','pooled_d'),[[30030,60060,120120]])
        self.assertEqual(fit_ecm_costs.fit_groups(ds,'g1','pooled_d'),[[30030],[60060],[120120]])
        rows=fixture()
        for row in rows:
            row['D']=row['features']['D']
            row['phases']['baby']*=row['D']/30030
        for group in fit_ecm_costs.fit_groups(ds,'multiple','per_d'):
            local=[r for r in rows if r['D'] in group]
            rate=fit_ecm_costs.fit_phase_medians(local)['baby']
            self.assertTrue(math.isclose(rate,1e-7*group[0]/30030,rel_tol=1e-12))

    def test_small_ladder_latency_floor(self):
        rows=[]
        for count in (300,600,1200,2400):
            f=model.features(60060,60060*(count-2),4423,8192)
            phases={phase:f[key]*1e-7 for phase,key in PHASE_FEATURES.items() if phase!='glue'}
            w=giant_work(f);phases.update(giant=0.7*w['ladder_launches']+1e-6*w['ladder'],residual=0.01,name=0)
            rows.append(dict(features=f,phases=phases))
        rates=fit(rows)['giant']
        self.assertTrue(math.isclose(rates['ladder_launch'],0.7,rel_tol=1e-10))
        self.assertTrue(math.isclose(rates['ladder'],1e-6,rel_tol=1e-10))

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
            elif key=='giant_coverage':self.assertEqual(actual[key],value)
            else:self.assertTrue(math.isclose(actual[key],value,rel_tol=1e-12,abs_tol=1e-12))


if __name__=='__main__':unittest.main()
