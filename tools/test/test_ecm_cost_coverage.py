"""Coverage regressions: complete flags/counts cannot hide missing cases."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from ecm_cost_coverage import expected_measurements,check_study_coverage,check_blind_coverage,check_replay_labels,independent_scope_counts


def fixture():
    controls=dict(bits=[2203,4423,8191],d=[30030,60060,120120],repeats=2,stage1_batch=[1,12],
        b2=3000000000,train_b2=[3000000000,6000000000],holdout_b2=3750000000,resident_mb=640,
        chain_min=8192,g1=True,g2=True,bridge=True,holdout_all_d=True)
    stage1=[dict(bits=bits,batch=batch,name=f's1_m{bits}_batch{batch}_r{rep}',warmup=rep==0) for bits in controls['bits']
        for batch in controls['stage1_batch'] for rep in range(controls['repeats']+1)]
    rows=expected_measurements(controls)
    return dict(controls=controls,stage1=stage1,stage2=copy.deepcopy(rows),planned_stage2_cases=copy.deepcopy(rows),complete=True)


class CoverageTests(unittest.TestCase):
    def test_full_matrix(self):
        self.assertEqual(check_study_coverage(fixture()),dict(stage2_observations=666,stage1_batches=18))
    def test_duplicate_replaces_missing_same_count(self):
        data=fixture();data['stage2'][0]=copy.deepcopy(data['stage2'][1])
        with self.assertRaisesRegex(ValueError,'missing, duplicated'):check_study_coverage(data)
    def test_both_plan_and_rows_deleted(self):
        data=fixture();data['stage2'].pop();data['planned_stage2_cases'].pop()
        with self.assertRaises(ValueError):check_study_coverage(data)
    def test_stage1_batch_missing(self):
        data=fixture();data['stage1'][0]=copy.deepcopy(data['stage1'][-1])
        with self.assertRaisesRegex(ValueError,'Stage1'):check_study_coverage(data)
    def test_duplicate_cold_stage1_same_group(self):
        data=fixture();data['stage1'][2]=copy.deepcopy(data['stage1'][1])
        with self.assertRaisesRegex(ValueError,'Stage1'):check_study_coverage(data)
    def test_duplicate_blind_replaces_missing(self):
        cases=[dict(scope_id='same',bits=8191,D=30030,B2=3000000000,owner_mb=0,rep=r,kind='blind') for r in range(2)]
        frozen=dict(cases=cases);runs=[dict(case=copy.deepcopy(c)) for c in cases]
        self.assertEqual(check_blind_coverage(frozen,runs),2)
        runs[1]=copy.deepcopy(runs[0])
        with self.assertRaisesRegex(ValueError,'missing or duplicated'):check_blind_coverage(frozen,runs)
    def test_replay_cannot_be_renamed_blind(self):
        study=dict(stage2=[dict(bits=8191,D=30030,owner_mb=0,kind='train',features=dict(I=2880))])
        c=dict(bits=8191,D=30030,owner_mb=0,kind='shape_replay',features=dict(I=2880),scope_id='g1')
        check_replay_labels(study,[c]);self.assertEqual(independent_scope_counts([c]),{})
        c['kind']='blind'
        with self.assertRaisesRegex(ValueError,'replay label'):check_replay_labels(study,[c])


if __name__=='__main__':unittest.main()
