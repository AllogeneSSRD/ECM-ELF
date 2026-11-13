"""Exercise real matrix aggregation/import identity without running GPU work."""
from pathlib import Path
import json
import sys
import tomllib
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from stage2_tune_matrix import aggregate, sample_groot_layout, write_profile
import import_stage2_benchmark_tune as importer


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
        self.assertEqual(document['profile']['format'], 16)
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

    def test_s4_actual_rows_stay_separate(self):
        conditions=[dict(uuid_hex='fixture'),dict(uuid_hex='fixture',env_s4_coop_reduce=32)]
        wide=dict(target_bits=9216,arithmetic_bits=9216,carrier_exponent=0,modulus_kind='generic')
        rows=[self.row('scalar',10,reduce_coop=0,**wide),self.row('old-request',8,condition=1,**wide)]
        rows += [self.row('observed-'+str(k),7-k,condition=1,reduce_coop=32,reduce_threads=128,reduce_kernel=k,**wide) for k in (1,2,3,4)]
        document=tomllib.loads(write_profile(conditions,rows))
        self.assertEqual(len(document['condition']),1)
        columns,*rows=document['width']['9216']['matrix']
        self.assertEqual({(r[columns.index('reduce_coop')],r[columns.index('reduce_kernel')]) for r in rows},{(0,0),(255,0),(32,1),(32,2),(32,3),(32,4)})

    def test_seed_rows_and_legacy_requests_stay_separate(self):
        document=tomllib.loads(write_profile([dict(uuid_hex='fixture',env_giant_seed_coop=32),dict(uuid_hex='fixture')],
            [self.row('legacy-request',3),self.row('observed-scalar',2,condition=1,seed_coop=0)]))
        self.assertEqual(len(document['condition']),1)
        columns,*rows=document['width']['5700']['matrix']
        self.assertEqual({r[columns.index('seed_coop')] for r in rows},{0,255})
        wide=dict(target_bits=9216,arithmetic_bits=9216,carrier_exponent=0,modulus_kind='generic')
        result=aggregate([self.row('scalar',10,seed_coop=0,**wide),self.row('cooperative',5,seed_coop=32,**wide)])
        self.assertEqual(len(result),2)

    def test_runtime_import_cap_is_per_actual_backend(self):
        receipts=[]
        for i,mode in enumerate((0,0,32,32,32,32)):
            sample=self.row(str(i),10,target_bits=9216,arithmetic_bits=9216,carrier_exponent=0,
                modulus_kind='generic',groot_balanced=0,baby_coop=mode,giant_coop=0,
                clean=1,bad=0,hits=0,selftest_cases=1,checked=1,predicted_seconds=10,
                measured_model_ratio=1,prediction_basis='before_measurement',
                phase_accounting='exclusive_engine_v1',init_seconds=10,main_seconds=0)
            for phase in ('shape','setup','baby','ftree','main_setup','inverse_setup','giant_loop','descent','accum','finalize'):
                sample['phase_'+phase+'_seconds']=10 if phase=='baby' else 0
            receipts.append(dict(status='stage2_completed',hits=0,bad_factors=0,factors=[],
                N_hex=format((1<<9215)+1,'x'),carrier_exponent=0,B1=sample['b1'],B2=sample['b2'],
                tune_observation=dict(condition={'uuid_hex':'fixture'},sample=sample)))
        payload=''.join(json.dumps(row)+'\n' for row in receipts).encode('utf-8')
        # Exercise the real import quota, replacing only input I/O/normalization.
        with patch.object(Path,'read_bytes',return_value=payload),patch.object(importer,'normalize_result',side_effect=lambda row:row):
            _,rows,_=importer.import_runtime([Path('fixture-receipts.jsonl')],sample_limit=2)
        self.assertEqual({r['baby_coop']:r['repeats'] for r in rows},{0:2,32:2})


if __name__ == '__main__':
    unittest.main()
