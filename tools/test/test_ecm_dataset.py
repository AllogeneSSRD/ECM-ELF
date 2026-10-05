"""Small exact database/parser/bound regressions; no GPU or GP required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('ecm_dataset_library',Path(__file__).resolve().parents[1]/'ecm_dataset/dataset.py')
d=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(d)


class DatasetTests(unittest.TestCase):
    def test_prime_powers_and_torsion(self):
        self.assertEqual(d.bounds([(2,5),(3,2),(7,1),(11,1)],1),(32,[(32,0)]))
        self.assertEqual(d.bounds([(2,5),(3,2),(7,1),(11,1)],12),(11,[(8,11),(11,0)]))
        self.assertEqual(d.bounds([(17,2),(101,1)],1),(289,[(289,0)]))
        self.assertEqual(d.bounds([(2,3),(3,1),(599,1),(114713,1)],1),
                         (114713,[(599,114713),(114713,0)]))

    def test_import_initial_nulls_and_idempotence(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'f.html';p.write_text('<td id="M11"><a href="/factor/23">23</a><td id="M11"><a href="/factor/89">89</a>')
            db=d.connect(Path(td)/'d.sqlite')
            d.import_catalog(db,p);d.import_catalog(db,p)
            rows=list(db.execute('SELECT * FROM factors'))
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r['sigma'] is None and r['b1'] is None and r['group_order'] is None for r in rows))
            p.write_text('<td id="M11"><a href="/factor/29">29</a>')
            with self.assertRaises(ValueError):d.import_catalog(db,p)
            db.close()

    def test_incomparable_sigma_is_not_stored(self):
        with tempfile.TemporaryDirectory() as td:
            db=d.connect(Path(td)/'d.sqlite');d.ensure_factor(db,11,23)
            for sigma,b1,b2 in [(6,29,563),(7,31,401)]:
                result=dict(sigma=str(sigma),group_order='1',group_factorization='[]',
                            point_order='1',point_factorization=json.dumps([[str(b1),1],[str(b2),1]]))
                d.store_best(db,11,23,result)
            self.assertEqual(db.execute('SELECT sigma FROM factors WHERE exponent=11 AND value="23"').fetchone()[0],'6')
            self.assertEqual(db.execute('SELECT count(*) FROM factors').fetchone()[0],1)
            self.assertEqual({r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")},{'mersennes','factors'})
            db.close()

    def test_default_updates_only_on_dominance(self):
        with tempfile.TemporaryDirectory() as td:
            db=d.connect(Path(td)/'d.sqlite');d.ensure_factor(db,11,23)
            for sigma,parts,expected in [(6,[(29,1),(563,1)],'6'),(7,[(31,1),(401,1)],'6'),
                                         (9,[(3,2),(227,1)],'9'),(10,[(3,2),(227,1)],'9')]:
                result=dict(sigma=str(sigma),group_order='1',group_factorization='[]',point_order='1',
                            point_factorization=json.dumps([[str(p),e] for p,e in parts]))
                d.store_best(db,11,23,result)
                self.assertEqual(db.execute('SELECT sigma FROM factors WHERE exponent=11 AND value="23"').fetchone()[0],expected)
            db.close()

    def test_integer_exponent_inference(self):
        self.assertEqual(d.infer_exponent((1<<8171)-1),8171)
        with self.assertRaises(ValueError):d.infer_exponent(23)


if __name__=='__main__':unittest.main()
