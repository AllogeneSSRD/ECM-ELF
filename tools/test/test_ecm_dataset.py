"""Small exact database/parser/bound regressions; no GPU or GP required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

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

    def test_scan_accepts_either_improvement_and_limits_b2(self):
        with tempfile.TemporaryDirectory() as td:
            db=d.connect(Path(td)/'d.sqlite');d.ensure_factor(db,11,23)
            def store(sigma, parts):
                result=dict(sigma=str(sigma),group_order='1',group_factorization='[]',point_order='1',
                            point_factorization=json.dumps([[str(p),e] for p,e in parts]))
                return d.store_best(db,11,23,result,scan_policy=True)
            def selected():
                row=db.execute('SELECT sigma,b1,b2 FROM factors WHERE exponent=11 AND value="23"').fetchone()
                return tuple(row)
            self.assertTrue(store(6,[(29,1),(563,1)]))
            self.assertTrue(store(7,[(31,1),(401,1)]))  # B2 improves while B1 rises.
            self.assertEqual(selected(),('7','31','401'))
            self.assertTrue(store(8,[(29,1),(563,1)]))  # B1 improves while B2 rises.
            self.assertEqual(selected(),('8','29','563'))
            self.assertFalse(store(9,[(29,1),(563,1)]))
            db.close()
        with tempfile.TemporaryDirectory() as td:
            db=d.connect(Path(td)/'d.sqlite');d.ensure_factor(db,11,23)
            self.assertTrue(d.store_best(db,11,23,dict(sigma='6',group_order='1',group_factorization='[]',
                            point_order='1',point_factorization='[["2",2]]'),scan_policy=True))
            self.assertFalse(d.store_best(db,11,23,dict(sigma='7',group_order='1',group_factorization='[]',
                             point_order='1',point_factorization='[["2",1],["200000",1]]'),scan_policy=True))
            self.assertEqual(db.execute('SELECT sigma FROM factors WHERE exponent=11 AND value="23"').fetchone()[0],'6')
            db.close()
        with tempfile.TemporaryDirectory() as td:
            db=d.connect(Path(td)/'d.sqlite');d.ensure_factor(db,11,23)
            def result(sigma, b1, b2):
                return dict(sigma=str(sigma),group_order='1',group_factorization='[]',point_order='1',
                            point_factorization=json.dumps([[str(b1),1],[str(b2),1]]))
            self.assertTrue(d.store_best(db,11,23,result(6,29,700000)))
            self.assertTrue(d.store_best(db,11,23,result(7,23,500000),scan_policy=True))
            row=db.execute('SELECT sigma,b1,b2 FROM factors WHERE exponent=11 AND value="23"').fetchone()
            self.assertEqual(tuple(row),('7','23','500000'))
            db.close()

    def test_scan_policies(self):
        def result(sigma, parts):
            return dict(sigma=str(sigma),group_order='1',group_factorization='[]',point_order='1',
                        point_factorization=json.dumps([[str(p),1] for p in parts]))
        with tempfile.TemporaryDirectory() as td:
            db=d.connect(Path(td)/'d.sqlite');d.ensure_factor(db,11,23)
            self.assertTrue(d.store_best(db,11,23,result(6,(41,2441)),scan_policy='strict'))
            self.assertFalse(d.store_best(db,11,23,result(7,(29,590137)),scan_policy='strict'))
            self.assertTrue(d.store_best(db,11,23,result(8,(37,2003)),scan_policy='strict'))
            self.assertEqual(db.execute('SELECT sigma FROM factors').fetchone()[0],'8')
            db.close()
        with tempfile.TemporaryDirectory() as td:
            db=d.connect(Path(td)/'d.sqlite');d.ensure_factor(db,11,23)
            self.assertTrue(d.store_best(db,11,23,result(6,(41,2441)),scan_policy='normal'))
            self.assertFalse(d.store_best(db,11,23,result(7,(29,590137)),scan_policy='normal'))
            self.assertFalse(d.store_best(db,11,23,result(8,(43,492761)),scan_policy='normal'))
            self.assertTrue(d.store_best(db,11,23,result(9,(43,1637)),scan_policy='normal'))
            self.assertEqual(db.execute('SELECT sigma FROM factors').fetchone()[0],'9')
            db.close()
        with tempfile.TemporaryDirectory() as td:
            db=d.connect(Path(td)/'d.sqlite');d.ensure_factor(db,11,23)
            stage1_only=result(6,(2,))
            stage1_only['point_factorization']='[["2",7]]'
            self.assertTrue(d.store_best(db,11,23,stage1_only,scan_policy='normal'))
            self.assertEqual(d.bound_score((128,0)),16384)
            self.assertFalse(d.store_best(db,11,23,result(7,(41,2441)),scan_policy='normal'))
            self.assertTrue(d.store_best(db,11,23,result(8,(2,1000)),scan_policy='normal'))
            self.assertEqual(db.execute('SELECT sigma FROM factors').fetchone()[0],'8')
            db.close()

    def test_integer_exponent_inference(self):
        self.assertEqual(d.infer_exponent((1<<8171)-1),8171)
        with self.assertRaises(ValueError):d.infer_exponent(23)

    def test_connect_waits_for_temporary_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'d.sqlite'
            db=d.connect(path);db.close()
            original_connect=d.sqlite3.connect
            def short_timeout(*args, **kwargs):
                kwargs['timeout']=0.05
                return original_connect(*args, **kwargs)
            ready=threading.Event()
            def lock_then_release():
                locked=original_connect(path)
                try:
                    locked.execute('BEGIN EXCLUSIVE')
                    ready.set()
                    time.sleep(0.2)
                finally:
                    locked.rollback();locked.close()
            worker=threading.Thread(target=lock_then_release)
            worker.start();self.assertTrue(ready.wait(1))
            try:
                with mock.patch.object(d.sqlite3,'connect',side_effect=short_timeout):
                    contender=d.connect(path)
                contender.close()
            finally:
                worker.join()

    def test_reader_does_not_block_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'d.sqlite'
            reader=d.connect(path)
            d.ensure_factor(reader,11,23);d.ensure_factor(reader,11,89)
            cursor=reader.execute('SELECT value FROM factors ORDER BY value')
            cursor.fetchone()
            writer=d.sqlite3.connect(path,timeout=0.1)
            try:
                with writer:
                    writer.execute('UPDATE factors SET primality_verified=1 WHERE value="23"')
            finally:
                writer.close();cursor.close();reader.close()

    def test_concurrent_connections_can_write(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'d.sqlite'
            db=d.connect(path);db.close()
            def write(factor):
                connection=d.connect(path)
                try:
                    d.ensure_factor(connection,11,factor)
                finally:
                    connection.close()
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(write,range(20,28)))
            db=d.connect(path)
            self.assertEqual(db.execute('SELECT count(*) FROM factors').fetchone()[0],8)
            db.close()


if __name__=='__main__':unittest.main()
