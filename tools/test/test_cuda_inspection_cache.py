"""Cache invalidation and unchanged production cuobjdump validation gates."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/packaging'))
from cuda_inspection_cache import InspectionCache
import package_release as packaging


class CacheTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'build_release_contract_tests';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.temp.name)
        self.cuda=self.root/'cuda';(self.cuda/'bin').mkdir(parents=True)
        self.tool=self.cuda/'bin/cuobjdump.exe';self.tool.write_bytes(b'tool')
        self.binary=self.root/'worker.exe';self.binary.write_bytes(b'worker')
        self.cache_root=self.root/'cache'

    def tearDown(self):self.temp.cleanup()
    def cache(self,force=False):return InspectionCache(self.cache_root,self.cuda,force=force)

    def test_images_reuse_content_then_invalidate_and_force(self):
        def scan(command,**kwargs):
            suffix='cubin' if command[1]=='--list-elf' else 'ptx'
            arch='60' if suffix=='cubin' else '50'
            return f'ELF file 1: worker.sm_{arch}.{suffix}'.encode()
        with patch.object(packaging.subprocess,'check_output',side_effect=scan) as process:
            first=self.cache();packaging.images(self.cuda,self.binary,'50','60',first)
            self.assertEqual(process.call_count,2)
            second=self.cache();packaging.images(self.cuda,self.binary,'50','60',second)
            self.assertEqual((process.call_count,second.hits),(2,1))
            # Same bytes at another path, and changed timestamp, remain reusable.
            copy=self.root/'copy.exe';copy.write_bytes(self.binary.read_bytes());self.binary.touch()
            packaging.images(self.cuda,copy,'50','60',self.cache());self.assertEqual(process.call_count,2)
            # Even with identical size/timestamp, content changes invalidate.
            self.binary.write_bytes(b'WORKER')
            packaging.images(self.cuda,self.binary,'50','60',self.cache());self.assertEqual(process.call_count,4)
            self.tool.write_bytes(b'TOOL')
            packaging.images(self.cuda,self.binary,'50','60',self.cache());self.assertEqual(process.call_count,6)
            packaging.images(self.cuda,self.binary,'50','60',self.cache(True));self.assertEqual(process.call_count,8)
            with self.assertRaisesRegex(ValueError,'unexpected ptx'):
                packaging.images(self.cuda,self.binary,'75','60',self.cache())
            self.assertEqual(process.call_count,8)

    def test_corruption_failure_and_concurrent_change_are_not_reused(self):
        scans=[]
        def scan():scans.append(1);return {'entries':['entry']}
        self.cache().get('ptx',self.binary,scan)
        file=next(self.cache_root.glob('*.json'));file.write_text('{partial')
        self.cache().get('ptx',self.binary,scan);self.assertEqual(len(scans),2)
        record=json.loads(file.read_text());record['result']['entries']=[];file.write_text(json.dumps(record))
        self.cache().get('ptx',self.binary,scan);self.assertEqual(len(scans),3)
        with self.assertRaisesRegex(RuntimeError,'tool failed'):
            self.cache().get('failure',self.binary,lambda:(_ for _ in ()).throw(RuntimeError('tool failed')))
        self.assertEqual(len(list(self.cache_root.glob('*.json'))),1)
        def changed():self.binary.write_bytes(b'changed');return {}
        with self.assertRaisesRegex(ValueError,'changed during inspection'):self.cache().get('changed',self.binary,changed)
        self.assertEqual(len(list(self.cache_root.glob('*.json'))),1)

    def test_object_hits_still_validate_current_curve_and_width(self):
        build=self.root/'build';build.mkdir();obj=build/'kernel.obj';obj.write_bytes(b'object')
        source=build/'cuda_modules/stage1/edwards_512.cu'
        def commands(path):
            (build/'compile_commands.json').write_text(json.dumps([dict(file=str(path),directory=str(build),command=f'nvcc -o "{obj}"')]))
        commands(source)
        entry=b'.version 7.8\n.target sm_50\n.visible .entry kernel_ParamsILj512ELj4E()\n'
        class Process:
            def __init__(self,*args,**kwargs):self.stdout=io.BytesIO(entry);self.stderr=io.BytesIO()
            def wait(self):return 0
        with patch.object(packaging.subprocess,'Popen',side_effect=Process) as process:
            packaging.inspect_modules(self.cuda,build,'stage1',build/'report.json',self.cache())
            packaging.inspect_modules(self.cuda,build,'stage1',build/'report.json',self.cache())
            self.assertEqual(process.call_count,1)
            commands(source.with_name('edwards_1024.cu'))
            with self.assertRaisesRegex(ValueError,'Unexpected curve/parameter/width'):
                packaging.inspect_modules(self.cuda,build,'stage1',build/'report.json',self.cache())
            self.assertEqual(process.call_count,1)

    def test_verifier_identity_change_invalidates(self):
        from cuda_inspection_cache import sha as real_sha
        calls=[]
        def scan():calls.append(1);return {}
        self.cache().get('test',self.binary,scan)
        def changed(path):return 'f'*64 if Path(path).name=='package_release.py' else real_sha(path)
        with patch('cuda_inspection_cache.sha',side_effect=changed):self.cache().get('test',self.binary,scan)
        self.assertEqual(len(calls),2)


if __name__=='__main__':unittest.main()
