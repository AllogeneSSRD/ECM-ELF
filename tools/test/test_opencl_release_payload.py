"""Release OpenCL payload identity and native dependency failure injection."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/packaging'))
import opencl_release_payload as payload


class PayloadTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'build_release_contract_tests';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.temp.name)
        self.build=self.root/'build';self.gmp=self.root/'gmp';self.ssl=self.root/'ssl'
        self.repo=self.root/'repo';self.source=self.put(self.repo/'source.cpp','source')
        self.resource=self.put(self.repo/'src/ecm/opencl/resources/test.cl','kernel')
        self.put(self.build/'opencl_resources/test.cl','kernel')
        self.put(self.build/'ecm.exe','worker')
        for name in payload.RUNTIMES:
            self.put((self.gmp if name=='gmp-10.dll' else self.ssl)/'bin'/name,name)
            self.put(self.build/name,name)
        self.put(self.ssl/'share/openssl/copyright','license')
        self.compiler=self.put(self.root/'compiler','compiler');self.cmake=self.put(self.root/'cmake','cmake')
        fields=dict(role='release',backend='opencl',gmp_root=str(self.gmp),openssl_root=str(self.ssl),compiler=str(self.compiler),
            compiler_sha256=payload.sha(self.compiler),cmake=str(self.cmake),cmake_sha256=payload.sha(self.cmake))
        framed=''.join(str(len(str(v)))+':'+str(v) for name in sorted(fields) for v in (name,fields[name]))
        self.put(self.build/'profile.json',json.dumps(dict(profile_sha256=hashlib.sha256(framed.encode()).hexdigest(),fields=fields)))
        self.put(self.build/'CMakeCache.txt','ECM_OPENCL_PORTABLE_RESOURCES:BOOL=ON\nECM_ENABLE_CUDA:BOOL=OFF\n')
        self.put(self.build/'compile_commands.json',json.dumps([dict(file='source.cpp',command='cl source.cpp')]))
        self.header=self.put(self.root/'sdk/header.h','header')
        self.obj=self.put(self.build/'CMakeFiles/ecm.dir/source.cpp.obj','object')
        self.dep=self.put(Path(str(self.obj)+'.d'),str(self.header)+'\n')
        self.patch_root=patch.object(payload,'ROOT',self.repo);self.patch_root.start()
        self.patch_sources=patch.object(payload,'sources',side_effect=lambda:{'source.cpp':payload.sha(self.source)});self.patch_sources.start()
        self.snapshot=self.put(self.build/'snapshot.json',json.dumps(payload.snapshot_inputs(self.build)))

    def tearDown(self):self.patch_sources.stop();self.patch_root.stop();self.temp.cleanup()
    def put(self,path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding='utf-8');return path
    def freeze(self):payload.freeze(self.build,self.snapshot)

    def test_freeze_and_verify_complete_payload(self):
        self.freeze();identity=payload.verify(self.build,self.gmp)
        self.assertEqual(set(identity['files_sha256']),{'ecm.exe',*payload.RUNTIMES,'opencl_resources/test.cl','licenses/OpenSSL.txt'})
        self.assertIn(str(self.header.resolve()),identity['compiler_inputs_sha256'])

    def test_changed_source_header_object_resource_and_worker_rejected(self):
        self.freeze()
        for path in (self.source,self.header,self.obj,self.build/'opencl_resources/test.cl',self.build/'ecm.exe'):
            with self.subTest(path=path):
                before=path.read_bytes();path.write_bytes(b'changed')
                try:
                    with self.assertRaises(ValueError):payload.verify(self.build,self.gmp)
                finally:path.write_bytes(before)

    def test_unstable_build_and_missing_resource_rejected(self):
        self.header.write_text('changed');self.dep.write_text('')
        with self.assertRaisesRegex(ValueError,'header changed during build'):self.freeze()
        self.header.write_text('header');self.dep.write_text(str(self.header)+'\n')
        (self.build/'opencl_resources/test.cl').unlink()
        with self.assertRaisesRegex(ValueError,'resource deployment'):self.freeze()

    def test_uninitialized_cuda_off_cache_and_no_cuda_commands(self):
        cache=self.build/'CMakeCache.txt';cache.write_text(cache.read_text().replace('ECM_ENABLE_CUDA:BOOL','ECM_ENABLE_CUDA:UNINITIALIZED'))
        self.freeze();payload.verify(self.build,self.gmp)
        self.put(self.build/'compile_commands.json',json.dumps([dict(file='source.cu',command='nvcc source.cu')]))
        with self.assertRaisesRegex(ValueError,'CUDA compile commands'):self.freeze()


if __name__=='__main__':unittest.main()
