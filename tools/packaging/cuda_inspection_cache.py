"""Disposable, content-addressed cuobjdump results; never a build identity gate."""
import hashlib
import json
from pathlib import Path
import uuid


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class InspectionCache:
    def __init__(self,root,cuda,*,force=False):
        self.root=Path(root);self.inspector=Path(cuda)/'bin/cuobjdump.exe';self.force=force
        self.identity=None;self.hits=0;self.misses=0

    def get(self,kind,binary,scan):
        if self.identity is None:
            # Parser/validation edits and in-place toolkit replacement invalidate
            # old entries, regardless of timestamps or installation directory.
            self.identity=dict(schema=1,tool=str(self.inspector.resolve()),tool_sha256=sha(self.inspector),
                verifier_sha256=sha(Path(__file__).with_name('package_release.py')),cache_sha256=sha(__file__))
        before=sha(binary)
        identity=dict(self.identity,kind=kind,input_sha256=before)
        path=self.root/(digest(identity)+'.json')
        if not self.force:
            try:
                record=json.loads(path.read_text(encoding='utf-8'))
                if record['identity']==identity and record['result_sha256']==digest(record['result']):
                    self.hits+=1
                    return record['result'],before
            except (OSError,ValueError,KeyError,TypeError):pass
        result=scan()
        if sha(binary)!=before:raise ValueError(f'CUDA input changed during inspection: {binary}')
        self.root.mkdir(parents=True,exist_ok=True)
        scratch=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        try:
            scratch.write_text(json.dumps(dict(identity=identity,result=result,result_sha256=digest(result))),encoding='utf-8')
            scratch.replace(path)
        finally:scratch.unlink(missing_ok=True)
        self.misses+=1
        return result,before
