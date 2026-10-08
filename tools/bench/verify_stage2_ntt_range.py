"""Verify a generated host-range build against the original GPU binary.

Dump both binaries with the same cuobjdump; require identical full SASS and
resources after normalizing only their anonymous TU namespace IDs. Instruction
text, encodings, schedule and all other symbols remain exact. No host timing claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


NAMESPACE = re.compile(rb'(_GLOBAL__N__)[0-9a-f]{8}(_18_ecm_cuda_stage2_cu_c66e9bf7)')


def normalized_sha(path):
    h=hashlib.sha256();symbols=set()
    with Path(path).open('rb') as stream:
        for line in stream:
            symbols.update(m.group(0).decode() for m in NAMESPACE.finditer(line))
            h.update(NAMESPACE.sub(rb'\g<1>00000000\g<2>',line))
    if len(symbols)!=1:raise ValueError('requires exactly one known anonymous CUDA TU identity')
    return h.hexdigest(),sorted(symbols)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--analyze-existing',action='store_true',help='Retain first rejected dumps; validate their original SHA before name-only analysis')
    p.add_argument('--cuobjdump',type=Path,default=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/cuobjdump.exe'))
    a=p.parse_args();exe=a.exe.resolve();project=a.project.resolve();out=exe.parent
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
    generated=read(project/'instrumentation.json');base=Path(generated['base_exe'])
    if sha(base)!=generated['base_binary_sha256']:raise ValueError('base binary changed')
    build=read(out/'build_manifest.json')
    if sha(exe)!=build['sha256'].lower():raise ValueError('instrumented binary changed')
    if {n:h.lower() for n,h in build['source_hashes'].items()}!=generated['generated_sources']:
        raise ValueError('instrumented closure differs')
    for name,want in generated['generated_sources'].items():
        if sha(project/name)!=want:raise ValueError('generated source changed: '+name)
    for stem,want in build['objects'].items():
        if sha(out/'_objects'/(stem+'.obj'))!=want.lower():raise ValueError('compiled object changed: '+stem)
    files={}
    initial=read(out/'gpu_equivalence_initial_rejection.json') if a.analyze_existing else None
    for label,binary in [('base',base),('instrumented',exe)]:
        for kind,flag in [('sass','--dump-sass'),('resources','--dump-resource-usage')]:
            dest=out/(label+'_'+kind+'.txt')
            if a.analyze_existing:
                if sha(dest)!=initial['files'][label+'_'+kind]['sha256']:raise ValueError('original rejected dump changed')
            else:
                if dest.exists():raise ValueError('retain old dump; use a new build/output directory')
                with dest.open('wb') as stream:
                    subprocess.run([str(a.cuobjdump),flag,str(binary)],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=180)
            files[label+'_'+kind]=dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size)
    exact=files['base_sass']['sha256']==files['instrumented_sass']['sha256']
    resources=files['base_resources']['sha256']==files['instrumented_resources']['sha256']
    normalized={k:normalized_sha(v['path']) for k,v in files.items()}
    same_sass=normalized['base_sass'][0]==normalized['instrumented_sass'][0]
    same_resources=normalized['base_resources'][0]==normalized['instrumented_resources'][0]
    count=len(re.findall(r'^ Function ',Path(files['instrumented_resources']['path']).read_text(),re.M))
    receipt=dict(complete=same_sass and same_resources and count==172,full_raw_identical=exact,
        normalized_raw_identical=same_sass,normalized_resources_identical=same_resources,
        normalized_files={k:dict(sha256=v[0],namespace=v[1]) for k,v in normalized.items()},
        resources_identical=resources,kernels=count,base_binary_sha256=sha(base),
        instrumented_binary_sha256=sha(exe),base_sass=files['base_sass']['path'],
        instrumented_sass=files['instrumented_sass']['path'],sass_sha256=files['base_sass']['sha256'],
        files=files,project_sha256=sha(project/'instrumentation.json'),tool_sha256=sha(__file__),
        scope='Full GPU SASS/resource output equal after only anonymous TU ID normalization. All instructions/encodings/scheduling bytes unchanged. Host-only ranges still need actual curve/output/check and geometry validation; not host-code or timing equivalence.')
    (out/'gpu_equivalence.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('complete','full_raw_identical','resources_identical','kernels','instrumented_binary_sha256')}))
    if not receipt['complete']:raise ValueError('GPU binary equivalence failed; retain all raw dumps')


if __name__=='__main__':main()
