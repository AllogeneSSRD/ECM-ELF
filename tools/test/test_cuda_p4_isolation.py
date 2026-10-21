"""P4 experimental source relocation and consumer boundary gates (CPU only)."""
from source_layout import current_source,current_files,LAYOUT
import argparse
import json
from pathlib import Path
import re
import subprocess
from stage2_native_source import ROOT

BASE = '2a2606e'
EXTRACTED = {'point_arithmetic.cuh','carry_reference.cuh','carry_cone.cuh'}


def expand(path):
    text = path.read_text(encoding='utf-8-sig')
    def sub(m):
        return expand(path.parent/m[1]) if m[1] in EXTRACTED else m[0]
    return re.sub(r'^\s*#include "([^"]+)"\s*$', sub, text, flags=re.M)


def normalized(text):
    text = re.sub(r'^\s*#\s*include[^\n]*\n', '', text, flags=re.M)
    text = re.sub(r'^\s*#pragma once[^\n]*\n', '', text, flags=re.M)
    text = re.sub(r'^\s*#define (?:NTT_POLY_PROBE_NO_MAIN(?: 1)?|STAGE2_TREE_GPU_NO_MAIN|NTT_TENSOR_GOLDILOCKS_PROBE_NO_MAIN)[^\n]*\n','',text,flags=re.M)
    # These exact guards surrounded only the old entrypoint. Keep all arithmetic
    # feature guards and macro values in the comparison.
    names=('STAGE2_TREE_GPU','NTT_POLY_PROBE','NTT_TENSOR_GOLDILOCKS_PROBE','NTT_TENSOR_LENGTH_PROBE')
    for name in names:
        found=re.search(r'^#ifndef '+name+r'_NO_MAIN\s*$',text,re.M)
        if found:
            text=text[:found.start()]+text[found.end():]
            at=text.rindex('#endif');text=text[:at]+text[at:].split('\n',1)[-1]
    lexemes=re.findall(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|\w+|[^\s]',text,re.S)
    return [token for token in lexemes if not token.startswith(('//','/*'))]


def check_boundary(path, text):
    for match in re.finditer(r'^\s*#\s*include\s*"([^"]+)"',text,re.M):
        target=(path.parent/match[1]).resolve()
        assert target.suffix not in ('.cu','.cpp','.inl'), ('implementation inclusion',path,match[1])
        if 'tools/experiments/' in path.as_posix():
            private=ROOT/'src/ecm/backend/cuda/stage2'
            allowed={current_source(old).resolve() for old in LAYOUT['moved'] if not old.startswith('src/cuda/stage2/')}
            assert not target.is_relative_to(private) or target in allowed, ('experiment includes private Stage2 host',path,target)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experiments-only',action='store_true',help='Intermediate experimental relocation gate; final gate includes probes')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=json.loads((Path(__file__).with_name('cuda_p4_migrations.json')).read_text())
    checked=[]
    for old,new in spec['mapping'].items():
        before=subprocess.check_output(['git','show',BASE+':'+old],cwd=ROOT).decode('utf-8-sig')
        after=expand(ROOT/new)
        if old in spec['entries']:after+='\n'+(ROOT/spec['entries'][old]).read_text(encoding='utf-8-sig')
        if old=='tools/test/ntt_coop_outer_probe.cu':
            wrapper='int main(int argc,char **argv) { return ntt_coop_probe_run(argc,argv); }'
            assert after.count(wrapper)==1
            after=after.replace(wrapper,'').replace('static int ntt_coop_probe_run(', 'int main(')
        assert normalized(before)==normalized(after), 'changed experimental body/default/macro: '+old
        checked.append(new)
    # The independent scale policy is an explicit experimental adapter. Its
    # body matches the existing production helper, with only includes relocated.
    assert normalized((current_source('src/cuda/stage2/scale_plain.cuh')).read_text())==normalized(
        (ROOT/'tools/experiments/cuda/stage2_tree/plain_scale.cuh').read_text())
    old=subprocess.check_output(['git','show',BASE+':src/cuda/stage2/stage2_d_model.cuh'],cwd=ROOT)
    assert old==(ROOT/'tools/experiments/cuda/stage2_tree/d_model_profiles.h').read_bytes()
    assert not (ROOT/'src/cuda/stage2/stage2_d_model.cuh').exists()
    count=0
    # These existing CPU-only studies/instrumentation are outside CUDA target
    # isolation. The allocator fixture includes generated CPU doubles, not a
    # GPU runtime or production build source fragment.
    cpu_only={'tools/test/stage2_s4_program_fixture.cpp','tools/bench/mers_karatsuba.cpp','tools/bench/mers_sqr_phases.cpp'}
    for folder in (('tools/experiments',) if a.experiments_only else ('tools/experiments','tools/test','tools/bench')):
        for path in (ROOT/folder).rglob('*'):
            if path.suffix in ('.cu','.cpp','.cuh','.h'):
                # One explicit adapter owns the private Stage1 template. The
                # probe itself sees only its finite diagnostic API.
                if path==ROOT/'tools/test/adapters/edwards_host_wait.cu':continue
                if path.relative_to(ROOT).as_posix() in cpu_only:continue
                check_boundary(path,path.read_text(encoding='utf-8-sig'));count+=1
    rejected=0
    for file,body in [
        (ROOT/'tools/experiments/bad.cuh','#include "../../src/ecm/backend/cuda/stage2/scale_plain.cuh"'),
        (ROOT/'tools/test/bad.cu','#include "../experiments/cuda/ntt_variants/main.cu"'),
        (ROOT/'tools/test/bad.cu','#include "../../src/ecm/backend/cuda/stage1/edwards/edwards_runtime.inl"')]:
        try:check_boundary(file,body)
        except AssertionError:rejected+=1
    assert rejected==3
    result=dict(complete=True,scope='experiments' if a.experiments_only else 'CUDA consumers',baseline=BASE,moved=checked,consumer_files=count,negative_boundaries=rejected,cpu_only_exclusions=sorted(cpu_only))
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,moved=len(checked),consumer_files=count,negative_boundaries=rejected)))


if __name__=='__main__':main()
