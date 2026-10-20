"""Guard the exact scope of NVCC file-identity normalization, CPU only."""
import argparse
import json
from pathlib import Path

from compare_cuda_sass import anonymous_name, fingerprints


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    old='_Z17edwards_naf_sliceIN63_GLOBAL__N__93314265_30_cgbn_edwards_instances_tpi4_cu_8db817da6ParamsILj128ELj4EEEEvP19cgbn_error_report_t'
    new=old.replace('93314265','bb567dfc').replace('8db817da','d35d0153')
    assert anonymous_name(old)==anonymous_name(new)
    for variant in (new.replace('ILj128E','ILj192E'),new.replace('Lj4E','Lj8E'),
                    new.replace('edwards_naf_slice','edwards_naf_other'),new.replace('tpi4_cu','tpix_cu')):
        assert anonymous_name(old)!=anonymous_name(variant)
    instruction='        /*0000*/ IADD3 R1, R2, R3, RZ; /* 0x00000000 */\n'
    def export(name,symbol,body):
        path=out/name;path.write_text('        Function : '+symbol+'\n'+body);return path
    before=fingerprints(export('old.sass',old,instruction))
    after=fingerprints(export('new.sass',new,instruction))
    assert list(before)==list(after)
    key=next(iter(before))
    assert before[key]['sha256']==after[key]['sha256']
    changed=fingerprints(export('changed.sass',new,instruction.replace('R2','R4')))
    assert before[key]['sha256']!=changed[key]['sha256']
    duplicate=export('collision.sass',old,instruction)
    with duplicate.open('a') as f:f.write('        Function : '+new+'\n'+instruction)
    try:fingerprints(duplicate)
    except AssertionError as e:assert 'collision' in str(e)
    else:raise AssertionError('ambiguous function identities merged')
    try:anonymous_name(new.replace('IN63_GLOBAL','IN64_GLOBAL'))
    except AssertionError as e:assert 'namespace length' in str(e)
    else:raise AssertionError('invalid namespace ABI length accepted')
    result=dict(complete=True,checks=10,instructions_preserved=True,template_geometry_preserved=True,
                filenames_preserved=True,ambiguity_rejected=True,gpu_runs=0)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
