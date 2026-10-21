"""Verify the complete planning TU, private layouts and exact Settings projection."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from test_result_publication_extraction import body
from test_stage1_host_modules import normalize

ROOT=Path(__file__).resolve().parents[2]


def masked(s):
    return re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*',lambda m:' '*len(m[0]),s,flags=re.S)


def balanced(s,at,left,right):
    m=masked(s);depth=0
    for i in range(at,len(s)):
        depth+=(m[i]==left)-(m[i]==right)
        if depth==0:return i+1
    raise AssertionError('unbalanced source')


def definitions(s):
    found=[];m=masked(s)
    for match in re.finditer(r'(?m)^([\w:<>]+(?:[ \t]+[\w:<>]+)*[ \t&*]+)(\w+)\(',m):
        end=balanced(s,m.index('(',match.start()),'(',')')
        while end<len(s) and s[end].isspace():end+=1
        if end<len(s) and s[end]=='{':found.append(match[2])
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=lambda name:subprocess.check_output(['git','show','da9953d:'+name],cwd=ROOT).decode('utf-8-sig')
    read=lambda name:(ROOT/name).read_text(encoding='utf-8-sig')
    original=old('src/ecm/application/cuda_stage2/ecm_stage2_portable_driver.inl')
    current=read('src/ecm/service/planning/stage2.cpp')
    names=definitions(original);assert names==definitions(current)
    for name in names:assert body(original,name)==body(current,name),name
    # Includes and one enclosing namespace change; every private memo, scope,
    # layout, initialization and helper remains in this single planning TU.
    extracted=current[current.index('namespace pt=ecm_stage2::tune::portable;'):current.rfind('\n}')]
    expected=original.replace('const Settings &settings','const TaskDefaults &settings')
    expected=expected.replace('const std::vector<Record> &range={}','const std::vector<Record> &range')
    assert normalize(expected)==normalize(extracted),'complete planning scope changed'
    before=old('src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp')
    main_source=read('src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp')
    main_names=definitions(before);assert main_names==definitions(main_source)
    for name in main_names:
        after=body(main_source,name)
        if name=='driver':after=after.replace(normalize('TaskDefaults{s.b2,s.arena,s.owner_mb}'),'s')
        assert body(before,name)==after,name
    types=read('src/ecm/service/stage2/runtime_types.h')
    for name in ['TaskInputError','Big','Record','Options','QueueSelection','Handle','RuntimeTuneObservation']:
        def structure(s):
            start=s.index('struct '+name+' ');return s[start:balanced(s,s.index('{',start),'{','}')+1]
        assert normalize(structure(before))==normalize(structure(types)),name
    assert set(re.findall(r'\bini_settings\.(\w+)',before))=={'arena','owner_mb'}
    assert set(re.findall(r'\bsettings\.(\w+)',original))=={'b2'}
    assert main_source.count('TaskDefaults{s.b2,s.arena,s.owner_mb}')==3
    assert 'struct TaskDefaults {uint64_t b2,arena,owner_mb;};' in types
    assert 'extern volatile LONG stop_requests;' in types and main_source.count('volatile LONG stop_requests = 0;')==1
    assert not re.search(r'ecm_toml|ecm_queue_config|application/configuration|\bSettings\b',masked(current))
    assert not (ROOT/'src/ecm/application/cuda_stage2/ecm_stage2_portable_driver.inl').exists()
    result=dict(complete=True,baseline='da9953d',planning_functions_unchanged=len(names),main_functions_unchanged=len(main_names),
                complete_planning_tu_unchanged=True,private_layouts_unchanged=7,settings_fields_passed=3,
                original_console_budget_source=True,global_stop_owners=1,gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
