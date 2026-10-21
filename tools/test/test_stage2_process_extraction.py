"""Verify narrow process input and exact synchronous domain feedback placement."""
import argparse,json,re,subprocess
from pathlib import Path
from test_stage2_planning_extraction import balanced,definitions,masked
from test_result_publication_extraction import body
from test_stage1_host_modules import normalize
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=lambda n:subprocess.check_output(['git','show','741cb66:'+n],cwd=ROOT).decode('utf-8-sig')
    read=lambda n:(ROOT/n).read_text(encoding='utf-8-sig')
    file='src/ecm/platform/process/stage2_curve_process.cpp';before=old(file);after=read(file)
    original=body(before,'child_run');actual=body(after,'run_curve_process')
    prefix=normalize('const auto &save=o.save,&results=o.results,&log=o.log; const auto &r=o.record;const auto &ini_settings=o.console_budget; const auto b2=o.b2,d=o.d,batch=o.batch;const int device=o.device;')
    assert actual.startswith('{'+prefix);actual='{'+actual[1+len(prefix):]
    token=normalize('try {');start=original.index(token,original.index(normalize('auto line=')))
    # Balanced brace parser works on normalized strings too, including catch.
    end=balanced(original,original.index('{',start),'{','}')
    catch=original.index('catch',end);end=balanced(original,original.index('{',catch),'{','}')
    feedback=original[start:end]
    expected=original[:start]+normalize('if(o.observe_selection)o.observe_selection(text);')+original[end:]
    level=normalize('ecm_cuda_stage2_default_log_level()==stage2_log::debug?stage2_log::debug:stage2_log::batches')
    expected=expected.replace(level,'o.engine_log_level')
    assert expected==actual,'process protocol/ordering changed'
    task=read('src/ecm/service/stage2/task.cpp');projection=body(task,'child_run')
    assert feedback in projection and level in projection
    names=set(re.findall(r'\bo\.(\w+)',before));mapped=set(re.findall(r'request\.(\w+)=o\.(\w+);',projection))
    assert mapped=={(n,n) for n in names} and len(names)==49
    for n in definitions(old('src/ecm/service/stage2/task.cpp')):assert body(old('src/ecm/service/stage2/task.cpp'),n)==body(task,n),n
    for n in definitions(before):
        if n!='child_run':assert body(before,n)==body(after,n),n
    planner='src/ecm/service/planning/stage2.cpp'
    assert normalize(old(planner)[old(planner).index('namespace s2prod {'):])==normalize(read(planner)[read(planner).index('namespace s2prod {'):])
    public=read('include/ecm/platform/process/curve_process.h')
    assert not re.search(r'\b(?:Options|Settings|HANDLE|LONG|cuda|mpz_t)\b',masked(public))
    assert not re.search(r'#include.*(?:service|backend|gmp|windows|cuda)',public)
    assert not re.search(r'#include.*(?:service|backend|gmp|configuration)',after)
    old_types=old('src/ecm/service/stage2/runtime_types.h');native=read('src/ecm/platform/process/stage2_curve_process.h')
    def handle(s):
        a=s.index('struct Handle {');return normalize(s[a:balanced(s,s.index('{',a),'{','}')+1])
    assert handle(old_types)==handle(native)
    report=dict(complete=True,baseline='741cb66',process_protocol_unchanged=True,synchronous_feedback_unchanged=True,
        projected_process_fields=49,complete_parent_worker_unchanged=True,planning_scope_unchanged=True,native_handle_unchanged=True,
        public_cuda_gmp_windows_types=0,gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
