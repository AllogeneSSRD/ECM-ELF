"""Check complete Stage2 flows against their pre-extraction implementation."""
import argparse,json,re,subprocess
from pathlib import Path
from test_stage2_planning_extraction import definitions,masked
from test_result_publication_extraction import body
from test_stage1_host_modules import normalize
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    old=lambda n:subprocess.check_output(['git','show','edf9060:'+n],cwd=ROOT).decode('utf-8-sig')
    read=lambda n:(ROOT/n).read_text(encoding='utf-8-sig')
    path='src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp';before=old(path);app=read(path)
    modules={n:read('src/ecm/'+n) for n in ['service/stage2/input.cpp','service/stage2/task.cpp',
        'service/planning/stage2.cpp','platform/process/stage2_curve_process.cpp','backend/cuda/stage2/host_policy.cpp']}
    all_sources=[app,*modules.values()];checked=[]
    for name in definitions(before):
        if name in ['driver','curve_worker']:continue
        owners=[v for v in all_sources if name in definitions(v)];assert len(owners)==1,(name,len(owners))
        assert body(before,name)==body(owners[0],name),name
        checked.append(name)
    task=modules['service/stage2/task.cpp']
    worker=body(task,'curve_worker')
    original=body(before,'curve_worker')
    start=original.index(normalize('std::vector<ecm_config::TomlLine> identity_lines;'))
    end=original.index(normalize('ecm_results::Submission submission;'),start)
    original=original[:start]+normalize('const auto identity=read_result_identity(identity_ini,o.worker,!o.ini.empty());')+original[end:]
    assert original==worker,'complete worker pipeline/order changed'
    # The new identity adapter performs precisely the original late read.
    identity=read('src/ecm/application/configuration/stage2_identity.cpp')
    assert 'explicit_config || fs::exists(identity_ini)' in identity
    assert 'ecm_config::cli_entries(identity_lines,worker,true)' in identity
    assert 'return {identity.result_user,identity.result_computer,identity.submission};' in identity
    old_driver=body(before,'driver')
    start=old_driver.index(normalize('const bool queue = o.save.empty();'))
    tail=old_driver[start:-1].replace(normalize('TaskDefaults{s.b2,s.arena,s.owner_mb}'),'defaults')
    tail=re.sub(r'\bs\.b2\b','defaults.b2',tail)
    expected=normalize('const auto &[cwd,ini,worktodo,finished,tmp,results,log,legacy_progress,progress_dir]=paths; const int device=o.device;const uint64_t d=o.d,batch=o.batch;')+tail
    assert body(task,'run_tasks')=='{'+expected+'}','complete parent pipeline/order changed'
    current_driver=body(app,'driver')
    call=normalize('return run_tasks(std::move(o),{cwd,ini,worktodo,finished,tmp,results,log,legacy_progress,progress_dir}, {cfg.b2_ratio,cfg.worktodo_priority,cfg.worktodo_skip,cfg.finished}, {s.b2,s.arena,s.owner_mb});')
    assert current_driver.count(call)==1
    current_driver=current_driver.replace(call,old_driver[start:-1])
    current_driver=current_driver.replace(normalize('const PackZeroDefaults pack_defaults{s.has_pack_zero,s.pack_zero,s.cost_profile}; configure_pack_zero(o,&pack_defaults);'),normalize('configure_pack_zero(o,&s);'))
    assert current_driver==old_driver,'application configuration/dispatch changed'
    previous_planner=old('src/ecm/service/planning/stage2.cpp');planner=modules['service/planning/stage2.cpp']
    for name in definitions(previous_planner):assert body(previous_planner,name)==body(planner,name),name
    assert old('src/ecm/application/cuda_stage2/ecm_stage2_console.h')==read('src/ecm/platform/ecm_stage2_console.h')
    assert not (ROOT/'src/ecm/application/cuda_stage2/ecm_stage2_console.h').exists()
    for name in ['service/stage2/task.cpp','platform/process/stage2_curve_process.cpp']:
        assert not re.search(r'ecm_toml|ecm_queue_config|ecm_config_generated|\bSettings\b',masked(modules[name]))
    assert sum(v.count('volatile LONG stop_requests = 0;') for v in all_sources)==1
    assert sum(v.count('HANDLE inherited_stop_event=nullptr;') for v in all_sources)==1
    report=dict(complete=True,baseline='edf9060',original_functions_unchanged=len(checked),existing_planner_functions_unchanged=len(definitions(previous_planner)),
                complete_parent_pipeline=True,complete_worker_pipeline=True,application_dispatch_unchanged=True,late_identity_read=True,
                console_projection_identical=True,stop_owners=1,gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
