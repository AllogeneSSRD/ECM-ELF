"""Compare complete GUI lifecycle/stop logic after moving it into Controller."""
import argparse,json,re,subprocess
from pathlib import Path
from test_result_publication_extraction import body as function_body
from test_stage1_host_modules import normalize
from test_stage2_planning_extraction import masked
ROOT=Path(__file__).resolve().parents[2]
def body(text,name):return function_body(re.sub(r'\)\s*const\b',')',text),name)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    path='src/ecm/application/gui/';before=subprocess.check_output(['git','show','13e932b:'+path+'app.cpp'],cwd=ROOT).decode('utf8')
    read=lambda n:(ROOT/path/n).read_text(encoding='utf8');controller=read('worker_controller.cpp');app=read('app.cpp');header=read('worker_controller.h')
    unchanged=['resolve_stage1_exe','stage2_mode','resolve_stage2_exe','worker_stage2','resolve_worker_exe','make_worker',
        'refresh_device_warnings','stop_all','stop_all_quietly','effective_config_text','worker_dir','request_close','request_stop','begin_graceful_exit','tick_graceful_stops',
        'is_stage2_exe','resolve_gui_exe','adjacent_stage2_exe','queue_task','parse_resume_pct','state_name','state_key','all_worker_states']
    for name in unchanged:assert body(before,name)==body(controller,name),name
    for name in ['path_dir','path_is_absolute','path_resolve']:assert body(before,name)==body(read('platform_win32.cpp'),name),name
    expected=body(before,'tick')
    for old,new in [('status_=loc_.t("results","factor_found")+": "+h.factor; w.hit_flash_until = now + 5000;','notify({WorkerNotice::Kind::Hit,w.index,h.factor,now});'),
        ('p95_pending_next_scan_ms_ = 0;','notify({WorkerNotice::Kind::P95,w.index,{},now});'),
        ('status_ = kOldDriverHint;','notify({WorkerNotice::Kind::OldDriver,w.index,kOldDriverHint,now});'),
        ('else status_="worker "+std::to_string(w.index)+": "+err;','else notify({WorkerNotice::Kind::Error,w.index,"worker "+std::to_string(w.index)+": "+err,now});')]:
        expected=expected.replace(normalize(old),normalize(new))
    assert expected==body(controller,'tick'),'complete poll/drain/merge/dispatch order changed'
    expected=body(before,'refresh_worker_for_queue').replace(normalize('w=std::move(replacement);'),normalize('w=std::move(replacement);notify({WorkerNotice::Kind::StageChanged,w.index,{},0});'))
    assert expected==body(controller,'refresh_worker_for_queue'),'phase state retention changed'
    assert not re.search(r'\b(?:proc|WorkerProc)\b',masked(app))
    assert 'controller_.start(w.index);' in app and 'controller_.force_close();' in app
    assert not re.search(r'ImGui|Localization|GpuMonitor|\bloc_\b|\bstatus_\b',masked(controller))
    assert 'const std::vector<WorkerView> &workers() const' in header
    assert header.index('private:')<header.index('std::unique_ptr<WorkerProc> proc;')
    assert 'worker_ui_[n.worker].hit_flash_until=n.at_ms+5000;' in app
    assert 'WorkerNotice::Kind::StageChanged)worker_ui_[n.worker]=WorkerUi{};' in app
    assert 'if(busy())throw' in body(controller,'configure')
    assert header.count('ecm_config::defaults::gui_graceful_stop_ms')==2
    assert header.count('ecm_config::defaults::gui_NumWorkers')==2
    assert not re.search(r'#include.*(?:toml_file|ecm_config_generated|ecm_toml)',header)
    report=dict(complete=True,baseline='13e932b',unchanged_functions=len(unchanged)+3,complete_tick_order_unchanged=True,
        stop_and_exit_bodies_unchanged=True,phase_retained_state_unchanged=True,ui_process_access=0,controller_imgui_localization_gpu_monitor_dependencies=0,
        readonly_borrowed_views=True,active_reconfiguration_rejected=True,gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
