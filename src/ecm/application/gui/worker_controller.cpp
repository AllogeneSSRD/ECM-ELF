#include "worker_controller.h"
#include "worker_proc.h"
#include "toml_file.h"
#include "../configuration/generated/ecm_config_generated.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
namespace ecmgui {
bool is_stage2_exe(std::string name) {
    name=file_name(name);for(char &c:name)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return name.find("stage2")!=name.npos;
}
std::string resolve_gui_exe(const std::string &value) {
    const auto local=path_resolve(exe_dir(),value);
    return file_exists(local) || value.find_first_of("\\/")!=value.npos?local:value;
}
std::string adjacent_stage2_exe() {
    const auto direct=path_join(exe_dir(),"ecm_cuda_stage2.exe");if(file_exists(direct))return direct;
    std::vector<std::string> candidates;std::error_code error;
    for(const auto &item:std::filesystem::directory_iterator(std::filesystem::u8path(exe_dir()),error)) {
        if(error)break;
        const auto name=item.path().filename().u8string();
        if(name.compare(0,11,"cuda-stage2"))continue;
        const auto exe=(item.path()/"ecm_cuda_stage2.exe").u8string();
        if(file_exists(exe))candidates.push_back(exe);
    }
    return candidates.size()==1?candidates.front():"ecm_cuda_stage2.exe";
}
std::string queue_task(const std::string &path,int worker,int priority=0,int skip=0) {
    std::ifstream in(std::filesystem::u8path(path));std::vector<std::pair<int,std::string>> tasks;
    std::string line;int section=1;
    while(std::getline(in,line)) {
        const auto a=line.find_first_not_of(" \t\r");if(a==line.npos)continue;line.erase(0,a);
        if(line.size()>=3 && static_cast<unsigned char>(line[0])==0xef &&
           static_cast<unsigned char>(line[1])==0xbb &&
           static_cast<unsigned char>(line[2])==0xbf)line.erase(0,3);
        if(line.empty())continue;
        auto lower=line;for(char &c:lower)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if(line.front()=='[') {
            const auto hash=lower.find('#'),end=lower.find(']');section=-1;
            if(lower.find("worker")!=lower.npos && hash!=lower.npos && end!=lower.npos) {
                const auto digits=lower.substr(hash+1,end-hash-1);section=std::atoi(digits.c_str());
            }
        } else if(line.front()!='#' && line.front()!=';')
            tasks.emplace_back(section,line);
    }
    std::string first,preferred;
    for(const auto &task:tasks)if(task.first==worker) {
        const int stage=task.second.compare(0,10,"ECMSTAGE2=")==0?2:1;
        if(stage==skip)continue;
        if(first.empty())first=task.second;
        if(stage==priority && preferred.empty())preferred=task.second;
    }
    return preferred.empty()?first:preferred;
}
bool parse_resume_pct(const std::string &line, double &pct) {
    const char *kResume = "Resuming from checkpoint:";
    const char *kLoaded = "Checkpoint loaded:";
    const std::size_t r = line.find(kResume);
    if (r != std::string::npos) {
        try {
            pct = std::stod(line.substr(r + std::strlen(kResume)));
            return pct >= 0.0 && pct <= 100.0;
        } catch (...) {
            return false;
        }
    }
    const std::size_t l = line.find(kLoaded);
    if (l != std::string::npos) {
        // "(23.8%)" -- take the number before the percent sign after the ratio.
        const std::size_t pctPos = line.find('%', l);
        if (pctPos == std::string::npos || pctPos == 0) return false;
        const std::size_t open = line.rfind('(', pctPos);
        if (open == std::string::npos) return false;
        try {
            pct = std::stod(line.substr(open + 1));
            return pct >= 0.0 && pct <= 100.0;
        } catch (...) {
            return false;
        }
    }
    return false;
}
const char *state_name(WorkerRunState s) {
    switch (s) {
        case WorkerRunState::Stopped: return "Stopped";
        case WorkerRunState::Starting: return "Starting";
        case WorkerRunState::Running: return "Running";
        case WorkerRunState::Restarting: return "Restarting";
        case WorkerRunState::Error: return "Error";
        case WorkerRunState::QueueEmpty: return "QueueEmpty";
    }
    return "?";
}
const char *state_key(WorkerRunState s) {
    switch (s) {
        case WorkerRunState::Stopped: return "state_stopped";
        case WorkerRunState::Starting: return "state_starting";
        case WorkerRunState::Running: return "state_running";
        case WorkerRunState::Restarting: return "state_restarting";
        case WorkerRunState::Error: return "state_error";
        case WorkerRunState::QueueEmpty: return "state_queue_empty";
    }
    return "state_stopped";
}
const WorkerRunState *all_worker_states(int *count) {
    static const WorkerRunState kStates[] = {
        WorkerRunState::Stopped,  WorkerRunState::Starting, WorkerRunState::Running,
        WorkerRunState::Restarting, WorkerRunState::Error,  WorkerRunState::QueueEmpty,
    };
    if (count != nullptr) *count = static_cast<int>(sizeof(kStates) / sizeof(kStates[0]));
    return kStates;
}
WorkerView::WorkerView()=default;
WorkerView::~WorkerView()=default;
WorkerView::WorkerView(WorkerView &&)noexcept=default;
WorkerView &WorkerView::operator=(WorkerView &&)noexcept=default;
bool WorkerView::has_process()const{return bool(proc);}
bool WorkerView::running()const{return proc && proc->running();}
WorkerRunState WorkerView::state()const{return proc?proc->state():WorkerRunState::Stopped;}
const std::string &WorkerView::state_text()const{static const std::string empty;return proc?proc->state_text():empty;}
std::string WorkerView::command_line()const{return proc?proc->command_line():"-";}
int WorkerView::restarts()const{return proc?proc->restarts():0;}
void WorkerController::set_sinks(std::function<void(const std::string &)> trace,std::function<void(const WorkerNotice &)> notice) {
    trace_=std::move(trace);notice_=std::move(notice);
}
bool WorkerController::busy()const {
    for(const auto &w:workers_)if(w.proc && (w.proc->running() || w.proc->state()==WorkerRunState::Restarting))return true;
    return false;
}
void WorkerController::configure(const WorkerSettings &settings) {
    if(busy())throw std::runtime_error("cannot reconfigure running or restarting workers");
    ini_path_=settings.ini_path;worker_exe_=settings.worker_exe;stage2_exe_=settings.stage2_exe;
    priority_=settings.priority;exit_confirm_=settings.exit_confirm;num_workers_=settings.num_workers;
    graceful_stop_ms_=settings.graceful_stop_ms;
    workers_.clear();resolved_stage2_exe_.clear();
    for(int i=1;i<=num_workers_;++i)workers_.push_back(make_worker(i));
    refresh_device_warnings();
}
void WorkerController::cancel_close(){exit_phase_=ExitPhase::Idle;}
void WorkerController::force_close(){
    stop_all_quietly();for(auto &w:workers_)w.stop_requested=false;
    exit_phase_=ExitPhase::Idle;quit_requested_=true;
}
bool WorkerController::start(int index,bool detailed_trace) {
    for(auto &w:workers_)if(w.index==index) {
        std::string err;refresh_worker_for_queue(w);
        if(w.proc && w.proc->start(err)) {
            trace("worker "+std::to_string(w.index)+": started pid="+std::to_string(w.proc->pid())+
                  (detailed_trace?" cmd="+w.proc->command_line():""));return true;
        }
        if(detailed_trace)trace("worker "+std::to_string(w.index)+": start failed: "+err);
        notify({WorkerNotice::Kind::Error,w.index,"worker "+std::to_string(w.index)+": "+err,0});return false;
    }
    return false;
}
std::string WorkerController::resolve_stage1_exe() const {
    if (!worker_exe_.empty() && !is_stage2_exe(worker_exe_)) {
        // Relative paths resolve against the GUI's own directory.
        if (worker_exe_.find('\\') == std::string::npos &&
            worker_exe_.find('/') == std::string::npos) {
            const std::string local = path_join(exe_dir(), worker_exe_);
            if (file_exists(local)) return local;
        }
        return resolve_gui_exe(worker_exe_);
    }
    for (const char *name : {"ecm_cuda.exe", "ecm.exe"}) {
        const std::string local = path_join(exe_dir(), name);
        if (file_exists(local)) return local;
    }
    return "ecm_cuda.exe";                 // last resort: PATH lookup
}
bool WorkerController::stage2_mode() const {
    return worker_stage2(1);
}
std::string WorkerController::resolve_stage2_exe() const {
    if(resolved_stage2_exe_.empty())resolved_stage2_exe_=!stage2_exe_.empty()?resolve_gui_exe(stage2_exe_):
        is_stage2_exe(worker_exe_)?resolve_gui_exe(worker_exe_):adjacent_stage2_exe();
    return resolved_stage2_exe_;
}
bool WorkerController::worker_stage2(int worker_index) const {
    const auto cfg=ecm_config::read_stage1(ecm_config::cli_entries(ini_.lines(),worker_index,true));
    const auto path=path_resolve(path_dir(ini_path_),cfg.worktodo);
    const auto task=queue_task(path,worker_index,cfg.worktodo_priority,cfg.worktodo_skip);
    return task.compare(0,10,"ECMSTAGE2=")==0;
}
std::string WorkerController::resolve_worker_exe(int worker_index) const {
    return (worker_index>0?worker_stage2(worker_index):stage2_mode())?resolve_stage2_exe():resolve_stage1_exe();
}
WorkerView WorkerController::make_worker(int index) const {
    const std::string section="workers."+std::to_string(index);
    auto gui_entries=ecm_config::section_entries(ini_.lines(),section);
    const auto worker=ecm_config::read_worker(gui_entries);
    WorkerView w;
    w.index=index;w.stage2=worker_stage2(index);
    w.exe=resolve_worker_exe(index);
    // Stage1 resolves paths beside its executable; Stage2 resolves TOML paths
    // beside the shared TOML. An adjacent pair therefore shares one work directory.
    w.working_dir=w.stage2?path_dir(ini_path_):path_dir(w.exe);
    if(w.working_dir.empty())w.working_dir=exe_dir();
    w.name=worker.name;
    if(!ini_.has(section,"name")) {
        const auto token=w.name.find("{N}");
        if(token!=w.name.npos)w.name.replace(token,3,std::to_string(index));
    }
    const auto entries=ecm_config::cli_entries(ini_.lines(),index,true);
    const auto stage1=ecm_config::read_stage1(entries);
    w.device=stage1.device;w.stage1_device=stage1.device;
    w.curve_model=stage1.curve_model;w.execution_target=stage1.execution_target;
    w.stage1_target=stage1.execution_target;w.gpu_param=stage1.gpu_param;
    w.gpucurves=worker.gpucurves;
    w.worktodo=path_resolve(path_dir(ini_path_),stage1.worktodo);
    const auto log=index>1 && !stage1.log_file_explicit && stage1.log_file==ecm_config::defaults::stage1_log_file?
        ecm_config::worker_file(stage1.log_file,index):stage1.log_file;
    w.log_file=path_resolve(w.working_dir,log);
    if(w.stage2) {
        const auto stage2=ecm_config::read_stage2(entries);
        w.execution_target="gpu";
        if(stage2.device>=0)w.device=stage2.device;
        w.log_file=stage2.has_log?path_resolve(path_dir(ini_path_),stage2.log):
            path_resolve(path_dir(ini_path_),ecm_config::worker_file(ecm_config::defaults::stage2_stage2_log_file,index));
    }
    w.task=queue_task(w.worktodo,index,stage1.worktodo_priority,stage1.worktodo_skip);
    w.extra_args=worker.extra_args;w.autostart=worker.autostart!=0;
    w.effective_config=effective_config_text(index);
    WorkerSpawn spawn;
    spawn.exe=w.exe;spawn.ini=ini_path_;spawn.working_dir=w.working_dir;
    spawn.worker_index=index;spawn.priority=priority_;spawn.stage2=w.stage2;
    if(w.stage2)spawn.extra_args={"--gui-events","--log-level","batches"};
    std::string arg;bool quoted=false;
    for(char c:w.extra_args) {
        if(c=='"'){quoted=!quoted;continue;}
        if(!quoted && (c==' ' || c=='\t')) {
            if(!arg.empty()){spawn.extra_args.push_back(arg);arg.clear();}
        } else arg.push_back(c);
    }
    if(!arg.empty())spawn.extra_args.push_back(arg);
    w.proc.reset(new WorkerProc());w.proc->configure(spawn,WorkerLimits{});
    w.proc->set_autostart(w.autostart);
    return w;
}
void WorkerController::refresh_worker_for_queue(WorkerView &w) {
    if (!w.proc || w.proc->running() || w.proc->state()==WorkerRunState::Restarting ||
        worker_stage2(w.index)==w.stage2) return;
    auto replacement=make_worker(w.index);
    replacement.events=std::move(w.events);
    replacement.raw=std::move(w.raw);
    replacement.hits=w.hits;
    replacement.tasks_done=w.tasks_done;
    replacement.progress_lines=w.progress_lines;
    replacement.p95=w.p95;
    w.proc->stop();
    trace("worker "+std::to_string(w.index)+": dispatch "+
          (replacement.stage2?"ECMSTAGE2":"ECMSTAGE1"));
    w=std::move(replacement);
    notify({WorkerNotice::Kind::StageChanged,w.index,{},0});
}
void WorkerController::refresh_device_warnings() {
    std::map<int,int> seen;
    for(const auto &w:workers_)if(w.execution_target=="gpu")++seen[w.device];
    for(auto &w:workers_)w.duplicate_device=w.execution_target=="gpu" && seen[w.device]>1;
}
void WorkerController::start_all() {
    for(const auto &w:workers_)if(w.proc && !w.proc->running())start(w.index,true);
}
void WorkerController::stop_all() {
    // The explicit "Stop all" button keeps its meaning (stop everything now) but still asks
    // for a checkpoint first: dropping up to one checkpoint interval of GPU work for a
    // button labelled "Stop" surprised the user (2026-09-28).
    for (WorkerView &w : workers_) {
        if (w.proc == nullptr) continue;
        if (w.proc->running() || w.proc->state()==WorkerRunState::Restarting) request_stop(w.index);
    }
}
void WorkerController::stop_all_quietly() {
    // Exit path (docs/usage/GUI.md 5.6). Requirements from the user: closing the GUI
    // must terminate the workers silently -- no restart, no "Error" state, no crash
    // accounting, nothing in the status line, and no waiting around.
    int running = 0;
    for (WorkerView &w : workers_) {
        if (w.proc == nullptr) continue;
        if (w.proc->running()) ++running;
        // WorkerProc::stop() is an immediate Job-object termination (TerminateJobObject),
        // so it kills the worker and any grandchildren right here and returns without
        // waiting for the driver to do anything. It also sets the state to Stopped and
        // clears the state text, so nothing downstream can turn this into a crash.
        w.proc->stop();
    }
    if (running > 0) {
        // One line only: this is what a test asserts on (nothing else may appear between
        // it and the process exit -- no Restarting/Error/restart lines).
        trace("shutdown: terminated " + std::to_string(running) + " running worker(s)");
    }
}
void WorkerController::tick() {
    const unsigned long long now = mono_ms();
    last_tick_ms_ = now;
    tick_graceful_stops(now);
    for (WorkerView &w : workers_) {
        if (w.proc == nullptr) continue;
        // Read the state BEFORE the autostart attempt, so the transition to Running
        // is observed below and reaches the trace/UI like any other transition.
        const WorkerRunState before = w.proc->state();
        const int restarts_before = w.proc->restarts();
        if (w.proc->autostart() && !w.proc->running() && before == WorkerRunState::Stopped &&
            !w.proc->breaker_tripped() && !autostart_done_) {
            std::string err;
            if (w.proc->start(err)) {
                trace("worker " + std::to_string(w.index) + ": autostart pid=" +
                      std::to_string(w.proc->pid()) + " cmd=" + w.proc->command_line());
            } else {
                trace("worker " + std::to_string(w.index) + ": autostart failed: " + err);
            }
        }
        w.proc->poll(now);
        if(w.proc->running() && w.version_pid!=w.proc->pid()) {
            w.version_pid=w.proc->pid();
            w.application_version.clear();
        }
        WorkerProc::DrainedOutput d;
        w.proc->drain(d);
        if(!d.application_version.empty())w.application_version=d.application_version;
        w.progress_lines += d.progress_lines;
        if (d.last_progress.valid) {
            w.progress = d.last_progress;
            // Sample for the chart. The driver emits progress lines at a decaying rate, so
            // every line is worth keeping when it carries new information. The series holds
            // SECONDS PER CURVE (see WorkerView::hist_s_per_curve).
            if (w.progress.s_per_curve > 0.0 && (!w.stage2 || d.speed_sample)) {
                w.hist_s_per_curve.push_back(static_cast<float>(w.progress.s_per_curve));
            }
            const std::size_t cap = 600;
            if (w.hist_s_per_curve.size() > cap) {
                w.hist_s_per_curve.erase(
                    w.hist_s_per_curve.begin(),
                    w.hist_s_per_curve.begin() +
                        static_cast<std::ptrdiff_t>(w.hist_s_per_curve.size() - cap));
            }
            // Trace one sample per 1% step at most: a script must be able to check the
            // parsed numbers without the log growing by thousands of lines.
            if (w.progress.pct - w.last_traced_pct >= 1.0 || w.last_traced_pct < 0.0) {
                w.last_traced_pct = w.progress.pct;
                char buf[220];
                std::snprintf(buf, sizeof(buf),
                              "progress pct=%.1f s_per_curve=%.3f eta=%.1f done=%llu total=%llu bits=%llu gpu=%d",
                              w.progress.pct, w.progress.s_per_curve, w.progress.eta_s,
                              w.progress.curves_done, w.progress.curves_total, w.progress.bits,
                              w.progress.gpu ? 1 : 0);
                trace("worker " + std::to_string(w.index) + ": " + buf);
            }
        }
        if (!d.last_start_line.empty()) {
            w.task = d.last_start_line;
            trace("worker " + std::to_string(w.index) + ": task " + w.task);
        }
        if(d.stage2.valid) {
            w.stage2_status=d.stage2;
            if(d.last_start_line.empty() && (d.stage2_task_started || w.task.empty())) {
                const auto cfg=ecm_config::read_stage1(ecm_config::cli_entries(ini_.lines(),w.index,true));
                w.task=queue_task(w.worktodo,w.index,cfg.worktodo_priority,cfg.worktodo_skip);
            }
        }
        // "Resuming from checkpoint: 23.8% complete" / "Checkpoint loaded: … (23.8%)" tells
        // the GUI where the task already stands. That matters because the driver's redirected
        // progress lines are gated on the (restored) batch counter, so the first one after a
        // resume can be a long way off -- without this the bar sat empty and the panel said
        // "waiting for the first progress line" for the whole session (measured 2026-09-29).
        if (w.resume_pct < 0.0) {
            for (const std::string &ev : d.events) {
                double pct = 0.0;
                if (parse_resume_pct(ev, pct)) {
                    w.resume_pct = pct;
                    w.progress.pct = pct;          // the bar starts where the work is
                    w.progress.valid = false;      // but no speed/ETA numbers yet
                    char buf[128];
                    std::snprintf(buf, sizeof(buf), "progress seed=%.1f (from checkpoint)", pct);
                    trace("worker " + std::to_string(w.index) + ": " + buf);
                    break;
                }
            }
        }
        w.hits += static_cast<long long>(d.hits.size());
        for (const HitInfo &h : d.hits) {
            notify({WorkerNotice::Kind::Hit,w.index,h.factor,now});   // highlight the row for 5 s
        }
        w.events.insert(w.events.end(), d.events.begin(), d.events.end());
        w.raw.insert(w.raw.end(), d.raw.begin(), d.raw.end());
        // Prime95 handoff notices (docs/usage/GUI.md 18): keep the newest one per worker
        // plus a trace line, so a script can prove what the strip shows.
        for (const P95Notice &notice : d.p95) {
            w.p95 = notice;
            notify({WorkerNotice::Kind::P95,w.index,{},now});
            const char *lvl = notice.level == P95Notice::Level::Ok ? "ok"
                            : notice.level == P95Notice::Level::Warn ? "warn"
                            : notice.level == P95Notice::Level::Pending ? "pending" : "ready";
            trace("worker " + std::to_string(w.index) + ": p95 " + lvl + " worker=" +
                  std::to_string(notice.worker) + " added=" + std::to_string(notice.added) +
                  " pending=" + std::to_string(notice.pending_at_start + notice.lines) +
                  (notice.note.empty() ? "" : " note=" + notice.note) +
                  (notice.error.empty() ? "" : " error=" + notice.error));
        }
        // "No input number on stdin" means the worker exe never entered queue mode: it is
        // an ecm_cuda built before the D1/D2 changes, which treats `--worker N` as task
        // arguments. Say so once (status line + trace) instead of letting the user watch
        // the worker restart until the breaker trips.
        if (d.old_driver && !w.old_driver) {
            w.old_driver = true;
            notify({WorkerNotice::Kind::OldDriver,w.index,kOldDriverHint,now});
            trace("worker " + std::to_string(w.index) + ": DIAGNOSIS: " + std::string(kOldDriverHint));
        }
        // Keep the panes bounded (the supervisor already caps the pipe buffer).
        if (w.events.size() > 4000) {
            w.events.erase(w.events.begin(),
                           w.events.begin() + static_cast<std::ptrdiff_t>(w.events.size() - 4000));
        }
        if (w.raw.size() > 4000) {
            w.raw.erase(w.raw.begin(),
                        w.raw.begin() + static_cast<std::ptrdiff_t>(w.raw.size() - 4000));
        }
        if (w.proc->state() != before) {
            trace("worker " + std::to_string(w.index) + ": state " +
                  state_name(w.proc->state()) +
                  (w.proc->state_text().empty() ? "" : " (" + w.proc->state_text() + ")"));
        }
        if (w.proc->restarts() != restarts_before) {
            trace("worker " + std::to_string(w.index) + ": restart #" +
                  std::to_string(w.proc->restarts()));
        }
        if(before!=WorkerRunState::QueueEmpty && w.proc->state()==WorkerRunState::QueueEmpty &&
           !w.stop_requested && exit_phase_==ExitPhase::Idle) {
            const auto cfg=ecm_config::read_stage1(ecm_config::cli_entries(ini_.lines(),w.index,true));
            const auto next=queue_task(w.worktodo,w.index,cfg.worktodo_priority,cfg.worktodo_skip);
            if(!next.empty() && (next.compare(0,10,"ECMSTAGE2=")==0)!=w.stage2) {
                refresh_worker_for_queue(w);
                std::string err;
                if(w.proc->start(err))trace("worker "+std::to_string(w.index)+": dispatched pid="+
                                            std::to_string(w.proc->pid())+" cmd="+w.proc->command_line());
                else notify({WorkerNotice::Kind::Error,w.index,"worker "+std::to_string(w.index)+": "+err,now});
            }
        }
    }
    autostart_done_ = true;
}
std::string WorkerController::effective_config_text(int worker_index) const {
    const auto section="workers."+std::to_string(worker_index);
    std::ostringstream out;
    out<<"next task = "<<(worker_stage2(worker_index)?"ECMSTAGE2":"ECMSTAGE1")<<"\n"
       <<"exe = "<<resolve_worker_exe(worker_index)<<"\n";
    std::set<std::string> keys;
    for(const auto &binding:ecm_config::stage1_bindings)keys.insert(binding.key);
    for(const auto &binding:ecm_config::stage2_bindings)keys.insert(binding.key);
    const auto entries=ecm_config::cli_entries(ini_.lines(),worker_index,true);
    for(const auto &entry:entries)if(keys.count(entry.first)) {
        bool overridden=false;
        for(const auto &line:ini_.lines())if(line.kind==TomlFile::Line::Kind::KeyValue &&
            line.worker_scope==worker_index && ecm_config::lower(line.key)==entry.first)overridden=true;
        out<<entry.first<<" = "<<entry.second<<(overridden?"   ["+section+"]":"   [global]")<<"\n";
    }
    return out.str();
}
std::string WorkerController::worker_dir(int worker_index) const {
    if(worker_index>0)for(const auto &w:workers_)if(w.index==worker_index)return w.working_dir;
    const auto directory=path_dir(resolve_stage1_exe());
    return directory.empty()?exe_dir():directory;
}
bool WorkerController::request_close() {
    // Called from WM_CLOSE and from File -> Quit (main_win32.cpp). Returns true when the
    // window may close NOW.
    int running = 0;
    for (const WorkerView &w : workers_) {
        if (w.proc != nullptr && w.proc->running()) ++running;
    }
    if (running == 0) {
        quit_requested_ = true;
        return true;
    }
    if (exit_confirm_ == "kill") {
        // Explicit opt-out ([gui] exit_confirm = kill): terminate immediately.
        trace("exit: " + std::to_string(running) + " worker(s) running, exit_confirm=kill");
        stop_all_quietly();
        quit_requested_ = true;
        return true;
    }
    if (exit_confirm_ == "stop") {
        // Scripted/unattended: no modal, but still checkpoint before terminating.
        trace("exit: " + std::to_string(running) + " worker(s) running, exit_confirm=stop");
        begin_graceful_exit();
        return false;                       // the modal-free stopping phase is drawn below
    }
    // ask (default): open the confirmation modal; the window stays open.
    if (exit_phase_ == ExitPhase::Idle) {
        exit_phase_ = ExitPhase::Confirm;
        trace("exit: confirmation requested (" + std::to_string(running) + " worker(s) running)");
    }
    return false;
}
void WorkerController::request_stop(int worker_index) {
    for (WorkerView &w : workers_) {
        if (w.index != worker_index) continue;
        if (w.proc == nullptr) return;
        if(!w.proc->running()) {if(w.proc->state()==WorkerRunState::Restarting)w.proc->stop();return;}
        w.stop_requested = true;
        w.ckpt_seen = false;
        w.ckpt_baseline = w.stage2?0:newest_checkpoint_mtime(worker_dir(w.index), nullptr);
        w.stop_deadline_ms = mono_ms() + graceful_stop_ms_;
        w.proc->request_checkpoint_stop();
        trace("worker " + std::to_string(w.index) +
              (w.stage2?": stop requested (finishing current Stage2 curve, max ":": stop requested (waiting for a checkpoint, max ") +
              std::to_string(graceful_stop_ms_ / 1000) + " s)");
        return;
    }
}
void WorkerController::begin_graceful_exit() {
    exit_phase_ = ExitPhase::Stopping;
    exit_deadline_ms_ = mono_ms() + graceful_stop_ms_;
    for (WorkerView &w : workers_) {
        if (w.proc == nullptr || !w.proc->running()) continue;
        w.stop_requested = true;
        w.ckpt_seen = false;
        w.ckpt_baseline = w.stage2?0:newest_checkpoint_mtime(worker_dir(w.index), nullptr);
        w.stop_deadline_ms = exit_deadline_ms_;
        w.proc->request_checkpoint_stop();
        trace("worker " + std::to_string(w.index) +
              (w.stage2?": graceful stop requested (finishing current Stage2 curve, max ":": graceful stop requested (waiting for a checkpoint, max ") +
              std::to_string(graceful_stop_ms_ / 1000) + " s)");
    }
}
void WorkerController::tick_graceful_stops(unsigned long long now) {
    for (WorkerView &w : workers_) {
        if (w.proc == nullptr || !w.stop_requested) continue;
        if (!w.proc->running()) {
            // It exited on its own (finished its queue, or the driver honours a stop
            // request) -- nothing left to wait for.
            w.stop_requested = false;
            continue;
        }
        if (!w.stage2 && !w.ckpt_seen) {
            std::string name;
            const long long m = newest_checkpoint_mtime(worker_dir(w.index), &name);
            if (m != 0 && m > w.ckpt_baseline) {
                w.ckpt_seen = true;
                trace("worker " + std::to_string(w.index) + ": checkpoint file updated (" + name +
                      "); waiting for worker exit");
            }
        }
        // The child exits after writing its checkpoint. Never terminate it merely
        // because a file appeared: rotation can publish a backup before the new
        // primary is committed, and another worker may share this directory.
        if (now >= w.stop_deadline_ms) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "worker %d: stop timeout; terminating worker",
                          w.index);
            trace(buf);
            w.proc->stop();
            w.stop_requested = false;
            w.ckpt_seen = false;
        }
    }
    if (exit_phase_ != ExitPhase::Stopping) return;
    // All workers done? Then the window may close.
    bool any_running = false;
    for (const WorkerView &w : workers_) {
        if (w.proc != nullptr && (w.proc->running() || w.stop_requested)) any_running = true;
    }
    if (!any_running) {
        trace("exit: all workers stopped, closing");
        exit_phase_ = ExitPhase::Idle;
        quit_requested_ = true;
    } else if (now >= exit_deadline_ms_) {
        // Deadline reached: the remaining workers never checkpointed. Terminate them (the
        // modal tells the user this is happening) instead of hanging forever.
        trace("exit: deadline reached, terminating without a fresh checkpoint");
        stop_all_quietly();
        for (WorkerView &w : workers_) w.stop_requested = false;
        exit_phase_ = ExitPhase::Idle;
        quit_requested_ = true;
    }
}
}
