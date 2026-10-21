#pragma once
#include "../../platform/generated/configuration_defaults.h"
#include "log_parse.h"
#include "platform.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace ecmgui {
class WorkerProc;
class TomlFile;
const char *state_name(WorkerRunState state);
const char *state_key(WorkerRunState state);
const WorkerRunState *all_worker_states(int *count);
// Borrowed read-only view; Controller owns records, logs and process lifetime.
struct WorkerView {
    int index = 1;                    // [Worker #N] number (also --worker N)
    std::string name;                 // [Worker #N] name= (GUI-only key)
    int device = 0;                   // effective device= after section override
    int stage1_device = 0;
    std::string curve_model = ecm_config::defaults::stage1_curve;
    std::string execution_target = ecm_config::defaults::stage1_target;
    std::string stage1_target = ecm_config::defaults::stage1_target;
    int gpu_param = ecm_config::defaults::stage1_gpu_param;
    int gpucurves = 0;
    std::string worktodo;
    std::string log_file;
    std::string extra_args;           // [Worker #N] extra_args= (space separated)
    std::string exe;
    std::string working_dir;
    bool autostart = false;

    // Live state (M2).
    ProgressInfo progress;
    long long progress_lines = 0;
    long long tasks_done = 0;
    long long hits = 0;
    std::string task;                 // current worktodo line (from START:)
    bool stage2=false;
    Stage2Status stage2_status;
    std::string application_version;  // observed banner of this worker process
    std::vector<std::string> events;  // event layer (timestamped, ANSI-free)
    std::vector<std::string> raw;     // everything, for the "raw output" toggle
    // Progress history for the sparkline (one sample per accepted progress line).
    // NOTE: the speed series is SECONDS PER CURVE, not curves/second: the driver prints
    // s/curve, and "bigger is worse" is the axis the operator compares against the ETA
    // (user request 2026-09-29: "曲线/秒 改为 秒/曲线").
    std::vector<float> hist_s_per_curve;
    unsigned long long last_sample_ms = 0;
    bool duplicate_device = false;    // another worker uses the same device (warning)
    bool old_driver = false;          // worker exe printed "No input number on stdin"
    // ---- Prime95 handoff notices (docs/usage/GUI.md 18) ---------------------------
    // The driver prints one `p95_add:` line per finished task; the notice strip shows the
    // most severe one. A Pending notice (and a non-empty pending file on disk) stays red
    // until a later delivery succeeds.
    P95Notice p95;                    // last notice of this worker
    // Percent the driver reported when it resumed from a checkpoint ("Resuming from
    // checkpoint: 23.8% complete"). A resumed run starts its progress line only after a
    // long batch gap (see docs/usage/GUI.md 7.2), so the GUI shows this instead of an
    // empty bar in the meantime. -1 = no resume seen.
    double resume_pct = -1.0;
    // ---- graceful stop (per worker) ------------------------------------------------
    // A stop request does NOT kill the process right away: the driver checkpoints at its own
    // interval, so a new checkpoint is only a notice; termination occurs only at the stop deadline.
    // The normal path waits for worker exit. See docs/usage/GUI.md.
    bool stop_requested = false;
    std::string effective_config;     // "effective config" text for the detail panel
    WorkerView();
    ~WorkerView();
    WorkerView(WorkerView &&) noexcept;
    WorkerView &operator=(WorkerView &&) noexcept;
    bool has_process() const;
    bool running() const;
    WorkerRunState state() const;
    const std::string &state_text() const;
    std::string command_line() const;
    int restarts() const;
private:
    friend class WorkerController;
    std::unique_ptr<WorkerProc> proc;
    unsigned long version_pid = 0;
    double last_traced_pct = -1.0;
    unsigned long long stop_deadline_ms = 0;
    long long ckpt_baseline = 0;      // newest checkpoint mtime when the stop was requested
    bool ckpt_seen = false;           // a newer checkpoint appeared -> safe to terminate
};
struct WorkerSettings {
    std::string ini_path,worker_exe,stage2_exe,priority,exit_confirm;
    int num_workers=ecm_config::defaults::gui_NumWorkers;
    unsigned long long graceful_stop_ms=static_cast<unsigned long long>(ecm_config::defaults::gui_graceful_stop_ms);
};
struct WorkerNotice {
    enum class Kind {Hit,P95,OldDriver,Error,StageChanged};
    Kind kind;int worker;std::string text;unsigned long long at_ms;
};
class WorkerController {
public:
    enum class ExitPhase {Idle,Confirm,Stopping};
    explicit WorkerController(const TomlFile &config):ini_(config){}
    const std::vector<WorkerView> &workers() const{return workers_;}
    void set_sinks(std::function<void(const std::string &)> trace,std::function<void(const WorkerNotice &)> notice);
    // Reject active/restarting workers before changing any settings or ownership.
    void configure(const WorkerSettings &settings);
    bool busy() const;
    bool start(int index,bool detailed_trace=false);
    void start_all();void stop_all();void stop_all_quietly();void tick();
    void request_stop(int worker_index);
    bool request_close();void begin_graceful_exit();void cancel_close();void force_close();
    bool quit_requested() const{return quit_requested_;}
    ExitPhase exit_phase() const{return exit_phase_;}
    unsigned long long exit_deadline() const{return exit_deadline_ms_;}
    void suppress_autostart(){autostart_done_=true;}
    std::string resolve_stage1_exe() const;std::string resolve_stage2_exe() const;
    bool stage2_mode() const;bool worker_stage2(int worker_index) const;
    std::string resolve_worker_exe(int worker_index=0) const;
    std::string worker_dir(int worker_index=0) const;
private:
    WorkerView make_worker(int index) const;
    void refresh_worker_for_queue(WorkerView &worker);void refresh_device_warnings();
    void tick_graceful_stops(unsigned long long now);
    std::string effective_config_text(int worker_index) const;
    void trace(const std::string &line) const {if(trace_)trace_(line);}
    void notify(const WorkerNotice &notice) const {if(notice_)notice_(notice);}
    const TomlFile &ini_;
    std::vector<WorkerView> workers_;
    std::string ini_path_,worker_exe_,stage2_exe_,priority_,exit_confirm_;
    mutable std::string resolved_stage2_exe_;
    int num_workers_=ecm_config::defaults::gui_NumWorkers;
    unsigned long long graceful_stop_ms_=static_cast<unsigned long long>(ecm_config::defaults::gui_graceful_stop_ms);
    unsigned long long exit_deadline_ms_=0,last_tick_ms_=0;
    bool autostart_done_=false,quit_requested_=false;
    ExitPhase exit_phase_=ExitPhase::Idle;
    std::function<void(const std::string &)> trace_;
    std::function<void(const WorkerNotice &)> notice_;
};
}
