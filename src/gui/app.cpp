#include "app.h"
#include "platform.h"
#include "../core/generated/ecm_config_generated.h"

#include "imgui.h"
#include "imgui_internal.h"   // ImGui::DockBuilder* for the initial layout

#include <algorithm>
#include <cctype>              /* std::tolower for the start_tab value */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

namespace ecmgui {

// Bumped whenever the default layout changes: an ini carrying an older number gets
// the new default layout instead of keeping the old arrangement.
static const int kLayoutVersion = ecm_config::layout_version;

namespace {

// Directory part of a path ("" when it has no separator). Used by the Prime95 handoff
// panel: worktodo.add lives next to the worktodo.txt the ini points at.
std::string path_dir(const std::string &p) {
    const std::size_t s = p.find_last_of("\\/");
    return (s == std::string::npos) ? std::string() : p.substr(0, s);
}

// True for "C:\dir\file", "C:/dir/file" and "\\server\share" -- i.e. a path that must NOT be
// joined onto another directory.
bool path_is_absolute(const std::string &p) {
    if (p.size() >= 2 && p[1] == ':') return true;
    if (p.size() >= 1 && (p[0] == '\\' || p[0] == '/')) return true;
    return false;
}

// An ini path value resolved against `base` (the driver's directory, like the driver does):
// an absolute value is used as it is instead of being prefixed.
std::string path_resolve(const std::string &base, const std::string &value) {
    if (value.empty()) return std::string();
    if (path_is_absolute(value)) return value;
    return path_join(base, value);
}

// The layout blob (ImGui's own settings: dock nodes, panel sizes, viewport
// positions) is stored as ONE escaped [GUI] key, because values are single-line.
std::string escape_blob(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c);
        }
    }
    return out;
}

std::string unescape_blob(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            out.push_back(s[i]);
            continue;
        }
        switch (s[++i]) {
            case 'n': out.push_back('\n'); break;
            case 't': out.push_back('\t'); break;
            case '\\': out.push_back('\\'); break;
            default: out.push_back(s[i]); break;
        }
    }
    return out;
}

std::string int_list(const int *v, int n) {
    std::string out;
    for (int i = 0; i < n; ++i) {
        if (i) out += ",";
        out += std::to_string(v[i]);
    }
    return out;
}

bool parse_int_list(const std::string &s, int *v, int n) {
    std::size_t pos = 0;
    for (int i = 0; i < n; ++i) {
        const std::size_t comma = s.find(',', pos);
        const std::string tok = s.substr(pos, (comma == std::string::npos) ? std::string::npos
                                                                          : comma - pos);
        char *end = nullptr;
        const long parsed = std::strtol(tok.c_str(), &end, 10);
        if (tok.empty() || end == nullptr || *end != '\0') return false;
        v[i] = static_cast<int>(parsed);
        if (comma == std::string::npos) return i == n - 1;
        pos = comma + 1;
    }
    return true;
}

} // namespace

// Worker state -> a) the plain name used in --trace / the docs, b) the localization id.
// Both are explicit tables on purpose: the previous version DERIVED the id from the name
// by inserting '_' before every uppercase letter, which turned "Stopped" into
// "state__stopped" (leading capital!) and every state label rendered as the raw
// fallback text "workers.state__stopped" (user report, 2026-09-28). With an explicit
// table the ids are greppable, and --selftest verifies every one of them resolves.
// Percent the driver reported when it resumed from a checkpoint ("Resuming from
// checkpoint: 23.8% complete"); defined below, declared here because tick() uses it.
bool parse_resume_pct(const std::string &line, double &pct);

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

bool App::init(const std::string &ini_path, const std::string &language, std::string &err) {
    ini_path_ = ini_path.empty() ? default_ini_path() : ini_path;
    err.clear();

    if (!ini_.load(ini_path_, err)) {
        // First run: start from an empty file. The queue manager creates the
        // template; the GUI only ever adds its own [GUI] keys.
        ini_.reset(ini_path_);
        status_ = "new ini: " + ini_path_;
    }

    const auto gui=ecm_config::read_gui(ecm_config::section_entries(ini_.lines(),"GUI"));
    loc_dir_ = gui.localization_dir;
    if (loc_dir_.empty()) {
        loc_dir_ = default_localization_dir();
        // Running from a build directory: fall back to the source tree layout.
        const std::string alt = path_join(path_join(exe_dir(), ".."), "localization");
        if (!file_exists(path_join(loc_dir_, "english.xml")) &&
            file_exists(path_join(alt, "english.xml"))) {
            loc_dir_ = alt;
        }
    }

    language_ = language.empty() ? gui.language : language;
    // NOTE: "the UI language cannot be drawn" is handled in ONE place now -- the font
    // layer (apply_ui_font in main_win32.cpp) tests the font it actually loaded and, if
    // the language is not drawable, calls fall_back_to_english() before the first frame
    // is rendered. The old guard here was hardcoded to chineseSimplified and would have
    // let any other CJK language render as "???".
    std::string loc_err;
    if (!loc_.load(loc_dir_, language_, loc_err)) {
        status_ = "localization: " + loc_err;
    } else if (!loc_.missing_keys().empty()) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%zu key(s) missing vs english, using the baseline",
                      loc_.missing_keys().size());
        status_ = buf;
    }

    refresh_hz_=gui.refresh_hz;
    gpu_poll_ms_=gui.gpu_poll_ms;
    priority_=gui.priority;
    win_x_=gui.window[0];win_y_=gui.window[1];win_w_=gui.window[2];win_h_=gui.window[3];
    layout_blob_loaded_=unescape_blob(gui.dock_layout);
    num_workers_=gui.NumWorkers;
    worker_exe_=gui.exe;
    // Font: 0 = auto (the host scales 15 px by the window DPI), or an explicit pixel
    // size; `font` names a .ttf/.ttc, empty lets the host pick (CJK font for a CJK
    // language, else a Latin system font). A FRACTIONAL size is allowed on purpose:
    // at 150 % the automatic size is 22.5 px, and rounding it changes how crisp the
    // text looks, so the user can try e.g. 23.5 or 24.
    font_size_px_=gui.font_size;
    font_path_=gui.font;
    font_snap_=gui.font_snap;
    exit_confirm_=gui.exit_confirm;
    graceful_stop_ms_=static_cast<unsigned long long>(gui.graceful_stop_ms);
    // Result files: [GUI] results_json / results_txt, defaulting next to the exe. The
    // JSONL is append-only (the durable record), results.txt is derived from it.
    results_json_ = gui.results_json;
    if (results_json_.empty()) {
        results_json_ = path_join(exe_dir(), ecm_config::defaults::gui_results_json);
    }
    results_txt_ = gui.results_txt;
    if (results_txt_.empty()) {
        results_txt_ = path_join(exe_dir(), ecm_config::defaults::gui_results_txt);
    }
    // Prime95 handoff: the GUI reads these two [queue] keys itself so the notice strip can
    // say "not configured" before any worker runs, and so the two "open" buttons know
    // where to look (docs/DEV_ECM_GUI.md 18).
    p95_worktodo_path_ = ini_.get("", "p95_worktodo_path");
    p95_add_workers_ = ini_.get("", "p95_add_workers");
    {
        std::string r_err;
        if (!results_.init(results_json_, results_txt_, r_err)) {
            status_ = "results: " + r_err;
            trace("results: init failed: " + r_err);
        } else {
            results_baseline_hits_ = results_.hit_count();
            trace("results: " + results_json_ + " (" + std::to_string(results_baseline_hits_) +
                  " hit(s) already recorded)");
        }
    }
    // GPU monitoring: strictly optional, and never blocks the UI (own thread).
    if (!gpu_.start(gpu_poll_ms_)) {
        trace("gpu: NVML unavailable: " + gpu_.reason());
    } else {
        const std::vector<GpuInfo> devs = gpu_.devices();
        trace("gpu: nvml ok (" + gpu_.nvml_source() + "), " + std::to_string(devs.size()) +
              " device(s)");
        for (const GpuInfo &d : devs) {
            trace("gpu " + std::to_string(d.index) + ": " + d.name + " (" +
                  std::to_string(d.mem_total_mb) + " MB, " + std::to_string(d.cores) +
                  " CUDA cores)");
        }
    }
    rebuild_workers();
    return true;
}

void App::record_hit(const WorkerView &w, const HitInfo &hit) {
    HitRecord r;
    r.factor = hit.factor;
    r.param = hit.param;
    r.method = hit.method;
    r.curve = hit.curve;
    r.sigma = hit.sigma;
    r.has_sigma = hit.has_sigma;
    r.save = hit.save;
    r.worker = w.index;
    r.device = w.device;
    r.task = w.task;
    r.n_expr = w.task.empty() ? std::string() : std::string();
    // Exponent and B1 come from the save name when it follows the m{n}_{b1}.save
    // contract (the same rule the driver uses); otherwise from the task's N.
    std::string b1_text;
    int exponent = 0;
    if (!hit.save.empty() && ResultsStore::parse_save_name(hit.save, &exponent, &b1_text)) {
        r.exponent = exponent;
        r.b1_text = b1_text;
        r.b1 = std::atof(b1_text.c_str());
    }
    if (r.exponent == 0 && !w.task.empty()) {
        // ECMSTAGE2=1,2,677,... -> the third field is the exponent
        const std::size_t eq = w.task.find('=');
        const std::string fields = (eq == std::string::npos) ? w.task : w.task.substr(eq + 1);
        std::vector<std::string> parts;
        std::string cur;
        for (char c : fields) {
            if (c == ',') {
                parts.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        parts.push_back(cur);
        // [aid,]k,b,n,c,...
        for (std::size_t i = 0; i + 2 < parts.size(); ++i) {
            const int n = std::atoi(parts[i + 2].c_str());
            if (n > 0) {
                exponent = n;
                break;
            }
        }
        r.exponent = exponent;
    }

    std::string err;
    if (!results_.add(r, err)) {
        status_ = "results: " + err;
        trace("results: add failed: " + err);
        return;
    }
    if (!results_.flush(err)) {
        status_ = "results: " + err;
        trace("results: flush failed: " + err);
        return;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "factor found: %s (worker %d, %d factor(s) total)",
                  r.factor.c_str(), w.index, static_cast<int>(results_.factors().size()));
    status_ = buf;
    trace(std::string("results: ") + buf + " -> " + results_.txt_path());
}

void App::trace(const std::string &line) const {
    if (trace_) {
        trace_(line);
        return;
    }
    // No sink yet (App::init runs before the host installs one): keep the early lines.
    auto *self = const_cast<App *>(this);
    if (self->pending_trace_.size() < 200) self->pending_trace_.push_back(line);
}

void App::set_trace(std::function<void(const std::string &)> fn) {
    trace_ = std::move(fn);
    if (!trace_) return;
    for (const std::string &l : pending_trace_) trace_(l);
    pending_trace_.clear();
}

std::string App::resolve_worker_exe() const {
    if (!worker_exe_.empty()) {
        // Relative paths resolve against the GUI's own directory.
        if (worker_exe_.find('\\') == std::string::npos &&
            worker_exe_.find('/') == std::string::npos) {
            const std::string local = path_join(exe_dir(), worker_exe_);
            if (file_exists(local)) return local;
        }
        return worker_exe_;
    }
    for (const char *name : {"ecm_cuda.exe", "ecm.exe"}) {
        const std::string local = path_join(exe_dir(), name);
        if (file_exists(local)) return local;
    }
    return "ecm_cuda.exe";                 // last resort: PATH lookup
}

void App::rebuild_workers() {
    workers_.clear();
    const std::string exe = resolve_worker_exe();
    WorkerLimits limits;
    for (int i = 1; i <= num_workers_; ++i) {
        const std::string sec = "Worker #" + std::to_string(i);
        WorkerView w;
        w.index = i;
        auto worker_entries=ecm_config::section_entries(ini_.lines(),sec);
        if(!ini_.has(sec,"gpucurves") && ini_.has("","gpucurves"))
            ecm_config::replace_entry(worker_entries,"gpucurves",ini_.get("","gpucurves"));
        const auto worker=ecm_config::read_worker(worker_entries);
        w.name=worker.name;
        if(!ini_.has(sec,"name")) {
            const auto token=w.name.find("{N}");
            if(token!=w.name.npos)w.name.replace(token,3,std::to_string(i));
        }
        w.device=ini_.get_int(sec,"device",ini_.get_int("","device",ecm_config::defaults::stage1_device));
        w.method=ini_.get(sec,"method",ini_.get("","method",ecm_config::defaults::stage1_method));
        w.gpucurves=worker.gpucurves;
        w.worktodo=ini_.get(sec,"worktodo",ini_.get("","worktodo",ecm_config::defaults::stage1_worktodo));
        w.log_file=ini_.get(sec,"log_file",ini_.get("","log_file",ecm_config::defaults::stage1_log_file));
        if(i>1&&!ini_.has(sec,"log_file")&&!ini_.has("","log_file"))
            w.log_file=ecm_config::worker_file(ecm_config::defaults::stage1_log_file,i);
        w.extra_args=worker.extra_args;
        w.autostart=worker.autostart!=0;
        w.effective_config = effective_config_text(i);

        WorkerSpawn spawn;
        spawn.exe = exe;
        spawn.ini = ini_path_;
        spawn.worker_index = i;
        spawn.priority = priority_;
        // Run the driver in ITS OWN directory. The driver builds the checkpoint name with
        // no path (kernels/cuda/cgbn_stage1.cu: get_checkpoint_filename -> ".ecm_ckpt_<n>_…dat"),
        // so it is created relative to the current directory: without this the checkpoints
        // of a GUI started from anywhere else land next to the GUI instead of next to the
        // driver -- which also broke the "wait for a fresh checkpoint" exit flow, because
        // the GUI watches <driver dir>. This matches how a user runs ecm_cuda by hand.
        {
            const std::size_t slash = exe.find_last_of("\\/");
            if (slash != std::string::npos) spawn.working_dir = exe.substr(0, slash);
        }
        {
            // extra_args is space separated; quotes are honoured so a path can
            // contain spaces.
            const std::string &s = w.extra_args;
            std::string cur;
            bool in_quotes = false;
            for (char c : s) {
                if (c == '"') { in_quotes = !in_quotes; continue; }
                if (!in_quotes && (c == ' ' || c == '\t')) {
                    if (!cur.empty()) { spawn.extra_args.push_back(cur); cur.clear(); }
                    continue;
                }
                cur.push_back(c);
            }
            if (!cur.empty()) spawn.extra_args.push_back(cur);
        }
        w.proc.reset(new WorkerProc());
        w.proc->configure(spawn, limits);
        w.proc->set_autostart(w.autostart);
        workers_.push_back(std::move(w));
    }
    // Duplicate-device warning (docs/DEV_ECM_GUI.md 5.7): the driver cannot see
    // other workers, so this check only exists here.
    std::map<int, int> seen;
    for (const WorkerView &w : workers_) {
        if (w.method == "gpu") seen[w.device]++;
    }
    for (WorkerView &w : workers_) {
        w.duplicate_device = (w.method == "gpu" && seen[w.device] > 1);
    }
}

void App::start_all() {
    for (WorkerView &w : workers_) {
        if (w.proc == nullptr) continue;
        if (w.proc->running()) continue;
        std::string err;
        if (w.proc->start(err)) {
            trace("worker " + std::to_string(w.index) + ": started pid=" +
                  std::to_string(w.proc->pid()) + " cmd=" + w.proc->command_line());
        } else {
            trace("worker " + std::to_string(w.index) + ": start failed: " + err);
            status_ = "worker " + std::to_string(w.index) + ": " + err;
        }
    }
}

void App::stop_all() {
    // The explicit "Stop all" button keeps its meaning (stop everything now) but still asks
    // for a checkpoint first: dropping up to one checkpoint interval of GPU work for a
    // button labelled "Stop" surprised the user (2026-09-28).
    for (WorkerView &w : workers_) {
        if (w.proc == nullptr) continue;
        if (w.proc->running()) request_stop(w.index);
    }
}

void App::stop_all_quietly() {
    // Exit path (docs/DEV_ECM_GUI.md 5.6). Requirements from the user: closing the GUI
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

void App::tick() {
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
        WorkerProc::DrainedOutput d;
        w.proc->drain(d);
        w.progress_lines += d.progress_lines;
        if (d.last_progress.valid) {
            w.progress = d.last_progress;
            // Sample for the chart. The driver emits progress lines at a decaying rate, so
            // every line is worth keeping when it carries new information. The series holds
            // SECONDS PER CURVE (see WorkerView::hist_s_per_curve).
            if (w.progress.s_per_curve > 0.0) {
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
            // One record per hit: the JSONL keeps them all, results.txt merges them.
            record_hit(w, h);
            w.hit_flash_until = now + 5000;   // highlight the row for 5 s
        }
        w.events.insert(w.events.end(), d.events.begin(), d.events.end());
        w.raw.insert(w.raw.end(), d.raw.begin(), d.raw.end());
        // Prime95 handoff notices (docs/DEV_ECM_GUI.md 18): keep the newest one per worker
        // plus a trace line, so a script can prove what the strip shows.
        for (const P95Notice &notice : d.p95) {
            w.p95 = notice;
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
            status_ = kOldDriverHint;
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
    }
    autostart_done_ = true;
}

std::string App::effective_config_text(int worker_index) const {
    const std::string sec = "Worker #" + std::to_string(worker_index);
    // Resolve "section -> global -> default" exactly like the driver does, and show
    // which layer each key came from: this is the panel that answers "I changed the
    // ini, why did nothing happen?".
    std::vector<std::string> keys;
    std::set<std::string> seen;
    const auto collect = [&](const std::string &section) {
        for (const IniFile::Line &l : ini_.lines()) {
            if (l.kind != IniFile::Line::Kind::KeyValue || l.section != section) continue;
            if (seen.insert(l.key).second) keys.push_back(l.key);
        }
    };
    collect("");
    collect(sec);

    std::ostringstream oss;
    for (const std::string &k : keys) {
        const bool in_sec = ini_.has(sec, k);
        const bool in_global = ini_.has("", k);
        const std::string v = in_sec ? ini_.get(sec, k) : ini_.get("", k);
        oss << k << " = " << v
            << (in_sec ? "   [" + sec + "]"
                       : (in_global ? "   [global]" : "   [built-in]" ))
            << "\n";
    }
    if (keys.empty()) oss << "(the ini defines no keys yet)\n";
    return oss.str();
}

void App::on_imgui_ready() {
    ImGuiIO &io = ImGui::GetIO();
    // The GUI owns the layout: ImGui must not write its own imgui.ini, the layout
    // is persisted into [GUI] dock_layout (docs/DEV_ECM_GUI.md 7.1).
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    if (!layout_blob_loaded_.empty()) {
        ImGui::LoadIniSettingsFromMemory(layout_blob_loaded_.c_str(), layout_blob_loaded_.size());
    }
}

void App::capture_layout() {
    std::size_t len = 0;
    const char *blob = ImGui::SaveIniSettingsToMemory(&len);
    if (blob == nullptr) return;
    ini_.set("GUI", "dock_layout", escape_blob(std::string(blob, len)));
    ini_.set_int("GUI", "dock_layout_ver", kLayoutVersion);
    int rect[4] = {win_x_, win_y_, win_w_, win_h_};
    ini_.set("GUI", "window", int_list(rect, 4));
    ini_.set_int("GUI", "NumWorkers", num_workers_);
    ini_.set("GUI", "language", language_);
    ini_.set_int("GUI", "refresh_hz", refresh_hz_);
    ini_.set_int("GUI", "gpu_poll_ms", gpu_poll_ms_);
    ini_.set("GUI", "priority", priority_);
}

bool App::shutdown(std::string &err) {
    // Closing the GUI stops the workers (docs/DEV_ECM_GUI.md 5.6), silently: the job
    // objects would kill them when this process exits anyway, so the GUI terminates them
    // explicitly, immediately and without any UI/state noise.
    // NOTE: this is TerminateJobObject -- an immediate kill. The driver does NOT get to
    // flush a checkpoint here (the old comment claimed otherwise); the loss is bounded by
    // its checkpoint interval ([Worker #N] ckpt_seconds, default 600 s, the production ini
    // uses 120 s). A checkpoint-on-signal protocol would be a driver change (D6, not done).
    stop_all_quietly();
    // Persist whatever the results store still owes the disk.
    std::string r_err;
    if (!results_.flush(r_err)) trace("results: final flush failed: " + r_err);
    capture_layout();
    if (!ini_.save(err)) return false;
    return true;
}

void App::reload_localization() {
    // The font was picked for the language in effect at startup, so a language change
    // must let the frame loop re-apply it: switching TO a CJK language with a Latin
    // font loaded renders every label as "???" (measured cause of the user report
    // "Chinese shows ??? in my run but renders fine in the tests").
    font_reload_requested_ = true;
    std::string loc_err;
    if (!loc_.load(loc_dir_, language_, loc_err)) {
        status_ = "localization: " + loc_err;
        return;
    }
    if (loc_.missing_keys().empty()) {
        status_ = "localization reloaded: " + language_;
    } else {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%zu key(s) missing vs english, using the baseline",
                      loc_.missing_keys().size());
        status_ = buf;
    }
}

void App::set_language(const std::string &language) {
    language_ = language.empty() ? std::string("english") : language;
    reload_localization();
}

void App::fall_back_to_english(const std::string &reason) {
    // We could not obtain a font that can draw the requested language, so showing it
    // would produce "???" boxes. English is always drawable -> switch to it, keep the
    // reason visible in the status line, and reload the (ASCII) strings.
    if (language_ == "english") return;
    language_ = "english";
    status_ = reason;
    trace("font: " + reason);
    reload_localization();
}

std::vector<std::string> App::available_languages() const {
    return Localization::available(loc_dir_);
}

void App::set_window_rect(int x, int y, int w, int h) {
    if (w <= 0 || h <= 0) return;
    win_x_ = x;
    win_y_ = y;
    win_w_ = w;
    win_h_ = h;
}

void App::draw() {
    tick();

    // ---- Prime95 notification strip reserves its own row ------------------------
    // The strip sits directly under the menu bar and the dockspace starts BELOW it. That
    // reservation is the fix for "四种颜色的真实切换，我实测看不出" (2026-09-29): the strip used
    // to be drawn one row lower than WorkPos, i.e. on top of the dockspace area, where the
    // docked panels are painted over it -- it was there, just invisible.
    const ImGuiViewport *vp = ImGui::GetMainViewport();
    // The height comes from the strip's OWN last measured height, not from a guess: at 150 %
    // DPI the content row (text + small buttons) is taller than GetFrameHeight(), and a
    // too-small reservation let the window grow downwards over the dockspace -- measured
    // 2026-09-29: strip y=156..204 while the host started at y=187, i.e. 17 px of overlap
    // again. Self-measuring makes the reservation exactly right for any DPI/padding, and the
    // first frame (before the measurement exists) is covered by the frame-height fallback.
    const float strip_h = (p95_strip_h_ > 0.0f) ? p95_strip_h_ : ImGui::GetFrameHeight();
    const ImVec2 host_pos(vp->WorkPos.x, vp->WorkPos.y + strip_h);
    const ImVec2 host_size(vp->WorkSize.x, vp->WorkSize.y - strip_h);

    // The menu bar and the strip are TWO text bands stacked at the top of the client area.
    // Traced because the pixel test (tools/test/test_gui_cjk_pixels.ps1) has to know where the
    // menu bar ends: with the strip right below it, an automatic "first text band" scan merged
    // the two rows and measured ASCII strip glyphs as if they were Chinese menu glyphs
    // (median cell width dropped from 30 px to 23 px, measured 2026-09-29).
    {
        const float menu_bar_h = vp->WorkPos.y - vp->Pos.y;
        char mbuf[160];
        std::snprintf(mbuf, sizeof(mbuf),
                      "layout: menu_bar_h=%.0f strip_top=%.0f strip_h=%.0f host_top=%.0f",
                      menu_bar_h, vp->WorkPos.y, strip_h, host_pos.y);
        if (menu_bar_trace_last_ != mbuf) {
            menu_bar_trace_last_ = mbuf;
            trace(mbuf);
        }
    }

    // [GUI] start_tab: focus the wanted panel so its dock TAB gets selected. Focusing is the
    // only mechanism that works here -- DockNodeUpdate() copies the node's NavWindow back into
    // the tab bar every frame (third_party/imgui/imgui.cpp:19889 "Apply NavWindow focus back
    // to the tab bar"), so writing node->SelectedTabId is silently overwritten (measured
    // 2026-09-29: node kept Workers selected with the generator hidden). Retried until the
    // panel reports itself visible, bounded so a typo in the key cannot loop forever.
    if (!start_tab_applied_ && !start_tab_id_.empty() && frame_counter_ < 240) {
        ImGui::SetWindowFocus(start_tab_id_.c_str());
    }

    // ---- dockspace host -------------------------------------------------------
    // One full-viewport window owns the dockspace. Panels dock into it, so the user
    // can rearrange everything and the arrangement is persisted in [GUI] dock_layout.
    ImGui::SetNextWindowPos(host_pos);
    ImGui::SetNextWindowSize(host_size);
    ImGui::SetNextWindowViewport(vp->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    const ImGuiWindowFlags host_flags =
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;
    ImGui::Begin("###ecm_gui_host", nullptr, host_flags);
    ImGui::PopStyleVar(3);
    // A NoMove/NoResize window stops accepting SetNextWindowPos/Size after its first
    // frame, and the first frame's viewport WorkSize can still be stale -- so pin the
    // host rect explicitly every frame. Without this the panels end up offset (measured:
    // panels reaching x=1609 in a 1478 px viewport).
    ImGui::SetWindowPos(host_pos);
    ImGui::SetWindowSize(host_size);
    const ImGuiID dockspace_id = ImGui::GetID("ecm_gui_dockspace");
    // Build the layout once the viewport size has settled (frame 3) and whenever the
    // stored blob predates the current layout version -- so an ini written by an older
    // build (or by the "everything stacked" first run) gets repaired instead of pinned.
    if (!layout_built_ && frame_counter_ >= 2 &&
        (ini_.get_int("GUI", "dock_layout_ver", ecm_config::defaults::gui_dock_layout_ver) < kLayoutVersion ||
         ImGui::DockBuilderGetNode(dockspace_id) == nullptr)) {
        build_default_layout(dockspace_id);
    }
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
    ImGui::End();

    // Stage tracing of the FIRST frames (permanent, not a temporary debug aid): when the GUI
    // dies inside draw(), the last "draw: <stage> ok" line names the panel. That is how the
    // 2026-09-29 0xC0000005 was pinned to the GPU panel, and the tests assert that the whole
    // sequence plus "frame 1 rendered" is present -- so a first-frame crash can never again
    // pass as "the GUI started fine".
    const bool trace_stages = frame_counter_ < 3;
    draw_menu_bar();
    if (trace_stages) trace("draw: menu bar ok");
    draw_p95_notice(strip_h, host_pos.y);
    if (trace_stages) trace("draw: p95 notice ok");
    draw_workers_table();
    if (trace_stages) trace("draw: workers table ok");
    draw_gpu_panel();
    if (trace_stages) trace("draw: gpu panel ok");
    draw_results_panel();
    if (trace_stages) trace("draw: results panel ok");
    draw_detail_panel();
    if (trace_stages) trace("draw: detail panel ok");
    draw_gen_panel();
    if (trace_stages) trace("draw: gen panel ok");
    draw_worker_panes();
    if (trace_stages) trace("draw: worker panes ok");
    // Last so it is on top of everything; also keeps the dockspace from stealing input.
    draw_exit_modal();
    if (trace_stages) trace("draw: exit modal ok");

    // The focus/selection is applied at the START of App::draw (see the SetWindowFocus call
    // there); nothing to do here beyond recording that it worked.

    ++frame_counter_;
    if (panels_traced_at_frame_ < 0 && frame_counter_ > 30) trace_panel_rects();
}

// Extracts the percentage from the driver's checkpoint lines:
//   "Checkpoint loaded: s_partial=89173178/375102575 (23.8%), age=24348 seconds"
//   "Resuming from checkpoint: 23.8% complete (s_partial=89173178/375102575)"
// Returns false when the line is neither.
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

std::string App::worker_dir() const {
    // The driver resolves its relative paths against its OWN directory, and that is where
    // it writes .ecm_ckpt_*.dat -- which the graceful stop watches to know that the
    // worker's work is safely on disk.
    const std::string exe = resolve_worker_exe();
    const std::size_t slash = exe.find_last_of("\\/");
    if (slash == std::string::npos) return exe_dir();
    return exe.substr(0, slash);
}

bool App::request_close() {
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
        // Explicit opt-out ([GUI] exit_confirm = kill): terminate immediately.
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

void App::request_stop(int worker_index) {
    for (WorkerView &w : workers_) {
        if (w.index != worker_index) continue;
        if (w.proc == nullptr || !w.proc->running()) return;
        w.stop_requested = true;
        w.ckpt_seen = false;
        w.ckpt_baseline = newest_checkpoint_mtime(worker_dir(), nullptr);
        w.stop_deadline_ms = mono_ms() + graceful_stop_ms_;
        trace("worker " + std::to_string(w.index) +
              ": stop requested (waiting for a checkpoint, max " +
              std::to_string(graceful_stop_ms_ / 1000) + " s)");
        return;
    }
}

void App::begin_graceful_exit() {
    exit_phase_ = ExitPhase::Stopping;
    exit_deadline_ms_ = mono_ms() + graceful_stop_ms_;
    for (WorkerView &w : workers_) {
        if (w.proc == nullptr || !w.proc->running()) continue;
        w.stop_requested = true;
        w.ckpt_seen = false;
        w.ckpt_baseline = newest_checkpoint_mtime(worker_dir(), nullptr);
        w.stop_deadline_ms = exit_deadline_ms_;
        trace("worker " + std::to_string(w.index) +
              ": graceful stop requested (waiting for a checkpoint, max " +
              std::to_string(graceful_stop_ms_ / 1000) + " s)");
    }
}

void App::tick_graceful_stops(unsigned long long now) {
    for (WorkerView &w : workers_) {
        if (w.proc == nullptr || !w.stop_requested) continue;
        if (!w.proc->running()) {
            // It exited on its own (finished its queue, or the driver honours a stop
            // request) -- nothing left to wait for.
            w.stop_requested = false;
            continue;
        }
        if (!w.ckpt_seen) {
            std::string name;
            const long long m = newest_checkpoint_mtime(worker_dir(), &name);
            if (m != 0 && m > w.ckpt_baseline) {
                w.ckpt_seen = true;
                trace("worker " + std::to_string(w.index) + ": checkpoint written (" + name +
                      "), safe to stop");
            }
        }
        if (w.ckpt_seen || now >= w.stop_deadline_ms) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "worker %d: stopping (checkpoint %s)",
                          w.index, w.ckpt_seen ? "written" : "NOT written (timeout)");
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

bool App::draw_exit_modal() {
    if (exit_phase_ == ExitPhase::Idle) return false;
    const char *title = "###exit_modal";
    if (exit_phase_ == ExitPhase::Confirm) {
        if (!ImGui::IsPopupOpen(title)) ImGui::OpenPopup(title);
    }
    bool stay_open = true;
    // A fixed, centered modal: no docking interaction while the process is winding down.
    const ImGuiViewport *vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->GetCenter().x, vp->GetCenter().y), ImGuiCond_Always,
                            ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(560.0f * ImGui::GetIO().FontGlobalScale, 0.0f),
                            ImGuiCond_Always);
    if (ImGui::BeginPopupModal(title, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (exit_phase_ == ExitPhase::Confirm) {
            int running = 0;
            for (const WorkerView &w : workers_) {
                if (w.proc != nullptr && w.proc->running()) ++running;
            }
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "%d worker(s) are still running. Stop them (each one writes a "
                          "checkpoint first) and quit?", running);
            ImGui::TextWrapped("%s", buf);
            ImGui::Separator();
            if (ImGui::Button("Stop and quit", ImVec2(160.0f, 0.0f)) ||
                ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)) {
                trace("exit: confirmed by the user");
                ImGui::CloseCurrentPopup();
                begin_graceful_exit();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120.0f, 0.0f)) ||
                ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                trace("exit: cancelled by the user");
                exit_phase_ = ExitPhase::Idle;
                ImGui::CloseCurrentPopup();
                stay_open = false;
            }
        } else {
            // Reaching here means the popup is still registered but the phase moved on
            // (the stopping UI below owns that phase now).
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    // While the stopping phase runs there is no popup left to keep open, so the progress is
    // its own small window. Escape (or the button) forces the exit without a fresh
    // checkpoint: the wait is bounded but a user must always be able to get out.
    if (exit_phase_ == ExitPhase::Stopping) {
        const unsigned long long now = mono_ms();
        const long long left_ms =
            (exit_deadline_ms_ > now) ? static_cast<long long>(exit_deadline_ms_ - now) : 0;
        ImGui::SetNextWindowPos(ImVec2(vp->GetCenter().x, vp->GetCenter().y), ImGuiCond_Always,
                                ImVec2(0.5f, 0.5f));
        ImGui::Begin("Stopping workers###exit_stopping", nullptr,
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings);
        ImGui::TextWrapped("Waiting for the workers to write a checkpoint (up to %lld s left) ...",
                           left_ms / 1000);
        ImGui::Separator();
        for (const WorkerView &w : workers_) {
            if (w.proc == nullptr) continue;
            const bool running = w.proc->running();
            const char *state = !running      ? "stopped"
                                : w.ckpt_seen ? "checkpoint written, stopping"
                                              : "waiting for checkpoint";
            ImGui::Text("Worker #%d: %s", w.index, state);
        }
        ImGui::Separator();
        if (ImGui::Button("Force quit (no fresh checkpoint)", ImVec2(280.0f, 0.0f)) ||
            ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            trace("exit: forced by the user (no fresh checkpoint)");
            stop_all_quietly();
            for (WorkerView &w : workers_) w.stop_requested = false;
            exit_phase_ = ExitPhase::Idle;
            quit_requested_ = true;
            stay_open = false;
        }
        ImGui::End();
    }
    return stay_open && exit_phase_ != ExitPhase::Idle;
}

void App::build_default_layout(unsigned int dockspace_id) {
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    // Size the node from the space the host window actually has now (the viewport rect
    // is settled by the time this runs), not from a first-frame estimate.
    ImVec2 size = ImGui::GetContentRegionAvail();
    if (size.x < 100.0f || size.y < 100.0f) size = ImGui::GetMainViewport()->WorkSize;
    ImGui::DockBuilderSetNodeSize(dockspace_id, size);

    // Layout (agreed with the user):
    //   left 60% : top half  workers table (+ detail as a tab in the same node)
    //              bottom    per-worker output, one tab per worker
    //   right 40%: top half  GPU panel
    //              bottom    results table
    ImGuiID left = 0, right = 0;
    ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.60f, &left, &right);
    ImGuiID left_top = 0, left_bottom = 0;
    ImGui::DockBuilderSplitNode(left, ImGuiDir_Up, 0.50f, &left_top, &left_bottom);
    ImGuiID right_top = 0, right_bottom = 0;
    ImGui::DockBuilderSplitNode(right, ImGuiDir_Up, 0.50f, &right_top, &right_bottom);

    ImGui::DockBuilderDockWindow("###workers", left_top);
    // Detail shares the workers node as a tab, so it is "aside" the table without
    // stealing space from it (Workers is the selected tab). The generator does the same:
    // it is used occasionally (paste assignments, preview, apply), so it must not take
    // permanent space away from the table the user watches.
    ImGui::DockBuilderDockWindow("###detail", left_top);
    ImGui::DockBuilderDockWindow("###gen", left_top);
    ImGui::DockBuilderDockWindow("###gpu", right_top);
    ImGui::DockBuilderDockWindow("###results", right_bottom);
    for (const WorkerView &w : workers_) {
        const std::string id = "###log" + std::to_string(w.index);
        ImGui::DockBuilderDockWindow(id.c_str(), left_bottom);
    }
    ImGui::DockBuilderFinish(dockspace_id);
    // Which tab of the shared left-top node is selected at startup: [GUI] start_tab =
    // workers (default) | detail | gen. The generator is used in bursts (paste, preview,
    // apply), so being able to open the GUI straight on it is worth one ini key -- and it is
    // also what lets a test measure the panel's controls, because a non-selected tab is
    // skipped by ImGui and reports no geometry.
    std::string tab = ini_.get("GUI", "start_tab", ecm_config::defaults::gui_start_tab);
    for (char &c : tab) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const char *tab_id = (tab == "gen" || tab == "generator") ? "###gen"
                       : (tab == "detail") ? "###detail"
                                           : "###workers";
    // The selection is applied LATER (App::draw, after the panels were submitted once):
    // DockBuilderDockWindow() makes a placeholder for a window that does not exist yet, and
    // FindWindowByName() right here would return null for ###gen -- measured 2026-09-29: the
    // layout then kept Workers selected and the generator stayed hidden.
    start_tab_id_ = tab_id;
    start_tab_applied_ = false;
    layout_built_ = true;
    trace(std::string("layout: default dock layout built (left=workers+detail+gen/output, "
                      "right=gpu/results), start_tab=") + tab_id);
}

void App::trace_panel_rects() {
    // The panel rectangles as ImGui laid them out. A script can then assert that the
    // panels are inside the window and do not overlap -- the measurable version of
    // "the initial layout is not a pile of windows".
    std::vector<std::pair<std::string, std::string>> panels;   // {label, window id}
    panels.emplace_back("host", "###ecm_gui_host");
    panels.emplace_back("workers", "###workers");
    panels.emplace_back("results", "###results");
    panels.emplace_back("gpu", "###gpu");
    panels.emplace_back("detail", "###detail");
    for (const WorkerView &w : workers_) {
        panels.emplace_back("log" + std::to_string(w.index), "###log" + std::to_string(w.index));
    }

    const ImGuiViewport *vp = ImGui::GetMainViewport();
    {
        char vb[192];
        std::snprintf(vb, sizeof(vb),
                      "layout: viewport pos=(%d,%d) size=%dx%d work pos=(%d,%d) work size=%dx%d",
                      static_cast<int>(vp->Pos.x), static_cast<int>(vp->Pos.y),
                      static_cast<int>(vp->Size.x), static_cast<int>(vp->Size.y),
                      static_cast<int>(vp->WorkPos.x), static_cast<int>(vp->WorkPos.y),
                      static_cast<int>(vp->WorkSize.x), static_cast<int>(vp->WorkSize.y));
        trace(vb);
    }
    for (const auto &p : panels) {
        ImGuiWindow *win = ImGui::FindWindowByName(p.second.c_str());
        if (win == nullptr) {
            trace("layout: " + p.first + " " + p.second + " not created");
            continue;
        }
        char buf[160];
        std::snprintf(buf, sizeof(buf), "layout: %s %s x=%d y=%d w=%d h=%d", p.first.c_str(),
                      p.second.c_str(), static_cast<int>(win->Pos.x),
                      static_cast<int>(win->Pos.y), static_cast<int>(win->Size.x),
                      static_cast<int>(win->Size.y));
        trace(buf);
    }
    panels_traced_at_frame_ = frame_counter_;
    trace_font_metrics();
}

void App::trace_font_metrics() {
    if (font_metrics_traced_) return;
    font_metrics_traced_ = true;
    // "文件" = U+6587 U+4EF6, written as escapes so this source stays ASCII-only.
    ImGuiIO &io = ImGui::GetIO();
    const ImVec2 latin = ImGui::CalcTextSize("WW");
    const ImVec2 cjk = ImGui::CalcTextSize(u8"\u6587\u4EF6");
    ImFont *font = io.FontDefault;

    // Width alone cannot prove the UI is not drawing "tofu": ImGui's dynamic atlas
    // silently substitutes U+FFFD for a codepoint the loaded font lacks, and that
    // box has a nonzero advance too. So ask the atlas directly:
    //   * IsGlyphInFont()        -> the TTF actually maps the codepoint;
    //   * FindGlyphNoFallback()  -> it was really baked into this size's atlas
    //                               (FindGlyph() would hand back the fallback box).
    // U+E123 (private use) is the negative control: if *it* reported present, the
    // two checks above would be meaningless.
    bool baked_wen = false, baked_jian = false, map_wen = false, map_jian = false;
    bool neg_ctl_baked = true, neg_ctl_map = true;
    if (font != nullptr) {
        map_wen = font->IsGlyphInFont(0x6587);
        map_jian = font->IsGlyphInFont(0x4EF6);
        neg_ctl_map = font->IsGlyphInFont(0xE123);
        ImFontBaked *baked = font->GetFontBaked(ImGui::GetFontSize());
        if (baked != nullptr) {
            baked_wen = baked->FindGlyphNoFallback(0x6587) != nullptr;
            baked_jian = baked->FindGlyphNoFallback(0x4EF6) != nullptr;
            neg_ctl_baked = baked->FindGlyphNoFallback(0xE123) != nullptr;
        }
    }
    const bool tofu_free = map_wen && map_jian && baked_wen && baked_jian &&
                           !neg_ctl_map && !neg_ctl_baked;
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "font: measured latin 'WW'=%.0fx%.0f cjk 2-glyphs=%.0fx%.0f cjk_ok=%d "
                  "map=%d%d baked=%d%d negctl=%d%d",
                  latin.x, latin.y, cjk.x, cjk.y, tofu_free ? 1 : 0, map_wen ? 1 : 0,
                  map_jian ? 1 : 0, baked_wen ? 1 : 0, baked_jian ? 1 : 0,
                  neg_ctl_map ? 1 : 0, neg_ctl_baked ? 1 : 0);
    trace(buf);

    // Which localization file is actually in effect: distinguishes "the UI is in
    // English" from "the UI is Chinese but the glyphs are boxes".
    std::snprintf(buf, sizeof(buf),
                  "localization: dir=%s language=%s keys=%zu baseline=%zu missing=%zu cjk=%d",
                  loc_.dir().c_str(), loc_.language().c_str(), loc_.count(),
                  loc_.baseline_count(), loc_.missing_keys().size(),
                  loc_.needs_cjk_font() ? 1 : 0);
    trace(buf);
    if (loc_.count() > 0) {
        // Sample an actual translated string: proves the XML text (not the fallback
        // key) reached the UI layer.
        trace("localization: sample workers.title='" + loc_.t("workers", "title") + "'");
    }
}

void App::trace_gpu_history(int dev, const std::vector<GpuSample> &hist, float p_lo, float p_hi,
                            float c_lo, float c_hi) {
    // Traced when the retained window grows by ~20 samples (10 s at the default 500 ms
    // poll) plus once early on, so a test can look at the LAST line and see real history.
    if (hist.size() < 4) return;
    if (gpu_hist_reported_.size() <= static_cast<std::size_t>(dev)) {
        gpu_hist_reported_.resize(static_cast<std::size_t>(dev) + 1, 0);
    }
    const int reported = gpu_hist_reported_[static_cast<std::size_t>(dev)];
    if (reported != 0 && static_cast<int>(hist.size()) < reported + 20) return;
    gpu_hist_reported_[static_cast<std::size_t>(dev)] = static_cast<int>(hist.size());

    // Distinct values on rounded quantities: a flat curve means "1 distinct value", a
    // curve that merely *looks* flat (tiny jitter) means more than one.
    auto stats = [](const std::vector<GpuSample> &h, bool power, float *mn, float *mx, int *distinct) {
        std::set<long long> seen;
        *mn = 0.0f;
        *mx = 0.0f;
        bool first = true;
        for (const GpuSample &s : h) {
            const float v = power ? static_cast<float>(s.power_w)
                                  : static_cast<float>(s.clock_sm_mhz);
            if (first || v < *mn) *mn = v;
            if (first || v > *mx) *mx = v;
            first = false;
            seen.insert(static_cast<long long>(power ? v * 10.0f + 0.5f : v + 0.5f));
        }
        *distinct = static_cast<int>(seen.size());
    };
    float p_mn = 0.0f, p_mx = 0.0f, c_mn = 0.0f, c_mx = 0.0f;
    int p_distinct = 0, c_distinct = 0, u_distinct = 0;
    stats(hist, true, &p_mn, &p_mx, &p_distinct);
    stats(hist, false, &c_mn, &c_mx, &c_distinct);
    {
        std::set<long long> seen;
        for (const GpuSample &s : hist) {
            seen.insert(static_cast<long long>(s.util_gpu + 0.5f));
        }
        u_distinct = static_cast<int>(seen.size());
    }
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "gpu: history dev=%d samples=%zu util=%d distinct power=%d distinct "
                  "clock=%d distinct flat=%d",
                  dev, hist.size(), u_distinct, p_distinct, c_distinct,
                  (p_distinct <= 1 && c_distinct <= 1) ? 1 : 0);
    trace(buf);
    std::snprintf(buf, sizeof(buf),
                  "gpu: history dev=%d ranges power=%.1f..%.1f plot=%.1f..%.1f "
                  "clock=%.0f..%.0f plot=%.0f..%.0f",
                  dev, static_cast<double>(p_mn), static_cast<double>(p_mx),
                  static_cast<double>(p_lo), static_cast<double>(p_hi),
                  static_cast<double>(c_mn), static_cast<double>(c_mx),
                  static_cast<double>(c_lo), static_cast<double>(c_hi));
    trace(buf);
}

void App::draw_menu_bar() {
    if (!ImGui::BeginMainMenuBar()) return;

    if (ImGui::BeginMenu(loc_.t("menu", "file").c_str())) {
        if (ImGui::MenuItem(loc_.t("menu", "open_ini").c_str())) {
            open_in_explorer(ini_path_);
        }
        if (ImGui::MenuItem(loc_.t("menu", "open_localization").c_str())) {
            open_in_explorer(loc_dir_);
        }
        ImGui::Separator();
        if (ImGui::MenuItem(loc_.t("menu", "save_settings").c_str())) {
            std::string err;
            capture_layout();
            if (ini_.save(err)) {
                status_ = "saved " + ini_path_;
            } else {
                status_ = "save failed: " + err;
            }
        }
        ImGui::Separator();
        if (ImGui::MenuItem(loc_.t("menu", "start_all").c_str())) start_all();
        if (ImGui::MenuItem(loc_.t("menu", "stop_all").c_str())) stop_all();
        ImGui::Separator();
        if (ImGui::MenuItem(loc_.t("menu", "quit").c_str())) {
            // Same flow as WM_CLOSE: with workers running this asks and checkpoints first.
            request_close();
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(loc_.t("menu", "view").c_str())) {
        for (WorkerView &w : workers_) {
            ImGui::MenuItem((loc_.t("workers", "log_pane") + " #" + std::to_string(w.index)).c_str(),
                            nullptr, &w.show_pane);
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(loc_.t("menu", "language").c_str())) {
        for (const std::string &lang : available_languages()) {
            const bool sel = (lang == language_);
            if (ImGui::MenuItem(lang.c_str(), nullptr, sel)) {
                set_language(lang);
            }
        }
        ImGui::Separator();
        if (ImGui::MenuItem(loc_.t("menu", "reload_localization").c_str())) {
            reload_localization();
        }
        ImGui::EndMenu();
    }

    // Right-aligned status line.
    const std::string &msg = status_;
    if (!msg.empty()) {
        const float w = ImGui::CalcTextSize(msg.c_str()).x;
        ImGui::SameLine(ImGui::GetWindowWidth() - w - 20.0f);
        ImGui::TextDisabled("%s", msg.c_str());
    }
    ImGui::EndMainMenuBar();
}

// ---- Prime95 handoff (docs/DEV_ECM_GUI.md 18) ---------------------------------------

std::string App::p95AddPath() const {
    if (p95_worktodo_path_.empty()) return std::string();
    return path_join(path_dir(p95WorktodoPath()), "worktodo.add");
}

std::string App::p95PendingPath() const {
    return path_join(worker_dir(), "p95_add_pending.txt");
}

long long App::p95PendingCount() const {
    // Read straight from disk (not from a notice) so the red state is also correct when
    // the pending file was left by an earlier GUI session or by a hand-run driver.
    std::ifstream in(p95PendingPath());
    if (!in) return 0;
    long long n = 0;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) ++n;
    }
    return n;
}

void App::draw_p95_notice(float height, float host_top) {
    // A full-width strip right under the menu bar. The user asked for the handoff state to
    // be impossible to miss ("如果存在任何异常情况包括 pending 我希望都要通知 GUI 并使用
    // 显著颜色"), so the strip is always visible -- green when a task was handed over, grey
    // when the feature is off, yellow for a warning, red while anything is still parked.
    //
    // 2026-09-29 (user: "四种颜色的真实切换，我实测看不出；没有配置 p95_worktodo_path 时也没有
    // 明显提示"): the strip is now drawn at WorkPos (its row is RESERVED, see App::draw) so no
    // panel can cover it, it carries an ASCII severity marker, and the "not configured" case
    // gets the same treatment as the others plus a one-click way to fix it. Its rect is traced
    // so a test can prove it is on screen and above the dockspace.
    const long long parked = p95PendingCount();

    // Most severe wins; the rest are counted.
    int level = -1;                        // 0 grey/none, 1 green, 2 yellow, 3 red
    std::string text;
    std::string detail;
    int extra = 0;
    bool saw_notice = false;               // a worker reported something this session
    for (const WorkerView &w : workers_) {
        if (!w.p95.valid) continue;
        saw_notice = true;
        const int l = w.p95.level == P95Notice::Level::Pending ? 3
                    : w.p95.level == P95Notice::Level::Warn ? 2
                    : 1;                       // Ok and Ready are both "fine"
        const std::string msg =
            w.p95.level == P95Notice::Level::Pending
                ? w.p95.error
                : (w.p95.level == P95Notice::Level::Warn ? w.p95.note : std::string());
        if (l > level) {
            if (level >= 1 && !text.empty()) ++extra;
            level = l;
            text = msg;
        } else if (l == level) {
            ++extra;
        }
    }
    if (parked > 0) level = 3;             // disk state beats any stale green notice

    char buf[512];
    ImVec4 fg, bg;                          // text colour, band colour
    std::string marker;
    if (level < 0 && p95_worktodo_path_.empty()) {
        level = 0;                         // grey: nothing is configured
        std::snprintf(buf, sizeof(buf), "%s", loc_.t("p95", "not_configured").c_str());
        fg = ImVec4(0.78f, 0.78f, 0.82f, 1.0f);
        bg = ImVec4(0.16f, 0.16f, 0.20f, 1.0f);
        marker = "[-] ";
    } else if (level == 3) {
        std::snprintf(buf, sizeof(buf), loc_.t("p95", "pending").c_str(),
                      static_cast<int>(parked > 0 ? parked : 1));
        fg = ImVec4(1.0f, 0.45f, 0.45f, 1.0f);
        bg = ImVec4(0.30f, 0.05f, 0.05f, 1.0f);
        marker = "[FAIL] ";
        detail = text;
    } else if (level == 2) {
        std::snprintf(buf, sizeof(buf), "%s", loc_.t("p95", "warn").c_str());
        fg = ImVec4(1.0f, 0.80f, 0.25f, 1.0f);
        bg = ImVec4(0.26f, 0.20f, 0.03f, 1.0f);
        marker = "[WARN] ";
        detail = text;
    } else {
        // Configured and nothing wrong: green. "ready" before the first task finished,
        // "ok" once a delivery actually succeeded -- both are the healthy state.
        level = 1;
        std::snprintf(buf, sizeof(buf), "%s",
                      loc_.t("p95", saw_notice ? "ok" : "ready").c_str());
        fg = ImVec4(0.55f, 1.0f, 0.55f, 1.0f);
        bg = ImVec4(0.05f, 0.20f, 0.07f, 1.0f);
        marker = "[OK] ";
    }
    std::string line = marker + buf;
    if (!detail.empty()) line += " - " + detail;
    if (extra > 0) line += " (+" + std::to_string(extra) + ")";

    const ImGuiViewport *vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x, height));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                                   ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBringToFrontOnFocus;
    ImGui::PushStyleColor(ImGuiCol_WindowBg, bg);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 2.0f));
    ImGui::Begin("###p95_notice", nullptr, flags);
    ImGui::PushStyleColor(ImGuiCol_Text, fg);
    ImGui::TextUnformatted(line.c_str());
    ImGui::PopStyleColor();
    if (!p95_worktodo_path_.empty()) {
        ImGui::SameLine();
        if (ImGui::SmallButton(loc_.t("p95", "open_dir").c_str())) {
            open_in_explorer(path_dir(p95WorktodoPath()));
        }
        ImGui::SameLine();
        if (ImGui::SmallButton(loc_.t("p95", "open_pending").c_str())) {
            open_in_explorer(p95PendingPath());
        }
        ImGui::SameLine();
        ImGui::TextDisabled("p95_add_workers=%s",
                            p95_add_workers_.empty() ? "(none)" : p95_add_workers_.c_str());
    } else {
        // One click to fix the missing key instead of hunting for ecm.ini.
        ImGui::SameLine();
        if (ImGui::SmallButton(loc_.t("p95", "open_ini").c_str())) {
            open_in_explorer(ini_path_);
        }
    }
    const ImVec2 wmin = ImGui::GetWindowPos();
    const ImVec2 wmax = ImGui::GetWindowSize();
    // Remember the real height so App::draw can reserve exactly this row next frame (and cap
    // it, so a pathological style cannot eat the whole window).
    p95_strip_h_ = wmax.y;
    const float cap = ImGui::GetFrameHeight() * 4.0f;
    if (p95_strip_h_ > cap) p95_strip_h_ = cap;
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    // Trace what the strip shows AND where it is: level, parked count, text, and the geometry
    // (the strip must be above the dockspace; otherwise it is painted over and invisible).
    // On every change, and at least every ~2 s while it stays the same.
    {
        const char *lvl = level == 3 ? "red" : level == 2 ? "yellow" : level == 1 ? "green" : "grey";
        char geo[256];
        std::snprintf(geo, sizeof(geo),
                      " p95 notice: level=%s parked=%lld rect=%.0f,%.0f,%.0fx%.0f host_top=%.0f",
                      lvl, parked, wmin.x, wmin.y, wmax.x, wmax.y, host_top);
        const std::string key = geo + std::string(" text=") + line;
        if (key != p95_trace_last_ || frame_counter_ % 120 == 0) {
            p95_trace_last_ = key;
            trace(key);
        }
    }
}

// ---- worktodo generator (M6 scope A, docs/DEV_ECM_GUI.md 19) -------------------------

std::string App::genWorktodoPath() const {
    // The file the driver's queue reads: the ini's `worktodo` key, resolved against the
    // worker executable's directory (what the driver does with a relative value). An absolute
    // value must NOT be prefixed again -- measured 2026-09-29 in the test sandbox:
    // "D:\...\sandbox\D:\...\sandbox\worktodo.txt" was displayed and would have been written.
    return path_resolve(worker_dir(), ini_.get("", "worktodo", ecm_config::defaults::stage1_worktodo));
}

std::string App::p95WorktodoPath() const {
    return path_resolve(worker_dir(), p95_worktodo_path_);
}

void App::gen_make_preview() {
    gen_has_preview_ = false;
    gen_text_.clear();

    GenOptions opt;
    opt.valid = true;
    opt.save_pattern = gen_save_pattern_.empty() ? "m{n}_{b1}.save" : gen_save_pattern_;
    opt.sort_by = gen_sort_by_;
    opt.use_recommended = gen_use_recommended_;
    opt.blocks_per_sm = gen_blocks_per_sm_;
    opt.dedup = gen_dedup_;
    for (const WorkerView &w : workers_) {
        opt.worker_devices.push_back(std::make_pair(w.index, w.device));
    }
    opt.target_device = gen_target_device_;

    GpuProfile gpu;
    if (gen_use_recommended_) {
        // The tier table and the SM count come from the driver itself (D4), so a copied
        // table can never drift from the kernels. --gpu-info writes nothing.
        std::string cmd = "\"" + resolve_worker_exe() + "\" --gpu-info -d " +
                          std::to_string(gen_target_device_);
        std::string out;
        int code = -1;
        std::string err;
        if (!run_capture(cmd, out, code, 20000) || code != 0) {
            gen_status_ = loc_.t("gen", "gpu_info_failed") + " (exit " + std::to_string(code) + ")";
            gpu.error = out.empty() ? "no output from --gpu-info" : out;
            trace("gen: gpu-info failed exit=" + std::to_string(code) + " out=" + out);
        } else if (!parse_gpu_info(out, gpu, err)) {
            gen_status_ = loc_.t("gen", "gpu_info_failed") + " (" + err + ")";
            trace("gen: gpu-info unparsable: " + err);
        } else {
            trace("gen: gpu-info device=" + std::to_string(gpu.device) + " sm=" +
                  std::to_string(gpu.sm_count) + " tiers=" + std::to_string(gpu.tiers.size()) +
                  " carry=" + std::to_string(gpu.carry_bits));
        }
    }

    const GenResult r = generate(gen_input_, gpu, opt);
    if (!r.ok) {
        gen_status_ = r.error;
        gen_warnings_.clear();
        for (const std::string &e : r.parse_errors) gen_warnings_ += e + "\n";
        trace("gen: preview failed: " + r.error);
        return;
    }

    gen_text_ = r.text;
    gen_lines_ = 0;
    for (const GenSegment &s : r.segments) gen_lines_ += static_cast<int>(s.lines.size());
    gen_segments_ = static_cast<int>(r.segments.size());
    gen_duplicates_ = r.duplicates;
    gen_read_ = r.read;
    gen_skipped_ = r.skipped_comment + r.skipped_other + r.skipped_unknown;
    gen_target_ = genWorktodoPath();
    gen_warnings_.clear();
    for (const std::string &e : r.parse_errors) gen_warnings_ += e + "\n";
    for (const std::string &w : r.warnings) gen_warnings_ += w + "\n";

    // The preview is only valid for THIS state of the target file: apply re-validates it.
    const bool exists = stamp_file(gen_target_, gen_stamp_);
    gen_has_preview_ = true;

    char buf[256];
    std::snprintf(buf, sizeof(buf), "gen: preview segments=%d lines=%d read=%d skipped=%d dup=%d target_exists=%d",
                  gen_segments_, gen_lines_, gen_read_, gen_skipped_, gen_duplicates_,
                  exists ? 1 : 0);
    trace(buf);
    gen_status_ = loc_.t("gen", "preview_ready");
}

void App::gen_apply() {
    if (!gen_has_preview_ || gen_text_.empty()) {
        gen_status_ = loc_.t("gen", "no_preview");
        trace("gen: apply refused (no preview)");
        return;
    }
    std::string err;
    size_t bytes = 0;
    if (!apply_append(gen_target_, gen_text_, gen_stamp_, err, &bytes)) {
        gen_status_ = err;
        trace("gen: apply failed reason=" + err);
        return;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "gen: applied bytes=%zu target=%s", bytes, gen_target_.c_str());
    trace(buf);
    gen_status_ = loc_.t("gen", "applied");
    // Re-stamp: a second click must be preceded by a new preview (the queue may have
    // consumed lines in the meantime).
    stamp_file(gen_target_, gen_stamp_);
    gen_has_preview_ = false;
}

void App::draw_gen_panel() {
    // A docked panel that is not the selected tab of its node is SKIPPED by ImGui: Begin()
    // returns false, items are not laid out, and GetItemRectSize() reports stale numbers.
    // Everything (including the geometry trace the tests read) therefore happens only when the
    // tab is really visible. `[GUI] start_tab = gen` makes it the selected tab at startup.
    const bool visible = ImGui::Begin((loc_.t("gen", "title") + "###gen").c_str());
    if (!visible) {
        // A hidden tab reports no geometry, so the trace that the tests read cannot exist.
        // Say why (once every ~2 s) instead of leaving a silent hole -- that is how the
        // "start_tab selected but the panel never appeared" case was found (2026-09-29).
        if (frame_counter_ % 120 == 0) {
            if (const ImGuiWindow *gw = ImGui::FindWindowByName("###gen")) {
                char dbg[192];
                std::snprintf(dbg, sizeof(dbg),
                              "gen: panel hidden (docked=%d node_selected=%u my_tab=%u)",
                              gw->DockNode != nullptr ? 1 : 0,
                              gw->DockNode != nullptr ? gw->DockNode->SelectedTabId : 0u,
                              gw->TabId);
                trace(dbg);
            } else {
                trace("gen: panel window does not exist yet");
            }
        }
        ImGui::End();
        return;
    }
    if (start_tab_id_ == "###gen" && !start_tab_applied_) {
        start_tab_applied_ = true;         // the wanted tab really is on screen now
        trace("layout: start_tab is visible: ###gen");
    }

    // The widget's char buffer is refreshed only when the model changed from outside
    // (a loaded file); while the user types, ImGui's own state is authoritative and the
    // model is updated below from the buffer.
    if (!gen_input_sync_) {
        gen_input_buf_.assign(gen_input_.begin(), gen_input_.end());
        gen_input_buf_.push_back('\0');
        gen_input_sync_ = true;
    }

    // ---- input -----------------------------------------------------------------------
    ImGui::TextDisabled("%s", loc_.t("gen", "paste_hint").c_str());
    if (ImGui::Button(loc_.t("gen", "load_file").c_str())) {
        std::string path = gen_input_path_;
        if (browse_for_file(path, loc_.t("gen", "load_file"),
                            "Assignments (*.txt;*.csv)\0*.txt;*.csv\0All files\0*.*\0\0")) {
            std::ifstream in(path, std::ios::binary);
            if (in) {
                std::ostringstream ss;
                ss << in.rdbuf();
                gen_input_ = ss.str();
                gen_input_path_ = path;
                gen_input_sync_ = false;      // refresh the widget's char buffer
                gen_status_ = path;
                trace("gen: loaded input " + path);
            } else {
                gen_status_ = "cannot read " + path;
                trace("gen: cannot read input " + path);
            }
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%s", gen_input_path_.empty() ? "-" : gen_input_path_.c_str());

    ImGui::InputTextMultiline("##gen_input", gen_input_buf_.data(), gen_input_buf_.size(),
                              ImVec2(-1.0f, 120.0f));
    gen_input_ = gen_input_buf_.empty() ? std::string() : std::string(gen_input_buf_.data());
    // ---- options ---------------------------------------------------------------------
    ImGui::Checkbox(loc_.t("gen", "use_recommended").c_str(), &gen_use_recommended_);
    ImGui::SameLine();
    // `blocks/SM` box. `kBlocksStep` MUST stay 0: ImGui reserves two GetFrameHeight() wide step
    // buttons INSIDE the item width, so at 150 % DPI they ate ~63 px of an 80 px box and the
    // value was invisible ("没有宽度，无法显示数字", 2026-09-29). With step 0 the whole box edits
    // the number, and the two traced widths below (frame / editable) let a test catch a
    // regression to a non-zero step.
    const int kBlocksStep = 0;
    const std::string blocks_label = loc_.t("gen", "blocks_per_sm");
    ImGui::SetNextItemWidth(90.0f);
    ImGui::InputInt(blocks_label.c_str(), &gen_blocks_per_sm_, kBlocksStep, 0);
    if (gen_blocks_per_sm_ < 1) gen_blocks_per_sm_ = 1;
    if (gen_blocks_per_sm_ > 64) gen_blocks_per_sm_ = 64;
    // GetItemRectSize() covers the frame AND the label drawn next to it; the frame is what the
    // number is edited in.
    const float blocks_box_w = ImGui::GetItemRectSize().x - ImGui::CalcTextSize(blocks_label.c_str()).x -
                               ImGui::GetStyle().ItemInnerSpacing.x;
    const float blocks_edit_w = blocks_box_w -
                                ((kBlocksStep != 0) ? 2.0f * ImGui::GetFrameHeight() : 0.0f);
    ImGui::SameLine();
    ImGui::Checkbox(loc_.t("gen", "dedup").c_str(), &gen_dedup_);

    // Save pattern + sort field: two short text boxes.
    {
        char pattern[128];
        std::snprintf(pattern, sizeof(pattern), "%s", gen_save_pattern_.c_str());
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::InputText(loc_.t("gen", "save_pattern").c_str(), pattern, sizeof(pattern))) {
            gen_save_pattern_ = pattern;
        }
        char sortf[64];
        std::snprintf(sortf, sizeof(sortf), "%s", gen_sort_by_.c_str());
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90.0f);
        if (ImGui::InputText(loc_.t("gen", "sort_by").c_str(), sortf, sizeof(sortf))) {
            gen_sort_by_ = sortf;
        }
    }

    // Target GPU: the devices the ini's workers use (the recommendation depends on it).
    {
        std::vector<int> devices;
        for (const WorkerView &w : workers_) {
            if (std::find(devices.begin(), devices.end(), w.device) == devices.end()) {
                devices.push_back(w.device);
            }
        }
        std::sort(devices.begin(), devices.end());
        if (devices.empty()) devices.push_back(0);
        if (std::find(devices.begin(), devices.end(), gen_target_device_) == devices.end()) {
            gen_target_device_ = devices.front();
        }
        std::string current = "GPU " + std::to_string(gen_target_device_);
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::BeginCombo(loc_.t("gen", "target_device").c_str(), current.c_str())) {
            for (int d : devices) {
                const std::string label = "GPU " + std::to_string(d);
                if (ImGui::Selectable(label.c_str(), d == gen_target_device_)) {
                    gen_target_device_ = d;
                    gen_has_preview_ = false;      // the recommendation is device specific
                }
            }
            ImGui::EndCombo();
        }
    }

    // ---- actions: generate and apply are deliberately SEPARATE ---------------------
    if (ImGui::Button(loc_.t("gen", "preview").c_str())) gen_make_preview();
    ImGui::SameLine();
    if (!gen_has_preview_) ImGui::BeginDisabled();
    if (ImGui::Button(loc_.t("gen", "apply").c_str())) gen_apply();
    if (!gen_has_preview_) ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("%s", loc_.t("gen", "target").c_str());
    ImGui::SameLine();
    ImGui::Text("%s", genWorktodoPath().c_str());

    if (!gen_status_.empty()) {
        ImGui::TextWrapped("%s", gen_status_.c_str());
    }

    // ---- preview ---------------------------------------------------------------------
    if (gen_has_preview_) {
        ImGui::Separator();
        ImGui::Text("%s: %d, %s: %d, %s: %d, (%s %d)",
                    loc_.t("gen", "col_lines").c_str(), gen_lines_,
                    loc_.t("gen", "col_segments").c_str(), gen_segments_,
                    loc_.t("gen", "col_dup").c_str(), gen_duplicates_,
                    loc_.t("gen", "col_skipped").c_str(), gen_skipped_);
        std::string view = gen_text_;
        if (view.size() > 20000) view = view.substr(0, 20000) + "\r\n... (truncated)";
        view.push_back('\0');
        ImGui::InputTextMultiline("##gen_preview", view.data(), view.size(),
                                  ImVec2(-1.0f, 200.0f), ImGuiInputTextFlags_ReadOnly);
    }

    if (!gen_warnings_.empty()) {
        ImGui::Separator();
        ImGui::TextDisabled("%s", loc_.t("gen", "errors").c_str());
        std::string view = gen_warnings_;
        view.push_back('\0');
        ImGui::InputTextMultiline("##gen_warn", view.data(), view.size(),
                                  ImVec2(-1.0f, 70.0f), ImGuiInputTextFlags_ReadOnly);
    }

    ImGui::TextDisabled("%s", loc_.t("gen", "recommended_note").c_str());

    // One trace line per state change so a test can assert the panel's numbers AND that the
    // controls are wide enough to show their values (the blocks/SM box was unusable once:
    // 2026-09-29, see the InputInt call above). Not on frame 0/1: the panel has no size yet
    // and GetItemRectSize() reports a stretched item there.
    if (frame_counter_ >= 2) {
        char buf[384];
        std::snprintf(buf, sizeof(buf),
                      "gen: panel preview=%d lines=%d segments=%d blocks_box_w=%.0f blocks_edit_w=%.0f blocks_per_sm=%d target=%s",
                      gen_has_preview_ ? 1 : 0, gen_lines_, gen_segments_, blocks_box_w,
                      blocks_edit_w, gen_blocks_per_sm_, genWorktodoPath().c_str());
        if (buf != gen_trace_last_) {
            gen_trace_last_ = buf;
            trace(buf);
        }
    }
    // The panel is visible for exactly one frame bundle per change, so the trace above is the
    // measurement a test can trust: `blocks_box_w` must be a real input width (see the
    // InputInt call), and `target` must be the resolved file, never a doubled path.
    ImGui::End();
}

// ---- one line chart for every panel (docs/DEV_ECM_GUI.md 8) --------------------------
// Replaces ImGui::PlotLines, which the user found hard to read ("优化所有折线图使其更美观易读",
// 2026-09-29). What this adds over PlotLines:
//   * a rounded card with a dark inset background, so the chart reads as one object;
//   * a subtle 3-line grid and dim min/max labels, so a value can be read off the chart;
//   * a filled area under the curve plus a 2 px line: the shape is visible at a glance;
//   * the CURRENT value + unit as the headline (no need to guess from the axis);
//   * an optional reference line (e.g. the power limit) with its own label;
//   * a hover read-out: vertical guide + tooltip with the value and its age;
//   * "collecting…" instead of an empty box while the series is still short.
void App::draw_metric_plot(const MetricPlot &p, const std::vector<float> &values,
                          float *used_lo, float *used_hi) {
    const ImU32 color = (p.color != 0) ? p.color : IM_COL32(90, 170, 255, 255);
    const ImU32 bg = IM_COL32(18, 20, 24, 255);
    const ImU32 bg_top = IM_COL32(26, 29, 35, 255);
    const ImU32 grid = IM_COL32(255, 255, 255, 22);
    const ImU32 dim = IM_COL32(170, 175, 185, 200);
    const ImU32 fill = (color & 0x00FFFFFFu) | (60u << IM_COL32_A_SHIFT);

    const float line_h = ImGui::GetTextLineHeight();
    // The card is "headline row + plot band + min/max row". The heights in MetricPlot are given in
    // 15 px-font units, so they must scale with the font: at 150 % DPI (font 22.5 px, line height
    // ~31 px) a fixed 58 px card made the two text rows OVERLAP each other and cover the curve
    // (measured 2026-09-29 -- the user runs 150 %). The plot band gets its own slot and a floor.
    const float ui_scale = (font_size_px() > 0.0f) ? (font_size_px_ / 15.0f) : 1.0f;
    float h = p.height * ui_scale;
    const float rows_h = line_h * 2.0f + 8.0f;
    if (h < rows_h + 24.0f) h = rows_h + 24.0f;
    ImGui::PushID(p.id.c_str());
    ImGui::InvisibleButton("##plot", ImVec2(-1.0f, h));
    const ImVec2 p0 = ImGui::GetItemRectMin();
    const ImVec2 p1 = ImGui::GetItemRectMax();
    ImDrawList *dl = ImGui::GetWindowDrawList();
    // Card background: a subtle vertical gradient (AddRectFilledMultiColor takes no rounding
    // argument, so the rounded card is the AddRect below).
    dl->AddRectFilledMultiColor(p0, p1, bg_top, bg_top, bg, bg);
    dl->AddRect(p0, p1, IM_COL32(255, 255, 255, 18), 5.0f, 0, 1.0f);

    const float pad_x = 6.0f;
    const ImVec2 a0(p0.x + pad_x, p0.y + line_h + 3.0f);   // below the headline row
    const ImVec2 a1(p1.x - pad_x, p1.y - line_h - 2.0f);   // above the min/max row

    // Range: fixed, or the observed window with padding (so a nearly constant signal still
    // shows its shape instead of a dead flat line).
    float lo = p.lo, hi = p.hi;
    if (!p.fixed_range && !values.empty()) {
        float mn = values[0], mx = values[0];
        for (float v : values) { mn = (v < mn ? v : mn); mx = (v > mx ? v : mx); }
        float span = mx - mn;
        const float floor_span = (mx > 0.0f) ? mx * 0.02f : 1.0f;
        if (span < floor_span) span = floor_span;
        lo = mn - span * 0.12f;
        hi = mx + span * 0.12f;
        if (lo < 0.0f && mn >= 0.0f) lo = 0.0f;
    }
    if (hi <= lo) hi = lo + 1.0f;

    // Grid + axis labels.
    for (int g = 1; g <= 3; ++g) {
        const float y = a1.y - (a1.y - a0.y) * (static_cast<float>(g) / 4.0f);
        dl->AddLine(ImVec2(a0.x, y), ImVec2(a1.x, y), grid, 1.0f);
    }

    if (values.size() < 2) {
        // Not enough samples: say so instead of drawing a 1-point "line".
        const char *msg = p.empty_text.empty() ? "collecting..." : p.empty_text.c_str();
        const ImVec2 ts = ImGui::CalcTextSize(msg);
        dl->AddText(ImVec2((p0.x + p1.x - ts.x) * 0.5f, (p0.y + p1.y - ts.y) * 0.5f), dim, msg);
    } else {
        const float span = hi - lo;
        const float step_x = (a1.x - a0.x) / static_cast<float>(values.size() - 1);
        const auto y_of = [&](float v) {
            float t = (v - lo) / span;
            t = (t < 0.0f) ? 0.0f : (t > 1.0f ? 1.0f : t);
            return a1.y - (a1.y - a0.y) * t;
        };

        // Area fill: one thin column per sample step (a filled area under a polyline is not
        // convex, so AddConvexPolyFilled cannot be used).
        for (std::size_t i = 1; i < values.size(); ++i) {
            const float x0 = a0.x + step_x * static_cast<float>(i - 1);
            const float x1 = x0 + step_x + 0.5f;
            const float y = (y_of(values[i - 1]) + y_of(values[i])) * 0.5f;
            if (y >= a1.y) continue;
            dl->AddRectFilled(ImVec2(x0, y), ImVec2(x1, a1.y), fill);
        }

        // Reference line (dashed) with its label on the left. A reference that lies OUTSIDE the
        // auto-ranged window must not be silently dropped -- measured 2026-09-29: the enforced
        // 285 W limit against a 141..165 W observed window meant the dashed line never appeared
        // ("where is the limit line?"). It is pinned to the edge it lies beyond, marked with a
        // small triangle pointing that way, so "the curve sits far below the limit" stays visible
        // without stretching the range and flattening the curve.
        if (p.has_ref) {
            const bool above = (p.ref > hi);
            const bool below = (p.ref < lo);
            const float y = above ? a0.y + 0.5f : (below ? a1.y - 0.5f : y_of(p.ref));
            const ImU32 ref_col = (above || below) ? IM_COL32(255, 120, 120, 110)
                                                  : IM_COL32(255, 120, 120, 150);
            for (float x = a0.x; x < a1.x; x += 8.0f) {
                dl->AddLine(ImVec2(x, y), ImVec2((x + 4.0f < a1.x ? x + 4.0f : a1.x), y), ref_col,
                            1.0f);
            }
            if (!p.ref_label.empty()) {
                const ImVec2 ts = ImGui::CalcTextSize(p.ref_label.c_str());
                // Off-scale labels stay INSIDE the plot band (never over the headline row).
                const float ty = below ? (a1.y - ts.y - 1.5f) : (y + 1.5f);
                dl->AddRectFilled(ImVec2(a0.x + 2.0f, ty - 1.0f),
                                  ImVec2(a0.x + ts.x + 6.0f, ty + ts.y + 1.0f),
                                  IM_COL32(40, 20, 20, 220), 2.0f);
                dl->AddText(ImVec2(a0.x + 4.0f, ty), IM_COL32(255, 150, 150, 230),
                            p.ref_label.c_str());
                if (above || below) {
                    // A drawn arrow, not a glyph: no dependency on the font covering U+2191.
                    const float ax = a0.x + ts.x + 10.0f;
                    const float ay = ty + ts.y * 0.5f;
                    const float r = ts.y * 0.28f;
                    if (above) {
                        dl->AddTriangleFilled(ImVec2(ax, ay - r), ImVec2(ax - r, ay + r),
                                              ImVec2(ax + r, ay + r), ref_col);
                    } else {
                        dl->AddTriangleFilled(ImVec2(ax, ay + r), ImVec2(ax - r, ay - r),
                                              ImVec2(ax + r, ay - r), ref_col);
                    }
                }
            }
        }

        // The curve itself.
        std::vector<ImVec2> pts;
        pts.reserve(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            pts.push_back(ImVec2(a0.x + step_x * static_cast<float>(i), y_of(values[i])));
        }
        dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), color, 0, 2.0f);

        // The newest sample gets a dot: "where are we now" without reading the axis.
        const ImVec2 last = pts.back();
        dl->AddCircleFilled(last, 3.0f, color);
        dl->AddCircle(last, 4.0f, IM_COL32(0, 0, 0, 120), 0, 1.0f);

        // Hover: vertical guide + tooltip.
        if (ImGui::IsItemHovered() && ImGui::GetIO().MousePos.x >= a0.x &&
            ImGui::GetIO().MousePos.x <= a1.x) {
            const int idx = static_cast<int>((ImGui::GetIO().MousePos.x - a0.x) / step_x + 0.5f);
            const int clamped = (idx < 0) ? 0 : (idx >= static_cast<int>(values.size())
                                                     ? static_cast<int>(values.size()) - 1
                                                     : idx);
            const ImVec2 hp(pts[static_cast<std::size_t>(clamped)].x, a0.y);
            dl->AddLine(ImVec2(hp.x, a0.y), ImVec2(hp.x, a1.y), IM_COL32(255, 255, 255, 70), 1.0f);
            dl->AddCircleFilled(pts[static_cast<std::size_t>(clamped)], 2.5f,
                                IM_COL32(255, 255, 255, 220));
            char buf[160];
            if (p.ms_per_sample > 0.0f) {
                const float age_s = (static_cast<float>(values.size() - 1 - clamped)) *
                                    p.ms_per_sample / 1000.0f;
                std::snprintf(buf, sizeof(buf), "%s: %.2f %s\n%.1f s ago", p.label.c_str(),
                              static_cast<double>(values[static_cast<std::size_t>(clamped)]),
                              p.unit.c_str(), static_cast<double>(age_s));
            } else {
                std::snprintf(buf, sizeof(buf), "%s: %.2f %s\n#%d of %d", p.label.c_str(),
                              static_cast<double>(values[static_cast<std::size_t>(clamped)]),
                              p.unit.c_str(), clamped + 1, static_cast<int>(values.size()));
            }
            ImGui::SetTooltip("%s", buf);
        }
    }

    // Headline: label on the left, current value + unit on the right; min/max dim underneath.
    // (line_h comes from the layout block above -- the card height is built from it.)
    dl->AddText(ImVec2(p0.x + 8.0f, p0.y + 2.0f), dim, p.label.c_str());
    if (!values.empty()) {
        char vbuf[96];
        std::snprintf(vbuf, sizeof(vbuf), "%.*f %s", p.decimals,
                      static_cast<double>(values.back()), p.unit.c_str());
        const ImVec2 ts = ImGui::CalcTextSize(vbuf);
        dl->AddText(ImVec2(p1.x - ts.x - 8.0f, p0.y + 2.0f), color, vbuf);
    }
    {
        char b1[64], b2[64];
        std::snprintf(b1, sizeof(b1), "%.*f", p.decimals, static_cast<double>(lo));
        std::snprintf(b2, sizeof(b2), "%.*f", p.decimals, static_cast<double>(hi));
        dl->AddText(ImVec2(a0.x, p1.y - line_h - 1.0f), IM_COL32(140, 145, 155, 180), b1);
        const ImVec2 ts2 = ImGui::CalcTextSize(b2);
        dl->AddText(ImVec2(a1.x - ts2.x, p1.y - line_h - 1.0f),
                    IM_COL32(140, 145, 155, 180), b2);
    }
    ImGui::PopID();

    if (used_lo != nullptr) *used_lo = lo;
    if (used_hi != nullptr) *used_hi = hi;
    // Trace the chart so a test can assert it exists with a sane range and the reference line
    // (the panels have no other machine-readable footprint). Rate-limited per chart below.
    //
    // The geometry fields exist because of a REAL layout bug (2026-09-29): the card height was a
    // constant (58/62 px) while the text row height scales with the font, so at 150 % DPI
    // (font 22.5 px, line_h ~31 px) the headline row and the min/max row OVERLAPPED and both
    // covered the curve. `band_top`/`band_h` let a test prove the three bands are disjoint at
    // whatever DPI it runs at: band_top >= line_h + 2 and band_top + band_h <= h - line_h.
    {
        char tb[384];
        const float band_top = a0.y - p0.y;
        const float band_h = a1.y - a0.y;
        if (p.has_ref) {
            std::snprintf(tb, sizeof(tb),
                          "plot: %s label=\"%s\" n=%d lo=%.2f hi=%.2f last=%.2f ref=%.2f "
                          "h=%.0f line_h=%.0f band_top=%.0f band_h=%.0f",
                          p.id.c_str(), p.label.c_str(), static_cast<int>(values.size()),
                          static_cast<double>(lo), static_cast<double>(hi),
                          values.empty() ? 0.0 : static_cast<double>(values.back()),
                          static_cast<double>(p.ref), static_cast<double>(h),
                          static_cast<double>(line_h), static_cast<double>(band_top),
                          static_cast<double>(band_h));
        } else {
            std::snprintf(tb, sizeof(tb),
                          "plot: %s label=\"%s\" n=%d lo=%.2f hi=%.2f last=%.2f "
                          "h=%.0f line_h=%.0f band_top=%.0f band_h=%.0f",
                          p.id.c_str(), p.label.c_str(), static_cast<int>(values.size()),
                          static_cast<double>(lo), static_cast<double>(hi),
                          values.empty() ? 0.0 : static_cast<double>(values.back()),
                          static_cast<double>(h), static_cast<double>(line_h),
                          static_cast<double>(band_top), static_cast<double>(band_h));
        }
        // Rate limit PER CHART (a single shared timestamp starved every chart but the first
        // one: util changes on every sample and consumed the window, so power/clock/the worker
        // speed chart never got traced -- measured 2026-09-29).
        const unsigned long long now = mono_ms();
        const bool changed = (plot_trace_last_[p.id] != tb);
        // The first few changes are always traced (a test needs the early state), after that at
        // most one line per 2 s per chart.
        const bool early = (plot_trace_count_[p.id] < 3);
        const bool due = (now - plot_trace_ms_[p.id] >= 2000ull);
        if (changed && (early || due)) {
            plot_trace_last_[p.id] = tb;
            plot_trace_ms_[p.id] = now;
            ++plot_trace_count_[p.id];
            trace(tb);
        }
    }
}

void App::draw_workers_table() {
    // Stable window id (###workers): the docking layout must survive a language switch,
    // otherwise the titles change and ImGui sees a different window.
    ImGui::Begin((loc_.t("workers", "title") + "###workers").c_str());

    // Global controls: start/stop everything, plus the resolved worker executable
    // (so a wrong [GUI] exe= is visible instead of mysterious).
    if (ImGui::Button(loc_.t("workers", "start_all").c_str())) start_all();
    ImGui::SameLine();
    if (ImGui::Button(loc_.t("workers", "stop_all").c_str())) stop_all();
    ImGui::SameLine();
    ImGui::TextDisabled("%s %s", loc_.t("workers", "exe").c_str(),
                        resolve_worker_exe().c_str());
    ImGui::Separator();

    // Column widths are explicit (derived from sample text, so they follow the DPI-scaled
    // font) for everything except the progress bar, which takes the leftover space. Why:
    // the Task column used to hold the whole worktodo line, which is ~80 characters, so
    // the table auto-sized that column to its content and pushed the progress bar / speed
    // / ETA columns past the right edge of the panel -- the user reported "progress bar
    // and ETA do not show at all" (2026-09-28). The task now goes on its own line under
    // the row (see below).
    const float w_index = ImGui::GetFrameHeight() * 0.9f;
    // State width fits the state word only: the long state_text ("crashed 3 times in
    // 300 s - stopped (exit code 1)") goes to a tooltip + the Detail panel, because a
    // column sized for it costs ~100 px that the progress bar needs more.
    const float w_state = ImGui::CalcTextSize("Restarting*").x + 10.0f;
    const float w_gpu = ImGui::CalcTextSize("GPU 0 *").x + 8.0f;
    const float w_speed = ImGui::CalcTextSize("9999.99 s/c").x + 8.0f;
    const float w_eta = ImGui::CalcTextSize("99999 s").x + 8.0f;
    const float w_hits = ImGui::CalcTextSize("0000").x + 8.0f;
    // Nine columns, but only the applicable action button is drawn (Stop while running,
    // Start otherwise): two buttons side by side cost ~50 px for nothing.
    const float w_actions = ImGui::CalcTextSize("Stop").x + 40.0f;
    // Right edge of the table, captured before BeginTable: the task line spans to it.
    const float table_right_x = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    // Give the progress bar a guaranteed minimum by shrinking the NAME column first: at the
    // user's panel width (885 px) the fixed columns plus a 12-character name left the bar
    // only 45 px wide, which the test caught (fits=0). Deterministic beats "hope ImGui
    // distributes it nicely": the bar is what the user actually watches.
    const float kProgressMin = 120.0f;
    const float column_padding = ImGui::GetStyle().CellPadding.x * 2.0f * 9.0f + 34.0f;
    const float table_left_x = ImGui::GetCursorScreenPos().x;
    const float table_w = table_right_x - table_left_x;
    float w_name = ImGui::CalcTextSize("4070Ti-110e6").x + 8.0f;
    {
        const float fixed = w_index + w_state + w_gpu + w_speed + w_eta + w_hits +
                            w_actions + column_padding;
        if (fixed + w_name + kProgressMin > table_w) {
            w_name = table_w - fixed - kProgressMin;
            if (w_name < 64.0f) w_name = 64.0f;      // give up gracefully on tiny panels
        }
    }
    if (ImGui::BeginTable("workers", 9,
                          // Horizontal separators only: ImGui draws table borders AFTER the
                          // cell contents, so the vertical ("y") lines would strike through
                          // the full-width task line on its own row (user report
                          // 2026-09-28). Rows stay visually separated by the horizontal
                          // lines; nothing is lost, because the data columns are aligned by
                          // position and the header.
                          ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_BordersOuterH |
                              ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, w_index);
        ImGui::TableSetupColumn(loc_.t("workers", "col_state").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_state);
        ImGui::TableSetupColumn(loc_.t("workers", "col_gpu").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_gpu);
        ImGui::TableSetupColumn(loc_.t("workers", "col_name").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_name);
        ImGui::TableSetupColumn(loc_.t("workers", "col_progress").c_str(),
                                ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn(loc_.t("workers", "col_speed").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_speed);
        ImGui::TableSetupColumn(loc_.t("workers", "col_eta").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_eta);
        ImGui::TableSetupColumn(loc_.t("workers", "col_hits").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_hits);
        ImGui::TableSetupColumn(loc_.t("workers", "col_actions").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_actions);
        ImGui::TableHeadersRow();

        int task_lines = 0;
        // Measured on the first row: the real progress-bar x/width inside the panel. This
        // is the number the user could not see when the Task column pushed it out.
        float progress_x = 0.0f, progress_w = 0.0f, col_left = 0.0f;
        float state_x = 0.0f, name_x = 0.0f, speed_x = 0.0f, eta_x = 0.0f, actions_x = 0.0f;
        float task_wrap_w = 0.0f;   // usable width of the task line (must be most of the row)
        for (std::size_t i = 0; i < workers_.size(); ++i) {
            WorkerView &w = workers_[i];
            const WorkerRunState st = w.proc ? w.proc->state() : WorkerRunState::Stopped;
            ImGui::TableNextRow();
            // A hit flashes the row so the operator notices a long run paying off.
            if (w.hit_flash_until > mono_ms()) {
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(40, 120, 40, 120));
            }
            ImGui::TableSetColumnIndex(0);
            if (i == 0) col_left = ImGui::GetCursorScreenPos().x;
            ImGui::Text("%d", w.index);

            ImGui::TableSetColumnIndex(1);
            if (i == 0) state_x = ImGui::GetCursorScreenPos().x;
            ImVec4 col(0.8f, 0.8f, 0.8f, 1.0f);
            if (st == WorkerRunState::Running) col = ImVec4(0.4f, 0.9f, 0.4f, 1.0f);
            else if (st == WorkerRunState::Error) col = ImVec4(1.0f, 0.35f, 0.35f, 1.0f);
            else if (st == WorkerRunState::Restarting) col = ImVec4(1.0f, 0.8f, 0.3f, 1.0f);
            else if (st == WorkerRunState::QueueEmpty) col = ImVec4(0.5f, 0.7f, 1.0f, 1.0f);
            ImGui::TextColored(col, "%s", loc_.t("workers", state_key(st)).c_str());
            if (w.proc && !w.proc->state_text().empty()) {
                // Full text in the tooltip (and the Detail panel): the column is sized for
                // the state word so the progress bar keeps its width (see above).
                ImGui::SameLine();
                ImGui::TextDisabled("*");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", w.proc->state_text().c_str());
                }
            }

            ImGui::TableSetColumnIndex(2);
            if (w.method == "gpu") {
                ImGui::Text("GPU %d", w.device);
                if (w.duplicate_device) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "*");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("%s", loc_.t("workers", "duplicate_device").c_str());
                    }
                }
            } else {
                ImGui::TextUnformatted(w.method.c_str());
            }

            ImGui::TableSetColumnIndex(3);
            if (i == 0) name_x = ImGui::GetCursorScreenPos().x;
            ImGui::TextUnformatted(w.name.c_str());
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", w.name.c_str());
            if (w.old_driver) {
                // Visible, not just in the status line: the user's question is "why does
                // Start fail", and the answer is "this exe is too old" (see kOldDriverHint).
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "!");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kOldDriverHint);
            }

            ImGui::TableSetColumnIndex(4);
            if (i == 0) {
                progress_x = ImGui::GetCursorScreenPos().x;
                progress_w = ImGui::GetContentRegionAvail().x;
            }
            if (w.progress.valid) {
                ImGui::ProgressBar(static_cast<float>(w.progress.pct / 100.0), ImVec2(-1.0f, 0.0f));
            } else if (w.resume_pct >= 0.0) {
                // Resumed run: the driver has not printed a progress line yet (its schedule
                // is gated on the restored batch counter), but we know where it stands.
                char overlay[64];
                std::snprintf(overlay, sizeof(overlay), "%.1f%% (resumed)", w.resume_pct);
                ImGui::ProgressBar(static_cast<float>(w.resume_pct / 100.0), ImVec2(-1.0f, 0.0f),
                                   overlay);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", loc_.t("workers", "resume_note").c_str());
                }
            } else if (st == WorkerRunState::Running || st == WorkerRunState::Restarting) {
                // The driver emits its first progress line only after the first batch
                // (a resumed B1=2.6e8 task can take a while to reach it). Saying so beats
                // an empty bar, which reads as "the GUI is broken" (user report).
                ImGui::ProgressBar(0.0f, ImVec2(-1.0f, 0.0f), loc_.t("workers", "waiting_progress").c_str());
            } else {
                ImGui::ProgressBar(0.0f, ImVec2(-1.0f, 0.0f), "");
            }

            ImGui::TableSetColumnIndex(5);
            if (i == 0) speed_x = ImGui::GetCursorScreenPos().x;
            if (w.progress.s_per_curve > 0.0) {
                ImGui::Text("%.2f s/c", w.progress.s_per_curve);
            } else {
                ImGui::TextUnformatted("-");
            }

            ImGui::TableSetColumnIndex(6);
            if (i == 0) eta_x = ImGui::GetCursorScreenPos().x;
            if (w.progress.eta_s > 0.0) ImGui::Text("%.0f s", w.progress.eta_s);
            else ImGui::TextUnformatted("-");

            ImGui::TableSetColumnIndex(7);
            ImGui::Text("%lld", w.hits);

            ImGui::TableSetColumnIndex(8);
            if (i == 0) actions_x = ImGui::GetCursorScreenPos().x;
            const std::string tag = "##act" + std::to_string(w.index);
            const bool busy = (st == WorkerRunState::Running || st == WorkerRunState::Restarting);
            // Only the applicable button: the column is narrow on purpose (the progress bar
            // is more valuable than two buttons that are never both useful).
            if (busy) {
                if (ImGui::SmallButton((loc_.t("workers", "stop") + tag).c_str())) {
                    // Graceful: a checkpoint is written first (docs/DEV_ECM_GUI.md 5.6).
                    request_stop(w.index);
                }
            } else {
                if (ImGui::SmallButton((loc_.t("workers", "start") + tag).c_str())) {
                    std::string err;
                    if (w.proc && w.proc->start(err)) {
                        trace("worker " + std::to_string(w.index) + ": started pid=" +
                              std::to_string(w.proc->pid()));
                    } else {
                        status_ = "worker " + std::to_string(w.index) + ": " + err;
                    }
                }
            }

            if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                selected_worker_ = static_cast<int>(i);
            }

            // ---- the task line: its own row, spanning the whole table -----------------
            // The worktodo line ("ECMSTAGE2=7CF9…,1,2,3571,-1,"m3571_260e6.save",0,0,960,…")
            // is ~80 characters. Keeping it in a column made that column thousands of
            // pixels wide (with SizingStretchProp it also squeezed the progress bar away).
            // Instead: a second row per worker whose text starts at the table's left edge
            // and wraps at the table's right edge.
            //
            // Two details a first attempt got wrong -- together they made the line show only
            // its first couple of characters (user report 2026-09-28):
            //   * the cell's clip rect is the narrow FIRST COLUMN, and PushClipRect(..., 
            //     intersect=true) intersects with it instead of widening it -> pass false so
            //     the text can run across the remaining columns;
            //   * PushTextWrapPos() takes a WINDOW-LOCAL x, not a screen x (a screen
            //     coordinate silently disabled wrapping).
            if (!w.task.empty()) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                const ImVec2 start = ImGui::GetCursorScreenPos();
                const float text_h = ImGui::GetTextLineHeight();
                const float wrap_local = table_right_x - 8.0f - ImGui::GetWindowPos().x;
                task_wrap_w = table_right_x - 8.0f - start.x;
                ImGui::PushClipRect(start, ImVec2(table_right_x, start.y + text_h * 4.0f), false);
                ImGui::PushTextWrapPos(wrap_local);
                ImGui::TextDisabled("%s %s", loc_.t("workers", "col_task").c_str(),
                                    w.task.c_str());
                ImGui::PopTextWrapPos();
                ImGui::PopClipRect();
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", w.task.c_str());
                ++task_lines;
            }
        }
        ImGui::EndTable();

        // Geometry of the layout above, so a test can prove that the columns the user
        // could not see (progress / speed / ETA) are actually inside the panel.
        // Rate-limited (every 60 frames): the first frames still have the default panel
        // size, and one line per frame would bury the rest of the trace.
        if (frame_counter_ % 60 == 0) {
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                          "table: workers right=%.0f left=%.0f progress_x=%.0f progress_w=%.0f "
                          "state_x=%.0f name_x=%.0f speed_x=%.0f eta_x=%.0f actions_x=%.0f "
                          "task_wrap_w=%.0f task_lines=%d rows=%zu fits=%d",
                          static_cast<double>(table_right_x), static_cast<double>(col_left),
                          static_cast<double>(progress_x),
                          static_cast<double>(progress_w), static_cast<double>(state_x),
                          static_cast<double>(name_x), static_cast<double>(speed_x),
                          static_cast<double>(eta_x), static_cast<double>(actions_x),
                          static_cast<double>(task_wrap_w), task_lines, workers_.size(),
                          (progress_w >= 60.0f && progress_x + progress_w <= table_right_x + 2.0f)
                              ? 1
                              : 0);
            trace(buf);
        }
    }

    ImGui::Separator();
    ImGui::TextDisabled("%s", loc_.t("workers", "m2_note").c_str());
    ImGui::End();
}

void App::draw_gpu_panel() {
    ImGui::Begin((loc_.t("gpu", "title") + "###gpu").c_str());
    if (!gpu_.available()) {
        ImGui::TextDisabled("%s", loc_.t("gpu", "nvml_missing").c_str());
        ImGui::TextWrapped("%s", gpu_.reason().c_str());
        ImGui::Separator();
        ImGui::Text("%s: %d ms   %s: %d Hz", loc_.t("gpu", "poll").c_str(), gpu_poll_ms_,
                    loc_.t("gpu", "refresh").c_str(), refresh_hz_);
        ImGui::End();
        return;
    }

    const std::vector<GpuInfo> devices = gpu_.devices();
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const GpuInfo &info = devices[i];
        GpuSample s;
        const bool have = gpu_.latest(static_cast<int>(i), s);
        // One collapsible card per device.
        const std::string head = "#" + std::to_string(i) + "  " + info.name + "###gpu" +
                                 std::to_string(i);
        if (!ImGui::CollapsingHeader(head.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) continue;

        if (!have) {
            ImGui::TextDisabled("...");
            continue;
        }
        ImGui::Text("%s %.0f%%   %s %.1f W / %.0f W (%.0f%%)", loc_.t("gpu", "util").c_str(),
                    static_cast<double>(s.util_gpu), loc_.t("gpu", "power").c_str(), s.power_w,
                    s.power_limit_w,
                    s.power_limit_w > 0.0 ? 100.0 * s.power_w / s.power_limit_w : 0.0);
        // Order must match the format string exactly: a %s that receives an integer makes
        // vsnprintf dereference it as a pointer (this exact line crashed the GUI with
        // 0xC0000005 as soon as NVML reported a non-zero VRAM clock, 2026-09-29).
        ImGui::Text("%s %u MHz   %s %u MHz   %s %d C   %s %u/%u MB",
                    loc_.t("gpu", "clock_sm").c_str(), s.clock_sm_mhz,
                    loc_.t("gpu", "clock_mem").c_str(), s.clock_mem_mhz,
                    loc_.t("gpu", "temp").c_str(), s.temp_c,
                    loc_.t("gpu", "memory").c_str(), s.mem_used_mb, info.mem_total_mb);
        const std::string th = GpuMonitor::throttle_text(s.throttle_reasons);
        if (!th.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f), "%s: %s",
                               loc_.t("gpu", "throttle").c_str(), th.c_str());
        } else {
            ImGui::TextDisabled("%s: -", loc_.t("gpu", "throttle").c_str());
        }
        if (info.cores > 0) {
            ImGui::SameLine();
            ImGui::TextDisabled("| %d %s", info.cores, loc_.t("gpu", "cores").c_str());
        }

        // History: utilisation, power and SM clock over the retained window (the
        // sampling thread keeps ~2 minutes at the default poll interval).
        const std::vector<GpuSample> hist_raw = gpu_.history(static_cast<int>(i));
        // Physically impossible readings are dropped from the CHART (they are documented in
        // docs/DEV_ECM_GUI.md 8.1): the 4060 Laptop intermittently reports 590 W against a 55 W
        // enforced limit, and ONE such sample stretched the power range to 0..660 W so the real
        // 1.5..9.4 W curve was drawn as a dead flat line at the bottom (measured 2026-09-29).
        // The same rule already guards the whole-machine total below, and the card's text row
        // still shows the raw reading.
        std::vector<GpuSample> hist;
        hist.reserve(hist_raw.size());
        for (const GpuSample &h : hist_raw) {
            if (h.power_limit_w > 0.0 && h.power_w > 3.0 * h.power_limit_w) continue;
            hist.push_back(h);
        }
        if (hist.size() >= 2) {
            std::vector<float> util, power, clock;
            util.reserve(hist.size());
            power.reserve(hist.size());
            clock.reserve(hist.size());
            for (const GpuSample &h : hist) {
                util.push_back(static_cast<float>(h.util_gpu));
                power.push_back(static_cast<float>(h.power_w));
                clock.push_back(static_cast<float>(h.clock_sm_mhz));
            }
            // The charts themselves (grid, headline value, reference line, hover read-out)
            // are drawn by App::draw_metric_plot, shared with the worker speed chart; power and
            // SM clock auto-range over the OBSERVED window (a nearly constant signal then still
            // shows its shape instead of a dead flat line -- measured: idle power 9.2 vs 9.6 W).
            const float poll_ms = static_cast<float>(gpu_poll_ms_);
            const float power_limit = static_cast<float>(s.power_limit_w);

            MetricPlot pu;
            pu.id = "gpu" + std::to_string(i) + "/util";
            pu.label = loc_.t("gpu", "util");
            pu.unit = "%";
            pu.color = IM_COL32(90, 200, 255, 255);
            pu.fixed_range = true;       // a percentage always reads 0..100
            pu.lo = 0.0f;
            pu.hi = 100.0f;
            pu.decimals = 0;
            pu.ms_per_sample = poll_ms;
            pu.empty_text = loc_.t("gpu", "collecting");
            draw_metric_plot(pu, util);

            MetricPlot pp;
            pp.id = "gpu" + std::to_string(i) + "/power";
            pp.label = loc_.t("gpu", "power");
            pp.unit = "W";
            pp.color = IM_COL32(255, 190, 70, 255);
            pp.lo = 0.0f;                // fallback range when the window is still short
            pp.hi = (power_limit > 0.0f) ? power_limit : 100.0f;
            pp.decimals = 1;
            pp.ms_per_sample = poll_ms;
            pp.empty_text = loc_.t("gpu", "collecting");
            if (power_limit > 0.0f) {
                pp.has_ref = true;       // the enforced power limit is the number to compare to
                pp.ref = power_limit;
                pp.ref_label = loc_.t("gpu", "power_limit");
            }
            // The limit is what that dashed reference line claims. Trace it (when it changes)
            // so a script can cross-check the chart's `ref=` against the real NVML value
            // instead of trusting the picture.
            {
                char lim[96];
                std::snprintf(lim, sizeof(lim), "gpu: limits dev=%d power_limit_w=%.1f",
                              static_cast<int>(i), s.power_limit_w);
                if (gpu_limit_trace_[static_cast<int>(i)] != lim) {
                    gpu_limit_trace_[static_cast<int>(i)] = lim;
                    trace(lim);
                }
            }
            float p_lo = 0.0f, p_hi = 0.0f;
            draw_metric_plot(pp, power, &p_lo, &p_hi);   // real range, reported by the trace

            MetricPlot pc;
            pc.id = "gpu" + std::to_string(i) + "/clock";
            pc.label = loc_.t("gpu", "clock_sm");
            pc.unit = "MHz";
            pc.color = IM_COL32(120, 230, 150, 255);
            pc.lo = 0.0f;
            pc.hi = 3000.0f;             // fallback; the real range is the observed window
            pc.decimals = 0;
            pc.ms_per_sample = poll_ms;
            pc.empty_text = loc_.t("gpu", "collecting");
            float c_lo = 0.0f, c_hi = 0.0f;
            draw_metric_plot(pc, clock, &c_lo, &c_hi);

            trace_gpu_history(static_cast<int>(i), hist, p_lo, p_hi, c_lo, c_hi);
        }
    }

    // Whole-machine power: the sum of the newest sample of every card -- the number a
    // user actually wants when asking "is the box at its limit". Physically impossible
    // readings are excluded (see docs/DEV_ECM_GUI.md 8.1).
    {
        double total = 0.0;
        int counted = 0;
        for (std::size_t i = 0; i < devices.size(); ++i) {
            GpuSample s;
            if (!gpu_.latest(static_cast<int>(i), s)) continue;
            if (s.power_limit_w > 0.0 && s.power_w > 3.0 * s.power_limit_w) continue;
            total += s.power_w;
            ++counted;
        }
        if (counted > 0) {
            ImGui::Text("%s: %.1f W", loc_.t("gpu", "total_power").c_str(), total);
        }
    }

    ImGui::Separator();
    ImGui::Text("%s: %d ms   %s: %d Hz   %s", loc_.t("gpu", "poll").c_str(), gpu_poll_ms_,
                loc_.t("gpu", "refresh").c_str(), refresh_hz_, gpu_.nvml_source().c_str());
    ImGui::TextDisabled("%s", loc_.t("gpu", "m4_note").c_str());
    ImGui::End();
}

void App::draw_results_panel() {
    ImGui::Begin((loc_.t("results", "title") + "###results").c_str());
    if (ImGui::Button(loc_.t("results", "rebuild").c_str())) {
        std::string err;
        if (results_.rebuild_from_jsonl(err)) {
            status_ = "results.txt rebuilt from the JSONL";
        } else {
            status_ = "rebuild failed: " + err;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button(loc_.t("results", "open_folder").c_str())) {
        open_in_explorer(results_txt_.empty() ? exe_dir() : results_txt_);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%lld hit(s), %d factor(s)", results_.hit_count(),
                        static_cast<int>(results_.factors().size()));

    const std::vector<MergedFactor> &factors = results_.factors();
    if (factors.empty()) {
        ImGui::TextDisabled("%s", loc_.t("results", "empty").c_str());
    } else if (ImGui::BeginTable("results", 6,
                                 // Column order (user request): the SCALAR fields first
                                 // (factor, bits, hits, first seen), then the LIST fields
                                 // (curves, sigmas) last -- they are the wide ones and are
                                 // truncated inside their column with the full list in the
                                 // tooltip. ScrollX/ScrollY: the lists can be long, so the
                                 // table scrolls instead of pushing the panel wide.
                                 ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                                     ImGuiTableFlags_SizingFixedFit,
                                 ImVec2(0.0f, ImGui::GetContentRegionAvail().y - 24.0f))) {
        const float w_factor = ImGui::CalcTextSize("1943118631").x + 12.0f;   // 10-digit factor
        const float w_count = ImGui::CalcTextSize("0000").x + 12.0f;
        const float w_time = ImGui::CalcTextSize("2026-09-28 23:59").x + 12.0f;
        const float w_list = ImGui::CalcTextSize("0000000000000000000").x + 12.0f;  // ~19 digits
        ImGui::TableSetupColumn(loc_.t("results", "factor").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_factor);
        ImGui::TableSetupColumn(loc_.t("results", "bits").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_count);
        ImGui::TableSetupColumn(loc_.t("results", "hits").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_count);
        ImGui::TableSetupColumn(loc_.t("results", "first_seen").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_time);
        ImGui::TableSetupColumn(loc_.t("results", "curves").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_list);
        ImGui::TableSetupColumn(loc_.t("results", "sigmas").c_str(),
                                ImGuiTableColumnFlags_WidthFixed, w_list);
        ImGui::TableSetupScrollFreeze(4, 1);      // keep the scalars visible while scrolling
        ImGui::TableHeadersRow();
        for (const MergedFactor &f : factors) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "%s", f.factor.c_str());
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", ResultsStore::format_line(f).c_str());
            }
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%d", f.bits);
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%d", f.hits);
            ImGui::TableSetColumnIndex(3);
            ImGui::TextUnformatted(f.first_seen.c_str());
            ImGui::TableSetColumnIndex(4);
            std::string curves;
            for (std::size_t i = 0; i < f.curves.size(); ++i) {
                if (i) curves += ",";
                curves += std::to_string(f.curves[i]);
            }
            ImGui::TextUnformatted(curves.empty() ? "-" : curves.c_str());
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", curves.c_str());
            ImGui::TableSetColumnIndex(5);
            std::string sigmas;
            for (std::size_t i = 0; i < f.sigmas.size(); ++i) {
                if (i) sigmas += ",";
                sigmas += std::to_string(f.sigmas[i]);
            }
            ImGui::TextUnformatted(sigmas.empty() ? "-" : sigmas.c_str());
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", sigmas.c_str());
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled("%s: %s", loc_.t("results", "json_file").c_str(), results_.json_path().c_str());
    ImGui::End();
}

void App::draw_detail_panel() {
    ImGui::Begin((loc_.t("detail", "title") + "###detail").c_str());
    if (selected_worker_ >= 0 && selected_worker_ < static_cast<int>(workers_.size())) {
        WorkerView &w = workers_[static_cast<std::size_t>(selected_worker_)];
        ImGui::Text("%s #%d (%s)", loc_.t("workers", "col_name").c_str(), w.index, w.name.c_str());
        ImGui::Separator();
        ImGui::TextUnformatted(loc_.t("detail", "command_line").c_str());
        ImGui::BeginChild("cmd", ImVec2(0, 40), true);
        ImGui::TextWrapped("%s", w.proc ? w.proc->command_line().c_str() : "-");
        ImGui::EndChild();
        if (w.proc) {
            ImGui::Text("%s %s   %s %d", loc_.t("workers", "col_state").c_str(),
                        state_name(w.proc->state()), "restarts", w.proc->restarts());
        }
        // Diagnostics for "the progress bar does not move": how many progress lines the
        // GUI actually received from this worker, and what the last one said. A count of 0
        // while the worker is Running means the driver has not printed one yet (it emits
        // the first line after the first batch), not that the GUI is deaf.
        ImGui::Text("%s %lld   %s %.1f%%  %.2f s/c  ETA %.0f s", "progress lines",
                    w.progress_lines, "last", w.progress.pct, w.progress.s_per_curve,
                    w.progress.eta_s);
        ImGui::Separator();
        ImGui::TextUnformatted(loc_.t("detail", "effective_config").c_str());
        ImGui::BeginChild("eff", ImVec2(0, 200), true);
        ImGui::TextUnformatted(w.effective_config.c_str());
        ImGui::EndChild();
    } else {
        ImGui::TextDisabled("-");
    }
    ImGui::End();
}

void App::draw_worker_panes() {
    for (WorkerView &w : workers_) {
        if (!w.show_pane) continue;
        const std::string title = loc_.t("workers", "log_pane") + " #" +
                                  std::to_string(w.index) + " (" + w.name + ")###log" +
                                  std::to_string(w.index);
        ImGui::Begin(title.c_str(), &w.show_pane);
        ImGui::Separator();
        // Progress panel: the numbers come from the parsed progress line (the ASCII bar
        // in the raw output is NOT what the user sees here).
        // Only the SPEED is charted. The progress-% chart was dropped on request
        // (2026-09-29: "worker 日志窗口删去进度%"): the percentage is already the bar in the
        // Workers table and the headline number right here, and it is monotone, so a second
        // chart of it carried no information. The speed chart is seconds per curve (the unit
        // the driver prints and the one the ETA is built from), not curves/second.
        if (w.progress.valid) {
            ImGui::Text("%.1f%%   %.2f s/curve   ETA %.0f s", w.progress.pct,
                        w.progress.s_per_curve, w.progress.eta_s);
            if (!w.hist_s_per_curve.empty()) {
                MetricPlot plot;
                plot.id = "worker" + std::to_string(w.index) + "/s_per_curve";
                plot.label = loc_.t("workers", "speed_history");
                plot.unit = "s/curve";
                plot.color = IM_COL32(180, 150, 255, 255);
                plot.height = 62.0f;
                // Two decimals: a curve takes tens of seconds on a big B1, and the useful
                // differences between progress lines are in the hundredths… but a range of
                // 1.40..1.44 s/curve must not collapse into "1.4".
                plot.fixed_range = false;
                plot.decimals = 2;
                plot.empty_text = loc_.t("workers", "collecting");
                draw_metric_plot(plot, w.hist_s_per_curve);
            }
        }
        ImGui::TextDisabled("%s: %s", loc_.t("workers", "log_file").c_str(), w.log_file.c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton((loc_.t("workers", "open_saves") + "##" +
                                std::to_string(w.index)).c_str())) {
            open_in_explorer(exe_dir());
        }
        ImGui::SameLine();
        ImGui::Checkbox((loc_.t("workers", "show_raw") + "##" + std::to_string(w.index)).c_str(),
                        &w.show_raw);
        ImGui::Separator();
        const std::vector<std::string> &lines = w.show_raw ? w.raw : w.events;
        if (lines.empty()) {
            ImGui::TextDisabled("%s", loc_.t("workers", "no_output").c_str());
        } else {
            ImGui::BeginChild("log", ImVec2(0, 0), false,
                              ImGuiWindowFlags_HorizontalScrollbar);
            for (const std::string &line : lines) {
                // ANSI already stripped; the events keep the driver timestamp.
                ImGui::TextUnformatted(line.c_str());
            }
            if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f) {
                ImGui::SetScrollHereY(1.0f);
            }
            ImGui::EndChild();
        }
        ImGui::End();
    }
}

} // namespace ecmgui
