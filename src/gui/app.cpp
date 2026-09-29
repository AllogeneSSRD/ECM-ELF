#include "app.h"
#include "platform.h"

#include "imgui.h"
#include "imgui_internal.h"   // ImGui::DockBuilder* for the initial layout

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>

namespace ecmgui {

// Bumped whenever the default layout changes: an ini carrying an older number gets
// the new default layout instead of keeping the old arrangement.
static const int kLayoutVersion = 3;

namespace {

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

    loc_dir_ = ini_.get("GUI", "localization_dir");
    if (loc_dir_.empty()) {
        loc_dir_ = default_localization_dir();
        // Running from a build directory: fall back to the source tree layout.
        const std::string alt = path_join(path_join(exe_dir(), ".."), "localization");
        if (!file_exists(path_join(loc_dir_, "english.xml")) &&
            file_exists(path_join(alt, "english.xml"))) {
            loc_dir_ = alt;
        }
    }

    language_ = language.empty() ? ini_.get("GUI", "language", "english") : language;
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

    refresh_hz_ = ini_.get_int("GUI", "refresh_hz", 10);
    if (refresh_hz_ < 1) refresh_hz_ = 1;
    if (refresh_hz_ > 60) refresh_hz_ = 60;
    gpu_poll_ms_ = ini_.get_int("GUI", "gpu_poll_ms", 500);
    if (gpu_poll_ms_ < 100) gpu_poll_ms_ = 100;
    priority_ = ini_.get("GUI", "priority", "below_normal");

    int rect[4] = {win_x_, win_y_, win_w_, win_h_};
    if (parse_int_list(ini_.get("GUI", "window"), rect, 4)) {
        win_x_ = rect[0];
        win_y_ = rect[1];
        win_w_ = rect[2];
        win_h_ = rect[3];
    }
    layout_blob_loaded_ = unescape_blob(ini_.get("GUI", "dock_layout"));

    num_workers_ = ini_.get_int("GUI", "NumWorkers", 1);
    if (num_workers_ < 1) num_workers_ = 1;
    if (num_workers_ > 64) num_workers_ = 64;
    worker_exe_ = ini_.get("GUI", "exe");
    // Font: 0 = auto (the host scales 15 px by the window DPI), or an explicit pixel
    // size; `font` names a .ttf/.ttc, empty lets the host pick (CJK font for a CJK
    // language, else a Latin system font). A FRACTIONAL size is allowed on purpose:
    // at 150 % the automatic size is 22.5 px, and rounding it changes how crisp the
    // text looks, so the user can try e.g. 23.5 or 24.
    {
        const std::string fs = ini_.get("GUI", "font_size", "auto");
        if (fs != "auto" && !fs.empty()) {
            try {
                font_size_px_ = std::stof(fs);
            } catch (...) {
                font_size_px_ = 0.0f;      // unparsable -> auto
            }
        }
        if (font_size_px_ < 6.0f) font_size_px_ = 0.0f;
        if (font_size_px_ > 96.0f) font_size_px_ = 96.0f;
        font_path_ = ini_.get("GUI", "font");
        const std::string snap = ini_.get("GUI", "font_snap", "1");
        font_snap_ = !(snap == "0" || snap == "off" || snap == "false" || snap == "no");
    }
    // Exit policy: what happens when the window is closed while workers run.
    //   ask  (default) : confirmation modal, then checkpoint, then exit
    //   stop           : no modal (scripts/unattended), still checkpoint, then exit
    //   kill           : terminate immediately (the old behaviour)
    exit_confirm_ = ini_.get("GUI", "exit_confirm", "ask");
    if (exit_confirm_ != "stop" && exit_confirm_ != "kill") exit_confirm_ = "ask";
    {
        const std::string v = ini_.get("GUI", "graceful_stop_ms", "300000");
        try {
            long long ms = std::stoll(v);
            if (ms < 1000) ms = 1000;
            if (ms > 3600000) ms = 3600000;
            graceful_stop_ms_ = static_cast<unsigned long long>(ms);
        } catch (...) {
            graceful_stop_ms_ = 300000;
        }
    }
    // Result files: [GUI] results_json / results_txt, defaulting next to the exe. The
    // JSONL is append-only (the durable record), results.txt is derived from it.
    results_json_ = ini_.get("GUI", "results_json");
    if (results_json_.empty()) {
        results_json_ = path_join(exe_dir(), "results.json.txt");
    }
    results_txt_ = ini_.get("GUI", "results_txt");
    if (results_txt_.empty()) {
        results_txt_ = path_join(exe_dir(), "results.txt");
    }
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
        w.name = ini_.get(sec, "name", "Worker #" + std::to_string(i));
        w.device = ini_.get_int(sec, "device", ini_.get_int("", "device", 0));
        w.method = ini_.get(sec, "method", ini_.get("", "method", "gpu"));
        w.gpucurves = ini_.get_int(sec, "gpucurves", ini_.get_int("", "gpucurves", 0));
        w.worktodo = ini_.get(sec, "worktodo", ini_.get("", "worktodo", "worktodo.txt"));
        w.log_file = ini_.get(sec, "log_file", ini_.get("", "log_file", "screen.log"));
        if (i > 1 && !ini_.has(sec, "log_file") && !ini_.has("", "log_file")) {
            // Mirror the driver's per-worker default (D1).
            w.log_file = "screen_" + std::to_string(i) + ".log";
        }
        w.extra_args = ini_.get(sec, "extra_args");
        w.autostart = ini_.get_int(sec, "autostart", 0) != 0;
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
            // Sample for the sparkline. The driver emits progress lines at a decaying
            // rate, so every line is worth keeping when it carries new information.
            w.hist_pct.push_back(static_cast<float>(w.progress.pct));
            if (w.progress.s_per_curve > 0.0) {
                w.hist_speed.push_back(static_cast<float>(1.0 / w.progress.s_per_curve));
            }
            const std::size_t cap = 600;
            if (w.hist_pct.size() > cap) {
                w.hist_pct.erase(w.hist_pct.begin(),
                                 w.hist_pct.begin() + static_cast<std::ptrdiff_t>(w.hist_pct.size() - cap));
            }
            if (w.hist_speed.size() > cap) {
                w.hist_speed.erase(w.hist_speed.begin(),
                                   w.hist_speed.begin() + static_cast<std::ptrdiff_t>(w.hist_speed.size() - cap));
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

    // ---- dockspace host -------------------------------------------------------
    // One full-viewport window owns the dockspace. Panels dock into it, so the user
    // can rearrange everything and the arrangement is persisted in [GUI] dock_layout.
    const ImGuiViewport *vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
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
    ImGui::SetWindowPos(vp->WorkPos);
    ImGui::SetWindowSize(vp->WorkSize);
    const ImGuiID dockspace_id = ImGui::GetID("ecm_gui_dockspace");
    // Build the layout once the viewport size has settled (frame 3) and whenever the
    // stored blob predates the current layout version -- so an ini written by an older
    // build (or by the "everything stacked" first run) gets repaired instead of pinned.
    if (!layout_built_ && frame_counter_ >= 2 &&
        (ini_.get_int("GUI", "dock_layout_ver", 0) < kLayoutVersion ||
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
    draw_workers_table();
    if (trace_stages) trace("draw: workers table ok");
    draw_gpu_panel();
    if (trace_stages) trace("draw: gpu panel ok");
    draw_results_panel();
    if (trace_stages) trace("draw: results panel ok");
    draw_detail_panel();
    if (trace_stages) trace("draw: detail panel ok");
    draw_worker_panes();
    if (trace_stages) trace("draw: worker panes ok");
    // Last so it is on top of everything; also keeps the dockspace from stealing input.
    draw_exit_modal();
    if (trace_stages) trace("draw: exit modal ok");

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
    // stealing space from it (Workers is the selected tab).
    ImGui::DockBuilderDockWindow("###detail", left_top);
    ImGui::DockBuilderDockWindow("###gpu", right_top);
    ImGui::DockBuilderDockWindow("###results", right_bottom);
    for (const WorkerView &w : workers_) {
        const std::string id = "###log" + std::to_string(w.index);
        ImGui::DockBuilderDockWindow(id.c_str(), left_bottom);
    }
    ImGui::DockBuilderFinish(dockspace_id);
    // Make Workers the visible tab of the shared node.
    if (ImGuiWindow *w = ImGui::FindWindowByName("###workers")) {
        if (w->DockNode != nullptr) w->DockNode->SelectedTabId = w->TabId;
    }
    layout_built_ = true;
    trace("layout: default dock layout built (left=workers+detail/output, right=gpu/results)");
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
        const std::vector<GpuSample> hist = gpu_.history(static_cast<int>(i));
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
            const std::string tag = "##h" + std::to_string(i);
            // Auto-range power and clock around the OBSERVED window (with 10 % padding):
            // with the fixed 0..0 "auto" range ImGui scales to the data, but a nearly
            // constant signal then looks dead flat -- an explicit padded range shows the
            // small variations (measured: idle power 9.2 vs 9.6 W, SM clock 210 vs 2595).
            const auto padded_range = [](const std::vector<float> &v, float lo_fallback,
                                         float hi_fallback, float *lo, float *hi) {
                float mn = v.empty() ? lo_fallback : v[0];
                float mx = mn;
                for (float x : v) {
                    mn = (x < mn ? x : mn);
                    mx = (x > mx ? x : mx);
                }
                float span = mx - mn;
                if (span < (mx > 0.0f ? mx * 0.02f : 1.0f)) span = (mx > 0.0f ? mx * 0.02f : 1.0f);
                *lo = mn - span * 0.10f;
                *hi = mx + span * 0.10f;
                if (*lo < 0.0f) *lo = 0.0f;
            };
            float p_lo = 0.0f, p_hi = 0.0f, c_lo = 0.0f, c_hi = 0.0f;
            padded_range(power, 0.0f, 1.0f, &p_lo, &p_hi);
            padded_range(clock, 0.0f, 1.0f, &c_lo, &c_hi);
            ImGui::PlotLines((tag + "u").c_str(), util.data(), static_cast<int>(util.size()), 0,
                             loc_.t("gpu", "util").c_str(), 0.0f, 100.0f, ImVec2(-1.0f, 50.0f));
            ImGui::PlotLines((tag + "p").c_str(), power.data(), static_cast<int>(power.size()), 0,
                             loc_.t("gpu", "power").c_str(), p_lo, p_hi, ImVec2(-1.0f, 50.0f));
            ImGui::PlotLines((tag + "c").c_str(), clock.data(), static_cast<int>(clock.size()), 0,
                             loc_.t("gpu", "clock_sm").c_str(), c_lo, c_hi, ImVec2(-1.0f, 50.0f));
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
        if (w.progress.valid) {
            ImGui::Text("%.1f%%   %.2f s/curve   ETA %.0f s", w.progress.pct,
                        w.progress.s_per_curve, w.progress.eta_s);
            if (!w.hist_speed.empty()) {
                ImGui::PlotLines(("##speed" + std::to_string(w.index)).c_str(),
                                 w.hist_speed.data(), static_cast<int>(w.hist_speed.size()),
                                 0, loc_.t("workers", "speed_history").c_str(), 0.0f, 0.0f,
                                 ImVec2(-1.0f, 60.0f));
            }
            if (!w.hist_pct.empty()) {
                ImGui::PlotLines(("##pct" + std::to_string(w.index)).c_str(), w.hist_pct.data(),
                                 static_cast<int>(w.hist_pct.size()), 0,
                                 loc_.t("workers", "progress_history").c_str(), 0.0f, 100.0f,
                                 ImVec2(-1.0f, 60.0f));
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
