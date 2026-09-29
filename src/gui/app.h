#pragma once

// Application state and panels of ecm_gui (milestone M1: skeleton).
//
// The panels are real (docking + viewports, geometry persisted into [GUI]), but
// the DATA is deliberately config-driven for now: the worker list comes from
// NumWorkers + the [Worker #N] sections of ecm.ini, the log panes say so instead
// of inventing lines, and the GPU panel reports what the sample source gives it
// (nothing until M4 wires NVML). M2 replaces the placeholders with live process
// output; nothing in this header needs to change for that.

#include "gpu_monitor.h"
#include "ini_file.h"
#include "localization.h"
#include "log_parse.h"
#include "results.h"
#include "worker_proc.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ecmgui {

// Worker state -> the plain name (used by --trace and the docs) and the localization id
// under the "workers" panel. The ids are an EXPLICIT table (not derived from the name):
// deriving them produced "state__stopped" (double underscore from the leading capital),
// which is not in the XML files, so the table showed the raw fallback text
// "workers.state__stopped" instead of "Stopped" (user report, 2026-09-28).
// --selftest checks that every id here resolves in every localization file.
const char *state_name(WorkerRunState s);
const char *state_key(WorkerRunState s);
// Every state, for the self-test's completeness check.
const WorkerRunState *all_worker_states(int *count);

// One row of the Workers table plus its log pane.
struct WorkerView {
    int index = 1;                    // [Worker #N] number (also --worker N)
    std::string name;                 // [Worker #N] name= (GUI-only key)
    int device = 0;                   // effective device= after section override
    std::string method = "gpu";
    int gpucurves = 0;
    std::string worktodo;
    std::string log_file;
    std::string extra_args;           // [Worker #N] extra_args= (space separated)
    bool autostart = false;

    // Live state (M2).
    std::unique_ptr<WorkerProc> proc;
    ProgressInfo progress;
    long long progress_lines = 0;
    long long tasks_done = 0;
    long long hits = 0;
    std::string task;                 // current worktodo line (from START:)
    std::vector<std::string> events;  // event layer (timestamped, ANSI-free)
    std::vector<std::string> raw;     // everything, for the "raw output" toggle
    // Progress history for the sparkline (one sample per accepted progress line).
    std::vector<float> hist_pct;
    std::vector<float> hist_speed;    // curves/s (1 / s_per_curve)
    unsigned long long last_sample_ms = 0;
    double last_traced_pct = -1.0;
    bool show_raw = false;
    bool show_pane = true;
    bool duplicate_device = false;    // another worker uses the same device (warning)
    bool old_driver = false;          // worker exe printed "No input number on stdin"
    // Percent the driver reported when it resumed from a checkpoint ("Resuming from
    // checkpoint: 23.8% complete"). A resumed run starts its progress line only after a
    // long batch gap (see docs/DEV_ECM_GUI.md 7.2), so the GUI shows this instead of an
    // empty bar in the meantime. -1 = no resume seen.
    double resume_pct = -1.0;
    // ---- graceful stop (per worker) ------------------------------------------------
    // A stop request does NOT kill the process right away: the driver checkpoints at its own
    // interval, so the worker is terminated only after a checkpoint newer than the request
    // appeared (or the deadline passed). See App::request_stop / docs/DEV_ECM_GUI.md 5.6.
    bool stop_requested = false;
    unsigned long long stop_deadline_ms = 0;
    long long ckpt_baseline = 0;      // newest checkpoint mtime when the stop was requested
    bool ckpt_seen = false;           // a newer checkpoint appeared -> safe to terminate
    unsigned long long hit_flash_until = 0;  // row highlight after a hit
    std::string effective_config;     // "effective config" text for the detail panel
};

class App {
public:
    // Reads the ini (default: <exe dir>/ecm.ini) and the localization files.
    // Returns false with `err` only when the ini cannot be read at all; a missing
    // localization file is reported through status() but is not fatal.
    bool init(const std::string &ini_path, const std::string &language, std::string &err);

    // Called by the imgui backend once the context exists (loads the layout blob).
    void on_imgui_ready();
    // Stops the workers and saves the layout + settings back into the ini.
    bool shutdown(std::string &err);

    void draw();
    // Drives the worker supervisor (called once per frame from draw()).
    void tick();

    // Optional hook so the host can mirror worker state transitions into --trace.
    // Lines produced before the hook is installed (App::init runs first) are buffered
    // and flushed here, so nothing is lost depending on the call order.
    void set_trace(std::function<void(const std::string &)> fn);
    // Set by File -> Quit (or WM_CLOSE): the host polls it and closes the window.
    bool quit_requested() const { return quit_requested_; }
    // ---- exit flow -----------------------------------------------------------------
    // Closing the window while workers run must NOT silently kill them: request_close()
    // opens the confirmation modal and returns false (the host keeps the window alive);
    // it returns true only when there is nothing to stop. On consent the workers are asked
    // to checkpoint first, and the GUI waits for that (bounded) before terminating them.
    // [GUI] exit_confirm = ask | stop | kill, [GUI] graceful_stop_ms = <ms>.
    bool request_close();
    // Stop button / row action: same graceful path for a single worker.
    void request_stop(int worker_index);

    Localization &loc() { return loc_; }
    const Localization &loc() const { return loc_; }
    IniFile &ini() { return ini_; }
    const std::string &status() const { return status_; }
    // The host may report a startup problem the App cannot know about (e.g. driver-only
    // command line arguments passed to the GUI): it shows up in the status line.
    void set_status(const std::string &s) { status_ = s; }
    const std::string &language() const { return language_; }
    bool needs_cjk_font() const { return loc_.needs_cjk_font(); }
    // [GUI] font_size (0 = auto = DPI-scaled) / [GUI] font (path, empty = pick one) /
    // [GUI] font_snap (snap glyph advances to whole pixels; crisper, see 10.3).
    // A fractional size is allowed: at 150 % the automatic size is 22.5 px.
    float font_size_px() const { return font_size_px_; }
    bool font_snap() const { return font_snap_; }
    const std::string &font_path() const { return font_path_; }

    // ---- runtime font reload ------------------------------------------------------
    // The font is chosen for the UI language, but the language can be switched while
    // the GUI runs (Language menu). Without re-applying the font, switching *to* a CJK
    // language would keep the Latin font and every label would render as "???".
    // set_language()/reload_localization() flag this; the frame loop calls
    // apply_ui_font() (main_win32.cpp) and then clear_font_reload_request().
    bool font_reload_requested() const { return font_reload_requested_; }
    void clear_font_reload_request() { font_reload_requested_ = false; }
    void request_font_reload() { font_reload_requested_ = true; }
    // Called by the font layer when no font we could load can draw the current UI
    // language: fall back to English (readable) instead of showing "???" boxes.
    void fall_back_to_english(const std::string &reason);
    const std::vector<WorkerView> &workers() const { return workers_; }

    // Starts/stops every worker (menu + the per-row buttons).
    void start_all();
    void stop_all();
    // Shutdown variant: terminate the workers immediately, quietly (no trace per worker,
    // no state/crash accounting, one summary line) -- see the user requirement in
    // docs/DEV_ECM_GUI.md 5.6.
    void stop_all_quietly();
    // Resolves the worker executable: [GUI] exe=, else ecm_cuda.exe next to the GUI,
    // then ecm.exe next to it; falls back to the bare name (PATH lookup).
    std::string resolve_worker_exe() const;

    // Reloads the current language file (the "Reload localization" button).
    void reload_localization();
    void set_language(const std::string &language);
    std::vector<std::string> available_languages() const;

    bool show_demo = false;           // Tools menu: only if imgui_demo.cpp is vendored

    // Main window rectangle + refresh rate: the Win32 host owns the real window and
    // feeds its final rectangle back before shutdown so [GUI] window= is updated.
    int window_x() const { return win_x_; }
    int window_y() const { return win_y_; }
    int window_w() const { return win_w_; }
    int window_h() const { return win_h_; }
    void set_window_rect(int x, int y, int w, int h);
    int refresh_hz() const { return refresh_hz_; }

private:
    void rebuild_workers();
    void draw_menu_bar();
    void draw_workers_table();
    void draw_gpu_panel();
    void draw_results_panel();
    void draw_detail_panel();
    void draw_worker_panes();
    void capture_layout();
    // Builds the initial dock layout the first time (or when the stored one predates
    // the dockspace): without it every panel opens stacked at the same spot.
    void build_default_layout(unsigned int dockspace_id);
    // One-shot trace of the panel rectangles, so a script can prove the panels are
    // arranged (and not overlapping) without a screenshot.
    void trace_panel_rects();
    // Exit confirmation + graceful-stop modal (drawn from draw()). Returns true while the
    // modal is open, i.e. the window must not close yet.
    bool draw_exit_modal();
    // Starts the graceful stop for every running worker and switches to the waiting phase.
    void begin_graceful_exit();
    // Advances the per-worker graceful stop (checkpoint watch + deadline) from tick().
    void tick_graceful_stops(unsigned long long now);
    // Directory holding the worker executable: where the driver writes .ecm_ckpt_*.dat.
    std::string worker_dir() const;
    // One-shot measurement of the default font (can it draw CJK?), traced so the tests
    // can check "Chinese renders" without a human looking at the screen.
    void trace_font_metrics();
    // Periodically traces what the GPU history actually holds (sample count, min/max and
    // the number of DISTINCT values per series), so "the power/clock curve is a dead flat
    // line" can be told apart from "the sampled signal really is constant".
    void trace_gpu_history(int dev, const std::vector<GpuSample> &hist, float p_lo, float p_hi,
                           float c_lo, float c_hi);
    std::string effective_config_text(int worker_index) const;
    void trace(const std::string &line) const;
    // Records one hit in the two result files and refreshes the status line.
    void record_hit(const WorkerView &w, const HitInfo &hit);

    IniFile ini_;
    Localization loc_;
    GpuMonitor gpu_;
    ResultsStore results_;
    std::string results_json_;
    std::string results_txt_;
    long long results_baseline_hits_ = 0;   // hits already in the files at startup
    std::vector<WorkerView> workers_;
    std::string ini_path_;
    std::string loc_dir_;
    std::string language_ = "english";
    std::string status_;
    std::string layout_blob_loaded_;
    std::string worker_exe_;              // [GUI] exe= (empty = auto)
    float font_size_px_ = 0.0f;           // [GUI] font_size= (0 = auto)
    bool font_snap_ = true;               // [GUI] font_snap= (PixelSnapH)
    std::string font_path_;               // [GUI] font= (empty = pick automatically)
    bool font_reload_requested_ = false;  // language changed: re-apply the font
    bool font_metrics_traced_ = false;
    std::vector<int> gpu_hist_reported_;   // per device: sample count at the last trace
    int selected_worker_ = 0;
    int refresh_hz_ = 10;
    int gpu_poll_ms_ = 500;
    std::string priority_ = "below_normal";
    int num_workers_ = 1;
    bool layout_dirty_ = false;
    bool autostart_done_ = false;
    bool quit_requested_ = false;
    // Exit flow: Idle -> (Confirm) -> Stopping -> quit_requested_
    enum class ExitPhase { Idle, Confirm, Stopping };
    ExitPhase exit_phase_ = ExitPhase::Idle;
    unsigned long long exit_deadline_ms_ = 0;
    // [GUI] exit_confirm = ask | stop | kill (default ask)
    std::string exit_confirm_ = "ask";
    // [GUI] graceful_stop_ms: how long to wait for a fresh checkpoint before terminating.
    unsigned long long graceful_stop_ms_ = 300000;
    bool layout_built_ = false;          // default dock layout applied this run
    int panels_traced_at_frame_ = -1;    // frame counter when the rects were traced
    int frame_counter_ = 0;
    unsigned long long last_tick_ms_ = 0;
    std::function<void(const std::string &)> trace_;
    std::vector<std::string> pending_trace_;
    // Main window rectangle (Win32 owns it; we persist it in [GUI] window=).
    int win_x_ = 120, win_y_ = 80, win_w_ = 1500, win_h_ = 900;
};

} // namespace ecmgui
