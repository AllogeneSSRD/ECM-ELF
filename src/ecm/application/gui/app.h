#pragma once
#include "../configuration/generated/ecm_config_generated.h"

// Stage1/Stage2 worker supervision, readable progress, configuration and results.

#include "gpu_monitor.h"
#include "gen_input_text.h"
#include "probability_panel.h"
#include "toml_file.h"
#include "localization.h"
#include "log_parse.h"
#include "results.h"
#include "worker_controller.h"
#include "worktodo_gen.h"

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ecmgui {

bool run_worker_log_scroll_selftest(std::string &error);

// Telemetry charts go through App::draw_metric_plot so they all look
// and read the same way: rounded card, subtle grid, filled area under the curve, the current
// value + unit as the headline, min/max on the axis, an optional reference line (e.g. a power
// limit) and a hover read-out with a vertical guide.
struct MetricPlot {
    std::string id;            // must be unique (the "##" prefix is added internally)
    std::string label;         // headline, e.g. "Power"
    std::string unit;          // e.g. "W", "%", "MHz", "s/curve"
    unsigned int color = 0;   // curve + fill colour, IM_COL32(...) (0 = a default blue);
                              // unsigned int, not ImU32: this header deliberately does not
                              // include imgui.h (same reason App never uses ImGui types here)
    float height = 58.0f;
    bool fixed_range = false;  // true: use lo/hi as given; false: pad the observed window
    float lo = 0.0f;           // fixed range (or the fallback when the series is short)
    float hi = 100.0f;
    float ref = 0.0f;          // reference line value (NAN = none)
    bool has_ref = false;
    std::string ref_label;     // drawn next to the reference line
    std::string empty_text;    // shown while there are not enough samples yet
    int decimals = 1;          // decimals for the headline value
    float ms_per_sample = 0.0f; // hover read-out: age of a sample (0 = unknown)
};

class App {
public:
    // Reads the ini (default: <exe dir>/ecm.toml) and the localization files.
    // Returns false with `err` only when the ini cannot be read at all; a missing
    // localization file is reported through status() but is not fatal.
    bool init(const std::string &ini_path, const std::string &language, std::string &err);

    // Called by the imgui backend once the context exists (loads the layout blob).
    void on_imgui_ready();
    // Stops the workers and saves the layout + settings back into the ini.
    bool shutdown(std::string &err, bool save_docking_layout = true);

    void draw();
    // Optional hook so the host can mirror worker state transitions into --trace.
    // Lines produced before the hook is installed (App::init runs first) are buffered
    // and flushed here, so nothing is lost depending on the call order.
    void set_trace(std::function<void(const std::string &)> fn);
    // Set by File -> Quit (or WM_CLOSE): the host polls it and closes the window.
    bool quit_requested() const { return controller_.quit_requested(); }
    // ---- exit flow -----------------------------------------------------------------
    // Closing the window while workers run must NOT silently kill them: request_close()
    // opens the confirmation modal and returns false (the host keeps the window alive);
    // it returns true only when there is nothing to stop. On consent the workers are asked
    // to checkpoint first, and the GUI waits for that (bounded) before terminating them.
    // [GUI] exit_confirm = ask | stop | kill, [GUI] graceful_stop_ms = <ms>.
    bool request_close();
    Localization &loc() { return loc_; }
    const Localization &loc() const { return loc_; }
    TomlFile &ini() { return ini_; }
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
    // Discard geometry from a minimized/off-screen session before ImGui loads it.
    void discard_saved_layout();
    int refresh_hz() const { return refresh_hz_; }

private:
    void rebuild_workers();
    void draw_menu_bar();
    // Prominent full-width strip under the menu bar: the Prime95 handoff state in red /
    // yellow / green / grey, with the two "open" buttons (docs/usage/GUI.md 18).
    // `height` is the row reserved for it and `host_top` the top of the dockspace host,
    // which the strip must stay above -- otherwise the docked panels paint over it.
    void draw_p95_notice(float height, float host_top);
    // Worktodo generator (M6 scope A, docs/usage/GUI.md 19): paste assignments, preview
    // the queue the generator would write, then append it with a re-validated mtime.
    void draw_gen_panel();
    void draw_probability_panel();
    // The one line-chart implementation every panel uses (see MetricPlot).
    // `used_lo`/`used_hi` (optional) receive the y-range the chart actually drew with, which
    // the GPU history trace reports so a test can check the range covers the data.
    void draw_metric_plot(const MetricPlot &p, const std::vector<float> &values,
                          float *used_lo = nullptr, float *used_hi = nullptr);
    void gen_make_preview();
    void gen_apply();
    // The file the generated queue is appended to (the ini's `worktodo`, resolved against
    // the worker executable's directory -- what the driver itself uses).
    std::string genWorktodoPath() const;
    void draw_workers_table();
    void draw_stage2_status(const WorkerView &worker,bool compact=false);
    std::string task_text(const WorkerView &worker) const;
    bool select_workflow_ini(const std::string &path);
    void draw_gpu_panel();
    void draw_results_panel();
    void draw_detail_panel();
    void draw_worker_panes();
    void capture_layout(bool save_docking_layout = true);
    // Builds the initial dock layout the first time (or when the stored one predates
    // the dockspace): without it every panel opens stacked at the same spot.
    void build_default_layout(unsigned int dockspace_id);
    // One-shot trace of the panel rectangles, so a script can prove the panels are
    // arranged (and not overlapping) without a screenshot.
    void trace_panel_rects();
    // Exit confirmation + graceful-stop modal (drawn from draw()). Returns true while the
    // modal is open, i.e. the window must not close yet.
    bool draw_exit_modal();
    // ---- Prime95 handoff ([queue] p95_worktodo_path / p95_add_workers) ---------------
    // Read from the ini by the GUI itself (not from a worker's output), so the "not
    // configured" hint is visible before any worker runs.
    std::string p95_worktodo_path_;       // empty = the handoff is off
    std::string p95_add_workers_;         // "" | "3" | "1,3" | "1-8" | "auto"
    std::string p95_trace_last_;          // last traced notice-strip text (change detect)
    std::map<std::string, std::string> plot_trace_last_;  // per chart: last traced range
    std::map<std::string, unsigned long long> plot_trace_ms_;  // per chart trace rate limit
    std::map<std::string, int> plot_trace_count_;            // per chart: lines emitted
    std::map<int, std::string> gpu_limit_trace_;             // per device: last traced power limit
    float p95_strip_h_ = 0.0f;            // the strip's real height, reserved in App::draw
    std::string menu_bar_trace_last_;     // top-band geometry trace (change detect)
    // ---- worktodo generator (M6 scope A) --------------------------------------------
    GenInputText gen_input_;              // editable buffer and loaded file content
    std::string gen_input_path_;          // last file loaded into it ("" = pasted)
    std::string gen_set_b1_;              // blank keeps each assignment's B1
    std::string gen_set_b2_ = "0";        // GUI default overrides every assignment's B2
    std::string gen_save_pattern_ = "{curve}_{target}_{b1}.save";
    std::string gen_curve_type_ = "m0";
    std::string gen_sort_by_ = "n";
    int gen_blocks_per_sm_ = 2;           // the `n` in curves = n * sm_count * ipb
    bool gen_use_recommended_ = true;     // default ON
    bool gen_dedup_ = true;
    int gen_target_device_ = 0;
    int gen_worker_index_ = 1;
    std::string gen_text_;                // the preview (exactly what apply would write)
    std::string gen_status_;              // one-line result of the last action
    std::string gen_warnings_;            // rejected input lines, warnings
    FileStamp gen_stamp_;                 // target state when the preview was made
    std::string gen_target_;              // the file the preview would be appended to
    bool gen_has_preview_ = false;
    ProbabilityPanel probability_panel_;
    bool show_probability_=true,probability_focus_requested_=false;
    int gen_lines_ = 0, gen_segments_ = 0, gen_duplicates_ = 0, gen_read_ = 0,
        gen_skipped_ = 0;
    std::string gen_trace_last_;
    std::string p95AddPath() const;       // <dir of p95_worktodo_path>\worktodo.add
    std::string p95PendingPath() const;   // <worker dir>\p95_add_pending.txt
    std::string p95WorktodoPath() const;  // the ini value resolved against the worker dir
    long long p95PendingCount();          // cached count; external file checked every 500 ms
    // One-shot measurement of the default font (can it draw CJK?), traced so the tests
    // can check "Chinese renders" without a human looking at the screen.
    void trace_font_metrics();
    // Periodically traces what the GPU history actually holds (sample count, min/max and
    // the number of DISTINCT values per series), so "the power/clock curve is a dead flat
    // line" can be told apart from "the sampled signal really is constant".
    void trace_gpu_history(int dev, const std::vector<GpuSample> &hist, float p_lo, float p_hi,
                           float c_lo, float c_hi);
    void trace(const std::string &line) const;
    void refresh_result_sources();
    void copy_result_rows(const std::vector<std::size_t> &indices);

    TomlFile ini_;
    WorkerController controller_{ini_};
    const std::vector<WorkerView> &workers_=controller_.workers();
    struct WorkerUi {bool show_raw=false,follow_output=true,show_pane=true;unsigned long long hit_flash_until=0;};
    std::map<int,WorkerUi> worker_ui_;
    Localization loc_;
    GpuMonitor gpu_;
    ResultStore result_;
    ResultFilter result_filter_;
    char result_search_[256]{};
    std::set<std::string> result_selected_;
    std::string result_focus_;
    std::string result_error_;
    unsigned long long result_refresh_at_=0;
    std::string ini_path_;
    std::string loc_dir_;
    std::string language_ = ecm_config::defaults::gui_language;
    std::string status_;
    long long p95_pending_cached_ = 0;
    unsigned long long p95_pending_next_scan_ms_ = 0;
    std::string layout_blob_loaded_;
    std::string worker_exe_;              // [GUI] exe= (empty = auto)
    std::string stage2_exe_;
    float font_size_px_ = ecm_config::defaults::gui_font_size;           // [GUI] font_size= (0 = auto)
    bool font_snap_ = ecm_config::defaults::gui_font_snap;               // [GUI] font_snap= (PixelSnapH)
    std::string font_path_;               // [GUI] font= (empty = pick automatically)
    bool font_reload_requested_ = false;  // language changed: re-apply the font
    bool font_metrics_traced_ = false;
    std::vector<int> gpu_hist_reported_;   // per device: sample count at the last trace
    int selected_worker_ = 0;
    int refresh_hz_ = ecm_config::defaults::gui_refresh_hz;
    int gpu_poll_ms_ = ecm_config::defaults::gui_gpu_poll_ms;
    std::string priority_ = ecm_config::defaults::gui_priority;
    int num_workers_ = ecm_config::defaults::gui_NumWorkers;
    bool layout_dirty_ = false;
    // [GUI] exit_confirm = ask | stop | kill (default ask)
    std::string exit_confirm_ = ecm_config::defaults::gui_exit_confirm;
    // [GUI] graceful_stop_ms: how long to wait for a fresh checkpoint before terminating.
    unsigned long long graceful_stop_ms_ = ecm_config::defaults::gui_graceful_stop_ms;
    bool layout_built_ = false;          // default dock layout applied this run
    std::string start_tab_id_;           // [GUI] start_tab -> window id of the wanted tab
    bool start_tab_applied_ = false;     // the selection was applied (see draw())
    int panels_traced_at_frame_ = -1;    // frame counter when the rects were traced
    int frame_counter_ = 0;
    std::function<void(const std::string &)> trace_;
    std::vector<std::string> pending_trace_;
    // Main window rectangle (Win32 owns it; we persist it in [GUI] window=).
    int win_x_ = ecm_config::defaults::gui_window[0], win_y_ = ecm_config::defaults::gui_window[1], win_w_ = ecm_config::defaults::gui_window[2], win_h_ = ecm_config::defaults::gui_window[3];
};

} // namespace ecmgui
