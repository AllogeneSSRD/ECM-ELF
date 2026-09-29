#pragma once

// Platform boundary of ecm_gui (docs/DEV_ECM_GUI.md section 3).
//
// Everything the UI needs that is NOT portable lives behind these declarations:
// the window/message loop and font/shell helpers (`Platform`, implemented in
// platform_win32.cpp), worker processes (`WorkerProc`, milestone M2) and GPU
// sampling (`GpuMonitor`, milestone M4). The UI code (app.cpp) only ever touches
// this header, so a future Linux port adds platform_posix.cpp + the GLFW/OpenGL3
// backends without touching the panels.
//
// ASCII-only on purpose: the vendored/CUDA toolchain treats a BOM-less source with
// CJK comments as GBK, which eats the next line of code (see tools/README.md).

#include <cstdint>
#include <string>
#include <vector>

namespace ecmgui {

// ---------------------------------------------------------------------------
// Command line (parsed once from the real Win32 command line)
// ---------------------------------------------------------------------------

// Arguments of the running process, argv[0] excluded.
std::vector<std::string> command_line_args();
bool has_arg(const std::vector<std::string> &args, const std::string &name);
// Value of `name` (as "--name value"); empty when absent.
std::string arg_value(const std::vector<std::string> &args, const std::string &name);

// Newest mtime (seconds since the Unix epoch, 0 when none) of the driver's checkpoint
// files in `dir`: "<dir>\.ecm_ckpt_*.dat". Used by the graceful stop (docs/DEV_ECM_GUI.md
// 5.6): the GUI may only terminate a worker once a checkpoint newer than the stop request
// has appeared, otherwise up to one checkpoint interval of work would be lost.
// `name_out` (optional) receives the file name that was found.
long long newest_checkpoint_mtime(const std::string &dir, std::string *name_out);

// ---------------------------------------------------------------------------
// Platform services (implemented today)
// ---------------------------------------------------------------------------

// Directory containing the running ecm_gui.exe, WITHOUT a trailing separator.
// Empty string when it cannot be determined.
std::string exe_dir();

std::string path_join(const std::string &a, const std::string &b);
bool file_exists(const std::string &path);
std::string file_name(const std::string &path);

// Window setting names used by the [GUI] section, so the UI and the layout code
// cannot drift apart.
std::string default_ini_path();             // <exe dir>/ecm.ini
std::string default_localization_dir();     // <exe dir>/localization

// CJK-capable UI font search. Returns the first existing candidate ("" = none,
// in which case the UI falls back to the built-in ASCII font and the language is
// forced back to English -- docs/DEV_ECM_GUI.md 10.3).
std::vector<std::string> cjk_font_candidates();
std::string find_cjk_font_file();
// Latin system UI font (Segoe UI, Tahoma, Arial...): the built-in bitmap font is only
// ~13 px and scales badly, so a scalable system font is preferred when available.
std::vector<std::string> ui_font_candidates();
std::string find_ui_font_file();
// DPI scale of a window (1.0 = 96 dpi; 1.5 on a 150 % display). 1.0 when unavailable.
float window_dpi_scale(void *hwnd);

// Opens a folder (or selects a file) in the OS file manager. Used by the worker
// "open saves folder" button.
bool open_in_explorer(const std::string &path);

// Runs a short command and captures its stdout (used by the worktodo generator to read
// `ecm_cuda.exe --gpu-info`). No console window appears (CREATE_NO_WINDOW), the child is
// killed when `timeout_ms` passes, and only text on stdout is collected -- diagnostics go
// to stderr and are left alone. Returns false when the process cannot start or times out.
bool run_capture(const std::string &command_line, std::string &out, int &exit_code,
                 int timeout_ms = 20000);

// Native "open file" dialog (comdlg32). Returns false when the user cancels.
// `filter` uses the Win32 double-NUL format, e.g. "Text\0*.txt\0All\0*.*\0\0".
bool browse_for_file(std::string &path, const std::string &title, const std::string &filter);

// ---------------------------------------------------------------------------
// Worker processes -- milestone M2 (docs/DEV_ECM_GUI.md section 5)
//
// One worker == one `ecm_cuda.exe -ini <ini> --worker N` process:
//   * spawned with CREATE_NO_WINDOW, never through cmd.exe;
//   * stdout and stderr share ONE pipe write end, so the reader sees the real
//     interleaving and needs a single thread;
//   * a Job object with KILL_ON_JOB_CLOSE, so closing the GUI never leaves an
//     orphan holding the GPU;
//   * priority from [GUI] priority=;
//   * restart with a 5 s backoff and a 3-failures-in-5-minutes breaker.
// ---------------------------------------------------------------------------

struct WorkerSpawn {
    std::string exe;                 // ecm_cuda.exe / ecm.exe (absolute)
    std::string ini;                 // absolute path of ecm.ini
    int worker_index = 1;            // becomes --worker N (ini + worktodo section)
    std::string priority = "below_normal";
    std::string working_dir;         // empty = the GUI's own directory
    std::vector<std::string> extra_args;
};

enum class WorkerRunState { Stopped, Starting, Running, Restarting, Error, QueueEmpty };

struct WorkerStatus {
    WorkerRunState state = WorkerRunState::Stopped;
    unsigned long exit_code = 0;
    int restarts = 0;
    bool breaker_tripped = false;
};

// ---------------------------------------------------------------------------
// GPU sampling -- milestone M4
// ---------------------------------------------------------------------------

struct GpuInfo {
    int index = 0;
    std::string name;
    // CUDA cores as reported by nvmlDeviceGetNumGpuCores (7680 on a 60-SM Ada card,
    // i.e. 128 cores per SM) -- NOT the SM count. The SM count used by the gpucurves
    // recommender comes from the driver's --gpu-info switch (D4), which asks CUDA.
    int cores = 0;
    unsigned int mem_total_mb = 0;
};

struct GpuSample {
    bool valid = false;
    int util_gpu = 0;                 // percent
    int util_mem = 0;                 // percent
    unsigned int clock_sm_mhz = 0;
    unsigned int clock_mem_mhz = 0;
    double power_w = 0.0;
    double power_limit_w = 0.0;
    int temp_c = 0;
    unsigned int mem_used_mb = 0;
    unsigned long long throttle_reasons = 0;   // NVML bitfield
    unsigned long long ts_ms = 0;
};

} // namespace ecmgui
