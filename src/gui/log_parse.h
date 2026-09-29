#pragma once

// Parsing of the driver's stdout/stderr into the pieces the UI needs
// (docs/DEV_ECM_GUI.md sections 7.2 / 7.3).
//
// Everything here is a pure function over one line, so it is unit-testable without
// a GPU, a window or a worker process (see src/gui/log_parse_test.cpp).
//
// Facts this encodes (all measured from the driver source):
//   * every line printed by ecm_ts_fprintf() carries a "[YYYY-MM-DD HH:MM:SS] "
//     prefix (src/opencl_ecm_log.cpp:39);
//   * ANSI SGR codes ARE written into a pipe (default progress colour is
//     "\033[36m", src/opencl_ecm_log.cpp:27), so they must be stripped or shown;
//   * progress lines come in two shapes with the same fields:
//       stage1: [bar] 42.3%  42.3/100 (~1.20 s/curve)  elapsed 51.0s  ETA 69.0s
//       GPU: [bar] 42.3%  123456, +789 bits (~1.20 s/curve)  elapsed 51.0s  remaining 69.0s
//     (src/core/ecm_driver.cpp:1263 and kernels/cuda/cgbn_stage1.cu:102);
//   * a finished queue prints "===== queue done, N task(s) processed ====="
//     (src/core/ecm_driver.cpp, end of run_queue_manager());
//   * a hit prints "factor[i]=<decimal>" and "FACTOR FOUND aid=... task=...".

#include <string>

namespace ecmgui {

enum class LogKind {
    Raw,        // everything else: shown only when "raw output" is on
    Progress,   // stage1:/GPU: progress lines -> status columns, not the log
    Event,      // START:, Checkpoint, Resuming, FACTOR FOUND -> the log pane
    Error,      // ERROR: / # ERROR / FATAL: / warnings
    QueueDone,  // the queue manager finished its worktodo section
};

struct ProgressInfo {
    bool valid = false;
    bool gpu = false;                    // "GPU:" line instead of "stage1:"
    double pct = 0.0;
    double s_per_curve = 0.0;
    double elapsed_s = 0.0;
    double eta_s = 0.0;                  // "ETA 69.0s" or "remaining 69.0s"
    unsigned long long curves_done = 0;  // 42.3 -> 42, 123456 -> 123456
    unsigned long long curves_total = 0; // only in the stage1: form
    unsigned long long bits = 0;         // "+789 bits"
};

// One hit as reported by the driver's D3 hit line (docs/DEV_ECM_GUI.md section 11):
//   factor[i]=<decimal> curve=<i> sigma=<64-bit> param=<p> method=<m> save=<name>
// Only `factor` is guaranteed: the extra fields are optional on the line, and the CPU
// back-ends also print their own (different) hit lines, which set `factor` alone.
struct HitInfo {
    std::string factor;                  // decimal factor
    int curve = -1;
    unsigned long long sigma = 0;
    bool has_sigma = false;
    int param = -1;
    std::string method;                  // gpu | edwards | mont
    std::string save;                    // save-file name (may be empty)
};

// One `p95_add:` notice, i.e. the outcome of handing a finished task to Prime95 through
// worktodo.add (docs/DEV_ECM_GUI.md section 13). The driver emits exactly four shapes:
//   p95_add: ready workers="1-8" file="<...>" pending=<n>
//   p95_add: ok worker=<n> added=<n> pending_delivered=<n> file="<...>"
//   p95_add: warn worker=<n> added=<n> pending_delivered=<n> file="<...>" note="<text>"
//   p95_add: pending worker=<n> lines=<n> file="<...>" error="<text>"
// Values are quoted; the escaping is exactly what the driver's p95_quote() writes:
// `\"` is a quote in the text, `\\` is one backslash, and any other `\x` is literal (so a
// Windows path is NOT doubled and reads normally in the log pane).
struct P95Notice {
    enum class Level { Ready, Ok, Warn, Pending };
    bool valid = false;
    Level level = Level::Ready;
    int worker = 0;                       // section the line went to (0 = no header)
    long long added = 0;                  // lines appended by that delivery
    long long pending_delivered = 0;      // parked lines re-delivered with it
    long long lines = 0;                  // lines still parked (Pending shape)
    long long pending_at_start = 0;       // waiting when the worker started (Ready shape)
    std::string file;                     // worktodo.add that was written
    std::string note;                     // why the routing fell back (Warn)
    std::string error;                    // failure reason (Pending)
};

struct ParsedLine {
    LogKind kind = LogKind::Raw;
    std::string text;        // the line as received, ANSI codes removed
    std::string plain;       // text without the "[timestamp] " prefix
    std::string timestamp;   // "2026-09-28 18:42:41", empty when absent
    ProgressInfo progress;
    bool is_hit = false;
    std::string factor;      // decimal factor of a hit ("factor[i]=<dec>")
    HitInfo hit;             // full D3 hit details (factor + friends)
    bool queue_done = false;
    long long tasks_processed = -1;
    std::string start_line;  // the worktodo line of a START: event
    P95Notice p95;           // set when the line is a `p95_add:` notice
    // Set when the line proves the worker executable does NOT understand the queue-mode
    // invocation (`-ini … --worker N`). Measured case (2026-09-28): a driver built before
    // D1/D2 puts `--worker` into its positional list, takes the single-run path and dies
    // with "No input number on stdin" + a "mode: cpu-stub" banner -- which the GUI used to
    // report only as "worker keeps restarting". See kOldDriverHint.
    bool old_driver = false;
};

// What a `--worker`-unaware driver prints; the GUI turns `old_driver` into this hint.
extern const char *const kOldDriverHint;

// Removes ANSI escape sequences (CSI ... final byte, and the 2-byte forms).
std::string strip_ansi(const std::string &s);

// Splits "[2026-09-28 18:42:41] rest" -> timestamp + rest (rest = whole input when
// there is no timestamp). A malformed prefix is left in `plain`.
void split_timestamp(const std::string &s, std::string &timestamp, std::string &plain);

ParsedLine parse_line(const std::string &line);

// Incremental splitter for a byte stream from a pipe: handles "\r\n", lone "\n"
// and lone "\r" (the driver's in-place TTY updates). Returns complete lines only.
class LineSplitter {
public:
    void push(const char *data, std::size_t len);
    bool next(std::string &line);
    std::size_t pending() const { return buffer_.size(); }

private:
    std::string buffer_;
};

} // namespace ecmgui
