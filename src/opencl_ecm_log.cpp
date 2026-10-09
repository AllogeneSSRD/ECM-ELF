#include "opencl_ecm_log.h"
#include "opencl_ecm_runtime_config.h"

#include <chrono>
#include <cctype>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

std::mutex g_log_mutex;

// Optional extra output sink (screen.log in queue mode). Guarded by g_log_mutex.
FILE *g_log_mirror = nullptr;

// Progress cadence for the FILE (mirror) only -- ini key `progress_log_seconds`.
//   > 0 : at most one progress line per N seconds lands in the log file
//   = 0 : the log file gets no progress line at all
//   < 0 : every progress line (the pre-D4 behaviour)
// The pipe/console still gets every line (~200 ms cadence): the GUI tails the worker's
// stdout by complete lines and must not wait a minute for the first one. Whatever the
// cadence, a line reporting 100.0% is always written, so the file always shows the end of
// a task. See docs/usage/GUI.md
double g_progress_log_seconds = 60.0;
double g_last_progress_log_s = -1.0e18;

// Wall-clock seconds for the file gate. Monotonic (steady_clock) on purpose: a system
// clock jump (NTP, DST) must not silence the log for an hour or unlock a burst of lines.
double now_seconds_monotonic() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Both progress shapes carry an ASCII bar plus a percentage:
//     GPU: [====>       ]  42.1%  ...   (CUDA / OpenCL stage 1)
//     stage1: [====>    ]  42.1%  ...   (CPU stage 1, ecm_driver.cpp)
// Anything else (banners, results, warnings) is never a progress line.
bool line_is_progress(const char *text) {
    if (text == nullptr) return false;
    if (std::strstr(text, "GPU: [") == nullptr && std::strstr(text, "stage1: [") == nullptr) {
        return false;
    }
    return std::strchr(text, '%') != nullptr;
}

// The task-complete line ("... ] 100.0% ...") is written whatever the cadence.
bool line_is_progress_end(const char *text) {
    return line_is_progress(text) && std::strstr(text, " 100.0%") != nullptr;
}

// Does the file get this progress line? Updates the gate when it says yes.
bool file_wants_progress(const char *text) {
    if (g_progress_log_seconds < 0.0) return true;          // every line
    if (line_is_progress_end(text)) return true;            // always the 100% line
    if (g_progress_log_seconds == 0.0) return false;        // never
    const double now = now_seconds_monotonic();
    if (now - g_last_progress_log_s < g_progress_log_seconds) return false;
    g_last_progress_log_s = now;
    return true;
}

// Progress-bar colour (ANSI escape). Default cyan, matches the OpenCL bar.
std::string g_progress_color_code = "\033[36m";

std::string timestamp_prefix() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    std::ostringstream oss;
    oss << "[" << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << "] ";
    return oss.str();
}

class timestamped_streambuf : public std::streambuf {
public:
    explicit timestamped_streambuf(std::streambuf *target) : target_(target) {}

protected:
    int overflow(int ch) override {
        if (ch == traits_type::eof()) {
            return target_->sputc(ch);
        }
        std::lock_guard<std::mutex> lk(g_log_mutex);
        if (at_line_start_) {
            std::string p = timestamp_prefix();
            target_->sputn(p.data(), static_cast<std::streamsize>(p.size()));
            if (g_log_mirror) {
                std::fwrite(p.data(), 1, p.size(), g_log_mirror);
            }
            at_line_start_ = false;
        }
        target_->sputc(static_cast<char>(ch));
        if (g_log_mirror) {
            std::fputc(static_cast<char>(ch), g_log_mirror);
        }
        if (ch == '\n') {
            at_line_start_ = true;
            if (g_log_mirror) {
                std::fflush(g_log_mirror);
            }
        }
        return ch;
    }

    int sync() override {
        return target_->pubsync();
    }

private:
    std::streambuf *target_;
    bool at_line_start_ = true;
};

timestamped_streambuf *g_cout_buf = nullptr;
timestamped_streambuf *g_cerr_buf = nullptr;
bool g_installed = false;

} // namespace

void ecm_log_set_mirror(FILE *mirror) {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    g_log_mirror = mirror;
}

void ecm_log_set_progress_color(const char *name) {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    g_progress_color_code.clear();
    if (name == nullptr || *name == '\0') {
        return;
    }
    std::string n = name;
    for (char &c : n) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (n == "none") {
        // stay empty
    } else if (n == "red") {
        g_progress_color_code = "\033[31m";
    } else if (n == "green") {
        g_progress_color_code = "\033[32m";
    } else if (n == "yellow") {
        g_progress_color_code = "\033[33m";
    } else if (n == "blue") {
        g_progress_color_code = "\033[34m";
    } else if (n == "magenta") {
        g_progress_color_code = "\033[35m";
    } else if (n == "cyan") {
        g_progress_color_code = "\033[36m";
    } else if (n == "white") {
        g_progress_color_code = "\033[37m";
    } else if (n == "grey" || n == "gray") {
        g_progress_color_code = "\033[90m";
    }
    // unknown name → empty (no colour)
}

const char *ecm_log_progress_color_code() {
    return g_progress_color_code.c_str();
}

const char *ecm_log_progress_color_reset() {
    return g_progress_color_code.empty() ? "" : "\033[0m";
}

void ecm_enable_console_ansi() {
#ifdef _WIN32
    const HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE || h == nullptr) {
        return;
    }
    DWORD mode = 0;
    if (GetConsoleMode(h, &mode)) {
        SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
#else
    (void)0;
#endif
}

bool ecm_log_timestamp_enabled() {
    return ecm_runtime_config().log_timestamp;  // default ON; Use --no-log-timestamp to disable
}

void ecm_log_set_progress_log_seconds(double seconds) {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    g_progress_log_seconds = seconds;
    g_last_progress_log_s = now_seconds_monotonic();   // the next window starts now
}

double ecm_log_progress_log_seconds() {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    return g_progress_log_seconds;
}

void ecm_install_timestamped_iostreams() {
    if (!ecm_log_timestamp_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_log_mutex);
    if (g_installed) {
        return;
    }
    g_cout_buf = new timestamped_streambuf(std::cout.rdbuf());
    g_cerr_buf = new timestamped_streambuf(std::cerr.rdbuf());
    std::cout.rdbuf(g_cout_buf);
    std::cerr.rdbuf(g_cerr_buf);
    g_installed = true;
}

int ecm_ts_vfprintf(FILE *stream, const char *fmt, va_list ap) {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    const std::string p = ecm_log_timestamp_enabled() ? timestamp_prefix() : std::string();

    const auto emit = [&](FILE *f) {
        if (!p.empty()) {
            std::fputs(p.c_str(), f);
        }
        va_list apc;
        va_copy(apc, ap);
        std::vfprintf(f, fmt, apc);
        va_end(apc);
        std::fflush(f);
    };

    emit(stream);

    if (g_log_mirror != nullptr && g_log_mirror != stream) {
        /* The file is rate-limited for progress lines (ini progress_log_seconds): a task
           running for hours would otherwise fill screen.log with a line every 200 ms.
           The line has to be formatted HERE to be classified, so this renders into a
           buffer first and falls back to a second vfprintf for anything longer. */
        char buf[4096];
        va_list apc;
        va_copy(apc, ap);
        const int n = std::vsnprintf(buf, sizeof(buf), fmt, apc);
        va_end(apc);
        const bool fits = (n >= 0) && (static_cast<size_t>(n) < sizeof(buf));
        if (!fits || !line_is_progress(buf) || file_wants_progress(buf)) {
            emit(g_log_mirror);
        }
    }
    return 0;
}

int ecm_ts_fprintf(FILE *stream, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int rc = ecm_ts_vfprintf(stream, fmt, ap);
    va_end(ap);
    return rc;
}
