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
        emit(g_log_mirror);
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
