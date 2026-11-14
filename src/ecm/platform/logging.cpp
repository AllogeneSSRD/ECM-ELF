#include "ecm/platform/logging.h"
#include "ecm/platform/log_file.h"
#include "ecm/platform/log_read.h"
#include "io/files.h"
#include <fstream>

bool ecm_visit_log_lines(const std::string &path,const std::function<void(const std::string &)> &visitor) {
    std::ifstream in(path);if(!in)return false;std::string line;
    while(std::getline(in,line))visitor(line);return !in.bad();
}
#include "ecm/application/configuration/runtime.h"

#include <chrono>
#include <cctype>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

std::mutex g_log_mutex;

// Optional extra output sink (screen.log in queue mode). Guarded by g_log_mutex.
FILE *g_log_mirror = nullptr;
std::filesystem::path g_shared_log;
uint64_t g_log_max_kb = 0;
int g_log_worker = 1;
std::string g_stdio_pending;

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

std::string worker_log_prefix(int worker) {
    return "[Worker #" + std::to_string(worker) + "] " + timestamp_prefix();
}

bool mirror_enabled() { return g_log_mirror || !g_shared_log.empty(); }

void write_mirror(std::string_view text) {
    if (!g_shared_log.empty()) {
        try {
            ecm_append_worker_log(g_shared_log, g_log_worker, g_log_max_kb, text);
        } catch (const std::exception &error) {
            std::fprintf(stderr, "WARNING: %s\n", error.what());
        }
        return;
    }
    const auto formatted = ecm_format_worker_log_lines(text, g_log_worker);
    std::fwrite(formatted.data(), 1, formatted.size(), g_log_mirror);
    std::fflush(g_log_mirror);
}

void append_mirror(std::string &pending, std::string_view text) {
    pending.append(text);
    const auto end = pending.rfind('\n');
    if (end != std::string::npos) {
        write_mirror(std::string_view(pending).substr(0, end + 1));
        pending.erase(0, end + 1);
    }
}

void finish_mirror(std::string &pending) {
    if (!pending.empty()) {
        pending += '\n';
        write_mirror(pending);
        pending.clear();
    }
}

class timestamped_streambuf : public std::streambuf {
public:
    explicit timestamped_streambuf(std::streambuf *target) : target_(target) {}
    void finish_file_line() { finish_mirror(pending_); }

protected:
    int overflow(int ch) override {
        if (ch == traits_type::eof()) {
            return target_->sputc(ch);
        }
        std::lock_guard<std::mutex> lk(g_log_mutex);
        if (at_line_start_ && ecm_log_timestamp_enabled()) {
            std::string p = timestamp_prefix();
            target_->sputn(p.data(), static_cast<std::streamsize>(p.size()));
        }
        target_->sputc(static_cast<char>(ch));
        at_line_start_ = ch == '\n';
        if (mirror_enabled()) {
            pending_ += static_cast<char>(ch);
            if (at_line_start_) {
                write_mirror(pending_);
                pending_.clear();
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
    std::string pending_;
};

timestamped_streambuf *g_cout_buf = nullptr;
timestamped_streambuf *g_cerr_buf = nullptr;
bool g_installed = false;

namespace fs = std::filesystem;
using NativeLogName = fs::path::string_type;

bool same_log_name(std::basic_string_view<fs::path::value_type> first,
                   std::basic_string_view<fs::path::value_type> second) {
#ifdef _WIN32
    if (first.empty() || second.empty()) return first.empty() && second.empty();
    return CompareStringOrdinal(first.data(), static_cast<int>(first.size()), second.data(),
                                static_cast<int>(second.size()), TRUE) == CSTR_EQUAL;
#else
    return first == second;
#endif
}

uint64_t next_archive_index(const fs::path &path) {
    const auto parent = path.has_parent_path() ? path.parent_path() : fs::path(".");
    const auto prefix = path.stem().native() + fs::path(".").native();
    const auto suffix = path.extension().native();
    constexpr auto limit = (std::numeric_limits<uint64_t>::max)();
    uint64_t maximum = 0;
    for (const auto &entry : fs::directory_iterator(parent)) {
        const NativeLogName leaf = entry.path().filename().native();
        const std::basic_string_view<fs::path::value_type> name(leaf);
        if (name.size() <= prefix.size() + suffix.size() ||
            !same_log_name(name.substr(0, prefix.size()), prefix) ||
            !same_log_name(name.substr(name.size() - suffix.size()), suffix)) continue;
        const auto digits = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
        uint64_t index = 0;
        bool numeric = true;
        for (const auto digit : digits) {
            if (digit < '0' || digit > '9') { numeric = false; break; }
            const auto value = static_cast<uint64_t>(digit - '0');
            if (index > (limit - value) / 10) {
                index = limit;
            } else {
                index = index * 10 + value;
            }
        }
        if (numeric && index > maximum) maximum = index;
    }
    if (maximum == limit) throw std::runtime_error("log archive index exhausted");
    return maximum + 1;
}

// False means another writer already owns this archive name.
bool move_log_to_archive(const fs::path &path, const fs::path &archive) {
#ifdef _WIN32
    if (MoveFileExW(path.c_str(), archive.c_str(), MOVEFILE_WRITE_THROUGH)) return true;
    const auto error = GetLastError();
    if (error == ERROR_ALREADY_EXISTS || error == ERROR_FILE_EXISTS) return false;
    throw std::system_error(static_cast<int>(error), std::system_category(), "cannot rename log");
#else
    const int reservation = ::open(archive.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (reservation < 0) {
        if (errno == EEXIST) return false;
        throw std::system_error(errno, std::generic_category(), "cannot reserve log archive");
    }
    ::close(reservation);
    std::error_code error;
    fs::rename(path, archive, error);
    if (!error) return true;
    std::error_code ignored;
    fs::remove(archive, ignored);
    throw std::system_error(error, "cannot rename log");
#endif
}

void rotate_log(const fs::path &path) {
    try {
        for (unsigned attempt = 0; attempt < 8; ++attempt) {
            const auto archive = path.parent_path() /
                fs::path(path.stem().native() + fs::path("." + std::to_string(next_archive_index(path))).native() +
                         path.extension().native());
            if (move_log_to_archive(path, archive)) return;
        }
        throw std::runtime_error("log archive names keep changing");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "WARNING: cannot rotate log %s: %s\n", path.u8string().c_str(), error.what());
    }
}

std::ofstream open_log_for_append(const fs::path &path) {
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out) throw std::runtime_error("cannot open log: " + path.u8string());
    return out;
}

} // namespace

std::string ecm_format_worker_log_lines(std::string_view message, int worker) {
    if (message.empty()) return {};
    const auto prefix = worker_log_prefix(worker);
    std::string formatted;
    while (!message.empty()) {
        const auto newline = message.find('\n');
        formatted += prefix;
        if (newline == std::string_view::npos) {
            formatted.append(message);
            break;
        }
        formatted.append(message.substr(0, newline + 1));
        message.remove_prefix(newline + 1);
    }
    return formatted;
}

void ecm_append_worker_log(const std::filesystem::path &path, int worker,
                           uint64_t max_kb, std::string_view message) {
    if (message.empty()) return;
    if (max_kb > (std::numeric_limits<uint64_t>::max)() / 1000)
        throw std::invalid_argument("log size limit overflows bytes");
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    ecm_io::FileLock lock(path.u8string());
    auto size = fs::exists(path) ? fs::file_size(path) : 0;
    bool separator = false;
    if (size) {
        std::ifstream tail(path, std::ios::binary);
        tail.seekg(-1, std::ios::end);
        const auto last = tail.get();
        if (last == std::char_traits<char>::eof())
            throw std::runtime_error("cannot inspect log tail: " + path.u8string());
        separator = last != '\n';
    }
    const auto prefix = worker_log_prefix(worker);
    const auto limit = max_kb * 1000;
    std::ofstream out;
    while (!message.empty()) {
        if (!out.is_open()) out = open_log_for_append(path);
        const auto newline = message.find('\n');
        const auto line = message.substr(0, newline);
        if (separator) { out.put('\n'); ++size; separator = false; }
        out << prefix << line << '\n';
        if (!out) throw std::runtime_error("cannot append log: " + path.u8string());
        size += prefix.size() + line.size() + 1;
        if (limit && size > limit) {
            out.close();
            if (!out) throw std::runtime_error("cannot close log: " + path.u8string());
            rotate_log(path);
            size = fs::exists(path) ? fs::file_size(path) : 0;
        }
        if (newline == std::string_view::npos) break;
        message.remove_prefix(newline + 1);
    }
    if (out.is_open()) {
        out.close();
        if (!out) throw std::runtime_error("cannot close log: " + path.u8string());
    }
}

EcmLogMirror::~EcmLogMirror() {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    if (!file_ && shared_path_.empty()) return;
    finish_mirror(g_stdio_pending);
    if (g_cout_buf) g_cout_buf->finish_file_line();
    if (g_cerr_buf) g_cerr_buf->finish_file_line();
    if (file_) std::fclose(file_);
    g_log_mirror = nullptr;
    g_shared_log.clear();
}

bool EcmLogMirror::open(const std::string &path, int worker, uint64_t max_kb) {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    if (mirror_enabled()) return false;
    if (max_kb) {
        try {
            const auto candidate = fs::u8path(path);
            if (candidate.has_parent_path()) fs::create_directories(candidate.parent_path());
            ecm_io::FileLock lock(path);
            auto log = open_log_for_append(candidate);
            log.close();
            if (!log) return false;
            shared_path_ = candidate;
            g_shared_log = candidate;
            g_log_max_kb = max_kb;
        } catch (const std::exception &) {
            return false;
        }
    } else {
        file_ = std::fopen(path.c_str(), "a");
        if (!file_) return false;
        g_log_mirror = file_;
    }
    g_log_worker = worker;
    return true;
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

    if (mirror_enabled() && g_log_mirror != stream) {
        // Render separately so multiline file output gets a prefix on every line.
        char buf[4096];
        va_list apc;
        va_copy(apc, ap);
        const int n = std::vsnprintf(buf, sizeof(buf), fmt, apc);
        va_end(apc);
        if (n >= 0 && (!line_is_progress(buf) || file_wants_progress(buf))) {
            if (static_cast<size_t>(n) < sizeof(buf)) {
                append_mirror(g_stdio_pending, std::string_view(buf, n));
            } else {
                std::string message(static_cast<size_t>(n) + 1, '\0');
                va_copy(apc, ap);
                const int size = std::vsnprintf(message.data(), message.size(), fmt, apc);
                va_end(apc);
                if (size == n) append_mirror(g_stdio_pending, std::string_view(message).substr(0, n));
            }
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
