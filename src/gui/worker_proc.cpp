#include "worker_proc.h"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ecmgui {

unsigned long long mono_ms() {
#ifdef _WIN32
    return static_cast<unsigned long long>(GetTickCount64());
#else
    return 0;
#endif
}

namespace {

#ifdef _WIN32
DWORD priority_class_from_name(const std::string &name) {
    std::string n;
    for (char c : name) n.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    if (n == "idle") return IDLE_PRIORITY_CLASS;
    if (n == "high") return HIGH_PRIORITY_CLASS;
    if (n == "above_normal" || n == "abovenormal") return ABOVE_NORMAL_PRIORITY_CLASS;
    if (n == "normal") return NORMAL_PRIORITY_CLASS;
    return BELOW_NORMAL_PRIORITY_CLASS;      // default: stay out of the UI's way
}

// Quotes one argument for CreateProcessW's command line (the child parses it with
// the standard rules; paths here routinely contain spaces).
std::wstring widen(const std::string &s) {
    if (s.empty()) return std::wstring();
    const int need = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()),
                                         nullptr, 0);
    std::wstring out(static_cast<std::size_t>(need), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), &out[0], need);
    return out;
}

void append_quoted(std::wstring &cmd, const std::wstring &arg, bool first) {
    if (!first) cmd.push_back(L' ');
    const bool need_quotes = arg.find_first_of(L" \t\"") != std::wstring::npos;
    if (!need_quotes) {
        cmd += arg;
        return;
    }
    cmd.push_back(L'"');
    for (wchar_t c : arg) {
        if (c == L'"') cmd.push_back(L'\\');
        cmd.push_back(c);
    }
    cmd.push_back(L'"');
}

std::string narrow(const std::wstring &s) {
    if (s.empty()) return std::string();
    const int need = WideCharToMultiByte(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()),
                                         nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<std::size_t>(need), '\0');
    WideCharToMultiByte(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), &out[0], need,
                        nullptr, nullptr);
    return out;
}
#endif

} // namespace

WorkerProc::WorkerProc() {
#ifdef _WIN32
    // One job per worker: KILL_ON_JOB_CLOSE ties the whole child tree (and any
    // grandchildren, e.g. a stage-2 helper) to this process's lifetime.
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job != nullptr) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info, sizeof(info));
        job_ = job;
    }
#endif
}

WorkerProc::~WorkerProc() {
    stop();
    if (reader_.joinable()) reader_.join();
    if (job_ != nullptr) {
#ifdef _WIN32
        CloseHandle(static_cast<HANDLE>(job_));
#endif
        job_ = nullptr;
    }
}

void WorkerProc::configure(const WorkerSpawn &spawn, const WorkerLimits &limits) {
    spawn_ = spawn;
    limits_ = limits;
}

std::string WorkerProc::command_line() const {
    std::string cmd = spawn_.exe;
    if (!spawn_.ini.empty()) {
        cmd += " -ini " + spawn_.ini;
    }
    cmd += " --worker " + std::to_string(spawn_.worker_index);
    for (const std::string &a : spawn_.extra_args) {
        cmd += " " + a;
    }
    return cmd;
}

bool WorkerProc::start(std::string &err) {
    err.clear();
    if (running_) return true;
    if (breaker_tripped_) {
        err = "breaker tripped";
        return false;
    }
#ifdef _WIN32
    if (!spawn_locked(err)) {
        state_ = WorkerRunState::Error;
        state_text_ = err;
        return false;
    }
    want_running_ = true;
    saw_queue_done_ = false;
    state_ = WorkerRunState::Running;
    state_text_.clear();
    return true;
#else
    err = "worker processes are Windows-only for now";
    return false;
#endif
}

#ifdef _WIN32
bool WorkerProc::spawn_locked(std::string &err) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) {
        err = "CreatePipe failed";
        return false;
    }
    // The read end must not be inherited by the child, or the pipe never closes.
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = wr;      // stdout and stderr SHARE one write end: the reader
    si.hStdError = wr;       // thread then sees the driver's real write order.
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                             OPEN_EXISTING, 0, nullptr);
    si.hStdInput = nul;

    std::wstring cmd;
    append_quoted(cmd, widen(spawn_.exe), true);
    if (!spawn_.ini.empty()) {
        append_quoted(cmd, L"-ini", false);
        append_quoted(cmd, widen(spawn_.ini), false);
    }
    append_quoted(cmd, L"--worker", false);
    append_quoted(cmd, widen(std::to_string(spawn_.worker_index)), false);
    for (const std::string &a : spawn_.extra_args) {
        append_quoted(cmd, widen(a), false);
    }
    std::vector<wchar_t> mutable_cmd(cmd.begin(), cmd.end());
    mutable_cmd.push_back(L'\0');

    const std::wstring cwd = widen(spawn_.working_dir);
    PROCESS_INFORMATION pi{};
    const DWORD flags = CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT |
                        CREATE_SUSPENDED;   // assign to the job before it can run
    const BOOL ok = CreateProcessW(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE, flags,
                                   nullptr, cwd.empty() ? nullptr : cwd.c_str(), &si, &pi);
    CloseHandle(wr);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) {
        CloseHandle(rd);
        err = "CreateProcessW failed (" + std::to_string(GetLastError()) + ")";
        return false;
    }

    if (job_ != nullptr) {
        if (!AssignProcessToJobObject(static_cast<HANDLE>(job_), pi.hProcess)) {
            // Not fatal (a parent job may already own us), but say so.
            std::fprintf(stderr, "ecm_gui: AssignProcessToJobObject failed (%lu)\n",
                         GetLastError());
        }
    }
    SetPriorityClass(pi.hProcess, priority_class_from_name(spawn_.priority));

    close_handles();
    process_ = pi.hProcess;
    thread_ = pi.hThread;
    pid_ = pi.dwProcessId;
    pipe_read_ = rd;
    running_ = true;
    reader_stop_ = false;
    reader_ = std::thread(&WorkerProc::reader_main, this);
    ResumeThread(pi.hThread);
    return true;
}
#endif

void WorkerProc::close_handles() {
#ifdef _WIN32
    if (thread_ != nullptr) { CloseHandle(static_cast<HANDLE>(thread_)); thread_ = nullptr; }
    if (process_ != nullptr) { CloseHandle(static_cast<HANDLE>(process_)); process_ = nullptr; }
    if (pipe_read_ != nullptr) { CloseHandle(static_cast<HANDLE>(pipe_read_)); pipe_read_ = nullptr; }
#endif
}

void WorkerProc::reader_main() {
#ifdef _WIN32
    HANDLE rd = static_cast<HANDLE>(pipe_read_);
    if (rd == nullptr) return;
    LineSplitter splitter;
    char buf[4096];
    DWORD got = 0;
    while (!reader_stop_.load()) {
        if (!ReadFile(rd, buf, sizeof(buf), &got, nullptr) || got == 0) break;
        splitter.push(buf, got);
        std::string line;
        std::vector<std::string> batch;
        while (splitter.next(line)) batch.push_back(line);
        if (!batch.empty()) {
            std::lock_guard<std::mutex> lk(mu_);
            for (std::string &l : batch) {
                lines_.push_back(std::move(l));
                while (lines_.size() > limits_.max_lines) lines_.pop_front();
            }
        }
    }
#endif
}

void WorkerProc::stop() {
#ifdef _WIN32
    if (!running_ && !want_running_) return;
    want_running_ = false;
    if (job_ != nullptr) {
        // Kills the child and any grandchildren it started.
        TerminateJobObject(static_cast<HANDLE>(job_), 0);
    } else if (process_ != nullptr) {
        TerminateProcess(static_cast<HANDLE>(process_), 0);
    }
    reader_stop_ = true;
    if (pipe_read_ != nullptr) {
        // Unblock ReadFile so the reader thread can finish.
        CancelIoEx(static_cast<HANDLE>(pipe_read_), nullptr);
    }
    if (reader_.joinable()) reader_.join();
    close_handles();
    running_ = false;
    state_ = WorkerRunState::Stopped;
    state_text_.clear();
#endif
}

std::size_t WorkerProc::pending_lines() const {
    std::lock_guard<std::mutex> lk(const_cast<std::mutex &>(mu_));
    return lines_.size();
}

void WorkerProc::drain(DrainedOutput &out) {
    std::deque<std::string> local;
    {
        std::lock_guard<std::mutex> lk(mu_);
        local.swap(lines_);
    }
    for (const std::string &l : local) {
        // Both views are ANSI-free: the raw view keeps UNCLASSIFIED lines, not escape
        // codes (the driver writes colour into pipes; rendering escape sequences in a
        // GUI window is just noise). Events keep the timestamp, raw does not (the
        // timestamp column is drawn by the UI).
        const ParsedLine pl = parse_line(l);
        out.raw.push_back(pl.plain);
        if (pl.old_driver) out.old_driver = true;
        switch (pl.kind) {
            case LogKind::Progress:
                ++out.progress_lines;
                out.last_progress = pl.progress;
                break;
            case LogKind::Event:
            case LogKind::Error:
            case LogKind::QueueDone:
                out.events.push_back(pl.text);       // timestamped, ANSI-free
                if (pl.queue_done) saw_queue_done_ = true;
                if (!pl.start_line.empty()) out.last_start_line = pl.start_line;
                if (pl.is_hit) out.hits.push_back(pl.hit);
                if (pl.p95.valid) out.p95.push_back(pl.p95);
                break;
            case LogKind::Raw:
            default:
                break;                           // only in the raw view
        }
    }
}

void WorkerProc::poll(unsigned long long now_ms) {
#ifdef _WIN32
    if (running_) {
        // Has the process left? A zombie handle still answers WAIT_TIMEOUT until it
        // is really gone, so wait with a zero timeout.
        HANDLE p = static_cast<HANDLE>(process_);
        if (p != nullptr && WaitForSingleObject(p, 0) == WAIT_OBJECT_0) {
            DWORD code = 0;
            GetExitCodeProcess(p, &code);
            exit_code_ = code;
            reader_stop_ = true;
            if (reader_.joinable()) reader_.join();
            close_handles();
            running_ = false;

            // "Queue empty" is the healthy end: the driver's queue manager ran out
            // of lines and exited 0 (it may also exit 0 after a hit, so the text
            // marker decides -- exit codes cannot tell a hit from a finished queue).
            const bool clean = (code == 0);
            if (clean || saw_queue_done_) {
                state_ = WorkerRunState::QueueEmpty;
                state_text_ = "queue done";
                return;
            }
            // Any other exit is treated as a crash: restart with a backoff, unless
            // the breaker has seen too many of them recently.
            crash_times_ms_.push_back(now_ms);
            const unsigned long long window = static_cast<unsigned long long>(limits_.breaker_window_s) * 1000ull;
            while (!crash_times_ms_.empty() && now_ms - crash_times_ms_.front() > window) {
                crash_times_ms_.erase(crash_times_ms_.begin());
            }
            if (static_cast<int>(crash_times_ms_.size()) >= limits_.breaker_crashes) {
                breaker_tripped_ = true;
                state_ = WorkerRunState::Error;
                char buf[160];
                std::snprintf(buf, sizeof(buf),
                              "crashed %d times in %d s - stopped (exit code %lu)",
                              static_cast<int>(crash_times_ms_.size()), limits_.breaker_window_s,
                              static_cast<unsigned long>(code));
                state_text_ = buf;
                return;
            }
            restart_at_ms_ = now_ms + static_cast<unsigned long long>(limits_.restart_delay_s) * 1000ull;
            state_ = WorkerRunState::Restarting;
            char buf[96];
            std::snprintf(buf, sizeof(buf), "exit code %lu; restart in %d s",
                          static_cast<unsigned long>(code), limits_.restart_delay_s);
            state_text_ = buf;
            return;
        }
    }

    if (want_running_ && !running_ && !breaker_tripped_ &&
        state_ == WorkerRunState::Restarting && now_ms >= restart_at_ms_) {
        ++restarts_;
        std::string err;
        if (!spawn_locked(err)) {
            state_ = WorkerRunState::Error;
            state_text_ = err;
            return;
        }
        state_ = WorkerRunState::Running;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "restart #%d", restarts_);
        state_text_ = buf;
    }
#endif
}

} // namespace ecmgui
