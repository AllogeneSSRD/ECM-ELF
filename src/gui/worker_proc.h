#pragma once

// Worker process supervision (milestone M2, docs/DEV_ECM_GUI.md section 5).
//
// One worker == one `ecm_cuda.exe -ini ecm.ini --worker N` process:
//   * CreateProcessW with CREATE_NO_WINDOW, never through cmd.exe;
//   * stdout AND stderr point at the SAME pipe write end, so one reader thread
//     sees the real write order (no cross-pipe reordering);
//   * the child runs inside a Job object with KILL_ON_JOB_CLOSE: closing the GUI
//     (or crashing) never leaves an orphan holding the GPU;
//   * priority class from [GUI] priority=;
//   * non-"queue empty" exit -> restart after 5 s; 3 crashes inside 5 minutes ->
//     breaker trips, the worker stops and is flagged red.
//
// Threading: start()/stop()/poll() are called from the UI thread; the pipe reader
// runs on its own thread and hands complete lines to an internal queue guarded by
// a mutex. poll() never blocks.

#include "log_parse.h"
#include "platform.h"

#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ecmgui {

struct WorkerLimits {
    int restart_delay_s = 5;         // wait before restarting a crashed worker
    int breaker_crashes = 3;         // this many crashes ...
    int breaker_window_s = 300;      // ... inside this window trips the breaker
    std::size_t max_lines = 4000;    // ring buffer per worker (raw + events)
};

class WorkerProc {
public:
    WorkerProc();
    ~WorkerProc();

    WorkerProc(const WorkerProc &) = delete;
    WorkerProc &operator=(const WorkerProc &) = delete;

    void configure(const WorkerSpawn &spawn, const WorkerLimits &limits);

    // Starts (or restarts) the process unless the breaker has tripped.
    bool start(std::string &err);
    // Terminates the whole job (children included). Safe to call when stopped.
    void stop();
    // Autostart helper: start when the config says so and the worker is stopped.
    void set_autostart(bool on) { autostart_ = on; }
    bool autostart() const { return autostart_; }

    // UI-thread tick: drains captured output and applies the restart/breaker rules.
    // `now_ms` is a monotonic millisecond clock supplied by the caller.
    void poll(unsigned long long now_ms);

    WorkerRunState state() const { return state_; }
    const std::string &state_text() const { return state_text_; }
    unsigned long exit_code() const { return exit_code_; }
    int restarts() const { return restarts_; }
    bool breaker_tripped() const { return breaker_tripped_; }
    unsigned long pid() const { return pid_; }
    bool running() const { return running_; }

    // Output: moves the newly captured lines into the two views and reports what the
    // UI needs from them (last progress values, current task line, hit count).
    struct DrainedOutput {
        std::vector<std::string> events;   // event layer (timestamped, ANSI-free)
        std::vector<std::string> raw;      // unclassified lines, ANSI-free
        long long progress_lines = 0;
        ProgressInfo last_progress;
        std::string last_start_line;       // newest "START: <worktodo line>"
        std::vector<HitInfo> hits;         // every hit line seen in this batch (D3 fields)
        // Every `p95_add:` notice seen in this batch (Prime95 handoff, docs 13).
        std::vector<P95Notice> p95;
        // The worker printed "No input number on stdin": its executable predates the
        // --worker support (D1/D2) and never entered queue mode. See kOldDriverHint.
        bool old_driver = false;
    };
    void drain(DrainedOutput &out);
    std::size_t pending_lines() const;

    const WorkerSpawn &spawn() const { return spawn_; }

    // Test/introspection hook: the command line that would be run.
    std::string command_line() const;

private:
    void reader_main();
    bool spawn_locked(std::string &err);
    void close_handles();

    WorkerSpawn spawn_;
    WorkerLimits limits_;

    std::thread reader_;
    std::mutex mu_;
    std::deque<std::string> lines_;      // complete lines waiting for the UI
    std::atomic<bool> reader_stop_{false};
    std::atomic<bool> running_{false};

    void *job_ = nullptr;                // HANDLE
    void *process_ = nullptr;            // HANDLE
    void *thread_ = nullptr;             // HANDLE
    void *pipe_read_ = nullptr;          // HANDLE
    void *pipe_write_ = nullptr;         // HANDLE
    unsigned long pid_ = 0;

    WorkerRunState state_ = WorkerRunState::Stopped;
    std::string state_text_;
    unsigned long exit_code_ = 0;
    int restarts_ = 0;
    bool breaker_tripped_ = false;
    bool autostart_ = false;
    bool want_running_ = false;
    bool saw_queue_done_ = false;
    unsigned long long restart_at_ms_ = 0;
    std::vector<unsigned long long> crash_times_ms_;
};

// Monotonic milliseconds (GetTickCount64 on Windows).
unsigned long long mono_ms();

} // namespace ecmgui
