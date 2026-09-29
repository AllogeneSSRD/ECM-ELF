// Headless verification of the worker supervisor (milestone M2). Run with
//
//   ecm_gui.exe --worker-selftest --fake <path to ecm_gui_fake_worker.exe>
//
// It drives WorkerProc against the fake worker scenarios and checks the promises
// of docs/DEV_ECM_GUI.md section 5: output capture through ONE pipe, ANSI handling,
// "queue empty" vs crash classification, restart with backoff, the crash breaker,
// priority classes, and killing the whole child tree through the Job object.
//
// Exit code 0 = every check passed. Prints a line per check (and to
// <exe dir>\ecm_gui_worker_selftest.log).

#include "worker_selftest.h"

#include "log_parse.h"
#include "platform.h"
#include "worker_proc.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ecmgui {
namespace {

int g_pass = 0;
int g_fail = 0;
FILE *g_log = nullptr;

void say(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char buf[1024];
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fputs(buf, stdout);
    std::fflush(stdout);
    if (g_log) {
        std::fputs(buf, g_log);
        std::fflush(g_log);
    }
}

void check(bool ok, const std::string &what, const std::string &detail = std::string()) {
    if (ok) {
        ++g_pass;
        say("  [ok]   %s\n", what.c_str());
    } else {
        ++g_fail;
        say("  [FAIL] %s%s%s\n", what.c_str(), detail.empty() ? "" : " -- ", detail.c_str());
    }
}

struct RunResult {
    std::vector<std::string> events;
    std::vector<std::string> raw;
    long long progress_lines = 0;
    std::vector<HitInfo> hits;
    ProgressInfo last_progress;
    std::string last_start_line;
    unsigned long long ms_to_finish = 0;
};

void drain_into(WorkerProc &w, RunResult &r);

// Pumps the worker until the predicate says the interesting moment arrived.
RunResult drive(WorkerProc &w, int timeout_ms, const std::function<bool(const RunResult &)> &stop_when) {
    RunResult r;
    const unsigned long long t0 = mono_ms();
    while (mono_ms() - t0 < static_cast<unsigned long long>(timeout_ms)) {
        w.poll(mono_ms());
        drain_into(w, r);
        if (stop_when(r)) break;
#ifdef _WIN32
        Sleep(20);
#endif
    }
    drain_into(w, r);
    r.ms_to_finish = mono_ms() - t0;
    return r;
}

// Moves the supervisor output into the accumulated result.
void drain_into(WorkerProc &w, RunResult &r) {
    WorkerProc::DrainedOutput d;
    w.drain(d);
    r.events.insert(r.events.end(), d.events.begin(), d.events.end());
    r.raw.insert(r.raw.end(), d.raw.begin(), d.raw.end());
    r.progress_lines += d.progress_lines;
    r.hits.insert(r.hits.end(), d.hits.begin(), d.hits.end());
    if (d.last_progress.valid) r.last_progress = d.last_progress;
    if (!d.last_start_line.empty()) r.last_start_line = d.last_start_line;
}

WorkerSpawn make_spawn(const std::string &fake, const std::string &scenario,
                       const std::string &priority, const std::string &marker = "") {
    WorkerSpawn s;
    s.exe = fake;
    s.ini = "";                       // the fake ignores -ini
    s.worker_index = 2;
    s.priority = priority;
    s.extra_args.push_back("--scenario");
    s.extra_args.push_back(scenario);
    // Four progress lines -> the last one is 75% / 30 curves / +3 bits, so the value
    // assertions in step 1 compare against round numbers.
    s.extra_args.push_back("--lines");
    s.extra_args.push_back("4");
    if (!marker.empty()) {
        s.extra_args.push_back("--marker");
        s.extra_args.push_back(marker);
    }
    return s;
}

bool find_event(const std::vector<std::string> &v, const char *needle) {
    for (const std::string &s : v) {
        if (s.find(needle) != std::string::npos) return true;
    }
    return false;
}

bool has_ansi(const std::vector<std::string> &v) {
    for (const std::string &s : v) {
        if (s.find('\033') != std::string::npos) return true;
    }
    return false;
}

// Fixture lines (fake_worker diagnostics) are not driver output, so they land in the
// raw view, not in the event layer: look in both.
long long event_value(const std::vector<std::string> &v, const char *key) {
    const std::string k = std::string(key) + "=";
    for (const std::string &s : v) {
        const std::size_t p = s.find(k);
        if (p == std::string::npos) continue;
        return std::atoll(s.c_str() + p + k.size());
    }
    return -1;
}

#ifdef _WIN32
bool process_gone(unsigned long pid) {
    if (pid == 0) return true;
    HANDLE h = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (h == nullptr) return true;              // no handle -> gone (or no rights)
    const DWORD rc = WaitForSingleObject(h, 0);
    DWORD code = 0;
    GetExitCodeProcess(h, &code);
    CloseHandle(h);
    return rc == WAIT_OBJECT_0 || code != STILL_ACTIVE;
}
#endif

} // namespace

int run_worker_selftest(const std::string &fake_path, const std::string &log_path) {
    fopen_s(&g_log, log_path.c_str(), "w");
    say("ecm_gui worker supervisor self-test\n  fake worker: %s\n", fake_path.c_str());

    if (!file_exists(fake_path)) {
        say("FAIL: fake worker not found\n");
        return 2;
    }

    WorkerLimits fast;
    fast.restart_delay_s = 1;      // keep the self-test short; production default is 5
    fast.breaker_window_s = 300;
    fast.breaker_crashes = 3;
    fast.max_lines = 4000;

    // ---- 1. normal run: capture + classification -------------------------
    say("[1] scripted run: output capture, classification, queue-empty state\n");
    {
        WorkerProc w;
        w.configure(make_spawn(fake_path, "ok", "below_normal"), fast);
        std::string err;
        check(w.start(err), "start() spawns the worker", err);
        RunResult r = drive(w, 20000, [&](const RunResult &) { return w.state() == WorkerRunState::QueueEmpty; });
        check(w.state() == WorkerRunState::QueueEmpty, "state becomes QueueEmpty",
              w.state_text());
        check(w.exit_code() == 0, "exit code 0", std::to_string(w.exit_code()));
        check(w.restarts() == 0, "no restart for a clean run");
        check(r.raw.size() >= 12, "all output lines captured",
              std::to_string(r.raw.size()) + " lines");
        check(r.progress_lines >= 3, "progress lines recognised",
              std::to_string(r.progress_lines) + " progress lines");
        check(!has_ansi(r.raw), "ANSI escapes removed from captured text");
        check(find_event(r.events, "START: ECMSTAGE2="), "START event kept in the event layer");
        check(find_event(r.events, "FACTOR FOUND"), "FACTOR FOUND kept in the event layer");
        check(find_event(r.events, "Checkpoint saved"), "checkpoint event kept");
        const long long prio = event_value(r.raw, "priority");
        check(prio == static_cast<long long>(BELOW_NORMAL_PRIORITY_CLASS),
              "priority class applied (below_normal)", std::to_string(prio));
        // The queue-done line stays visible in the log pane on purpose: it is the
        // only line that explains WHY the worker stopped.
        check(find_event(r.events, "queue done"),
              "queue-done line is shown in the log pane (explains the stop)");
        // Progress VALUES parsed from a real pipe stream (the fake worker prints the
        // driver's GPU-progress shape): the table columns are fed by exactly these
        // fields, so they are asserted here rather than in the pure-function test.
        check(r.last_progress.valid, "a progress line was parsed");
        check(std::abs(r.last_progress.pct - 75.0) < 1e-9, "progress percentage",
              std::to_string(r.last_progress.pct));
        check(std::abs(r.last_progress.s_per_curve - 1.2) < 1e-9, "s/curve",
              std::to_string(r.last_progress.s_per_curve));
        check(std::abs(r.last_progress.eta_s - 60.0) < 1e-9, "remaining -> ETA",
              std::to_string(r.last_progress.eta_s));
        check(r.last_progress.curves_done == 30, "curve count",
              std::to_string(r.last_progress.curves_done));
        check(r.last_progress.bits == 3, "bits", std::to_string(r.last_progress.bits));
        check(r.last_progress.gpu, "GPU progress form recognised");
        check(r.last_start_line.compare(0, 10, "ECMSTAGE2=") == 0,
              "the task line was captured for the table", r.last_start_line);
        check(r.hits.size() >= 2, "hit lines captured", std::to_string(r.hits.size()));
    }

    // ---- 2. hit parsing ---------------------------------------------------
    say("[2] hit lines: factor extraction\n");
    {
        const ParsedLine a = parse_line(
            "[2026-09-28 18:42:41] factor[3]=1234567890123456789012345678907 curve=37 sigma=99");
        check(a.is_hit && a.factor == "1234567890123456789012345678907",
              "factor[i]=<decimal> parsed (D3 fields tolerated)", a.factor);
        const ParsedLine b = parse_line(
            "[2026-09-28 18:42:41] GPU: factor 12345 found in Step 1 with curve 3 (sigma 0:99)");
        check(b.is_hit, "CGBN hit line recognised");
        const ParsedLine c = parse_line(
            "[2026-09-28 18:42:41]   curve 7 sigma=12345 -> factor found");
        check(c.is_hit, "CPU hit line recognised");
    }

    // ---- 3. restart after a crash ----------------------------------------
    say("[3] crash then restart (backoff), then a clean finish\n");
    {
        WorkerProc w;
        const std::string marker = path_join(exe_dir(), "ecm_gui_worker_selftest.marker");
        std::remove(marker.c_str());
        w.configure(make_spawn(fake_path, "crash-once", "normal", marker), fast);
        std::string err;
        check(w.start(err), "start() spawns the worker", err);
        RunResult r = drive(w, 30000, [&](const RunResult &) { return w.state() == WorkerRunState::QueueEmpty; });
        check(w.restarts() == 1, "exactly one restart", std::to_string(w.restarts()));
        check(w.state() == WorkerRunState::QueueEmpty, "second run ends in QueueEmpty",
              w.state_text());
        check(w.exit_code() == 0, "final exit code 0", std::to_string(w.exit_code()));
        check(r.ms_to_finish >= 900, "restart waited for the backoff (>= 1 s here)",
              std::to_string(r.ms_to_finish) + " ms");
        check(find_event(r.events, "ERROR: stage1 failed"),
              "the crash's error line was captured");
        std::remove(marker.c_str());
    }

    // ---- 4. breaker ------------------------------------------------------
    say("[4] repeated crashes trip the breaker and stop the restarts\n");
    {
        WorkerProc w;
        w.configure(make_spawn(fake_path, "crash", "normal"), fast);
        std::string err;
        check(w.start(err), "start() spawns the worker", err);
        RunResult r = drive(w, 30000, [&](const RunResult &) { return w.breaker_tripped(); });
        check(w.breaker_tripped(), "breaker tripped");
        check(w.restarts() == 2, "restarted twice before tripping (3 crashes)",
              std::to_string(w.restarts()));
        check(w.state() == WorkerRunState::Error, "state is Error", w.state_text());
        check(w.state_text().find("crashed") != std::string::npos,
              "state text explains the breaker", w.state_text());
        check(w.exit_code() == 3, "last exit code reported", std::to_string(w.exit_code()));
        err.clear();
        check(!w.start(err), "start() refuses once tripped", err);
    }

    // ---- 5. job object kills the whole tree ------------------------------
    say("[5] stop() kills the child AND its grandchild (job object)\n");
    {
        WorkerProc w;
        w.configure(make_spawn(fake_path, "spawn-child", "normal"), fast);
        std::string err;
        check(w.start(err), "start() spawns the parent", err);
        RunResult r = drive(w, 10000, [&](const RunResult &acc) { return event_value(acc.raw, "child") > 0; });
        const long long child_pid = event_value(r.raw, "child");
        check(child_pid > 0, "the child reported its pid",
              std::to_string(child_pid));
#ifdef _WIN32
        if (child_pid > 0) {
            check(!process_gone(static_cast<unsigned long>(child_pid)),
                  "child is alive before stop()");
            w.stop();
            Sleep(500);
            check(process_gone(static_cast<unsigned long>(child_pid)),
                  "child is gone after stop() (job object worked)");
            check(!w.running(), "worker reports stopped");
            check(w.state() == WorkerRunState::Stopped, "state is Stopped", w.state_text());
        }
#endif
    }

    // ---- 6. stop() on a running worker -----------------------------------
    say("[6] stop() on a hanging worker\n");
    {
        WorkerProc w;
        w.configure(make_spawn(fake_path, "hang", "idle"), fast);
        std::string err;
        check(w.start(err), "start() spawns the hanging worker", err);
        RunResult r = drive(w, 3000, [&](const RunResult &) { return false; });
        check(r.raw.size() >= 1, "hanging worker's first line was captured",
              std::to_string(r.raw.size()) + " lines");
        check(w.state() == WorkerRunState::Running, "state is Running while it hangs",
              w.state_text());
        w.stop();
        check(!w.running() && w.state() == WorkerRunState::Stopped,
              "stop() ends it without a restart", w.state_text());
        w.poll(mono_ms() + 5000);
        check(w.state() == WorkerRunState::Stopped, "no restart after an explicit stop",
              w.state_text());
    }

    say("\npassed: %d   failed: %d\n  log: %s\n", g_pass, g_fail, log_path.c_str());
    if (g_log) {
        std::fclose(g_log);
        g_log = nullptr;
    }
    return g_fail == 0 ? 0 : 1;
}

} // namespace ecmgui
