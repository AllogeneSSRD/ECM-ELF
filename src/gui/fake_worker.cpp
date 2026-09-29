// fake_worker -- a stand-in for ecm_cuda.exe used by ecm_gui's process tests.
//
// It prints lines in the exact shapes the driver prints (timestamped, ANSI
// coloured, both progress forms, START/FACTOR/Checkpoint events, "queue done")
// and then behaves according to --scenario, so the worker supervisor can be tested
// without a GPU or a real multi-hour ECM run:
//
//   --scenario ok           print a scripted run, exit 0 with "queue done"
//   --scenario crash        print, exit 3              (crash handling / breaker)
//   --scenario crash-once   exit 3 the first time (marker file), then behave like ok
//                                                      (restart handling)
//   --scenario hang         print, then sleep forever   (stop / job object)
//   --scenario spawn-child  start a copy of itself in "hang" mode, print child=<pid>,
//                           then sleep                        (job-object kill test)
//   --scenario stale-driver print a pre-D1/D2 driver's "No input number on stdin"
//                           failure, exit 1      (GUI must diagnose the stale exe)
//   --scenario resumed      print a checkpoint-resume pair and then NO progress line
//                           (the GUI must seed the bar from the resume percentage)
//
// Extra switches:
//   --marker <path>   marker file used by crash-once
//   --lines <n>       how many filler lines to print (ring-buffer tests)
//   --delay-ms <n>    sleep between the scripted lines
//
// ASCII only on purpose (see tools/README.md) and every line is flushed, so a pipe
// reader sees it immediately.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {

int g_argc = 0;
char **g_argv = nullptr;

std::string arg_value(const char *name, const std::string &def = "") {
    for (int i = 1; i + 1 < g_argc; ++i) {
        if (std::strcmp(g_argv[i], name) == 0) return g_argv[i + 1];
    }
    return def;
}

void out(const char *fmt, ...) {
    // Same "[YYYY-MM-DD HH:MM:SS] " prefix the driver writes (ecm_ts_fprintf).
    SYSTEMTIME st;
    GetLocalTime(&st);
    std::printf("[%04d-%02d-%02d %02d:%02d:%02d] ", st.wYear, st.wMonth, st.wDay, st.wHour,
                st.wMinute, st.wSecond);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
    std::fflush(stdout);
}

void out_raw(const std::string &s) {
    std::fputs(s.c_str(), stdout);
    std::fflush(stdout);
}

void sleep_ms(int ms) {
#ifdef _WIN32
    Sleep(static_cast<DWORD>(ms));
#else
    usleep(static_cast<useconds_t>(ms) * 1000);
#endif
}

void print_scripted_run(int lines, int delay_ms) {
    const std::string worker = arg_value("--worker", "1");
    out("===== ECM queue manager =====\n");
    out("worker : %s (ini section [Worker #%s]; unprefixed keys are the global defaults)\n",
        worker.c_str(), worker.c_str());
    out("gpu backend : CUDA/CGBN, param0, device 0\n");
#ifdef _WIN32
    out("priority=%lu\n", static_cast<unsigned long>(GetPriorityClass(GetCurrentProcess())));
#endif
    out("START: ECMSTAGE2=1,2,5351,-1,\"m5351_110e6.save\",0,0,960\n");
    for (int i = 0; i < lines; ++i) {
        // One ANSI-coloured progress line per iteration, in the driver's log-mode
        // shape (colour code + ASCII bar + percentage + speed + ETA).
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "\033[36mGPU: [====>   ] %.1f%%  %d, +%d bits (~%.2f s/curve)  "
                      "elapsed %.1fs  remaining %.1fs\033[0m\r\n",
                      100.0 * i / (lines > 0 ? lines : 1), i * 10, i, 1.2, i * 1.2, 60.0);
        out_raw(buf);
        out("curve %d sigma=%d -> factor found\n", i, 12345 + i);
        // Same shape as the driver D3 hit line (curve/sigma/param/method/save), so the
        // GUI result path is exercised by the fake worker too.
        out("factor[%d]=123456789012345678901234567890%d curve=%d sigma=%d param=3 "
            "method=gpu save=fake_%d_1e4.save\n",
            i, i % 7, i, 1000000 + i, i);
        if (delay_ms > 0) sleep_ms(delay_ms);
    }
    out("Checkpoint saved: s_partial=1234/5678 (21.7%%)\n");
    out("factor[0]=999888777666555444333222111\n");
    out("FACTOR FOUND aid=N/A task=ECMSTAGE2=1,2,5351,-1,\"m5351_110e6.save\",0,0,960\n");
    out("===== queue done, 2 task(s) processed =====\n");
}

} // namespace

int main(int argc, char **argv) {
    g_argc = argc;
    g_argv = argv;
    const std::string scenario = arg_value("--scenario", "ok");
    const int lines = std::atoi(arg_value("--lines", "3").c_str());
    const int delay_ms = std::atoi(arg_value("--delay-ms", "0").c_str());
    const std::string marker = arg_value("--marker");

    if (scenario == "ok") {
        print_scripted_run(lines, delay_ms);
        return 0;
    }
    if (scenario == "crash") {
        out("START: ECMSTAGE2=garbage\n");
        out("ERROR: not enough fields (need at least 8 core columns)\n");
        out("FATAL: backend prepare failed; aborting queue.\n");
        return 3;
    }
    if (scenario == "resumed") {
        // A RESUMED run, in the driver's own words (user's production log, 2026-09-29). The
        // driver's redirected progress lines are gated on the batch counter RESTORED from the
        // checkpoint, so the GUI sees the resume percentage long before any progress line --
        // and has to show that instead of an empty bar.
        const double pct = std::atof(arg_value("--pct", "23.8").c_str());
        out("Checkpoint loaded: s_partial=89173178/375102575 (%.1f%%), age=24348 seconds\n", pct);
        out("Resuming from checkpoint: %.1f%% complete (s_partial=89173178/375102575)\n", pct);
        out("GPU: CGBN<16, 3584> kernel, N is 3375 bits (120 blocks x 128 threads)\n");
        for (;;) sleep_ms(500);      // alive, but no progress line at all
    }
    if (scenario == "stale-driver") {
        // Byte-for-byte what a pre-D1/D2 ecm_cuda prints when the GUI launches it as
        // `-ini <path> --worker N`: the unknown switch becomes a positional argument, so
        // it takes the single-run path and dies asking stdin for a number. Measured from
        // D:\code\GIMPS\ecm-win-x86_64-sm89-cuda13.3_param0\ecm_cuda.exe (2026-09-28).
        // The GUI must recognise this (kOldDriverHint) instead of just restarting.
        out("ecm driver starting\n");
        out("  mode: cpu-stub, gpucurves=0, ckpt=600s, device=0, group_order=off\n");
        out("No input number on stdin\n");
        return 1;
    }
    if (scenario == "crash-once") {
        FILE *f = marker.empty() ? nullptr : std::fopen(marker.c_str(), "rb");
        if (f != nullptr) {
            std::fclose(f);
            out("second run: queue has work again\n");
            print_scripted_run(lines, delay_ms);
            return 0;
        }
        if (!marker.empty()) {
            FILE *w = std::fopen(marker.c_str(), "wb");
            if (w != nullptr) {
                std::fputs("1", w);
                std::fclose(w);
            }
        }
        out("GPU: warning: only 12 blocks for 60 SMs - some SMs idle\n");
        out("ERROR: stage1 failed for task: ECMSTAGE2=1,2,5351,-1,\"m5351_110e6.save\",0,0,960\n");
        return 7;
    }
    if (scenario == "spawn-child") {
#ifdef _WIN32
        char self[MAX_PATH];
        if (GetModuleFileNameA(nullptr, self, MAX_PATH) > 0) {
            std::string cmd = std::string("\"") + self + "\" --scenario hang";
            STARTUPINFOA si{};
            si.cb = sizeof(si);
            si.dwFlags = STARTF_USESHOWWINDOW;
            si.wShowWindow = SW_HIDE;
            PROCESS_INFORMATION pi{};
            if (CreateProcessA(nullptr, const_cast<char *>(cmd.c_str()), nullptr, nullptr, FALSE,
                               CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
                out("child=%lu\n", static_cast<unsigned long>(pi.dwProcessId));
                CloseHandle(pi.hThread);
                CloseHandle(pi.hProcess);
            } else {
                out("child=0 (CreateProcess failed %lu)\n", GetLastError());
            }
        }
#endif
        out("parent sleeping\n");
        for (;;) sleep_ms(1000);
    }
    if (scenario == "hang") {
        out("START: ECMSTAGE2=1,2=hang\n");
        for (;;) sleep_ms(1000);
    }

    std::fprintf(stderr, "fake_worker: unknown scenario '%s'\n", scenario.c_str());
    return 2;
}
