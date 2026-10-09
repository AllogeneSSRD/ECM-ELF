// Unit test for the pure log/progress parsing layer (src/gui/log_parse.cpp).
// Build: cmake --build <dir> --target ecm_gui_log_parse_test
// Exit code: 0 = all checks passed.
//
// Every expected value here comes from a REAL driver line (the format strings are
// quoted in log_parse.h with their source locations), so a format change shows up
// as a failing test instead of a GUI that silently shows "—".

#include "log_parse.h"

#include <cstdio>
#include <string>

// The driver prefixes every line with this (src/opencl_ecm_log.cpp:39); tests must
// use it too, otherwise they exercise the "no timestamp" path by accident.
// A macro (not a variable) so TS "rest of the line" concatenates as a literal.
#define TS "[2026-09-28 18:42:41] "
#include <string>

using namespace ecmgui;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool ok, const std::string &what, const std::string &detail = std::string()) {
    if (ok) {
        ++g_pass;
        std::printf("  [ok]   %s\n", what.c_str());
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s%s%s\n", what.c_str(), detail.empty() ? "" : " -- ",
                    detail.c_str());
    }
}

bool near(double a, double b, double eps = 1e-6) {
    const double d = a - b;
    return (d < 0 ? -d : d) <= eps;
}

} // namespace

int main() {
    std::printf("[1] ANSI stripping\n");
    {
        const std::string s = strip_ansi("\033[36mGPU: [=>  ] 10.0%\033[0m\r");
        check(s == "GPU: [=>  ] 10.0%\r", "CSI sequences removed", s);
        check(strip_ansi("plain") == "plain", "plain text untouched");
        check(strip_ansi("\033]0;title\007rest") == "rest", "OSC sequence removed",
              strip_ansi("\033]0;title\007rest"));
    }

    std::printf("[2] timestamp split\n");
    {
        std::string ts, plain;
        split_timestamp("[2026-09-28 18:42:41] worker : 1", ts, plain);
        check(ts == "2026-09-28 18:42:41", "timestamp extracted", ts);
        check(plain == "worker : 1", "payload extracted", plain);
        split_timestamp("no timestamp here", ts, plain);
        check(ts.empty() && plain == "no timestamp here", "line without prefix");
        split_timestamp("[not a date] x", ts, plain);
        check(ts.empty() && plain == "[not a date] x", "malformed prefix left alone");
    }

    std::printf("[3] CPU progress line (ecm_driver.cpp:1263)\n");
    {
        const ParsedLine p = parse_line(
            "[2026-09-28 18:42:41] stage1: [====>                    ] 42.3%  42.3/100 "
            "(~1.20 s/curve)  elapsed 51.0s  ETA 69.0s");
        check(p.kind == LogKind::Progress, "classified as progress");
        check(p.progress.valid && !p.progress.gpu, "CPU form");
        check(near(p.progress.pct, 42.3), "percentage", std::to_string(p.progress.pct));
        check(p.progress.curves_done == 42 && p.progress.curves_total == 100,
              "curves done/total");
        check(near(p.progress.s_per_curve, 1.20), "s/curve",
              std::to_string(p.progress.s_per_curve));
        check(near(p.progress.elapsed_s, 51.0), "elapsed",
              std::to_string(p.progress.elapsed_s));
        check(near(p.progress.eta_s, 69.0), "ETA", std::to_string(p.progress.eta_s));
    }

    std::printf("[4] GPU progress line (cgbn_stage1.cu:102)\n");
    {
        const ParsedLine p = parse_line(
            "[2026-09-28 18:42:41] \033[36mGPU: [====>                    ] 42.3%  123456, "
            "+789 bits (~1.20 s/curve)  elapsed 51.0s  remaining 69.0s\033[0m\r");
        check(p.kind == LogKind::Progress, "classified as progress");
        check(p.progress.valid && p.progress.gpu, "GPU form");
        check(near(p.progress.pct, 42.3), "percentage", std::to_string(p.progress.pct));
        check(p.progress.curves_done == 123456, "curve count",
              std::to_string(p.progress.curves_done));
        check(p.progress.bits == 789, "bits", std::to_string(p.progress.bits));
        check(near(p.progress.s_per_curve, 1.20), "s/curve",
              std::to_string(p.progress.s_per_curve));
        check(near(p.progress.eta_s, 69.0), "'remaining' mapped to the ETA field",
              std::to_string(p.progress.eta_s));
        check(p.text.find('\033') == std::string::npos, "ANSI already stripped");
    }

    std::printf("[5] events and errors\n");
    {
        const ParsedLine s = parse_line(
            "[2026-09-28 18:42:41] START: ECMSTAGE2=AID,1,2,5351,-1,\"m5351_110e6.save\",0,0,960");
        check(s.kind == LogKind::Event && s.start_line.compare(0, 10, "ECMSTAGE2=") == 0,
              "START carries the worktodo line", s.start_line);
        check(parse_line(TS "Checkpoint saved: s_partial=1/2 (50.0%)") .kind == LogKind::Event,
              "checkpoint is an event");
        check(parse_line(TS "Resuming from checkpoint: 12.0% complete").kind == LogKind::Event,
              "resume is an event");
        const ParsedLine e = parse_line("[2026-09-28 18:42:41] ERROR: not enough fields");
        check(e.kind == LogKind::Error, "ERROR is an error line");
        check(parse_line(TS "FATAL: backend prepare failed; aborting queue.").kind ==
                  LogKind::Error, "FATAL is an error line");
        check(parse_line(TS "# ERROR ECMSTAGE2=garbage").kind == LogKind::Error,
              "a marked worktodo line is an error");
        check(parse_line(
                  TS "GPU: warning: only 12 blocks for 60 SMs - some SMs idle").kind ==
                  LogKind::Error, "driver warnings surface as warnings");
    }

    std::printf("[6] hits and queue completion\n");
    {
        const ParsedLine h = parse_line(TS "factor[3]=1234567890123456789012345678907");
        check(h.is_hit && h.factor == "1234567890123456789012345678907", "factor value",
              h.factor);
        const ParsedLine h2 = parse_line(
            TS "factor[0]=999 curve=37 sigma=12345 param=0 save=m5351_110e6.save");
        check(h2.is_hit && h2.factor == "999", "D3 hit fields tolerated", h2.factor);
        const ParsedLine f = parse_line(
            TS "FACTOR FOUND aid=N/A task=ECMSTAGE2=1,2,5351,-1,\"m5351_110e6.save\",0,0,960");
        check(f.is_hit && f.kind == LogKind::Event, "FACTOR FOUND is a hit event");
        const ParsedLine d = parse_line(TS "===== queue done, 2 task(s) processed =====");
        check(d.queue_done && d.tasks_processed == 2 && d.kind == LogKind::QueueDone,
              "queue completion + task count", std::to_string(d.tasks_processed));
        check(parse_line(TS "gpu backend : CUDA/CGBN, param0, device 1").kind == LogKind::Event,
              "banner line is an event");
        check(parse_line(TS "some unclassified chatter").kind == LogKind::Raw,
              "unknown lines stay raw");
    }

    std::printf("[7] pipe line splitter\n");
    {
        LineSplitter sp;
        sp.push("a\r\nb\nc", 6);
        std::string l;
        check(sp.next(l) && l == "a", "'\\r\\n' yields 'a'", l);
        check(sp.next(l) && l == "b", "'\\n' yields 'b'", l);
        check(!sp.next(l), "partial line is kept back");
        check(sp.pending() == 1, "one byte pending", std::to_string(sp.pending()));
        sp.push("d\r\n", 3);
        check(sp.next(l) && l == "cd", "partial line completes", l);
        // Bare CR (in-place TTY updates) also ends a line.
        sp.push("x\ry\r", 4);
        check(sp.next(l) && l == "x", "bare CR ends a line", l);
        check(sp.next(l) && l == "y", "second CR line", l);
        // Chunk boundary inside a multi-byte UTF-8 character must not split a line.
        sp.push("zh", 2);
        sp.push("\xE6\x96\x87", 3);
        sp.push("!\n", 2);
        check(sp.next(l) && l == "zh\xE6\x96\x87!", "UTF-8 across chunks reassembled", l);
    }

    // ---- "the worker exe is older than --worker support" ------------------------
    // Measured output of a pre-D1/D2 ecm_cuda launched as `-ini … --worker 1`: the
    // unknown switch lands in the positional list, so it never reaches the queue
    // manager and dies reading a task from stdin. Reproduced with the release build
    // from D:\code\GIMPS\... (2026-09-28).
    {
        const ParsedLine a = parse_line(TS "ecm driver starting");
        check(!a.old_driver, "the plain startup banner is not the old-driver marker");
        const ParsedLine b = parse_line(TS "  mode: cpu-stub, gpucurves=0, ckpt=600s, device=0, group_order=off");
        check(!b.old_driver, "the cpu-stub banner alone is not enough (it is a real mode too)");
        const ParsedLine c = parse_line("No input number on stdin");
        check(c.old_driver, "'No input number on stdin' flags a driver without --worker");
        check(c.kind == LogKind::Error, "and it is classified as an error");
        check(std::string(kOldDriverHint).find("--worker") != std::string::npos,
              "the hint names the missing switch");
        const ParsedLine d = parse_line(TS "===== ECM queue manager =====");
        check(!d.old_driver, "a queue-manager banner does not flag an old driver");
    }

    // ---- `p95_add:` notices (Prime95 handoff, docs/usage/GUI.md 13) -------------
    // The four shapes the driver emits, including a path with a space and an escaped quote
    // in the free-text reason: the strip must survive both.
    {
        const ParsedLine ready = parse_line(TS "p95_add: ready workers=\"1-8\" file=\"D:\\code\\GIMPS\\p95\\worktodo.add\" pending=2");
        check(ready.p95.valid, "the ready notice is recognised");
        check(ready.kind == LogKind::Event, "and it is an Event (it belongs in the log pane)");
        check(ready.p95.level == P95Notice::Level::Ready, "level = ready");
        check(ready.p95.pending_at_start == 2, "pending=2 is read", std::to_string(ready.p95.pending_at_start));
        check(ready.p95.file == "D:\\code\\GIMPS\\p95\\worktodo.add", "the quoted path is unescaped", ready.p95.file);

        const ParsedLine ok = parse_line(TS "p95_add: ok worker=2 added=1 pending_delivered=0 file=\"C:\\p95 dir\\worktodo.add\"");
        check(ok.p95.level == P95Notice::Level::Ok, "level = ok");
        check(ok.p95.worker == 2, "worker=2 is read", std::to_string(ok.p95.worker));
        check(ok.p95.added == 1, "added=1 is read");
        check(ok.p95.file == "C:\\p95 dir\\worktodo.add", "a path with a space survives", ok.p95.file);

        const ParsedLine warn = parse_line(TS "p95_add: warn worker=0 added=1 pending_delivered=0 file=\"D:\\p95\\worktodo.add\" note=\"worktodo.txt has none of the [Worker #3...] sections p95_add_workers asks for; appending without a section header\"");
        check(warn.p95.level == P95Notice::Level::Warn, "level = warn");
        check(warn.p95.worker == 0, "worker=0 (header-less) is read");
        check(warn.p95.note.find("without a section header") != std::string::npos,
              "the free-text note is kept whole", warn.p95.note);

        const ParsedLine pend = parse_line(TS "p95_add: pending worker=1 lines=3 file=\"D:\\p95\\worktodo.add\" error=\"worktodo.add is locked: held by another writer (waited 3000 ms)\"");
        check(pend.p95.level == P95Notice::Level::Pending, "level = pending");
        check(pend.p95.lines == 3, "lines=3 is read", std::to_string(pend.p95.lines));
        check(pend.p95.error.find("locked") != std::string::npos, "the error text is kept", pend.p95.error);
        // An escaped quote inside a value must not end it early: the driver writes \" for a
        // quote that belongs to the text. Single backslashes are literal (a path with
        // single backslashes needs no doubling), and a trailing \\ is one backslash.
        const ParsedLine esc = parse_line(TS "p95_add: pending worker=0 lines=1 file=\"D:\\p95\\worktodo.add\" error=\"cannot append: \\\"bad\\\" name\"");
        check(esc.p95.error.find("bad") != std::string::npos, "escaped quotes survive", esc.p95.error);
        check(esc.p95.error.find("cannot append: \"bad\" name") == 0, "and the escape itself is gone", esc.p95.error);
        // Bytes on the wire: file="D:\p95\\"  (the driver doubles a TRAILING backslash so
        // the closing quote can never be ambiguous).
        const ParsedLine tail = parse_line(TS "p95_add: pending worker=0 lines=1 file=\"D:\\p95\\\\\" error=\"x\"");
        check(tail.p95.file == "D:\\p95\\", "a trailing doubled backslash is one backslash", tail.p95.file);
        const ParsedLine unknown = parse_line(TS "p95_add: something new");
        check(unknown.p95.valid && unknown.p95.level == P95Notice::Level::Ready,
              "an unknown p95_add shape is still a notice (level = ready)");

        // A progress line is NOT a notice, and vice versa.
        const ParsedLine prog = parse_line(TS "GPU: [===>    ] 42.0%  100, +50 bits (~1.00 s/curve)  elapsed 1.0s  remaining 1.0s");
        check(!prog.p95.valid, "a progress line is not a p95 notice");
        check(prog.kind == LogKind::Progress, "and it is still a progress line");
        check(!ok.progress.valid, "a notice is not mistaken for progress");
    }

    std::printf("\npassed: %d   failed: %d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
