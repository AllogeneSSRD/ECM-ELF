#pragma once
// p95_transfer.h — hand a FINISHED stage-1 task over to a running Prime95.
//
// After the driver has advanced its own queue (line moved to worktodo.finished.txt and the
// .save synchronized), the same task can be continued by Prime95's stage 2. The delivery
// channel is Prime95's documented drop box: append the line to `worktodo.add` next to
// Prime95's `worktodo.txt`; Prime95 pollutes it into worktodo.txt on its own schedule and
// deletes the file (undoc.txt, and incorporateWorkToDoAddFile() in commonc.c). Nothing here
// ever touches worktodo.txt, so Prime95's own bookkeeping stays intact.
//
// The line is appended VERBATIM — the AID and any known-factors field are what makes
// Prime95 report the eventual factor to the server under the right assignment, and
// ECMSTAGE2= is Prime95's own "continue stage 2 from a GMP-ECM stage 1" keyword
// (commonc.c: "ECMSTAGE2=k,b,n,c,filename[,B2-or-zero][,skip_curves][,num_curves]
// [,\"known-factors\"]").
//
// This is deliberately NOT ecm_p95feeder: that tool moves Edwards .tmp saves and stays
// standalone. This one belongs to the driver and only forwards completed queue lines.
//
// Failure policy: a handoff problem NEVER fails the task. The line is parked in a pending
// file and re-delivered with the next successful delivery; the GUI shows a red notice
// while that file is non-empty. See docs/DEV_ECM_GUI.md §18 and docs/DEV_ECM_WORKTODO.md §8.

#include <cstddef>
#include <string>
#include <vector>

struct P95TransferConfig {
    // ini `p95_worktodo_path`: Prime95's worktodo.txt. Empty = the feature is off.
    std::string worktodo_path;
    // ini `p95_add_workers`: "" (no section header) | "3" | "1,3" | "1-8" | "auto".
    std::string add_workers;
    // Where undelivered lines wait (usually "<exe dir>\p95_add_pending.txt").
    std::string pending_path;
};

struct P95TransferResult {
    bool enabled = false;         // false when p95_worktodo_path is empty
    bool ok = false;              // the line is in worktodo.add
    int  worker = 0;              // section it went to (0 = appended without a header)
    size_t added = 0;             // lines appended by THIS call
    size_t pending_delivered = 0; // previously parked lines that went out with it
    size_t pending_left = 0;      // lines still parked after this call
    std::string add_path;         // worktodo.add that was written
    std::string note;             // warning text (routing fell back), empty when clean
    std::string error;            // failure reason, empty on success
};

// Deliver one completed task line. Never throws, never blocks longer than ~3 s (the
// worktodo.add lock), and never modifies worktodo.txt.
P95TransferResult p95_transfer_deliver(const P95TransferConfig &cfg, const std::string &line);

// --- pieces the GUI/tests use to describe the state without delivering anything ---

// Parse `p95_add_workers` into worker numbers. Returns false when the spec is
// syntactically invalid (caller then falls back to a header-less append); entries outside
// 1..1024 are dropped and reported in `ignored`.
bool p95_parse_add_workers(const std::string &spec, std::vector<int> &workers, int &ignored);

// Path of the pending file for a given driver executable directory.
std::string p95_transfer_pending_path(const std::string &exe_dir);

// Path of the worktodo.add that goes with a given Prime95 worktodo.txt.
std::string p95_transfer_add_path(const std::string &worktodo_path);

// Number of lines waiting in the pending file (0 when it does not exist).
size_t p95_transfer_pending_count(const std::string &pending_path);
