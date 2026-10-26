#pragma once
// Prime95 delivery protocol, including legacy pending import.
#include <cstddef>
#include <string>
#include <vector>

struct P95TransferConfig {
    // ini `p95_worktodo_path`: Prime95's worktodo.txt. Empty = the feature is off.
    std::string worktodo_path;
    std::string sync_dir; // Dedicated, validated save destination (p95_sync_dir).
    // ini `p95_add_workers`: "" (no section header) | "3" | "1,3" | "1-8" | "auto".
    std::string add_workers;
    // Where undelivered lines wait (usually "<exe dir>\p95_add_pending.txt").
    std::string pending_path;
    // ── assignment keys (2026-09-29, measured in the operator's production run) ──────────
    // ini `p95_keep_aid` (default 1): append the line with its AID.
    //
    // Prime95 reports an assignment's progress to PrimeNet under that AID, and when PrimeNet
    // answers "Invalid assignment key" Prime95 DELETES the work unit from worktodo.txt
    // (commonc.c:6550-6569: "If we get an invalid assignment key, then the user probably
    // unreserved the exponent using the web forms - delete it from our work to do file").
    // A line WITHOUT an AID is never reported that way; Prime95 registers the work itself and
    // gets a fresh key (commonc.c:6532 PRIMENET_REGISTER_ASSIGNMENT), so the stage 2 runs and
    // is credited. Measured: the operator's M3583/M3571 lines carried stale AIDs, PrimeNet
    // rejected them ("ap: no such assignment key ... key: 2F1990C6…"), and both tasks
    // disappeared from Prime95's queue without ever running stage 2.
    //
    // So: an AID that PrimeNet has already rejected is dropped automatically (`rejected_aids`
    // below), which keeps the work alive; live AIDs are kept, which keeps the attribution.
    bool keep_aid = true;
    // Prime95's prime.log (and results.txt) are used to learn which AIDs PrimeNet rejected.
    // Empty = no detection (then every AID is kept).
    std::string prime_log_path;   // <p95 dir>\prime.log
    std::string results_txt_path; // <p95 dir>\results.txt
    // Ledger of the lines this driver delivered (usually "<exe dir>\p95_add_sent.txt"), used
    // to notice that Prime95 threw one away because of a rejected AID and to re-deliver it
    // WITHOUT the AID so no stage-1 work is lost.
    std::string sent_ledger_path;
    bool recover_lost = true; // ini `p95_recover_lost` (default 1)
};

struct P95TransferResult {
    bool enabled = false;               // false when p95_worktodo_path is empty
    bool ok = false;                    // the line is in worktodo.add
    bool publication_uncertain = false; // add published, but a later acknowledgement failed
    int worker = 0;                     // section it went to (0 = appended without a header)
    size_t added = 0;                   // lines appended by THIS call
    size_t pending_delivered = 0;       // previously parked lines that went out with it
    size_t pending_left = 0;            // lines still parked after this call
    size_t recovered = 0;               // lines re-delivered because Prime95 had dropped them
    bool aid_dropped = false;           // this line went over without its (rejected) AID
    std::string add_path;               // worktodo.add that was written
    std::string note;                   // warning text (routing fell back), empty when clean
    std::string error;                  // failure reason, empty on success
};

// Deliver one completed task line. Reports failures in the result and never
// modifies worktodo.txt. Each gate has a bounded budget: pending is fail-fast,
// add waits at most 3 s and ledger publication at most 5 s per operation.
P95TransferResult p95_transfer_deliver(const P95TransferConfig& cfg, const std::string& line);

// --- pieces the GUI/tests use to describe the state without delivering anything ---

// Parse `p95_add_workers` into worker numbers. Returns false when the spec is
// syntactically invalid (caller then falls back to a header-less append); entries outside
// 1..1024 are dropped and reported in `ignored`.
bool p95_parse_add_workers(const std::string& spec, std::vector<int>& workers, int& ignored);

// Path of the pending file for a given driver executable directory.
std::string p95_transfer_pending_path(const std::string& exe_dir);

// Path of the worktodo.add that goes with a given Prime95 worktodo.txt.
std::string p95_transfer_add_path(const std::string& worktodo_path);

// Number of lines waiting (0 when missing). Throws if an existing file is unreadable.
size_t p95_transfer_pending_count(const std::string& pending_path);

// Path of the delivered-lines ledger for a given driver executable directory.
std::string p95_transfer_ledger_path(const std::string& exe_dir);

// ── assignment keys ─────────────────────────────────────────────────────────────────

// The AIDs PrimeNet has rejected, as recorded by Prime95 itself:
//     PrimeNet error 43: Invalid assignment key
//     ap: no such assignment key, GUID: <guid>, key: <32 hex>
// `prime_log` and `results_txt` may be empty/missing (then the set is empty). The AIDs are
// returned upper-case. `err` (optional) receives why nothing could be read.
std::vector<std::string> p95_rejected_aids(const std::string& prime_log,
                                           const std::string& results_txt,
                                           std::string* err = nullptr);

// The AID of an ECMSTAGE2=/ECM2= line ("" when the line has none), and the same line with
// that AID removed -- what Prime95 needs to register the work itself instead of reporting it
// under a key PrimeNet has already rejected.
std::string p95_line_aid(const std::string& line);
std::string p95_strip_aid(const std::string& line);

// Lines from the ledger that Prime95 threw away (their AID was rejected and the line is no
// longer in Prime95's worktodo.txt or worktodo.add) -> AID-stripped, ready to re-deliver.
// Called automatically by p95_transfer_deliver; exposed for the tests.
std::vector<std::string> p95_transfer_recover_lines(const P95TransferConfig& cfg,
                                                    size_t* rejected_count = nullptr);

std::string p95_transfer_quote(const std::string&);

#include "ecm/io/save.h"

namespace ecm_sync {
// Persist responsibility before advancing the local queue. No task is published
// here. Retry validates the recorded source version before copying its save.
bool park_p95(const P95TransferConfig&, const std::string& stage2_line, const ecm_save::Range&,
              std::string& error);

// Publish the validated save first, then its worktodo.add assignment. Failures
// retain responsibility for retry; an ambiguous send is never blindly replayed.
bool flush_p95(const P95TransferConfig&, std::string& error);
size_t pending_p95_count(const P95TransferConfig&);
} // namespace ecm_sync
