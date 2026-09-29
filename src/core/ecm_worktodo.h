#pragma once

// Worktodo parsing + small file helpers used by the queue manager.
//
// Two line formats are supported (dispatched by prefix):
//
// 1) ECMSTAGE2= (CUDA-oriented, unchanged):
//      ECMSTAGE2=[<aid>,]<k>,<b>,<n>,<c>,<save_name>,<B2>,<skip_curves>,<curves_to_run>[,"factors"]
//    B2 and skip_curves are parsed but ignored (they are for a prime95 stage-2
//    consumer). B1 is not a field: it is extracted from save_name (m{n}_{b1}.save).
//
// 2) ECM= / ECM2= (Prime95 native ECM worktodo; both prefixes are equivalent —
//    added for the Edwards CPU path):
//      ECM=[<AID>,|N/A,|<nul>][FFT2=<fftl>,|<nul>]<k>,<b>,<n>,<c>,<B1>[,<B2>][,<curves_to_run>][,<specificsigma>][,"comma-separated-list-of-known-factors"]
//    B2 defaults to 0, curves_to_run defaults to 100 (matching Prime95).

#include <cstdint>
#include <string>
#include <vector>

#include <gmp.h>

struct EcmStage2Task {
    std::string raw_line;        // full original line, for finished/error output
    std::string aid;             // optional assignment id (empty if none)
    std::string k;               // integer string
    std::string b;               // integer string
    std::string c;               // signed integer string
    unsigned long n = 0;         // exponent
    std::string save_name;       // save file name (contains B1)
    uint32_t curves_to_run = 0;  // GPU curves per batch
    std::vector<std::string> factors; // known factors (optional)
};

// Prime95 ECM2= task (k*b^n+c, explicit B1/B2, optional 64-bit sigma + factors).
struct Ecm2Task {
    std::string raw_line;         // full original line
    std::string aid;              // "" | "N/A" | real assignment id
    std::string fft2;             // "" | FFT length string (informational only)
    std::string k;                // integer string
    std::string b;                // integer string
    std::string c;                // signed integer string
    unsigned long n = 0;          // exponent
    double B1 = 0.0;
    double B2 = 0.0;
    uint32_t curves_to_run = 0;
    bool has_sigma = false;
    uint64_t sigma = 0;           // optional specific sigma (64-bit)
    std::vector<std::string> factors; // known factors (optional)
};

// Parse one ECMSTAGE2 line into `task`. Returns false and fills `err` on failure.
bool ecm_parse_stage2_line(const std::string &line, EcmStage2Task &task, std::string &err);

// Parse one Prime95 ECM2= line into `task`. Returns false and fills `err` on failure.
bool ecm_parse_ecm2_line(const std::string &line, Ecm2Task &task, std::string &err);

// Extract B1 from `save_name` (the token between the last '_' and the trailing
// ".save"). The token is parsed with strtod, so "110e6", "1e7" and "1000000" all
// work. Returns false and fills `err` when the name has no usable B1 token.
bool ecm_extract_b1_from_save_name(const std::string &save_name, double *b1_out, std::string &err);

// Compute N = (k*b^n + c) / (f1 * f2 * ...) using exact integer arithmetic.
// Returns false and fills `err` when a field is malformed or a known factor does
// not divide (k*b^n + c) exactly.
bool ecm_compute_stage2_n(const EcmStage2Task &task, mpz_t N, std::string &err);

// Compute N = (k*b^n + c) / (f1 * f2 * ...) for a Prime95 ECM2= task.
bool ecm_compute_ecm2_n(const Ecm2Task &task, mpz_t N, std::string &err);

// Read the first non-empty, non-comment line of `path`. Returns false if the
// file is missing or has no task line.
bool ecm_worktodo_first_line(const std::string &path, std::string &line);

// Advance worktodo past the first task line: on success the line is removed; on
// error it is replaced in place by "# ERROR <line>" (kept as a comment so the
// user can inspect it). `first_line` must match the current first task line.
// Returns true if the file was rewritten.
enum class WorktodoAction { Remove, MarkError };
bool ecm_worktodo_advance(const std::string &path, const std::string &first_line,
                          WorktodoAction action);

// ---------------------------------------------------------------------------
// Sections (2026-10, D2 in docs/DEV_ECM_WORKTODO.md + docs/DEV_ECM_GUI.md)
//
// One worktodo file can serve several workers, Prime95 style:
//
//     ECMSTAGE2=...                 <- before any header: belongs to worker 1
//     [Worker #2]
//     ECMSTAGE2=...                 <- belongs to worker 2 only
//
// Rules:
//   * a worker consumes and advances ONLY its own section;
//   * headers, comments, blank lines and foreign sections are preserved verbatim
//     (and in order) when the file is rewritten;
//   * a repeated header for the same worker is fine (appended-to files do that);
//   * `worker <= 0` keeps the legacy behaviour: no section filter, i.e. the whole
//     file is one queue (existing single-worker setups are unaffected).
// ---------------------------------------------------------------------------

// Shared section-header syntax: "[Worker #N]" (case- and space-insensitive).
// Returns N (> 0) for a worker header, otherwise 0. When `is_bracket_line` is not
// null it reports whether the line was some other "[...]" line.
int ecm_worktodo_parse_worker_header(const std::string &line, bool *is_bracket_line);

// Section-aware variants of the two functions above.
bool ecm_worktodo_first_line(const std::string &path, int worker, std::string &line);
bool ecm_worktodo_advance(const std::string &path, int worker, const std::string &first_line,
                          WorktodoAction action);

// Distinct worker indices that own at least one task line (worker 1 when the file
// has no headers), ascending. The GUI uses this to flag sections that have no
// matching worker in ecm.ini. Returns false when the file cannot be opened.
bool ecm_worktodo_list_workers(const std::string &path, std::vector<uint32_t> &workers,
                               std::string &err);

// Append one line (with trailing newline) to `path`, creating parent dirs.
bool ecm_append_text_line(const std::string &path, const std::string &line);

// Copy *.save files from `dir` into `sync_dir_1` and `sync_dir_2` (empty dirs
// are skipped). With `full` every *.save is copied; otherwise only files whose
// mtime is strictly newer than `since` (seconds since epoch).
void ecm_sync_save_files(const std::string &dir, const std::string &sync_dir_1,
                         const std::string &sync_dir_2, bool full,
                         long long since_epoch_seconds);
