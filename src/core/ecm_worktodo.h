#pragma once

// ECMSTAGE2 worktodo parsing + small file helpers used by the queue manager.
//
// Line format (same as pipeline/ecm.py::to_stage2_line):
//   ECMSTAGE2=[<aid>,]<k>,<b>,<n>,<c>,<save_name>,<B2>,<skip_curves>,<curves_to_run>[,"factors"]
// B2 and skip_curves are parsed but ignored (they are for a prime95 stage-2
// consumer). B1 is not a field: it is extracted from save_name (m{n}_{b1}.save).

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

// Parse one ECMSTAGE2 line into `task`. Returns false and fills `err` on failure.
bool ecm_parse_stage2_line(const std::string &line, EcmStage2Task &task, std::string &err);

// Extract B1 from `save_name` (the token between the last '_' and the trailing
// ".save"). The token is parsed with strtod, so "110e6", "1e7" and "1000000" all
// work. Returns false and fills `err` when the name has no usable B1 token.
bool ecm_extract_b1_from_save_name(const std::string &save_name, double *b1_out, std::string &err);

// Compute N = (k*b^n + c) / (f1 * f2 * ...) using exact integer arithmetic.
// Returns false and fills `err` when a field is malformed or a known factor does
// not divide (k*b^n + c) exactly.
bool ecm_compute_stage2_n(const EcmStage2Task &task, mpz_t N, std::string &err);

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

// Append one line (with trailing newline) to `path`, creating parent dirs.
bool ecm_append_text_line(const std::string &path, const std::string &line);

// Copy *.save files from `dir` into `sync_dir_1` and `sync_dir_2` (empty dirs
// are skipped). With `full` every *.save is copied; otherwise only files whose
// mtime is strictly newer than `since` (seconds since epoch).
void ecm_sync_save_files(const std::string &dir, const std::string &sync_dir_1,
                         const std::string &sync_dir_2, bool full,
                         long long since_epoch_seconds);
