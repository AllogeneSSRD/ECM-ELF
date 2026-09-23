#pragma once
// p95_worktodo.h — Prime95 worktodo.txt / worktodo.add / prime.txt helpers.
//
// Used by the standalone handoff feeder (ecm_p95feeder), which moves locally
// produced stage-1 saves (e{n:07d}_c{k}.tmp) into a running Prime95 instance.
//
// File formats (verified against real Prime95 30.x worktodo.txt files):
//
//   worktodo.txt                     worktodo.add
//   --------------                   ------------
//   [Worker #1]                      [Worker #2]
//   ECM=...,1,2,12323,-1,55000000,0,664,"f1,f2"
//                                    ECM=...,1,2,991,-1,1000000,0,1,105413044550089
//   [Worker #2]
//   # commented-out line
//
// Prime95 documents worktodo.add as: "Prime95/mprime will periodically look for
// worktodo.add and append the entries from each '[Worker #]' section. Then the
// worktodo.add file will be deleted." (undoc.txt) — that is the race-free
// delivery channel, so the feeder never has to rewrite worktodo.txt.
//
// A line is "active work" when it is neither blank nor a '#' comment.
// A line is a "handoff" (stage-2-only continuation produced by our stage 1) when
// it is an ECM=/ECM2= line with curves == 1 and a non-zero specific sigma.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// One "[Worker #N]" section. Lines before the first section header are kept in a
// section with worker == 0 so a header-less worktodo survives a round trip.
struct P95WorkerSection {
    int worker = 0;
    std::vector<std::string> lines;   // raw lines, no trailing newline
};

// Parse a Prime95 worktodo file. Returns false (and fills `err`) if it cannot be
// read. A missing file yields an empty `sections` and returns true.
bool p95_read_worktodo(const std::string &path, std::vector<P95WorkerSection> &sections,
                       std::string &err);

// True when the line carries work (non-blank, not a '#' comment).
bool p95_line_is_active(const std::string &line);

// True when the line is an ECM=/ECM2= stage-2 continuation: curves == 1 and a
// non-zero specific sigma.
bool p95_line_is_handoff(const std::string &line);

size_t p95_count_active(const P95WorkerSection &s);
size_t p95_count_handoff(const P95WorkerSection &s);

// Number of handoff lines across all worker sections.
size_t p95_total_handoff(const std::vector<P95WorkerSection> &sections);

// Merge `assignments` (worker number -> ECM= line) into `path` as `[Worker #N]`
// sections, preserving the rest of the file. With append == false the file is
// rewritten from only the assignments. Returns false and fills `err` on I/O
// failure.
bool p95_write_worktodo_add(const std::string &path,
                            const std::vector<std::pair<int, std::string>> &assignments,
                            bool append, std::string &err);

// Read an integer key from a Prime95 prime.txt (e.g. MaxHighMemWorkers).
// Accepts "Key=value" with surrounding whitespace; '#' comments are skipped.
bool p95_read_prime_int(const std::string &prime_txt_path, const std::string &key,
                        long long &value);

// Prime95 save-file name for ECM work on (k=1,b=2,c=-1): "e{n:07d}".
std::string p95_ecm_save_name(uint32_t n);

// Read the `state` field (offset 0x40) of a Prime95 ECM save file.
// Returns 0xFFFFFFFF when the file is missing/unreadable.
uint32_t p95_read_save_state(const std::string &path);

// List "*.tmp" file names (name only, not the full path) in `dir`, sorted by
// modification time (oldest first). Returns false if the directory is unreadable.
bool p95_list_tmp_files(const std::string &dir, std::vector<std::string> &names);
