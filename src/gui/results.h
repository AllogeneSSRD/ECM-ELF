#pragma once

// Hit bookkeeping for ecm_gui (milestone M5, docs/DEV_ECM_GUI.md section 9).
//
// Two files, on purpose:
//
//   results.json.txt  APPEND-ONLY JSONL, one object per hit, never rewritten. This is
//                     the source of truth: a crash or a power cut costs at most the
//                     last line. Field names follow prime95's results.json.txt
//                     (status/worktype/exponent/factors/b1/...).
//   results.txt       The merged, human-readable table: ONE line per distinct factor,
//                     with the curves/Sigmas that hit it. Derived from the JSONL, so
//                     it can be rebuilt at any time (rebuild_from_jsonl) -- which is
//                     also what the test checks.
//
// Why the merge matters: one factor is often hit by several curves of the same batch
// (measured on M677/B1=1e6: 5 of 8 curves reported the same 31-bit factor), and a
// per-hit list would just repeat it.

#include "log_parse.h"

#include <string>
#include <vector>

namespace ecmgui {

// One hit, with everything the two files need.
struct HitRecord {
    std::string factor;        // decimal
    int exponent = 0;          // Mersenne exponent when known (else 0)
    std::string n_expr;        // the N expression of the task ("" when unknown)
    std::string b1_text;       // compact B1 as written in the save name ("1e6", "110e6")
    double b1 = 0.0;
    int param = -1;            // -1 = unknown
    std::string method;        // gpu | edwards | mont
    int curve = -1;
    unsigned long long sigma = 0;
    bool has_sigma = false;
    std::string save;          // save-file name
    int worker = 0;
    int device = 0;
    std::string task;          // the worktodo line
    std::string timestamp;     // UTC, "2026-09-28T19:20:31Z"
};

// One line of results.txt (a factor plus every hit of it).
struct MergedFactor {
    std::string factor;
    int bits = 0;
    int exponent = 0;
    std::string n_expr;
    std::string b1_text;
    double b1 = 0.0;
    int param = -1;
    std::string method;
    std::vector<int> curves;              // deduplicated, in hit order
    std::vector<unsigned long long> sigmas;
    int hits = 0;
    std::string first_seen;
    std::string last_seen;
    int worker = 0;
    int device = 0;
    std::string save;
};

class ResultsStore {
public:
    // Sets the two paths; loads an existing JSONL (so a restart keeps merging into
    // the same table). Missing files are fine (first run).
    bool init(const std::string &json_path, const std::string &txt_path, std::string &err);

    // Appends one hit to the JSONL and folds it into the merged table.
    bool add(const HitRecord &r, std::string &err);

    // Rewrites results.txt atomically (tmp + replace). Cheap enough to call once per
    // batch of hits, not once per hit.
    bool flush(std::string &err);

    // Rebuilds the merged table from the JSONL alone and rewrites results.txt.
    // Used as a repair path and by the tests ("results.txt must be reproducible").
    bool rebuild_from_jsonl(std::string &err);

    const std::vector<MergedFactor> &factors() const { return merged_; }
    long long hit_count() const { return hits_; }
    const std::string &json_path() const { return json_path_; }
    const std::string &txt_path() const { return txt_path_; }
    const std::string &last_error() const { return last_error_; }

    // Formats one merged line exactly as it is written (exposed for tests and for the
    // UI, which shows the same text).
    static std::string format_line(const MergedFactor &f);

    // "m5351_110e6.save" -> exponent 5351 / B1 text "110e6". Returns false when the
    // name does not follow the m{n}_{b1}.save contract.
    static bool parse_save_name(const std::string &save, int *exponent, std::string *b1_text);

    // Pulls the exponent out of a worktodo-derived N expression like "(1*2^677-1)".
    static int exponent_from_n_expr(const std::string &expr);

private:
    void merge(const HitRecord &r);
    std::string json_path_;
    std::string txt_path_;
    std::vector<MergedFactor> merged_;
    long long hits_ = 0;
    bool dirty_ = false;
    std::string last_error_;
};

} // namespace ecmgui
