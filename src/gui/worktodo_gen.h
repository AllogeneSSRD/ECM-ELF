#pragma once

// worktodo generator (docs/DEV_ECM_GUI.md 19, M6 scope A).
//
// The operator pastes PrimeNet assignments (ECM2= lines) and gets back a worktodo queue:
// parse -> filter -> dedup -> rewrite -> sort -> save-name check -> emit ECMSTAGE2= lines,
// with the curves of each line taken from the GPU tier that line's N would actually run on.
//
// The reference implementation is tools/ecm_worktodo/ecm.py, and the byte-for-byte test
// (tools/test/test_gui_generator.ps1) compares the two on the same input, so the C++ side
// mirrors ecm.py's field order, quoting and default save pattern exactly. What is NOT
// mirrored on purpose (documented, not silently different):
//   * the exact divisibility check of the known factors (needs bignum; the GUI links no
//     GMP) -- the generator checks the SHAPE (integer > 1) and says so;
//   * the CLI emitters (--emit-cli) and the ECM=/ECM2= Prime95 emitter: scope A is the
//     ECMSTAGE2= queue only;
//   * one invocation writes MULTIPLE [Worker #N] sections (ecm.py needs one run per worker).
//
// Why an estimate is enough for the tiers: the kernel picks the smallest CGBN container
// with `bits >= bits(N) + CARRY_BITS`, and a tier is 128 bits wide or more, so an error of
// a bit or two only matters near a boundary -- and near a boundary the recommendation is
// intentionally the larger tier anyway.

#include <string>
#include <utility>
#include <vector>

namespace ecmgui {

// ── GPU profile (`ecm_cuda.exe --gpu-info`, D4) ───────────────────────────────────────

struct GpuTier {
    int bits = 0;                 // CGBN container size
    int tpb = 0;                  // threads per block
    int tpi = 0;                  // threads per curve
    int ipb = 0;                  // curves per block (tpb / tpi)
    int blocks_per_sm = 0;        // register-allowed resident blocks per SM
    int blocks_min = 0;           // one block per SM
    long long curves_min = 0;     // blocks_min * ipb  (the kernel's under-occupancy value)
    int blocks_wave = 0;          // register-allowed block slots
    long long curves_wave = 0;    // blocks_wave * ipb (a whole multiple = no partial wave)
};

struct GpuProfile {
    bool valid = false;
    bool not_applicable = false;  // the OpenCL build: --gpu-info prints "not_applicable"
    int device = 0;
    int sm_count = 0;
    int carry_bits = 6;
    int gpu_param = 3;
    int fold = 0;
    std::string name;
    std::string error;            // why it is not valid
    std::vector<GpuTier> tiers;   // ascending `bits`

    // The tier the kernel would pick for an N of `bits` bits (nullptr = none is large
    // enough). Same rule as kernels/cuda/cgbn_stage1.cu and `--gpu-info --bits`.
    const GpuTier *pick(int bits) const;
};

// Parses the `key=value` text of --gpu-info (the `tier …` lines carry the tier fields).
bool parse_gpu_info(const std::string &text, GpuProfile &out, std::string &err);

// ── tasks ────────────────────────────────────────────────────────────────────────────

struct GenTask {
    int idx = 0;                        // input order (stable tie-break, like ecm.py)
    std::string keyword;                // "ECM2" | "ECM"
    std::string aid;                    // may be empty / "N/A"
    std::string fft2;
    std::string k, b, c;                // original spelling
    long long n = 0;
    std::string b1, b2 = "0";           // original spelling ("110e6" stays "110e6")
    long long curves = 100;
    std::string sigma;
    std::vector<std::string> factors;
    int bits = 0;                       // effective bits of N (filled by the pipeline)
    long long curves_out = 0;           // curves the emitted line carries
};

struct GenOptions {
    std::string save_pattern = "m{n}_{b1}.save";
    long long skip_curves = 0;
    // >= 0: use this many curves for every line (what ecm.py's --gpu-curves does, and what
    // the byte-for-byte test uses). < 0: use the recommendation.
    long long curves_fixed = -1;
    bool use_recommended = true;        // ☑ "use the recommended curves" (default on)
    int blocks_per_sm = 2;              // the `n` in curves = n * sm_count * ipb
    bool dedup = true;
    std::string sort_by = "n";          // "", "n", "n,k,b,c", "b1", … (ecm.py's field names)
    bool sort_desc = false;
    bool sort_factors = false;          // false keeps the input order (ecm.py default)
    std::string set_b1;                 // "" = keep
    std::string set_b2;                 // "" = keep
    bool set_has_na = false;
    long long min_n = -1, max_n = -1;              // -1 = no limit
    long long min_curves = -1, max_curves = -1;
    bool valid = false;
    std::string error;
    // (worker number, device) pairs read from the ini's [Worker #N] sections.
    std::vector<std::pair<int, int>> worker_devices;
    int target_device = 0;
};

struct GenSegment {
    int worker = 0;                     // 0 = no section header
    std::vector<std::string> lines;
};

struct GenResult {
    bool ok = false;
    std::string error;
    std::vector<GenTask> tasks;         // after the whole pipeline
    std::vector<std::string> warnings;
    std::vector<GenSegment> segments;   // only workers that got at least one line
    std::string text;                   // exactly what "apply append" would write (CRLF)
    int read = 0;                       // assignments seen
    int skipped_comment = 0;
    int skipped_other = 0;              // ECMSTAGE2=/other prefixes
    int skipped_unknown = 0;            // unparseable
    int filtered = 0;
    int duplicates = 0;
    std::vector<std::string> parse_errors;   // one line per rejected input line
};

// ── pipeline ─────────────────────────────────────────────────────────────────────────

// bits(k*b^n + c) - sum(bits(factor)), plain arithmetic (no bignum). Returns false with
// `err` when k/b/c/n are not usable.
bool effective_bits(const std::string &k, const std::string &b, long long n,
                    const std::string &c, const std::vector<std::string> &factors,
                    int &bits_out, std::string &err);

// Parses ONE ECM=/ECM2= line. Returns false with `err` (the same reasons ecm.py reports).
bool parse_assignment(const std::string &line, int idx, GenTask &out, std::string &err);

// The whole pipeline. `gpu` may be invalid when every line uses `curves_fixed`
// (use_recommended == false), otherwise it is required.
GenResult generate(const std::string &input, const GpuProfile &gpu, const GenOptions &opt);

// Save name: `pattern` with {k}{b}{c}{n}{b1}{b2} substituted (ecm.py's render_save_name).
std::string render_save_name(const std::string &pattern, const GenTask &t);
// nullptr when the name is fine, else the reason (mirrors ecm_extract_b1_from_save_name:
// must end with ".save" and carry a _<B1> token whose value is a positive number).
const char *check_save_name(const std::string &name);

// ── applying ─────────────────────────────────────────────────────────────────────────

// What the target file looked like when the preview was generated: apply_append() refuses
// to write when it changed in the meantime (the queue may be running and the user may have
// edited it), which is the "re-validate mtime before apply" rule.
struct FileStamp {
    bool exists = false;
    long long size = 0;
    long long mtime = 0;
};
bool stamp_file(const std::string &path, FileStamp &out);
// Appends `text` to `path` (creating it when missing), adding a separating newline when the
// file does not end with one -- exactly ecm.py's write_lines(append=True) rule.
bool apply_append(const std::string &path, const std::string &text, const FileStamp &expected,
                  std::string &err, size_t *appended = nullptr);

} // namespace ecmgui
