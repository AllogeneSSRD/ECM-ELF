#pragma once

// ECM queue-manager configuration, loaded from an INI file (ecm.ini).
//
// Format is deliberately minimal: `key = value`, lines starting with '#' are
// comments, blank lines are ignored. No third-party INI library is used.
//
// On first run, when the INI file is missing, the queue manager writes a default
// template (with bilingual comments, English first) and then proceeds with the
// built-in defaults.
//
// Layout rule (2026-09-24 cleanup): every knob is named after the ONE thing it
// controls, and mutually exclusive choices are ONE field with an enumerated value
// instead of several 0/1 fields.  Fields are grouped by the `method` they apply to,
// so "which key matters right now" is answerable from the group heading alone:
//
//   [queue]        always
//   [method]       selects the stage-1 engine (mutually exclusive)
//   [cpu]          method = edwards | mont
//   [edwards]      method = edwards only
//   [mont]         method = mont only
//   [task]         always (sigma)
//   [handoff]      ecm_p95feeder (Prime95 stage-2 handoff)
//   [gpu]          method = gpu only
//
// `ecm_p95feeder` parses its own, smaller key set from the same file
// (tmp_dir / p95_dir / worktodo / poll_seconds / max_in_flight / worker_allow /
// log_file / verbose / dry_run / keep_tmp) -- see src/core/ecm_p95feeder.cpp.

#include <cstdint>
#include <string>

struct EcmQueueConfig {
    // --- [queue] work queue, logging, saves ---
    std::string worktodo = "worktodo.txt";
    std::string finished = "worktodo.finished.txt";
    std::string log_file = "screen.log";         // empty = screen only
    std::string tmp_dir = ".";                   // stage-1 output dir
    std::string save_sync_dir_1;                 // empty = disabled
    std::string save_sync_dir_2;                 // empty = disabled
    std::string sync_mode = "incremental";       // incremental | full
    std::string progress_color = "cyan";         // none|red|green|yellow|blue|magenta|cyan|white|grey
    bool verbose = true;

    // --- [method] stage-1 engine (exactly one) ---
    //   gpu     : GPU batched stage 1               (the [gpu] group applies).
    //             WHICH GPU implementation runs is fixed at link time, not here:
    //             ecm.exe = OpenCL (src/opencl_backend_glue.cpp),
    //             ecm_cuda.exe = CUDA/CGBN (src/cuda/ecm_cuda_backend.cu).
    //             The startup banner prints the real one via ecm_backend_name().
    //   edwards : CPU Edwards / Atkin-Morain        (the [edwards] group applies)
    //   mont    : CPU Suyama-sigma Montgomery       (the [mont] group applies;
    //             same curve family as gmp-ecm -param 0 / Prime95 sigma_type=1)
    std::string method = "gpu";

    // --- [cpu] shared by method = edwards | mont ---
    // How the modular multiplication is carried out:
    //   auto : AVX512-IFMA 8-curve batch when the CPU supports it, else the scalar
    //          GMP/mpn path
    //   simd : force the batch, and fail loudly if the CPU lacks AVX512-IFMA
    //   gmp  : force the scalar mpn path (1 curve per task; also the A/B baseline)
    std::string backend = "auto";
    // Reduction domain of the SIMD field layer. Only meaningful for backend = simd:
    //   auto       : Mersenne fold when N = 2^k-1 (half the madds per multiply), else
    //                Montgomery reduction
    //   mersenne   : force the fold; N must be 2^k-1 or the run fails
    //   montgomery : force Montgomery reduction even for N = 2^k-1 (A/B baseline)
    std::string field = "auto";
    // Stage-1 worker threads. 0 = auto = min(#tasks, #cores); 1 = serial.  A task is
    // one 8-curve SIMD batch (backend = simd) or one curve (backend = gmp), so the
    // thread count is effectively capped by ceil(curves/8).
    uint32_t stage1_threads = 0;
    // Pin worker t to logical CPU list[t % len]; "" / none / auto = let the OS decide
    // (measured fastest on a hybrid-core laptop -- see docs/ECM_Montgomery_STAGE1.md §13).
    std::string affinity;
    // Save-file name written by the CPU paths.  {n} = Mersenne exponent when
    // N = 2^k-1 (else the bit length), {b1} = compact bound (1e5, 110e6, 12345).
    // NOTE: the reading side does NOT use this pattern -- it extracts B1 from
    // "the token between the last '_' and the trailing .save" (see
    // ecm_extract_b1_from_save_name), so keep that shape when changing it.
    std::string save_name_pattern = "m{n}_{b1}.save";
// On-disk cache for s = torsion*lcm(1..B1); empty = off, otherwise a directory.
// B1 = 260e6 needs ~10 s to build and ~0.3 s to load (validated), see
// src/core/ecm_stage1_exp_cache.h.
std::string exp_cache;

    // --- [edwards] method = edwards only ---
    // NAF window for the Edwards dictionary; 0 = the built-in default (12).
    // Dictionary size is 2^(w-2), so larger w = fewer point additions, more memory.
    int naf_w = 0;

    // --- [mont] method = mont only ---
    // Stage-1 exponent: which "torsion" convention to follow.  This decides which
    // producer a stage-2 handoff is compatible with (see docs §16.7):
    //   lcm      : s = lcm(1..B1)          -- gmp-ecm -param 0; our acceptance metric
    //   choose12 : s = 12 * lcm(1..B1)     -- Prime95 sigma_type=1; use this when
    //                                        Prime95 will run stage 2 on our point
    std::string exponent = "lcm";

    // --- [task] per-run task parameters ---
    // Fixed first sigma (decimal, 0 = random per run). Curve i uses sigma + i.
    // Kept to <= 2^63 on purpose: Prime95's ECMSTAGE2 reader parses SIGMA with
    // mpz_get_str() -> atoll() and a larger value silently rebuilds another curve.
    uint64_t sigma = 0;

    // --- [handoff] Prime95 stage-2 handoff ---
    // Informational only: ecm.exe never writes into the Prime95 directory, it just
    // prints a note when this is set.  The program that actually transfers stage-1
    // saves and appends ECM= worktodo lines is ecm_p95feeder, and it reads the SAME
    // key name from its own feeder.ini (see src/core/ecm_p95feeder.cpp).
    std::string p95_dir;

    // --- [ckpt] mid-stage-1 checkpoint interval ---
    // Seconds between mid-stage-1 checkpoint writes (0 = no periodic autosave; Ctrl+C
    // still saves).  Consumed by every method that has a mid-stage-1 checkpoint:
    //   gpu       : GPU curve buffer + s_partial (src/core/ecm_checkpoint.*; the same
    //               v4 header serves OpenCL and CUDA and records the parametrization)
    //   edwards   : per-curve <tmp_dir>/e{n:07d}_c{k}.ckpt
    //   montgomery: per-curve <tmp_dir>/m{n}_{b1}_c{k}.ckpt (docs/ECM_Montgomery_STAGE1.md §17)
    double ckpt_seconds = 600.0;

    // --- [gpu] method = gpu only (GPU path: OpenCL or CUDA/CGBN) ---
    int device = 0;                              // OpenCL device / CUDA device index
    // Curve parametrization of the GPU stage-1 path:
    //   0 = Suyama param0 (Prime95 sigma_type=1 / gmp-ecm -param 0): the same curves
    //       the CPU --method mont path runs, Z/12 torsion.  Effective divisor
    //       D ~ 21-23 vs ~6.4-7.6 (batch) = a ~3x ratio in D, but the per-curve
    //       success-rate ratio is far smaller (measured 1.30x-1.8x at B1=256 over
    //       bits 15-40; see docs/ECM_Montgomery_STAGE1.md 19.5).  Save file is
    //       written in param0 form (no PARAM=);
    //   3 = gmp-ecm batch parametrization (P=(2:1), d = sigma/2^32): the historical
    //       GPU path, Z/4 torsion, save file carries PARAM=3.
    // 0 is implemented for the CUDA/CGBN kernels; the OpenCL backend refuses it.
    // Default 3 keeps an ini without this key behaving exactly as before.
    int gpu_param = 3;
    uint32_t tpi = 8u;                           // OpenCL only: threads per instance
    int wg_size = 0;                             // OpenCL only: 0 = auto workgroup size
    std::string kernel_mul;                      // OpenCL only: operator overrides (id/alias/auto)
    std::string kernel_sqr;
    std::string kernel_add;
    std::string kernel_sub;
    std::string kernel_special_mult;
};

// Parse `key = value` lines from the INI file at `path` into `cfg`. Unknown keys
// and malformed values are ignored (the field keeps its default), except for the
// pre-2026-09-24 key names, which are mapped with a warning so existing ecm.ini
// files keep working -- see the migration table in the writer's template.
// Returns true if the file was read (even if empty); false if it could not be opened.
bool ecm_queue_config_load(const std::string &path, EcmQueueConfig &cfg);

// Write a commented default INI template to `path`. Returns true on success.
bool ecm_queue_config_write_default(const std::string &path);
