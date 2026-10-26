#include "../../../io/files.h"
#include "ecm/stage1/progress.h"
#include "ecm/stage1/cpu_runtime.h"
#include <future>
#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <utility>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <direct.h> /* _mkdir for the local save directory */
#include <windows.h>
#include <process.h>
#define access _access
#define getpid _getpid
#else
#include <unistd.h>
#endif

#ifdef _MSC_VER
#include <intrin.h> /* __rdtsc() for the Prime95-style sigma entropy mix */
#endif

#include <gmp.h>

#include "ecm/stage1/execution.h"
#include "ecm/curve/edwards.h"
#include "ecm/stage1/sigma.h"
#include "ecm/io/save.h"
#include "ecm/montgomery_stage1/opencl/checkpoint.h"
#include "../../../io/stage1_checkpoint.h"
#include "ecm/montgomery_stage1/cpu/checkpoint.h"
#include "ecm/edwards_stage1/cpu/ecm_edwards_cpu.h"
#include "ecm/montgomery_stage1/cpu/ecm_mont_cpu.h"
#include "ecm/montgomery_stage1/cpu/simd_mont_curve.h"
#include "ecm/cpu/arithmetic/ifma/simd_mont_ifma.h"
#include "../../../io/prime95_save.h"
#include "ecm/io/save.h"
#include "ecm/platform/logging.h"
#include <iomanip>
#include <regex>
#include "ecm/edwards_stage1/cpu/simd_edwards.h" /* SoA 批量 Edwards 层 (AVX512-IFMA TU, 见 §13.8) */
#include <intrin.h>
#include <windows.h>

namespace ecm_cpu_stage1 {
using ecm_stage1::ComputationResult;
using ecm_stage1::compute_batch_s;
using ecm_stage1::CurveKind;
using ecm_stage1::CurveOutcome;
using ecm_io::ensure_dir;
using ecm_stage1::ExecutionOptions;
using ecm_save::format_name;
using ecm_stage1::select_curve_sigma;

namespace {
bool fixed_sigma_batch_exceeds_prime95_range(uint64_t sigma, uint32_t curves) {
    return curves != 0 &&
           (sigma > static_cast<uint64_t>(INT64_MAX) ||
            static_cast<uint64_t>(curves - 1) > static_cast<uint64_t>(INT64_MAX) - sigma);
}

} // namespace
using ecm_stage1::stage1_now_ms;
using ecm_stage1::Stage1Options;
using ecm_stage1::stdout_is_tty_local;
using ecm_stage1::validate_stage1_factor;

// Run one stage-1 batch. N and the caller's n_expr are borrowed (not cleared);
// factors / array_found are allocated here and returned via `out` (caller frees).
/* ---------------------------------------------------------------------------
 * Shared automatic Stage1 save names (ecm.toml: save_name_pattern).
 *
 *   {curve}: e / m0 / m2 / m3; {target}: m<original exponent> / bit<actual bits>;
 *   {n}  : original assignment exponent when available, else actual N bits;
 *   {b1} : compact bound -- 100000 -> "1e5", 110000000 -> "110e6", 12345 -> "12345".
 *
 * ecm_extract_b1_from_save_name() documents the same pattern on the reading side.
 * ------------------------------------------------------------------------- */

/* Standalone codec fallback stem; the driver routes records into its batch. */
static std::string mont_save_stem(const std::string& pattern, const mpz_t N, double B1) {
    std::string f = format_name(pattern, N, B1);
    const size_t dot = f.rfind('.');
    if (dot != std::string::npos)
        f.erase(dot);
    return f;
}

// ---------------------------------------------------------------------------
// Suyama-sigma Montgomery stage 1 (Prime95 sigma_type = 1 / gmp-ecm -param 0).
//
// Same result contract as run_edwards_stage1: fills out->factors / out->array_found
// and returns an ECM_* code, so the CLI caller needs no changes.
//
//   * s = torsion * lcm(1..B1)   (torsion 1 matches gmp-ecm, 12 matches Prime95;
//     see docs/architecture/STAGE1.md 4.1)
//   * 8 curves per AVX512-IFMA batch when available, else the scalar MPN path
//   * ONE shared save file per task (same N and B1, one self-contained line per
//     curve -- the reference reader parses METHOD/B1/N from every line, so the
//     lines must stay self-contained; docs 9.2)
// ---------------------------------------------------------------------------
/* Default worker count for the Montgomery stage-1 path.
 *
 * The unit of parallel work differs per backend: the SIMD path runs 8 curves per
 * batch, so its parallelism is capped by the number of batches (curves/8) and the
 * scalar path by the number of curves.  Extra threads beyond that would only
 * contend, so clamp -- 8 curves under the IFMA SIMD implementation is one thread of real
 * work no matter how many cores the machine has. */
static uint32_t mont_default_threads(uint32_t curves, bool use_simd) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0)
        hw = 1;
    uint32_t tasks = use_simd ? ((curves + IFMA_LANES - 1) / IFMA_LANES) : curves;
    if (tasks == 0)
        tasks = 1;
    uint32_t t = (uint32_t)hw;
    if (t > tasks)
        t = tasks;
    return t;
}

/* ---------------------------------------------------------------------------
 * Montgomery stage-1 progress + checkpoint glue
 * ------------------------------------------------------------------------- */

/* Checkpoint / progress granularity: aim at ~512 callbacks per curve (a progress
   update every ~0.2% of the ladder) with a 4096-bit floor, rounded up to a power
   of two so the offsets in the checkpoint files stay readable.  It is derived
   from the exponent only, so a resumed run with the same command line computes
   the identical offsets and can match what the interrupted run wrote. */
static size_t mont_ckpt_chunk_bits(size_t nbits) {
    size_t p = 4096;
    const size_t want = nbits / 512;
    while (p < want && p < ((size_t)1 << 30))
        p <<= 1;
    return p;
}

/* Progress + pause decision for one SIMD batch.  Called from inside the ladder
   every `chunk` bits, with the batch holding the whole worker thread. */
struct MontProgCtx {
    std::atomic<uint64_t>* lane_bits; /* [curves]: exponent bits done per curve */
    const uint32_t* lane_curve;       /* [lanes] : curve index of each lane */
    uint32_t lanes;
    uint32_t total_curves;
    size_t nbits;
    std::atomic<long long>* last_ckpt_ms; /* shared by all workers: one timer */
    long long interval_ms;                /* < 0 = no periodic autosave */
};

static int mont_progress_cb(void* p, size_t bitnum) {
    MontProgCtx* pc = (MontProgCtx*)p;
    for (uint32_t k = 0; k < pc->lanes; k++)
        pc->lane_bits[pc->lane_curve[k]].store((uint64_t)bitnum, std::memory_order_relaxed);

    /* Work unit = one curve; the fractional part of a lane is bitnum/nbits.  A
       running sum over the per-curve counters is what keeps the number monotone
       when several workers report at different offsets, and it needs no lock. */
    uint64_t sum = 0;
    for (uint32_t i = 0; i < pc->total_curves; i++)
        sum += pc->lane_bits[i].load(std::memory_order_relaxed);
    const double done_f = pc->nbits ? (double)sum / (double)pc->nbits : 0.0;
    {
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        stage1_progress_set(done_f, pc->total_curves);
    }

    if (ecm_stage1_stop_requested())
        return 1; /* Ctrl+C: stop at this chunk */
    if (pc->interval_ms > 0) {
        const long long now = stage1_now_ms();
        if (now - pc->last_ckpt_ms->load(std::memory_order_relaxed) >= pc->interval_ms) {
            pc->last_ckpt_ms->store(now, std::memory_order_relaxed);
            return 1; /* autosave interval elapsed */
        }
    }
    return 0;
}

/* Scalar ladder wrapper: same policy, the offset arrives inside the state. */
static int mont_progress_cb_scalar(void* p, const mont_ladder_state_t* st) {
    return mont_progress_cb(p, st->bitnum);
}

ComputationResult run_mont_stage1(const ecm_stage1::Stage1Request& request, ExecutionOptions& opt,
                                  ecm_stage1_checkpoint::Context& checkpoint) {
    const auto N = request.modulus;
    const auto B1 = request.b1, B2 = request.b2;
    const auto curves = request.curves;
    ComputationResult result;
    result.failure = ecm_stage1::FailureSource::Preparation;
    if (curves == 0) {
        std::cerr << "curves must be > 0" << std::endl;
        return result;
    }

    ecm_edwards::Integer scalar;
    auto* s = scalar.v;
    const int torsion = opt.execution.selection.scalar_construction ==
                                ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12
                            ? 12
                            : 1;
    std::string s_status;
    const size_t s_bits = mont_build_s(s, (uint64_t)B1, (uint64_t)torsion, &s_status,
                                       opt.persistence.exponent_cache_dir);
    if (s_bits == 0) {
        /* only reachable for B1 > 5e9 or an allocation failure; say so instead of
           running a ladder over the torsion-only exponent */
        ecm_ts_fprintf(stderr,
                       "ERROR: cannot build the stage-1 exponent for B1=%.0f "
                       "(bound must be <= 5e9 and the prime sieve must fit in memory)\n",
                       B1);

        return result;
    }

    const bool isa = driver_simd_isa_ok();
    if (opt.execution.selection.cpu_implementation == ecm_stage1::CpuImplementation::IfmaSimd &&
        !isa) {
        ecm_ts_fprintf(stderr, "ERROR: --impl simd requested but this CPU lacks "
                               "AVX512-F/DQ/IFMA. Use auto or scalar.\n");

        return result;
    }
    const bool use_simd =
        (opt.execution.selection.cpu_implementation == ecm_stage1::CpuImplementation::IfmaSimd)
            ? true
        : (opt.execution.selection.cpu_implementation == ecm_stage1::CpuImplementation::GmpScalar)
            ? false
            : (isa && curves >= 2);
    ecm_ts_fprintf(stdout, "curve model     : montgomery (Suyama sigma, %s, torsion=%d)\n",
                   use_simd ? "AVX512-IFMA 8-lane batch" : "scalar mpn", torsion);
    ecm_ts_fprintf(stdout, "stage1 exponent : s_bits=%zu (lcm(1..%.0f) x %d) [%s]\n", s_bits, B1,
                   torsion, s_status.c_str());

    ecm_edwards::Array factors_owner(curves);
    auto* factors = factors_owner.data();
    std::vector<int> statuses(curves, ECM_NO_FACTOR_FOUND);
    auto* array_found = statuses.data();

    /* The exponent bit array depends only on (B1, torsion) -- not on N, not on sigma
       -- so it is built once here and read by every task below.  It is built before
       the sigmas because the checkpoint pre-pass needs s_bits to validate a resume. */
    size_t nbits = 0;
    std::unique_ptr<uint8_t, decltype(&std::free)> bits_owner(mont_expand_bits(s, &nbits),
                                                              std::free);
    auto* bits = bits_owner.get();
    if (!bits)
        return result;
    const size_t ckpt_chunk = mont_ckpt_chunk_bits(nbits);

    /* ---- mid-stage-1 checkpoints (docs/architecture/STAGE1.md) ----------
       One independently resumable record per curve in a shared batch file. The
       interval comes from ckpt_seconds / --ckpt; 0 = no autosave (Ctrl+C still
       checkpoints, which is what turns an interrupted run into a resumed one). */
    const bool use_ckpt = !opt.persistence.tmp_dir.empty();
    const std::string ckpt_stem = use_ckpt
                                      ? (opt.persistence.tmp_dir + "/" +
                                         mont_save_stem(opt.persistence.save_name_pattern, N, B1))
                                      : std::string();
    const long long ckpt_interval_ms =
        (opt.persistence.ckpt_ms > 0 && opt.persistence.ckpt_ms != ULONG_MAX)
            ? (long long)opt.persistence.ckpt_ms
            : -1;

    /* per-curve results: the normalised x for misses, the factor for hits */
    ecm_edwards::Array xs(curves);
    std::vector<int> hits(curves, 0);
    auto* hit = hits.data();

    /* curve state recovered from checkpoints */
    std::vector<size_t> bit_off(curves, 0); /* exponent bits already consumed */
    std::vector<char> ckpt_done(curves, 0); /* result restored: curve is finished */
    uint32_t n_done_ck = 0, n_inflight_ck = 0, n_seeded = 0;

    /* Sigmas: fixed base (from -sigma, then sigma+i) or random.
     *
     * Interoperability constraint (why this is not a full 64-bit random): Prime95's
     * ECMSTAGE2 path -- the one that runs stage 2 on a gmp-ecm/PARAM=0 save file --
     * parses SIGMA with  mpz_get_str() -> atoll()  (ecm.cpp:7375).  A sigma >= 2^63
     * overflows atoll, so Prime95 would rebuild a DIFFERENT curve from a mangled
     * sigma and quietly waste the stage-2 run (the bitmap/x-coordinate it loads came
     * from our curve).  Prime95 itself only ever generates sigmas below 2^53
     * (ecm.cpp:7416: (rand()&0x1F)<<48 + (rand()&0xFFFF)<<32 + rdtsc bits), so we use
     * the same generator the Edwards path already uses.  gmp-ecm's own reader is
     * mpz-based and has no such limit, but a sigma that works with every consumer is
     * strictly better -- and 2^53 still leaves ~9e15 curves to choose from. */
    std::vector<uint64_t> sigmas(curves);
    std::atomic<int> checkpoint_failed(0);
    for (uint32_t i = 0; i < curves; i++) {
        const uint64_t fixed_sigma = opt.identity.fixed_sigma64 + i;
        bool adopted = false;
        if (use_ckpt) {
            /* A checkpoint owns its curve's sigma.  Adopting it is what makes the
               identical command line continue the identical curve set; without this
               a resumed run would draw fresh random sigmas and the checkpoints
               would never match.  (The Edwards path decides the same way.) */
            mont_ckpt_t ck;
            mont_ckpt_init(&ck);
            std::string why;
            if (mont_ckpt_read(checkpoint, mont_ckpt_path(checkpoint, ckpt_stem, i), N, B1, torsion,
                               nbits, &ck, &why) &&
                ck.curve == i && (!opt.identity.sigma_fixed || ck.sigma == fixed_sigma)) {
                sigmas[i] = ck.sigma;
                adopted = true;
                if (ck.status == MONT_CKPT_DONE) {
                    ckpt_done[i] = 1;
                    n_done_ck++;
                    if (ck.hit) {
                        mpz_set(factors[i], ck.factor);
                        array_found[i] = ECM_FACTOR_FOUND_STEP1;
                        mpz_set(xs[i], ck.factor);
                        hit[i] = 1;
                    } else {
                        mpz_set(xs[i], ck.xout);
                    }
                } else if (ck.bitnum > 0) {
                    bit_off[i] = ck.bitnum;
                    n_inflight_ck++;
                } else {
                    n_seeded++;
                }
            }
            mont_ckpt_clear(&ck);
        }
        if (!adopted) {
            uint64_t saved = 0;
            checkpoint.saved_sigma(i, saved);
            sigmas[i] = checkpoint.sigma(
                i, select_curve_sigma(saved, opt.identity.sigma_fixed,
                                      opt.identity.fixed_sigma64, i));
            if (use_ckpt) {
                /* Seed record: pins the sigma before any ladder work happens, so a
                   run that is killed while this curve is still queued resumes the
                   SAME curve later instead of substituting a new random one. */
                mont_ckpt_t ck;
                mont_ckpt_init(&ck);
                ck.status = MONT_CKPT_INFLIGHT;
                ck.bitnum = 0;
                mont_ckpt_ident(&ck, i, sigmas[i], B1, torsion, nbits, use_simd ? "ifma" : "mpn");
                if (mont_ckpt_write(checkpoint, mont_ckpt_path(checkpoint, ckpt_stem, i), N, &ck) !=
                    0)
                    checkpoint_failed.store(1);
                mont_ckpt_clear(&ck);
                n_seeded++;
            }
        }
    }
    if (opt.identity.sigma_fixed &&
        fixed_sigma_batch_exceeds_prime95_range(opt.identity.fixed_sigma64, curves)) {
        ecm_ts_fprintf(
            stderr,
            "WARNING: sigma >= 2^63; Prime95's ECMSTAGE2 reader (atoll) cannot read it.\n"
            "         Use gmp-ecm for stage 2 with these sigmas, or pass a smaller -sigma.\n");
    }

    /* Work split.  A "task" is one SIMD batch (<= 8 curves, one thread) or, on the
       scalar backend, one curve.  Tasks are handed out by an atomic counter, so a
       worker that finishes a batch picks up the next one -- with uniform batch cost
       that is equivalent to a static split, but it degrades gracefully.

       Curves are grouped by "bits already consumed" (descending) before chunking,
       because the lanes of a batch share ONE exponent prefix: every lane has to
       start at the same offset.  On a fresh run every offset is 0, so this is
       exactly the old sequential split; after a resume it costs at most one partly
       empty batch per batch that the interrupted run was in the middle of. */
    std::vector<uint32_t> pending;
    for (uint32_t i = 0; i < curves; i++)
        if (!ckpt_done[i])
            pending.push_back(i);
    std::stable_sort(pending.begin(), pending.end(),
                     [&](uint32_t a, uint32_t b) { return bit_off[a] > bit_off[b]; });

    struct MontTask {
        uint32_t first, count;
        size_t start_bit;
    };

    std::vector<MontTask> task_list;
    {
        const size_t per_task = use_simd ? (size_t)IFMA_LANES : (size_t)1;
        for (size_t p = 0; p < pending.size();) {
            size_t q = p;
            while (q < pending.size() && (q - p) < per_task &&
                   bit_off[pending[q]] == bit_off[pending[p]])
                q++;
            MontTask tk = {(uint32_t)p, (uint32_t)(q - p), bit_off[pending[p]]};
            task_list.push_back(tk);
            p = q;
        }
    }
    const uint32_t tasks = (uint32_t)task_list.size();
    uint32_t nthreads = opt.execution.stage1_threads ? opt.execution.stage1_threads
                                                     : mont_default_threads(curves, use_simd);
    if (nthreads > tasks)
        nthreads = tasks;
    if (nthreads < 1)
        nthreads = 1;
    ecm_ts_fprintf(stdout, "stage1 threads  : %u worker(s) x %u task(s) of %s\n", nthreads, tasks,
                   use_simd ? "8 curves" : "1 curve");
    if (use_ckpt) {
        char iv[48];
        if (ckpt_interval_ms > 0)
            snprintf(iv, sizeof(iv), "%.3g s", ckpt_interval_ms / 1000.0);
        else
            snprintf(iv, sizeof(iv), "off (Ctrl+C still saves)");
        ecm_ts_fprintf(stdout,
                       "checkpoint      : %s_c*.ckpt  [%u done, %u mid-ladder, %zu to run]"
                       "  autosave %s\n",
                       ckpt_stem.c_str(), n_done_ck, n_inflight_ck, pending.size(), iv);
    }
    if (!opt.execution.affinity_cpus.empty()) {
        std::string a;
        for (size_t i = 0; i < opt.execution.affinity_cpus.size(); i++) {
            if (i)
                a += ",";
            a += std::to_string(opt.execution.affinity_cpus[i]);
        }
        ecm_ts_fprintf(stdout, "affinity        : %s (worker t -> cpu[%s][t %% %zu])\n", a.c_str(),
                       a.c_str(), opt.execution.affinity_cpus.size());
    }
    fflush(stdout);

    std::atomic<uint32_t> next_task(0);
    std::atomic<uint32_t> found_atomic(0);
    std::atomic<int> init_failed(0);
    std::atomic<int> paused_flag(0);
    std::atomic<long long> last_ckpt_ms(stage1_now_ms());
    /* per-curve progress in exponent bits: restored curves start at their own
       offset (done ones at nbits), so the bar is right from the first sample */
    std::vector<std::atomic<uint64_t>> lane_bits(curves);
    for (uint32_t i = 0; i < curves; i++)
        lane_bits[i].store(ckpt_done[i] ? (uint64_t)nbits : (uint64_t)bit_off[i],
                           std::memory_order_relaxed);

    auto worker = [&]() {
        if (use_simd) {
            mont_soa_ctx_t ctx; /* per-thread scratch pool */
            if (mont_soa_init(&ctx, N,
                              to_ifma_field_mode(opt.execution.selection.simd_reduction)) != 0) {
                init_failed.store(1);
                return;
            }
            /* Report the field layer the context actually resolved to: with
               field = auto the choice depends on N (Mersenne fold for 2^k-1), so
               printing the request alone would be misleading. */
            {
                static std::atomic<int> shown(0);
                if (shown.fetch_add(1) == 0)
                    ecm_ts_fprintf(stdout, "field layer     : %s\n", ifma_field_name(&ctx.mc));
            }
            const size_t lw = 8 * ctx.n;
            std::vector<uint64_t> state(mont_soa_state_words(&ctx));
            std::vector<mpz_t> bx(IFMA_LANES), bg(IFMA_LANES);
            for (unsigned k = 0; k < IFMA_LANES; k++) {
                mpz_inits(bx[k], bg[k], NULL);
            }
            for (;;) {
                const uint32_t t = next_task.fetch_add(1);
                if (t >= tasks)
                    break;
                const MontTask& tk = task_list[t];
                uint32_t cur[IFMA_LANES];
                uint64_t sg[IFMA_LANES];
                /* a short batch repeats its last sigma: unread lanes only waste
                   SIMD slots, the used lanes stay exactly as requested */
                for (uint32_t k = 0; k < IFMA_LANES; k++) {
                    const uint32_t c = pending[tk.first + (k < tk.count ? k : tk.count - 1)];
                    cur[k] = c;
                    sg[k] = sigmas[c];
                }

                /* Resume: every lane of the task shares the offset, so each lane's
                   p0/p1 pair comes from its own checkpoint file.  If any lane's
                   state cannot be read (deleted, truncated, hand-edited) the whole
                   task restarts from bit 0 -- the lanes cannot be at different
                   offsets, the ladder walks one shared bit sequence. */
                size_t start = tk.start_bit;
                if (start > 0) {
                    for (uint32_t k = 0; k < tk.count; k++) {
                        mont_ckpt_t ck;
                        mont_ckpt_init(&ck);
                        std::string why;
                        const int ok = mont_ckpt_read(checkpoint,
                                                      mont_ckpt_path(checkpoint, ckpt_stem, cur[k]),
                                                      N, B1, torsion, nbits, &ck, &why);
                        if (ok && ck.bitnum == start) {
                            ifma_from_mpz_lane(state.data() + 0 * lw, k, ck.X0, &ctx.mc);
                            ifma_from_mpz_lane(state.data() + 1 * lw, k, ck.Z0, &ctx.mc);
                            ifma_from_mpz_lane(state.data() + 2 * lw, k, ck.X1, &ctx.mc);
                            ifma_from_mpz_lane(state.data() + 3 * lw, k, ck.Z1, &ctx.mc);
                        } else {
                            start = 0;
                        }
                        mont_ckpt_clear(&ck);
                        if (!start)
                            break;
                    }
                }

                MontProgCtx pc;
                pc.lane_bits = lane_bits.data();
                pc.lane_curve = cur;
                pc.lanes = tk.count;
                pc.total_curves = curves;
                pc.nbits = nbits;
                pc.last_ckpt_ms = &last_ckpt_ms;
                pc.interval_ms = use_ckpt ? ckpt_interval_ms : -1;

                /* The ladder stops at a checkpoint boundary for two different
                   reasons -- the autosave interval elapsed, or SIGINT arrived --
                   and they must NOT be confused: an autosave pause persists the
                   state and carries on with the same batch, while SIGINT ends the
                   run after persisting.  (Getting this wrong turns a 1 s autosave
                   interval into "the run stops after 1 s".) */
                int rc = MONT_SOA_DONE;
                for (;;) {
                    size_t bitnum = start;
                    rc = mont_soa_stage1_bits_ex(&ctx, bits, nbits, sg, start, state.data(),
                                                 &bitnum, bx.data(), bg.data(), mont_progress_cb,
                                                 &pc, ckpt_chunk);
                    if (rc != MONT_SOA_PAUSED)
                        break;
                    if (use_ckpt) {
                        for (uint32_t k = 0; k < tk.count; k++) {
                            const uint32_t c = cur[k];
                            mont_ckpt_t ck;
                            mont_ckpt_init(&ck);
                            ck.status = MONT_CKPT_INFLIGHT;
                            ck.bitnum = bitnum;
                            mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, "ifma");
                            ifma_to_mpz_lane(ck.X0, state.data() + 0 * lw, k, &ctx.mc);
                            ifma_to_mpz_lane(ck.Z0, state.data() + 1 * lw, k, &ctx.mc);
                            ifma_to_mpz_lane(ck.X1, state.data() + 2 * lw, k, &ctx.mc);
                            ifma_to_mpz_lane(ck.Z1, state.data() + 3 * lw, k, &ctx.mc);
                            if (mont_ckpt_write(checkpoint,
                                                mont_ckpt_path(checkpoint, ckpt_stem, c), N,
                                                &ck) != 0)
                                checkpoint_failed.store(1);
                            mont_ckpt_clear(&ck);
                        }
                    }
                    if (ecm_stage1_stop_requested())
                        break;      /* SIGINT: leave after saving */
                    start = bitnum; /* autosave: continue from here */
                }
                if (rc == MONT_SOA_ERROR) {
                    init_failed.store(1);
                    break;
                }
                if (rc == MONT_SOA_PAUSED) {
                    paused_flag.store(1);
                    break;
                }

                uint32_t local = 0;
                for (uint32_t k = 0; k < tk.count; k++) {
                    const uint32_t c = cur[k];
                    lane_bits[c].store((uint64_t)nbits, std::memory_order_relaxed);
                    if (mpz_cmp_ui(bg[k], 1) > 0 && mpz_cmp(bg[k], N) < 0) {
                        mpz_set(factors[c], bg[k]);
                        array_found[c] = ECM_FACTOR_FOUND_STEP1;
                        mpz_set(xs[c], bg[k]);
                        hit[c] = 1;
                        local++;
                        if (!validate_stage1_factor(N, factors[c]))
                            checkpoint_failed.store(1);
                    } else {
                        mpz_set(xs[c], bx[k]);
                    }
                    if (use_ckpt) {
                        /* DONE record: if the process dies before the shared .save
                           is written, this is what keeps the curve from being
                           recomputed by the resumed run. */
                        mont_ckpt_t ck;
                        mont_ckpt_init(&ck);
                        ck.status = MONT_CKPT_DONE;
                        ck.hit = hit[c];
                        ck.bitnum = nbits;
                        mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, "ifma");
                        if (ck.hit)
                            mpz_set(ck.factor, factors[c]);
                        else
                            mpz_set(ck.xout, xs[c]);
                        if (mont_ckpt_write(checkpoint, mont_ckpt_path(checkpoint, ckpt_stem, c), N,
                                            &ck) != 0)
                            checkpoint_failed.store(1);
                        mont_ckpt_clear(&ck);
                    }
                }
                if (local)
                    found_atomic.fetch_add(local);
            }
            for (unsigned k = 0; k < IFMA_LANES; k++) {
                mpz_clears(bx[k], bg[k], NULL);
            }
            mont_soa_clear(&ctx);
        } else {
            for (;;) {
                const uint32_t t = next_task.fetch_add(1);
                if (t >= tasks)
                    break;
                const MontTask& tk = task_list[t];
                const uint32_t c = pending[tk.first];
                mpz_t g, x, Qx, Qz, inv;
                mpz_inits(g, x, Qx, Qz, inv, NULL);

                mont_ladder_state_t st;
                mont_ladder_state_init(&st);
                size_t start = tk.start_bit;
                if (start > 0) {
                    mont_ckpt_t ck;
                    mont_ckpt_init(&ck);
                    std::string why;
                    if (mont_ckpt_read(checkpoint, mont_ckpt_path(checkpoint, ckpt_stem, c), N, B1,
                                       torsion, nbits, &ck, &why) &&
                        ck.bitnum == start) {
                        st.bitnum = ck.bitnum;
                        mpz_set(st.X0, ck.X0);
                        mpz_set(st.Z0, ck.Z0);
                        mpz_set(st.X1, ck.X1);
                        mpz_set(st.Z1, ck.Z1);
                    } else {
                        start = 0;
                    }
                    mont_ckpt_clear(&ck);
                }

                uint32_t lane[1] = {c};
                MontProgCtx pc;
                pc.lane_bits = lane_bits.data();
                pc.lane_curve = lane;
                pc.lanes = 1;
                pc.total_curves = curves;
                pc.nbits = nbits;
                pc.last_ckpt_ms = &last_ckpt_ms;
                pc.interval_ms = use_ckpt ? ckpt_interval_ms : -1;

                /* autosave pause vs SIGINT: see the SIMD branch above */
                int rc = MONT_LADDER_MISS;
                for (;;) {
                    rc = mont_stage1_curve_bits_ex(g, Qx, Qz, N, sigmas[c], bits, nbits, start, &st,
                                                   mont_progress_cb_scalar, &pc, ckpt_chunk);
                    if (rc != MONT_LADDER_PAUSED)
                        break;
                    if (use_ckpt) {
                        mont_ckpt_t ck;
                        mont_ckpt_init(&ck);
                        ck.status = MONT_CKPT_INFLIGHT;
                        ck.bitnum = st.bitnum;
                        mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, "mpn");
                        mpz_set(ck.X0, st.X0);
                        mpz_set(ck.Z0, st.Z0);
                        mpz_set(ck.X1, st.X1);
                        mpz_set(ck.Z1, st.Z1);
                        if (mont_ckpt_write(checkpoint, mont_ckpt_path(checkpoint, ckpt_stem, c), N,
                                            &ck) != 0)
                            checkpoint_failed.store(1);
                        mont_ckpt_clear(&ck);
                    }
                    if (ecm_stage1_stop_requested())
                        break;         /* SIGINT: leave after saving */
                    start = st.bitnum; /* autosave: continue from here */
                }
                if (rc == MONT_LADDER_ERROR) {
                    init_failed.store(1);
                    mpz_clears(g, x, Qx, Qz, inv, NULL);
                    mont_ladder_state_clear(&st);
                    break;
                }
                if (rc == MONT_LADDER_PAUSED) {
                    paused_flag.store(1);
                    mpz_clears(g, x, Qx, Qz, inv, NULL);
                    mont_ladder_state_clear(&st);
                    break;
                }

                lane_bits[c].store((uint64_t)nbits, std::memory_order_relaxed);
                /* same normalisation as mont_stage1_curve_bits_x() */
                if (mpz_sgn(Qz) != 0 && mpz_invert(inv, Qz, N)) {
                    mpz_mul(x, Qx, inv);
                    mpz_mod(x, x, N);
                } else {
                    mpz_set(x, Qx);
                }
                if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, N) < 0) {
                    mpz_set(factors[c], g);
                    array_found[c] = ECM_FACTOR_FOUND_STEP1;
                    mpz_set(xs[c], g);
                    hit[c] = 1;
                    found_atomic.fetch_add(1);
                    if (!validate_stage1_factor(N, factors[c]))
                        checkpoint_failed.store(1);
                } else {
                    mpz_set(xs[c], x);
                }
                if (use_ckpt) {
                    mont_ckpt_t ck;
                    mont_ckpt_init(&ck);
                    ck.status = MONT_CKPT_DONE;
                    ck.hit = hit[c];
                    ck.bitnum = nbits;
                    mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, "mpn");
                    if (ck.hit)
                        mpz_set(ck.factor, factors[c]);
                    else
                        mpz_set(ck.xout, xs[c]);
                    if (mont_ckpt_write(checkpoint, mont_ckpt_path(checkpoint, ckpt_stem, c), N,
                                        &ck) != 0)
                        checkpoint_failed.store(1);
                    mont_ckpt_clear(&ck);
                }
                mpz_clears(g, x, Qx, Qz, inv, NULL);
                mont_ladder_state_clear(&st);
            }
        }
    };

    Stage1Progress progress;
    opt.stage1_t0_ms = stage1_now_ms();
    opt.stage1_curves = curves;
    stage1_progress_set((double)(curves - pending.size()), curves);

    const auto t0 = std::chrono::steady_clock::now();
    if (nthreads <= 1) {
        worker();
    } else {
        std::vector<std::future<void>> pool;
        pool.reserve(nthreads);
        for (uint32_t t = 0; t < nthreads; t++)
            pool.push_back(std::async(std::launch::async, [&worker, &opt, t]() {
                apply_thread_affinity(opt.execution.affinity_cpus, t);
                worker();
            }));
        for (auto& th : pool)
            th.get();
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    progress.finish();

    if (init_failed.load() || checkpoint_failed.load()) {
        ecm_ts_fprintf(stderr, "ERROR: mont stage-1 %s\n",
                       checkpoint_failed.load() ? "checkpoint write failed"
                                                : "SIMD context init failed");

        return result;
    }

    /* Interrupted: every worker wrote its lane's mid-ladder state before leaving,
       and the curves it never reached keep their seed/older record, so rerunning
       the identical command line continues instead of restarting. */
    if (paused_flag.load()) {
        ecm_ts_fprintf(stdout,
                       "stage-1 paused (checkpoint saved under %s); rerun the same command "
                       "line to resume\n",
                       opt.persistence.tmp_dir.c_str());

        result.stop();
        return result;
    }

    stage1_progress_set((double)curves, curves);

    /* Hit lines are printed here, in curve order, rather than from inside the worker
       loop: with N workers the old placement interleaved lines at random. */
    const uint32_t found = found_atomic.load();
    for (uint32_t i = 0; i < curves; i++) {
        if (hit[i])
            ecm_ts_fprintf(stdout, "  curve %u sigma=%llu -> factor found\n", i,
                           (unsigned long long)sigmas[i]);
    }

    // Move normalized points/factors out before releasing CPU worker storage.
    result.curves.resize(curves);
    for (uint32_t i = 0; i < curves; ++i) {
        auto& curve = result.curves[i];
        curve.sigma = sigmas[i];
        curve.kind = hit[i] ? CurveKind::Factor : CurveKind::Stage2Point;
        mpz_swap(curve.value, xs[i]);
    }

    if (n_done_ck || n_inflight_ck) {
        ecm_ts_fprintf(stdout,
                       "  resumed from checkpoint: %u curve(s) already done, %u mid-ladder\n",
                       n_done_ck, n_inflight_ck);
    }
    ecm_ts_fprintf(stdout, "  curves=%u  hits=%u  wall=%.2fs  (%.3f s/curve)\n", curves, found,
                   elapsed, curves ? elapsed / curves : 0.0);

    result.completion = ComputationResult::Completion::Completed;
    result.failure = ecm_stage1::FailureSource::None;
    result.legacy_code = found ? ECM_FACTOR_FOUND_STEP1 : ECM_NO_FACTOR_FOUND;
    result.validate(N);
    return result;
}

} // namespace ecm_cpu_stage1
