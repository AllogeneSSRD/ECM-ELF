#include "ecm/stage1/cpu_runtime.h"
#include "ecm/stage1/execution.h"
#include "ecm/stage1/progress.h"
#include "ecm/stage1/sigma.h"
#include "ecm/stage1/libecm_compat.h"
#include "ecm/curve/edwards.h"
#include "ecm/cpu/arithmetic/scalar/modular_mpn_limits.h"
#include "ecm/platform/logging.h"
#include "../../../../io/stage1_checkpoint.h"
#include "checkpoint.h"
#include "ecm_mont_cpu.h"
#include "gwnum_curve.h"
#include "simd_mont_curve.h"
#include "../prac/ecm_prac_plan.h"
#include "ecm/io/save.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace ecm_cpu_stage1 {
using ecm_stage1::ComputationResult;
using ecm_stage1::CurveKind;
using ecm_stage1::CurveOutcome;
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
using ecm_stage1::validate_stage1_factor;

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

/* Progress and pause decision for one task, in ladder bits or PRAC descriptors. */
struct MontProgCtx {
    std::atomic<uint64_t>* lane_bits;
    const uint32_t* lane_curve;       /* [lanes] : curve index of each lane */
    uint32_t lanes;
    uint32_t total_curves;
    size_t work_total;
    std::atomic<long long>* last_ckpt_ms; /* shared by all workers: one timer */
    long long interval_ms;                /* < 0 = no periodic autosave */
};

static int mont_progress_cb(void* p, size_t bitnum) {
    MontProgCtx* pc = (MontProgCtx*)p;
    for (uint32_t k = 0; k < pc->lanes; k++)
        pc->lane_bits[pc->lane_curve[k]].store((uint64_t)bitnum, std::memory_order_relaxed);

    /* Work unit = one curve; each lane contributes its completed fraction. */
    uint64_t sum = 0;
    for (uint32_t i = 0; i < pc->total_curves; i++)
        sum += pc->lane_bits[i].load(std::memory_order_relaxed);
    const double done_f = pc->work_total ? (double)sum / (double)pc->work_total : 0.0;
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

    const auto modulus_bits = mpz_sizeinbase(N, 2);
    const auto implementation = ecm_stage1::resolve_cpu_implementation(
        opt.execution.selection, modulus_bits, curves, driver_simd_isa_ok());
    if (!implementation) {
        if (opt.execution.selection.cpu_implementation == ecm_stage1::CpuImplementation::GmpScalar)
            ecm_ts_fprintf(stderr, "ERROR: Montgomery CPU Stage1 scalar GMP supports up to "
                                   "%zu bits. Use --impl auto or simd on an IFMA-capable CPU.\n",
                           ecm_cpu::gmp_fixed_width_bits);
        else
            ecm_ts_fprintf(stderr, "ERROR: Montgomery CPU Stage1 at %zu bits requires "
                                   "AVX512-F/DQ/IFMA for the selected implementation.\n",
                           modulus_bits);
        return result;
    }
    const bool use_simd = *implementation == ecm_stage1::CpuImplementation::IfmaSimd;
    const bool use_gwnum = *implementation == ecm_stage1::CpuImplementation::Gwnum;
    if (use_gwnum) {
        if (const char* error = mont_stage1_gwnum_load_error()) {
            ecm_ts_fprintf(stderr, "ERROR: %s\n", error);
            return result;
        }
    }
    const auto run_ladder = use_gwnum ? mont_stage1_curve_gwnum_bits_ex : mont_stage1_curve_bits_ex;
    const bool use_prac = opt.execution.selection.cpu_montgomery_chain ==
                          ecm_stage1::CpuMontgomeryChain::Prac;
    if (use_prac && (!use_simd || !(B1 >= 2 && B1 <= UINT32_MAX))) {
        ecm_ts_fprintf(stderr,
                       "ERROR: CPU Montgomery PRAC requires IFMA and B1 in 2..4294967295.\n");
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

    ecm_ts_fprintf(stdout, "curve model     : montgomery (Suyama sigma, %s, torsion=%d)\n",
                   use_simd ? "AVX512-IFMA 8-lane batch"
                            : use_gwnum ? "GWNUM 31.6" : "scalar mpn",
                   torsion);
    ecm_ts_fprintf(stdout, "stage1 exponent : s_bits=%zu (lcm(1..%.0f) x %d) [%s]\n", s_bits, B1,
                   torsion, s_status.c_str());

    EcmPracPlan prac;
    if (use_prac) {
        const auto search = ecm_prac_search_width(std::getenv("ECM_PRAC_SEARCH"));
        prac = ecm_prac_build(static_cast<uint32_t>(B1), torsion, s,
                              opt.persistence.exponent_cache_dir, opt.execution.stage1_threads, search);
        ecm_ts_fprintf(stdout, "stage1 chain    : PRAC [%s]\n", prac.status.c_str());
    } else if (use_simd || use_gwnum) {
        ecm_ts_fprintf(stdout, "stage1 chain    : binary ladder\n");
    }
    const std::string field = use_prac ? "prac-" + std::to_string(prac.identity)
                                     : use_simd ? "ifma" : use_gwnum ? "gwnum" : "mpn";
    const auto compatible_chain = [&](std::string_view saved_field) {
        return use_prac ? saved_field == field : saved_field.substr(0, 5) != "prac-";
    };

    ecm_edwards::Array factors_owner(curves);
    auto* factors = factors_owner.data();
    std::vector<int> statuses(curves, ECM_NO_FACTOR_FOUND);
    auto* array_found = statuses.data();

    /* Build the ladder's shared bit array now, or lazily if PRAC needs a replay. */
    size_t nbits = 0;
    std::unique_ptr<uint8_t, decltype(&std::free)> bits_owner(
        use_prac ? nullptr : mont_expand_bits(s, &nbits), std::free);
    auto* bits = bits_owner.get();
    if (!use_prac && !bits)
        return result;
    std::once_flag replay_once;
    const size_t work_total = use_prac ? prac.primes.size() : nbits;
    const size_t ckpt_chunk = use_prac ? std::max<size_t>(1, work_total / 256)
                                     : mont_ckpt_chunk_bits(nbits);

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
    std::vector<size_t> bit_off(curves, 0); /* exponent bits or PRAC entries already consumed */
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
                               work_total, &ck, &why) &&
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
                } else if (ck.bitnum > 0 && ck.bitnum < work_total && compatible_chain(ck.field)) {
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
                mont_ckpt_ident(&ck, i, sigmas[i], B1, torsion, work_total, field.c_str());
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
                       "checkpoint      : %s_c*.ckpt  [%u done, %u in flight, %zu to run]"
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
        lane_bits[i].store(ckpt_done[i] ? (uint64_t)work_total : (uint64_t)bit_off[i],
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
                                                      N, B1, torsion, work_total, &ck, &why);
                        if (ok && ck.bitnum == start && compatible_chain(ck.field)) {
                            ifma_from_mpz_lane(state.data() + 0 * lw, k, ck.X0, &ctx.mc);
                            ifma_from_mpz_lane(state.data() + 1 * lw, k, ck.Z0, &ctx.mc);
                            if (!use_prac) {
                                ifma_from_mpz_lane(state.data() + 2 * lw, k, ck.X1, &ctx.mc);
                                ifma_from_mpz_lane(state.data() + 3 * lw, k, ck.Z1, &ctx.mc);
                            }
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
                pc.work_total = work_total;
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
                    rc = use_prac
                             ? mont_soa_stage1_prac_ex(&ctx, prac.primes.data(), work_total, sg,
                                                       start, state.data(), &bitnum, bx.data(),
                                                       bg.data(), mont_progress_cb, &pc, ckpt_chunk)
                             : mont_soa_stage1_bits_ex(&ctx, bits, nbits, sg, start, state.data(),
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
                            mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, work_total, field.c_str());
                            ifma_to_mpz_lane(ck.X0, state.data() + 0 * lw, k, &ctx.mc);
                            ifma_to_mpz_lane(ck.Z0, state.data() + 1 * lw, k, &ctx.mc);
                            if (!use_prac) {
                                ifma_to_mpz_lane(ck.X1, state.data() + 2 * lw, k, &ctx.mc);
                                ifma_to_mpz_lane(ck.Z1, state.data() + 3 * lw, k, &ctx.mc);
                            }
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
                if (use_prac && (rc == MONT_SOA_REPLAY || rc == MONT_SOA_DONE)) {
                    const bool replay_all = rc == MONT_SOA_REPLAY;
                    bool replay_lane[IFMA_LANES]{};
                    bool needs_replay = false;
                    for (uint32_t k = 0; k < tk.count; ++k) {
                        replay_lane[k] = replay_all || mpz_cmp_ui(bg[k], 1) != 0;
                        needs_replay |= replay_lane[k];
                    }
                    if (needs_replay) {
                        std::call_once(replay_once, [&] {
                            bits_owner.reset(mont_expand_bits(s, &nbits));
                        });
                        if (!bits_owner) {
                            init_failed.store(1);
                            break;
                        }
                        ecm_ts_fprintf(stdout, "  PRAC: replaying exceptional lanes with the binary ladder\n");
                        ecm_edwards::Array replay_x(IFMA_LANES), replay_gcd(IFMA_LANES);
                        // The PRAC checkpoint remains a valid restart point if a replay is interrupted.
                        const auto stop_replay = [](void*, size_t) -> int {
                            return ecm_stage1_stop_requested() ? 1 : 0;
                        };
                        rc = mont_soa_stage1_bits_ex(&ctx, bits_owner.get(), nbits, sg, 0, state.data(),
                                                     nullptr, replay_x.data(), replay_gcd.data(),
                                                     stop_replay, nullptr, mont_ckpt_chunk_bits(nbits));
                        if (rc == MONT_SOA_DONE) {
                            for (uint32_t k = 0; k < tk.count; ++k) {
                                if (replay_lane[k]) {
                                    mpz_swap(bx[k], replay_x[k]);
                                    mpz_swap(bg[k], replay_gcd[k]);
                                }
                            }
                        }
                    }
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
                    lane_bits[c].store((uint64_t)work_total, std::memory_order_relaxed);
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
                        ck.bitnum = work_total;
                        mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, work_total, field.c_str());
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
                pc.work_total = nbits;
                pc.last_ckpt_ms = &last_ckpt_ms;
                pc.interval_ms = use_ckpt ? ckpt_interval_ms : -1;

                /* autosave pause vs SIGINT: see the SIMD branch above */
                int rc = MONT_LADDER_MISS;
                for (;;) {
                    rc = run_ladder(g, Qx, Qz, N, sigmas[c], bits, nbits, start, &st,
                                    mont_progress_cb_scalar, &pc, ckpt_chunk);
                    if (rc != MONT_LADDER_PAUSED)
                        break;
                    if (use_ckpt) {
                        mont_ckpt_t ck;
                        mont_ckpt_init(&ck);
                        ck.status = MONT_CKPT_INFLIGHT;
                        ck.bitnum = st.bitnum;
                        mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, field.c_str());
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
                    mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, field.c_str());
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
                       "  resumed from checkpoint: %u curve(s) already done, %u in flight\n",
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
