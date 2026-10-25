#include "ecm/io/results.h"
#include "../execution.h"
#include "../../../../service/stage1/preparation.h"
#include "io/stage1_checkpoint.h"
#include "ecm/platform/logging.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
using ecm_stage1::ComputationResult;
using ecm_stage1::compute_batch_s;
using ecm_stage1::CurveKind;
using ecm_stage1::ensure_dir;
using ecm_stage1::mont_format_save_name;
using ecm_stage1::random_sigma_u64;
using ecm_stage1::stage1_now_ms;
using ecm_stage1::Stage1Options;
using ecm_stage1::stdout_is_tty_local;
using ecm_stage1::validate_stage1_factor;
#include "api.h"
#include "naf.h"
#include "runtime.h"
#include "ecm/core/stage1/scalar.h"
#include "ecm/io/save.h"
#include "io/stage1_binary.h"
#include "ecm/platform/stage1_progress.h"
#include <iterator>
#include <stdexcept>

#include "checkpoint.h"
#include "host_math.h"

namespace ed_gpu_driver {
struct Progress {
    ecm_stage1_checkpoint::Context& checkpoint;
    using Clock = std::chrono::steady_clock;
    Batch& b;
    mpz_srcptr n, scalar;
    const Stage1Options& opt;
    std::string path;
    uint64_t b1, total, durable = 0;
    Clock::time_point last;
    const EcmEdwardsNafMetrics* metrics;
    bool failed = false;
    Clock::time_point initialization_start, loop_start, sample_start, last_print;
    uint64_t sampled = 0, delta = 0;
    bool printed = false, line_open = false;
    ecm_stage1_gpu_progress::SpeedWindow speed;

    Progress(ecm_stage1_checkpoint::Context& owner, Batch& batch, mpz_srcptr modulus,
             mpz_srcptr exponent, const Stage1Options& options,
             const std::string& checkpoint_path, uint64_t bound, uint64_t bits,
             const EcmEdwardsNafMetrics* measurements, Clock::time_point start)
        : checkpoint(owner), b(batch), n(modulus), scalar(exponent), opt(options), path(checkpoint_path), b1(bound),
          total(bits), last(Clock::now()), metrics(measurements), initialization_start(start) {
    }

    void finish() {
        if (line_open) {
            std::fputc('\n', stdout);
            std::fflush(stdout);
            line_open = false;
        }
    }

    void display(Clock::time_point now) {
        const auto text = ecm_stage1_gpu_progress::line(
            total ? 100.0 * b.done / total : 100.0, b.done, delta, b.window ? "columns" : "bits",
            speed.per_curve(total, b.count),
            std::chrono::duration<double>(now - loop_start).count(),
            speed.remaining(total, b.done));
        if (stdout_is_tty_local()) {
            std::fprintf(stdout, "\r%s%s%s", ecm_log_progress_color_code(), text.c_str(),
                         ecm_log_progress_color_reset());
            std::fflush(stdout);
            line_open = true;
        } else
            ecm_ts_fprintf(stdout, "%s\n", text.c_str());
        printed = true;
        last_print = now;
    }
};

static void prepared(void* context) {
    auto& p = *static_cast<Progress*>(context);
    const double dictionary = p.metrics->dictionary_wall_ms / 1000.0;
    const double preparation =
        std::chrono::duration<double>(Progress::Clock::now() - p.initialization_start).count();
    if (p.metrics->container_bits)
        ecm_ts_fprintf(
            stdout, "GPU Edwards geometry: container_bits=%u tpi=%u tpb=128 blocks=%llu\n",
            p.metrics->container_bits, p.metrics->tpi,
            static_cast<unsigned long long>((uint64_t(p.b.count) + 128 / p.metrics->tpi - 1) /
                                            (128 / p.metrics->tpi)));
    else
        ecm_ts_fprintf(
            stdout,
            "GPU Edwards completed checkpoint: host validation only; no device preparation\n");
    ecm_ts_fprintf(stdout, "GPU Edwards initialization: %.3f s\n",
                   (std::max)(0.0, preparation - dictionary));
    ecm_ts_fprintf(stdout, "GPU Edwards dictionary: %.3f s (gpu=%.3f ms)\n", dictionary,
                   p.metrics->dictionary_gpu_ms);
    p.total = p.b.window ? p.metrics->columns : p.total;
    // Restored work is the baseline, never a speed sample. Preparation and its
    // console I/O finish before the live loop clock starts.
    p.sampled = p.b.done;
    p.loop_start = p.sample_start = Progress::Clock::now();
    if (p.b.done == p.total)
        p.display(p.loop_start);
}

static bool progress(uint32_t done, void* context) {
    auto& p = *static_cast<Progress*>(context);
    p.b.done = done;
    const auto sample_time = Progress::Clock::now();
    p.delta = done - p.sampled;
    p.speed.add(p.delta,
                std::chrono::duration<double, std::milli>(sample_time - p.sample_start).count());
    p.sampled = done;
    const bool stop = ecm_stage1_stop_requested();
    const auto now = sample_time;
    if (!p.printed || stop || done == p.total ||
        sample_time - p.last_print >= std::chrono::milliseconds(200))
        p.display(sample_time);
    if (done < p.total &&
        (stop || (p.opt.persistence.ckpt_ms &&
                  now - p.last >= std::chrono::milliseconds(p.opt.persistence.ckpt_ms)))) {
        bool saved = true;
        try {
            if (!p.path.empty())
                saved = save_checkpoint(p.checkpoint, p.path, p.b, p.n, p.scalar, p.b1);
        } catch (const std::exception& error) {
            saved = false;
            ecm_ts_fprintf(stderr, "ERROR: %s\n", error.what());
        }
        if (!saved) {
            p.finish();
            p.failed = true;
            ecm_ts_fprintf(
                stderr, "ERROR: GPU Edwards checkpoint write failed: %s cursor=%u durable=%llu\n",
                p.path.c_str(), done, static_cast<unsigned long long>(p.durable));
            return false;
        }
        if (!p.path.empty())
            p.durable = done;
        p.last = now;
    }
    // The rate samples include each slice and its state readback, excluding
    // checkpoint/console callback time. Elapsed is the complete loop wall time.
    p.sample_start = Progress::Clock::now();
    if (stop && done < p.total) {
        p.finish();
        ecm_ts_fprintf(stdout, "GPU Edwards stopped at %s %u; %s\n",
                       p.b.window ? "NAF column" : "binary bit", done,
                       p.path.empty() ? "no checkpoint configured, progress is not recoverable"
                                      : "checkpoint committed");
    }
    return done == p.total || !stop;
}

} // namespace ed_gpu_driver

ecm_stage1::ComputationResult
ecm_cuda_stage1::run_edwards_stage1(const ecm_stage1::Stage1Request& request,
                                    ecm_stage1::ExecutionOptions& opt, ecm_stage1_checkpoint::Context& checkpoint) {
    const auto N = request.modulus;
    const auto n = N;
    const auto B1 = request.b1, B2 = request.b2, b1 = B1, b2 = B2;
    const auto curves = request.curves;
    const auto& n_expr = request.expression;
    ComputationResult result;

    using namespace ed_gpu_driver;
    if (!curves) {
        result.fail(ecm_stage1::FailureSource::Preparation);
        return result;
    }
    auto phase = ecm_stage1::FailureSource::Preparation;
    try {
        result.curves.resize(curves);
        (void)n_expr;
        // Prepare: scalar and explicit algorithm requirements define recovery admission.
        const auto initialization_start = Progress::Clock::now();
        if (opt.execution.naf_w != 0 && (opt.execution.naf_w < 2 || opt.execution.naf_w > 12))
            throw std::runtime_error("GPU Edwards NAF window must be 0 (auto) or 2..12");
        Integer scalar;
        if (!compute_batch_s(scalar.v, b1, 48, opt.persistence.exponent_cache_dir))
            throw std::runtime_error("cannot build Edwards exponent");
        if (!opt.persistence.tmp_dir.empty() && !ensure_dir(opt.persistence.tmp_dir))
            throw std::runtime_error("cannot create Edwards save directory");
        Batch batch(curves);
        batch.window = opt.execution.naf_w ? unsigned(opt.execution.naf_w) : 6;
        int mode_fixed = -1;
        if (const char* mode = std::getenv("ECM_EDWARDS_STAGE1_ALGO")) {
            const std::string value = mode;
            if (value != "binary" && value != "naf")
                throw std::runtime_error("ECM_EDWARDS_STAGE1_ALGO must be binary or naf");
            mode_fixed = value == "naf" ? 1 : 0;
            if (!mode_fixed) {
                if (opt.execution.naf_w)
                    throw std::runtime_error("binary Edwards mode cannot use --naf-w");
                batch.window = 0;
                batch.doubling = 0;
            }
        }
        const char* dbl_env = std::getenv("ECM_EDWARDS_DBL_ALGO");
        if (dbl_env) {
            if (!batch.window)
                throw std::runtime_error(
                    "binary Edwards mode cannot select a NAF doubling algorithm");
            const std::string value = dbl_env;
            if (value != "4" && value != "5")
                throw std::runtime_error("ECM_EDWARDS_DBL_ALGO must be 4 or 5");
            batch.doubling = value == "5" ? 5 : 4;
        }
        // Restore candidates: decode first; accept only a compatible construction/layout.
        bool selected_payload = false;
        checkpoint.validate_pending(
            [&](const std::vector<std::string>& records, std::vector<uint64_t>& sigmas) {
                using namespace ecm_stage1_checkpoint;
                if (records.size() != curves || sigmas.size() != curves)
                    return false;
                for (size_t i = 1; i < records.size(); ++i)
                    if (!records[i].empty())
                        return false;
                if (records[0].empty())
                    return true; // Valid seed-only batch.
                Batch candidate(curves);
                candidate.window = batch.window;
                candidate.doubling = batch.doubling;
                candidate.sigmas = sigmas;
                if (opt.identity.sigma_fixed)
                    for (uint32_t i = 0; i < curves; ++i)
                        candidate.sigmas[i] = opt.identity.fixed_sigma64 + i;
                const auto decoded = decode_checkpoint(
                    records[0], candidate, n, scalar.v, uint64_t(b1), opt.identity.sigma_fixed,
                    opt.execution.naf_w != 0, dbl_env != nullptr, mode_fixed);
                if (decoded.state == RestoreState::Incompatible ||
                    decoded.state == RestoreState::IoFailure)
                    throw Error(decoded.state, decoded.error);
                if (!decoded.restored())
                    return false;
                for (uint32_t i = 0; i < curves; ++i)
                    if (sigmas[i] && sigmas[i] != candidate.sigmas[i])
                        return false;
                sigmas = candidate.sigmas;
                batch.done = candidate.done;
                batch.window = candidate.window;
                batch.doubling = candidate.doubling;
                batch.sigmas = candidate.sigmas;
                batch.hit = candidate.hit;
                for (uint32_t i = 0; i < curves; ++i)
                    mpz_set(batch.factors[i], candidate.factors[i]);
                for (size_t i = 0; i < size_t(curves) * 9; ++i)
                    mpz_set(batch.state[i], candidate.state[i]);
                selected_payload = true;
                return true;
            });
        if (opt.identity.sigma_fixed)
            for (uint32_t i = 0; i < curves; ++i)
                batch.sigmas[i] = opt.identity.fixed_sigma64 + i;
        const auto path =
            opt.persistence.tmp_dir.empty() ? std::string{} : ecm_stage1_curve_record(0);
        bool restored = selected_payload;
        if (selected_payload)
            ecm_ts_fprintf(stdout, "GPU Edwards resume: columns=%u window=%u doubling=%d path=%s\n",
                           batch.done, batch.window, batch.doubling, path.c_str());
        if (!restored && !path.empty()) {
            const auto decoded =
                restore_checkpoint(checkpoint, path, batch, n, scalar.v, uint64_t(b1), opt.identity.sigma_fixed,
                                   opt.execution.naf_w != 0, dbl_env != nullptr, mode_fixed);
            if (decoded.state == ecm_stage1_checkpoint::RestoreState::Incompatible ||
                decoded.state == ecm_stage1_checkpoint::RestoreState::IoFailure)
                throw ecm_stage1_checkpoint::Error(decoded.state, decoded.error);
            restored = decoded.restored();
        }
        std::string existing;
        if (!restored && !path.empty() && !checkpoint.read_blob(path, existing)) {
            // Pre-container GPU Edwards used ed_<old-pattern>.save.gpu.ckpt.
            for (const auto& pattern :
                 {std::string("m{n}_{b1}.save"), opt.persistence.save_name_pattern}) {
                auto old = mont_format_save_name(pattern, n, b1);
                if (old.rfind("m0_", 0) == 0)
                    old.erase(0, 3);
                const auto legacy =
                    (std::filesystem::path(opt.persistence.tmp_dir) / ("ed_" + old + ".gpu.ckpt"))
                        .string();
                Batch saved(curves);
                saved.sigmas = batch.sigmas;
                saved.window = batch.window;
                saved.doubling = batch.doubling;
                if (restore_checkpoint(checkpoint, legacy, saved, n, scalar.v, uint64_t(b1),
                                       opt.identity.sigma_fixed, opt.execution.naf_w != 0,
                                       dbl_env != nullptr, mode_fixed)
                        .restored()) {
                    if (ecm_results::contains_stage1_curves(opt.persistence.results_file, n,
                                                            uint64_t(b1), true, 0, saved.sigmas))
                        continue;
                    batch.done = saved.done;
                    batch.window = saved.window;
                    batch.doubling = saved.doubling;
                    batch.sigmas = saved.sigmas;
                    batch.hit = saved.hit;
                    for (uint32_t i = 0; i < curves; ++i)
                        mpz_set(batch.factors[i], saved.factors[i]);
                    for (size_t i = 0; i < size_t(curves) * 9; ++i)
                        mpz_set(batch.state[i], saved.state[i]);
                    if (!checkpoint.import_payload(
                            checkpoint_body(batch, n, scalar.v, uint64_t(b1)), batch.sigmas))
                        throw std::runtime_error("cannot import legacy Edwards checkpoint");
                    ecm_ts_fprintf(
                        stdout,
                        "Stage1 legacy Edwards checkpoint imported: path=%s original_retained=1\n",
                        legacy.c_str());
                    restored = true;
                    break;
                }
            }
        }
        // Construct only when no compatible saved batch was accepted.
        if (!restored) {
            if (!path.empty() && checkpoint.read_blob(path, existing))
                throw std::runtime_error("GPU Edwards batch payload is invalid or unsupported");
            // Sigma selection and curve parameterization are separate operations.
            for (uint32_t i = 0; i < curves; ++i) {
                uint64_t sigma = 0;
                if (!checkpoint.saved_sigma(i, sigma))
                    sigma = opt.identity.sigma_fixed ? opt.identity.fixed_sigma64 + i
                                                     : random_sigma_u64();
                batch.sigmas[i] = checkpoint.sigma(i, sigma);
            }
            std::string error;
            uint32_t degenerate_curve = UINT32_MAX;
            if (!initialize(
                    batch, n, error,
                    [&](uint32_t lane, mpz_srcptr factor) {
                        result.set_curve(lane, CurveKind::Factor, factor, batch.sigmas[lane]);
                        if (validate_stage1_factor(n, factor))
                            return true;
                        result.fail(ecm_stage1::FailureSource::Publication);
                        return false;
                    },
                    &degenerate_curve)) {
                if (degenerate_curve < curves)
                    result.set_curve(degenerate_curve, CurveKind::Degenerate, nullptr,
                                     batch.sigmas[degenerate_curve]);
                throw std::runtime_error(error);
            }
        }
        ecm_ts_fprintf(stdout,
                       "GPU Edwards Stage1: device=%d curves=%u N_bits=%zu scalar_bits=%zu "
                       "window=%u doubling=%d\n",
                       opt.execution.device_index, curves, mpz_sizeinbase(n, 2),
                       mpz_sizeinbase(scalar.v, 2), batch.window, batch.doubling);
        EcmEdwardsNafMetrics metrics;
        Progress p{checkpoint, batch,
                   n,
                   scalar.v,
                   opt,
                   path,
                   uint64_t(b1),
                   mpz_sizeinbase(scalar.v, 2),
                   &metrics,
                   initialization_start};
        p.durable = restored ? batch.done : 0;
        phase = ecm_stage1::FailureSource::Execution;
        const auto arithmetic_start = std::chrono::steady_clock::now();
        // Execute with checkpoint safe points; final files belong to task.
        const auto run =
            execute_edwards_batch(n, scalar.v, batch.state.data(), curves, batch.done,
                                  {opt.execution.device_index, batch.window, batch.doubling, 0},
                                  progress, &p, &metrics, prepared);
        const int rc = run.code;
        const double arithmetic_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - arithmetic_start)
                .count();
        p.finish();
        ecm_ts_fprintf(stdout, "GPU Edwards arithmetic: algorithm=%s api_seconds=%.9f\n",
                       batch.window ? "naf" : "binary", arithmetic_seconds);
        if (batch.window)
            ecm_ts_fprintf(
                stdout,
                "GPU Edwards NAF: columns=%u additions=%u container_bits=%u tpi=%u "
                "dictionary_bytes=%llu dictionary_gpu_ms=%.6f loop_gpu_ms=%.6f launches=%u\n",
                metrics.columns, metrics.additions, metrics.container_bits, metrics.tpi,
                static_cast<unsigned long long>(metrics.dictionary_bytes),
                metrics.dictionary_gpu_ms, metrics.loop_gpu_ms, metrics.launches);
        if (rc < 0) {
            result.fail(run.failure == ecm_cuda_stage1::Failure::Device
                            ? ecm_stage1::FailureSource::Device
                        : run.failure == ecm_cuda_stage1::Failure::Preparation
                            ? ecm_stage1::FailureSource::Preparation
                            : ecm_stage1::FailureSource::Execution);
            throw std::runtime_error(run.error);
        }
        if (rc == 2) {
            if (p.failed)
                result.fail(ecm_stage1::FailureSource::Persistence);
            else
                result.stop();
            return result;
        }
        phase = ecm_stage1::FailureSource::Execution;
        // Collect owned factor/point/no-point outcomes from the final layout.
        Integer x, z, g, inv;
        bool found = false;
        for (uint32_t i = 0; i < curves; ++i) {
            auto* s = batch.state.data() + size_t(i) * 9;
            bool hit = batch.hit[i] != 0;
            if (!hit) {
                if (!point_valid(n, s[0], s + 5))
                    throw std::runtime_error("invalid GPU Edwards final point");
                mpz_add(x.v, s[7], s[6]);
                mpz_mod(x.v, x.v, n);
                mpz_sub(z.v, s[7], s[6]);
                mpz_mod(z.v, z.v, n);
                mpz_gcd(g.v, z.v, n);
                if (mpz_cmp_ui(g.v, 1) > 0 && mpz_cmp(g.v, n) < 0) {
                    hit = true;
                    mpz_set(x.v, g.v);
                } else if (mpz_cmp(g.v, n) == 0) {
                    ecm_ts_fprintf(stdout,
                                   "Edwards identity: sigma=%llu gcd(Qz,N)=N, no Stage2 point\n",
                                   static_cast<unsigned long long>(batch.sigmas[i]));
                    result.set_curve(i, CurveKind::CompletedWithoutPoint, nullptr, batch.sigmas[i]);
                    continue;
                } else {
                    mpz_invert(inv.v, z.v, n);
                    mpz_mul(x.v, x.v, inv.v);
                    mpz_mod(x.v, x.v, n);
                }
            }
            if (batch.hit[i])
                mpz_set(x.v, batch.factors[i]);
            result.set_curve(i, hit ? CurveKind::Factor : CurveKind::Stage2Point, x.v,
                             batch.sigmas[i]);
            found = found || hit;
        }
        result.completion = ComputationResult::Completion::Completed;
        result.validate(n);
        // Arithmetic state stays immutable while owned final values are published.
        phase = ecm_stage1::FailureSource::Persistence;
        if (!path.empty() && !save_checkpoint(checkpoint, path, batch, n, scalar.v, uint64_t(b1)))
            throw std::runtime_error("cannot retain completed Edwards batch");
        phase = ecm_stage1::FailureSource::Execution;
        // The task seals result publication before removing its resume state.
        result.completion = ComputationResult::Completion::Completed;
        result.legacy_code = found ? ECM_FACTOR_FOUND_STEP1 : ECM_NO_FACTOR_FOUND;
        (void)b2;
        return result;
    } catch (const std::exception& e) {
        if (const auto* checkpoint = dynamic_cast<const ecm_stage1_checkpoint::Error*>(&e))
            phase = checkpoint->state == ecm_stage1_checkpoint::RestoreState::IoFailure
                        ? ecm_stage1::FailureSource::Persistence
                        : ecm_stage1::FailureSource::Preparation;
        if (result.failure == ecm_stage1::FailureSource::None)
            result.fail(phase);
        ecm_ts_fprintf(stderr, "ERROR: GPU Edwards Stage1: %s\n", e.what());
        return result;
    }
}
