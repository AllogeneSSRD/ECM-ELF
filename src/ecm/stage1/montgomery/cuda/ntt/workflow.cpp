#include "workflow.h"
#include "ladder.h"
#include "tune.h"
#include "../host_math.h"
#include "../../../sigma.h"
#include "../../../../curve/edwards.h"
#include "../../../../../io/stage1_binary.h"
#include "../../../../../io/stage1_checkpoint.h"
#include "ecm/integer.h"
#include "ecm/platform/logging.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace ecm_cuda_stage1 {
namespace {
using namespace ecm_stage1;
using Clock = std::chrono::steady_clock;

struct Snapshot {
    unsigned cursor = 0;
    unsigned lanes = 0;
    std::vector<uint32_t> points;
};

constexpr std::string_view snapshot_magic = "ECMNTTS1";

bool read_snapshot(std::string_view bytes, unsigned count, unsigned p, unsigned curves,
                   Snapshot& out) {
    using ecm_stage1_binary::Reader;
    Reader reader(bytes);
    if (reader.left() < snapshot_magic.size() + 3 * 4 || reader.take(8) != snapshot_magic ||
        reader.u32() != count)
        return false;
    Snapshot decoded;
    decoded.cursor = reader.u32();
    decoded.lanes = reader.u32();
    if (decoded.cursor > count || !decoded.lanes || (curves < 32 && (decoded.lanes >> curves)))
        return false;
    const unsigned words = (p + 31) / 32;
    unsigned active = 0;
    for (unsigned i = 0; i < curves; ++i) active += (decoded.lanes >> i) & 1;
    const size_t size = size_t(4) * words * active;
    if (reader.left() != size * sizeof(uint32_t))
        return false;
    decoded.points.resize(size);
    for (auto& word : decoded.points) word = reader.u32();
    const unsigned top = p % 32 ? (1u << (p % 32)) - 1 : UINT32_MAX;
    for (size_t i = 0; i < decoded.points.size(); i += words) {
        const auto hi = decoded.points.begin() + i + words - 1;
        if (*hi > top || (*hi == top && std::all_of(decoded.points.begin() + i, hi,
                                                    [](auto v) { return v == UINT32_MAX; })))
            return false;
    }
    out = std::move(decoded);
    return true;
}

std::string encode_snapshot(unsigned count, unsigned cursor, unsigned lanes,
                            const std::vector<uint32_t>& points) {
    using ecm_stage1_binary::u32;
    std::string payload(snapshot_magic);
    payload.reserve(payload.size() + (3 + points.size()) * 4);
    u32(payload, count);
    u32(payload, cursor);
    u32(payload, lanes);
    for (auto word : points) u32(payload, word);
    return payload;
}

// Edwards saves already use the Montgomery x coordinate (1+y)/(1-y).
// Construct the Edwards curve and point before mapping into the shared ladder.
CurveKind prepare_edwards_lane(mpz_srcptr n, uint64_t sigma, mpz_ptr factor, uint32_t* initial,
                               unsigned words) {
    using ecm_edwards::Integer;
    using ecm_edwards::Setup;
    Integer d, px, py, a24, x, inv, den;
    const auto setup = ecm_edwards::construct(d.v, px.v, py.v, a24.v, factor, sigma, n);
    if (setup == Setup::Factor) return CurveKind::Factor;
    if (setup == Setup::Degenerate) return CurveKind::Degenerate;

    mpz_ui_sub(den.v, 1, py.v);
    const auto chart = ecm_edwards::inverse(inv.v, den.v, n, factor);
    if (chart == Setup::Factor) return CurveKind::Factor;
    if (chart == Setup::Degenerate) return CurveKind::CompletedWithoutPoint;
    mpz_add_ui(x.v, py.v, 1);
    mpz_mul(x.v, x.v, inv.v);
    mpz_mod(x.v, x.v, n);
    // The point with x=0 has order two and the Edwards scalar is even.
    if (mpz_sgn(x.v) == 0) return CurveKind::CompletedWithoutPoint;

    Integer one, aa, bb, e, x2, z2;
    mpz_set_ui(one.v, 1);
    mpz_add_ui(aa.v, x.v, 1);
    mpz_mul(aa.v, aa.v, aa.v);
    mpz_mod(aa.v, aa.v, n);
    mpz_sub_ui(bb.v, x.v, 1);
    mpz_mul(bb.v, bb.v, bb.v);
    mpz_mod(bb.v, bb.v, n);
    mpz_sub(e.v, aa.v, bb.v);
    mpz_mul(x2.v, aa.v, bb.v);
    mpz_mod(x2.v, x2.v, n);
    mpz_mul(z2.v, a24.v, e.v);
    mpz_add(z2.v, z2.v, bb.v);
    mpz_mul(z2.v, z2.v, e.v);
    mpz_mod(z2.v, z2.v, n);
    const mpz_srcptr slots[] = {x.v, one.v, x2.v, z2.v, a24.v, x.v};
    for (unsigned slot = 0; slot < 6; ++slot)
        mpz_export(initial + slot * words, nullptr, -1, sizeof(uint32_t), 0, 0, slots[slot]);
    return CurveKind::Stage2Point;
}
} // namespace

ComputationResult run_ntt_stage1(const Stage1Request& request, ExecutionOptions& opt,
                                 ecm_stage1_checkpoint::Context& checkpoint) {
    ComputationResult result;
    auto phase = FailureSource::Preparation;
    try {
        const auto N = request.modulus;
        const size_t p = mpz_sizeinbase(N, 2);
        const unsigned curves = request.curves;
        const bool edwards = opt.execution.selection.curve == CurveModel::Edwards;
        if (!curves || curves > stage1_ntt_max_curves || (!edwards && opt.execution.selection.gpu_param != 0) ||
            p < 32 || p > 262144 || mpz_sgn(N) <= 0 || mpz_scan0(N, 0) != p)
            throw std::invalid_argument(
                "NTT requires Edwards or Montgomery param0, a full 2^p-1 (32 <= p <= 262144), and 1-32 curves");
        ecm_edwards::Integer scalar;
        const unsigned torsion = edwards ? 48u :
            opt.execution.selection.scalar_construction == ScalarConstruction::PrimePowerLcmTimes12
                ? 12u : 1u;
        if (!compute_batch_s(scalar.v, request.b1, torsion, opt.persistence.exponent_cache_dir))
            throw std::runtime_error("cannot compute the Stage1 scalar");
        const size_t scalar_bits = mpz_sizeinbase(scalar.v, 2);
        if (scalar_bits > UINT32_MAX - 63)
            throw std::invalid_argument("Stage1 scalar exceeds the NTT cursor range");
        const unsigned count = unsigned(scalar_bits - 1);
        std::vector<uint32_t> packed((scalar_bits + 31) / 32);
        mpz_export(packed.data(), nullptr, -1, sizeof(uint32_t), 0, 0, scalar.v);

        Snapshot snapshot;
        bool has_checkpoint = false;
        phase = FailureSource::Persistence;
        checkpoint.validate_pending([&](const auto& records, auto& sigmas) {
            if (records.size() != curves || sigmas.size() != curves ||
                (!edwards && !montgomery_sigma_batch_fits(sigmas.front(), curves, 0)))
                return false;
            for (unsigned i = 0; i < curves; ++i)
                if ((edwards ? sigmas[i] == 0 : sigmas[i] != sigmas.front() + i) ||
                    (i && !records[i].empty()))
                    return false;
            has_checkpoint = read_snapshot(records.front(), count, unsigned(p), curves, snapshot);
            return has_checkpoint;
        });
        uint64_t saved_sigma = 0;
        checkpoint.saved_sigma(0, saved_sigma);
        std::vector<uint64_t> sigmas(curves);
        if (edwards) {
            for (unsigned i = 0; i < curves; ++i) {
                uint64_t saved = 0;
                checkpoint.saved_sigma(i, saved);
                sigmas[i] = select_curve_sigma(saved, opt.identity.sigma_fixed, opt.identity.fixed_sigma64, i);
            }
        } else {
            const uint64_t first = select_batch_sigma(saved_sigma, opt.identity.sigma_fixed,
                                                        opt.identity.fixed_sigma64, curves, 0);
            for (unsigned i = 0; i < curves; ++i) sigmas[i] = first + i;
        }
        phase = FailureSource::Preparation;
        result.curves.resize(curves);
        ecm_edwards::Array factors(curves);
        const unsigned words = unsigned((p + 31) / 32);
        std::vector<uint32_t> lanes_initial(size_t(curves) * 6 * words);
        std::vector<CurveKind> kinds(curves, CurveKind::Stage2Point);
        if (edwards) {
            for (unsigned i = 0; i < curves; ++i)
                kinds[i] = prepare_edwards_lane(N, sigmas[i], factors.data()[i],
                                                lanes_initial.data() + size_t(i) * 6 * words, words);
        } else {
            std::vector<int> hits(curves, ECM_NO_FACTOR_FOUND);
            uint32_t degenerate = curves;
            ecm_cuda_mont_host::ConstructionResults construction{
                factors.data(), hits.data(), nullptr, nullptr, &degenerate};
            size_t bytes = 0;
            const std::unique_ptr<uint32_t, decltype(&std::free)> initial_curves(
                ecm_cuda_mont_host::set_p_2p_suyama(N, curves, sigmas.front(), words * 32,
                                                    &bytes, &construction), std::free);
            if (degenerate < curves)
                throw std::runtime_error("degenerate param0 construction");
            for (unsigned i = 0; i < curves; ++i) {
                if (hits[i] == ECM_FACTOR_FOUND_STEP1) kinds[i] = CurveKind::Factor;
                constexpr unsigned slot_map[] = {3, 4, 5, 6, 1, 2};
                for (unsigned slot = 0; slot < 6; ++slot)
                    std::copy_n(initial_curves.get() + (7 * i + slot_map[slot]) * words,
                                words, lanes_initial.data() + (6 * i + slot) * words);
            }
        }
        std::vector<unsigned> active;
        unsigned lanes = 0;
        int legacy = ECM_NO_FACTOR_FOUND;
        auto edwards_identity = [&](unsigned lane) {
            ecm_ts_fprintf(stdout, "Edwards identity: sigma=%llu, no Stage2 point\n",
                           static_cast<unsigned long long>(sigmas[lane]));
            result.set_curve(lane, CurveKind::CompletedWithoutPoint, nullptr, sigmas[lane]);
        };
        for (unsigned i = 0; i < curves; ++i) {
            if (kinds[i] == CurveKind::Degenerate)
                throw std::runtime_error("degenerate Atkin-Morain construction on sigma " +
                                         std::to_string(sigmas[i]));
            if (kinds[i] == CurveKind::Factor) {
                result.set_curve(i, CurveKind::Factor, factors.data()[i], sigmas[i]);
                legacy = ECM_FACTOR_FOUND_STEP1;
            } else if (kinds[i] == CurveKind::CompletedWithoutPoint) {
                edwards_identity(i);
            } else {
                active.push_back(i);
                lanes |= 1u << i;
            }
        }
        if (has_checkpoint && snapshot.lanes != lanes)
            throw ecm_stage1_checkpoint::Error(ecm_stage1_checkpoint::RestoreState::Incompatible,
                                               "NTT checkpoint active curve mask differs");

        if (!active.empty()) {
            const size_t stride = words * active.size();
            std::vector<uint32_t> initial(6 * stride);
            for (unsigned slot = 0; slot < 6; ++slot)
                for (size_t i = 0; i < active.size(); ++i)
                    std::copy_n(lanes_initial.data() + (6 * active[i] + slot) * words,
                                words, initial.data() + slot * stride + i * words);
            if (has_checkpoint) std::copy(snapshot.points.begin(), snapshot.points.end(), initial.begin());

            ecm_mersenne_ntt::Control control;
            control.resume = snapshot.cursor;
            control.describe = [&](const ecm_mersenne_ntt::Plan& plan) {
                ecm_ts_fprintf(stdout, "stage1_ntt_plan: device=%d p=%zu curves=%zu length=%u bpw=%u digits=%u "
                    "forward_passes=%u inverse_passes=%u forward_layers=%s inverse_layers=%s "
                    "tile_layers=%u tile_elements=%u tile_threads=%u tile_smem_bytes=%u "
                    "outer_max_layers=%u outer_threads=%u warp_tail=%u graph_bits=%u "
                    "profile=%u source=%s resume=%u bits=%u convolution=%s curve=%s\n",
                    opt.execution.device_index, p, active.size(), plan.length, plan.bpw, plan.digits,
                    plan.passes, plan.passes, plan.forward.c_str(), plan.inverse.c_str(),
                    plan.tile, 1u << plan.tile, plan.tile_threads, 8u << plan.tile,
                    plan.outer_max, plan.outer_threads, unsigned(plan.warp_tail),
                    plan.window, opt.execution.ntt.profile, plan.source, control.resume, count,
                    plan.weighted ? "weighted" : "linear", edwards ? "edwards" : "mont0");
            };
            auto last_save = Clock::now(), last_progress = last_save;
            control.poll = [&](unsigned cursor) {
                const auto now = Clock::now();
                if (now - last_progress >= std::chrono::seconds(2)) {
                    ecm_ts_fprintf(stdout, "stage1_ntt_progress: bits=%u/%u (%.1f%%)\n",
                                   cursor, count, 100.0 * cursor / count);
                    last_progress = now;
                }
                const bool stop = ecm_stage1_stop_requested();
                const auto elapsed = std::chrono::duration<double, std::milli>(now - last_save).count();
                const bool due = opt.persistence.ckpt_ms && elapsed >= opt.persistence.ckpt_ms;
                return ecm_mersenne_ntt::Control::Action{!checkpoint.path().empty() && (stop || due), stop};
            };
            control.checkpoint = [&](unsigned cursor, const auto& points) {
                const auto payload = encode_snapshot(count, cursor, lanes, points);
                const bool ok = has_checkpoint ? checkpoint.write_blob(ecm_stage1_curve_record(0), payload)
                                               : checkpoint.import_payload(payload, sigmas);
                if (!ok)
                    throw ecm_stage1_checkpoint::Error(ecm_stage1_checkpoint::RestoreState::IoFailure,
                                                       "cannot persist NTT Stage1 checkpoint");
                has_checkpoint = true;
                last_save = Clock::now();
            };

            phase = FailureSource::Device;
            const auto selected = ecm_mersenne_ntt::select_tuned_plan(opt.execution.device_index,
                unsigned(p), unsigned(active.size()), opt.execution.ntt);
            if (ecm_stage1_stop_requested()) {
                result.stop();
                return result;
            }
            const auto run = ecm_mersenne_ntt::ladder(opt.execution.device_index, unsigned(p),
                unsigned(active.size()), initial, packed, count, selected, control);
            ecm_ts_fprintf(stdout, "stage1_ntt_memory: owned_bytes=%zu observed_bytes=%zu budget_bytes=%zu\n",
                run.owned_bytes, run.observed_bytes, run.budget_bytes);
            ecm_ts_fprintf(stdout, "stage1_ntt_timing: bits=%u device_ms=%.3f wall_ms=%.3f "
                "setup_ms=%.3f readback_ms=%.3f stopped=%u\n",
                run.completed_bits - control.resume, run.device_ms.front(), run.wall_ms.front(),
                1000 * run.setup_seconds, 1000 * run.readback_seconds, unsigned(run.stopped));
            if (run.stopped) {
                result.stop();
                return result;
            }

            phase = FailureSource::Execution;
            ecm_edwards::Integer x, z, inv;
            for (size_t i = 0; i < active.size(); ++i) {
                const unsigned lane = active[i];
                mpz_import(x.v, words, -1, sizeof(uint32_t), 0, 0, run.points.data() + i * words);
                mpz_import(z.v, words, -1, sizeof(uint32_t), 0, 0, run.points.data() + stride + i * words);
                if (!mpz_invert(inv.v, z.v, N)) {
                    mpz_gcd(inv.v, z.v, N);
                    if (mpz_cmp(inv.v, N) == 0) {
                        if (edwards) {
                            edwards_identity(lane);
                            continue;
                        }
                        result.set_curve(lane, CurveKind::Degenerate, nullptr, sigmas[lane]);
                        throw std::runtime_error("degenerate NTT endpoint on sigma " + std::to_string(sigmas[lane]));
                    }
                    result.set_curve(lane, CurveKind::Factor, inv.v, sigmas[lane]);
                    legacy = ECM_FACTOR_FOUND_STEP1;
                } else {
                    mpz_mul(x.v, x.v, inv.v);
                    mpz_mod(x.v, x.v, N);
                    result.set_curve(lane, CurveKind::Stage2Point, x.v, sigmas[lane]);
                }
            }
            if (opt.persistence.ckpt_ms && !checkpoint.path().empty() && control.resume != count)
                control.checkpoint(count, run.points);
        }
        result.legacy_code = legacy;
        result.completion = ComputationResult::Completion::Completed;
        result.validate(N);
    } catch (const ecm_stage1_checkpoint::Error& error) {
        result.fail(FailureSource::Persistence);
        ecm_ts_fprintf(stderr, "ERROR: GPU Stage1 NTT checkpoint: %s\n", error.what());
    } catch (const std::exception& error) {
        if (ecm_stage1_stop_requested()) {
            result.stop();
        } else {
            result.fail(phase);
            ecm_ts_fprintf(stderr, "ERROR: GPU Stage1 NTT: %s\n", error.what());
        }
    }
    return result;
}
} // namespace ecm_cuda_stage1
