#include "batches.h"
#include "tune.h"
#include "ecm/stage1/ntt_target.h"
#include "../../../edwards/memory.h"
#include "../../../sigma.h"
#include "../../../../../io/stage1_binary.h"
#include "../../../../../io/stage1_checkpoint.h"
#include "ecm/platform/logging.h"
#include <algorithm>
#include <iterator>

namespace ecm_cuda_stage1 {
namespace {
using namespace ecm_stage1;
using namespace ecm_stage1_binary;
using ecm_stage1_checkpoint::Context;
using ecm_stage1_checkpoint::Error;
using ecm_stage1_checkpoint::RestoreState;
constexpr std::string_view legacy_magic = "NTT-TASK-1";
constexpr std::string_view sequential_magic = "NTT-TASK-2";
constexpr std::string_view random_magic = "NTT-TASK-3";

struct TaskProgress {
    uint64_t first = 0;
    unsigned preferred = 0, curves = 0;
    std::string native, completed_bytes;
    ComputationResult completed;
    bool random = false;

    uint64_t sigma(uint32_t index) const {
        if (!random) return first + index;
        constexpr uint64_t mask = (uint64_t(1) << 53) - 1;
        uint64_t value = (first - 6 + index) % (mask - 5) + 6;
        // Cycle-walk a permutation so the task gets distinct 53-bit sigmas
        // and can reconstruct each one from the seed and its curve index.
        do {
            value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull) & mask;
            value = ((value ^ (value >> 27)) * 0x94d049bb133111ebull) & mask;
            value ^= value >> 31;
        } while (value < 6);
        return value;
    }

    std::string snapshot() const {
        std::string bytes(random ? random_magic : sequential_magic);
        u64(bytes, first);
        u32(bytes, preferred);
        u32(bytes, curves);
        u32(bytes, curves); // Keep the wire field for older snapshots.
        u32(bytes, uint32_t(completed.curves.size()));
        u64(bytes, native.size());
        bytes += native;
        bytes += completed_bytes;
        return bytes;
    }
};

void append_result(std::string& bytes, const CurveOutcome& curve, std::string& scratch);

bool restore(std::string_view bytes, const Stage1Request& request, TaskProgress& state) {
    try {
        Reader reader(bytes);
        const auto format = reader.take(legacy_magic.size());
        if (format != legacy_magic && format != sequential_magic && format != random_magic)
            return false;
        TaskProgress saved;
        saved.random = format == random_magic;
        saved.first = reader.u64();
        saved.preferred = reader.u32();
        saved.curves = reader.u32();
        const auto capacity = reader.u32();
        const auto completed = reader.u32();
        const auto native_size = reader.u64();
        const auto modulus_bytes = (mpz_sizeinbase(request.modulus, 2) + 7) / 8;
        const auto arithmetic_bytes = (ntt_arithmetic_bits(request.modulus, request.options) + 7) / 8;
        const uint64_t scheduled = uint64_t(completed) + saved.curves;
        if (scheduled > UINT32_MAX || scheduled >= uint64_t(request.curves) + stage1_ntt_max_curves ||
            (saved.curves && completed >= request.curves) ||
            !sigma_batch_fits(saved.first, std::max(request.curves, uint32_t(scheduled)), 1) ||
            (saved.random && (request.options.execution.selection.curve != CurveModel::Edwards ||
                              saved.first < 6 || saved.first >= (uint64_t(1) << 53))) ||
            (request.options.identity.sigma_fixed &&
             (saved.random || request.options.identity.fixed_sigma64 != saved.first)) ||
            saved.preferred > stage1_ntt_max_curves || capacity > saved.preferred ||
            saved.curves > capacity || bool(saved.curves) != bool(capacity) ||
            (native_size && !saved.curves) ||
            native_size > 6ull * saved.curves * (arithmetic_bytes + 3) + 256)
            return false;
        saved.native = reader.take(native_size);
        const auto results_start = reader.at;
        if (completed > reader.left() / 5)
            return false;
        saved.completed.curves.resize(completed);
        bool factor = false;
        for (uint32_t i = 0; i < completed; ++i) {
            auto& curve = saved.completed.curves[i];
            const auto kind = uint8_t(reader.take(1).front());
            if (kind != unsigned(CurveKind::Stage2Point) &&
                kind != unsigned(CurveKind::Factor) && kind != unsigned(CurveKind::CompletedWithoutPoint))
                return false;
            const auto value = reader.take(reader.u32());
            if (value.size() > modulus_bytes)
                return false;
            curve.kind = static_cast<CurveKind>(kind);
            curve.sigma = saved.sigma(i);
            mpz_import(curve.value, value.size(), -1, 1, 0, 0, value.data());
            if (curve.kind == CurveKind::Factor && format != legacy_magic) {
                for (auto* coordinate : {&curve.endpoint.x, &curve.endpoint.z}) {
                    const auto text = reader.take(reader.u32());
                    if (text.size() > 2 * modulus_bytes ||
                        text.find_first_not_of("0123456789abcdef") != text.npos) return false;
                    *coordinate = text;
                }
                if (curve.endpoint.x.empty() != curve.endpoint.z.empty()) return false;
            }
            factor |= curve.kind == CurveKind::Factor;
        }
        if (reader.left())
            return false;
        saved.completed.completion = ComputationResult::Completion::Completed;
        saved.completed.legacy_code = factor ? ECM_FACTOR_FOUND_STEP1 : ECM_NO_FACTOR_FOUND;
        if (completed)
            saved.completed.validate(request.modulus);
        if (format == legacy_magic) {
            std::string scratch(modulus_bytes, '\0');
            for (const auto& curve : saved.completed.curves)
                append_result(saved.completed_bytes, curve, scratch);
        } else saved.completed_bytes = bytes.substr(results_start);
        state = std::move(saved);
        return true;
    } catch (const std::runtime_error&) {
        return false;
    }
}

void append_result(std::string& bytes, const CurveOutcome& curve, std::string& scratch) {
    bytes.push_back(char(curve.kind));
    if (mpz_sgn(curve.value) == 0) {
        u32(bytes, 0);
        return;
    }
    size_t length = 0;
    mpz_export(scratch.data(), &length, -1, 1, 0, 0, curve.value);
    u32(bytes, uint32_t(length));
    bytes.append(scratch.data(), length);
    if (curve.kind == CurveKind::Factor) {
        for (const auto& coordinate : {&curve.endpoint.x, &curve.endpoint.z}) {
            u32(bytes, uint32_t(coordinate->size()));
            bytes += *coordinate;
        }
    }
}
} // namespace

ecm_stage1::ComputationResult run_ntt_batches(
    const ecm_stage1::Stage1Request& request, const ecm_stage1::TaskDependencies& deps,
    ecm_stage1_checkpoint::Context& session, const std::string& identity,
    const std::vector<std::string>& compatible_identities,
    const std::function<void(const ecm_stage1::CurveOutcome&, uint32_t)>& publish_factor) {
    using namespace ecm_stage1;
    using namespace ecm_edwards_memory;
    const auto free = host_free_bytes();
    const auto bits = ntt_arithmetic_bits(request.modulus, request.options);
    if (!bits)
        throw std::invalid_argument("NTT Stage1 task has no eligible arithmetic modulus");
    const auto budget = free ? usable_bytes(*free, 64 * mib) : UINT64_MAX;
    if (batch_host_bytes(request.curves, bits) > budget)
        throw std::runtime_error("NTT Stage1 task exceeds available host memory; reduce total curves");

    TaskProgress state;
    session.begin(identity, 1, false, 0, true, compatible_identities);
    session.validate_pending([&](const auto& records, auto&) {
        return records.size() == 1 && restore(records.front(), request, state);
    });
    const auto restored = uint32_t(state.completed.curves.size()) + state.curves;
    if (restored > request.curves && batch_host_bytes(restored, bits) > budget)
        throw std::runtime_error("NTT Stage1 checkpoint requires more available host memory");
    if (!state.first) {
        const auto& seed = request.options.identity;
        state.first = select_batch_sigma(0, seed.sigma_fixed, seed.fixed_sigma64, request.curves, 0);
        state.random = request.options.execution.selection.curve == CurveModel::Edwards &&
                       !seed.sigma_fixed && seed.random_sigma;
    }
    // Preserve Prime95 eligibility for Montgomery queue tasks whose requested range fits.
    const auto sigma_limit = request.options.persistence.retain_completed_checkpoint &&
        request.options.execution.selection.curve == CurveModel::Montgomery &&
        sigma_batch_fits(state.first, request.curves, 1, INT64_MAX) ? uint64_t(INT64_MAX) : UINT64_MAX;
    const auto persist = [&]() {
        return session.path().empty() ||
            session.write_blob(ecm_stage1_curve_record(0), state.snapshot());
    };
    const auto save_progress = [&]() {
        if (!persist())
            throw Error(RestoreState::IoFailure, "cannot commit NTT Stage1 task progress");
    };
    const auto publish_factors = [&](uint32_t first) {
        try {
            for (uint32_t i = first; i < state.completed.curves.size(); ++i) {
                const auto& completed = state.completed.curves[i];
                if (completed.kind == CurveKind::Factor)
                    publish_factor(completed, i);
            }
            return true;
        } catch (const std::exception& e) {
            ecm_ts_fprintf(stderr, "ERROR: Stage1 factor publication: %s\n", e.what());
            return false;
        }
    };
    auto failure = ComputationResult{};
    uint32_t offset = uint32_t(state.completed.curves.size());
    bool resumed = state.curves != 0;
    if (offset || resumed)
        ecm_ts_fprintf(stdout, "stage1_ntt_batch_resume: completed=%u total=%u\n", offset, request.curves);
    if (!publish_factors(0)) {
        failure.fail(FailureSource::Publication);
        return failure;
    }
    state.completed.curves.reserve(request.curves);
    std::string scratch((mpz_sizeinbase(request.modulus, 2) + 7) / 8, '\0');
    while (offset < request.curves) {
        if (ecm_stage1_stop_requested()) {
            failure.stop();
            return failure;
        }
        if (!state.curves) {
            const unsigned left = request.curves - offset;
            if (left >= state.preferred)
                state.preferred = ecm_mersenne_ntt::select_production_batch(request, left);
            if (!state.preferred || state.preferred > stage1_ntt_max_curves)
                throw std::logic_error("invalid NTT Stage1 scheduler batch size");
            state.curves = std::min(state.preferred, left);
            const uint64_t rounded = uint64_t(offset) + state.preferred;
            if (left < state.preferred && request.options.execution.ntt.pad_batch &&
                rounded <= UINT32_MAX && sigma_batch_fits(state.first, uint32_t(rounded), 1, sigma_limit) &&
                batch_host_bytes(uint32_t(rounded), bits) <= budget)
                state.curves = state.preferred;
            save_progress();
        }
        ecm_ts_fprintf(stdout, "stage1_ntt_batch: offset=%u total=%u curves=%u capacity=%u resumed=%u\n",
                       offset, request.curves, state.curves, state.curves, unsigned(resumed));
        auto options = request.options;
        options.identity.sigma_fixed = !state.random;
        options.identity.fixed_sigma64 = state.random ? 0 : state.sigma(offset);
        options.execution.ntt.observe_batch = true;
        const Stage1Request batch{request.modulus, request.b1, request.b2, state.curves, request.save_file,
                                  request.append_save, request.expression, options};
        ExecutionOptions execution(options);
        Context child;
        std::vector<uint64_t> sigmas(state.curves);
        for (uint32_t i = 0; i < state.curves; ++i) sigmas[i] = state.sigma(offset + i);
        child.begin_nested(session.path(), std::move(sigmas), state.native,
            [&](const std::string& checkpoint) {
                state.native = checkpoint;
                return persist();
            });
        auto result = deps.execute(batch, execution, child);
        child.close(false);
        resumed = false;
        if (result.retry_smaller_ntt_batch && state.native.empty() && state.curves > 1) {
            const auto left = request.curves - offset;
            state.curves = state.curves > left ? left : state.curves / 2;
            state.preferred = state.curves;
            ecm_ts_fprintf(stdout, "stage1_ntt_batch_retry: offset=%u curves=%u reason=budget\n",
                           offset, state.curves);
            save_progress();
            continue;
        }
        if (result.completion != ComputationResult::Completion::Completed)
            return result;
        result.validate(request.modulus);
        if (result.curves.size() != state.curves)
            throw std::runtime_error("NTT Stage1 batch result has the wrong curve count");
        for (uint32_t i = 0; i < state.curves; ++i) {
            if (result.curves[i].sigma != state.sigma(offset + i))
                throw std::runtime_error("NTT Stage1 batch result has the wrong sigma");
            append_result(state.completed_bytes, result.curves[i], scratch);
        }
        const auto previous = offset;
        offset += state.curves;
        if (result.legacy_code == ECM_FACTOR_FOUND_STEP1)
            state.completed.legacy_code = ECM_FACTOR_FOUND_STEP1;
        state.completed.curves.insert(state.completed.curves.end(),
            std::make_move_iterator(result.curves.begin()), std::make_move_iterator(result.curves.end()));
        state.curves = 0;
        state.native.clear();
        save_progress();
        if (!publish_factors(previous)) {
            failure.fail(FailureSource::Publication);
            return failure;
        }
        ecm_ts_fprintf(stdout, "stage1_ntt_batch_complete: completed=%u total=%u\n", offset, request.curves);
    }
    state.completed.completion = ComputationResult::Completion::Completed;
    if (state.completed.legacy_code != ECM_FACTOR_FOUND_STEP1)
        state.completed.legacy_code = ECM_NO_FACTOR_FOUND;
    return std::move(state.completed);
}
} // namespace ecm_cuda_stage1
