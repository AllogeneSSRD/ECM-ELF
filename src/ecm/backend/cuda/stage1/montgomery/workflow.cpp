#include "../execution.h"
#include "../../../../service/stage1/preparation.h"
#include "ecm/service/execution/legacy_gpu.h"
#include "../../../../service/stage1/gpu_common.h"
#include "ecm/storage/results/publication.h"
#include "../../../../storage/save/ecm_edwards_io.h"
#include "../../../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include "../../../../storage/checkpoint/ecm_stage1_legacy_gpu.h"
#include "../../runtime/cuda_ecm_shim.h"
#include "runtime.h"
#include "host_policy.h"
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iostream>

using namespace ecm_stage1;

namespace ecm_cuda_stage1 {
namespace {
struct ConstructionCapture {
    mpz_srcptr modulus;
    const uint64_t* sigma;
    LegacyExecutionResult* result;
};

bool capture_construction_factor(uint32_t lane, mpz_srcptr factor, void* context) {
    const auto& p = *static_cast<ConstructionCapture*>(context);
    p.result->set_curve(lane, LegacyExecutionResult::CurveKind::Factor, factor, *p.sigma + lane);
    return validate_stage1_factor(p.modulus, factor);
}
} // namespace

ecm_stage1::ComputationResult run_mont_stage1(const ecm_stage1::Stage1Request& request,
                                              ecm_stage1::ExecutionOptions& opt) {
    const auto N = request.modulus;
    const auto B1 = request.b1, B2 = request.b2;
    const auto curves = request.curves;
    LegacyExecutionResult raw;
    auto* out = &raw;

    *out = LegacyExecutionResult{};
    auto phase = FailureSource::Preparation;
    try {
        if (!curves)
            throw std::invalid_argument("gpucurves must be > 0");
        out->allocate_curves(curves);
        std::fill(out->curve_kinds.begin(), out->curve_kinds.end(),
                  LegacyExecutionResult::CurveKind::Failed);

        // Prepare: freeze the algorithm request before checkpoint inspection.
        const auto requested = ecm_cuda_mont_host::capture_mont_request(opt.gpu_algorithm.c_str());

        ecm_edwards::Integer owned_scalar;
        auto* batch_s = owned_scalar.v;
        if (!compute_batch_s(batch_s, B1,
                             opt.execution.selection.scalar_construction ==
                                     ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12
                                 ? 12u
                                 : 1u)) {
            std::cerr << "Failed to compute batch_s" << std::endl;
            out->fail(FailureSource::Preparation);
            return ecm_stage1::take_computation_result(std::move(raw), ECM_ERROR,
                                                       ecm_stage1_stop_requested() != 0);
        }
        // Restore candidates are inspected before fresh sigma generation.
        {
            const auto& algorithm = opt.gpu_algorithm;
            if (std::string(ecm_backend_name()) == "CUDA/CGBN" && algorithm == "ladder")
                ecm_stage1_import_legacy_gpu(
                    N, batch_s, curves, uint32_t(opt.execution.selection.gpu_param),
                    opt.identity.sigma_fixed, opt.identity.fixed_sigma64,
                    [&](uint64_t sigma, uint32_t count) {
                        std::vector<uint64_t> sigmas(count);
                        for (uint32_t i = 0; i < count; ++i)
                            sigmas[i] = sigma + i;
                        return ecm_results::contains_stage1_curves(
                            opt.persistence.results_file, N, uint64_t(B1), false,
                            unsigned(opt.execution.selection.gpu_param), sigmas);
                    });
        }

        {
            const int prep = ecm_backend_prepare(
                (size_t)mpz_sizeinbase(N, 2), opt.execution.verbose, opt.execution.device_index,
                opt.execution.gpu_mul_path.empty() ? nullptr : opt.execution.gpu_mul_path.c_str(),
                opt.execution.gpu_sqr_path.empty() ? nullptr : opt.execution.gpu_sqr_path.c_str(),
                opt.execution.gpu_add_path.empty() ? nullptr : opt.execution.gpu_add_path.c_str(),
                opt.execution.gpu_sub_path.empty() ? nullptr : opt.execution.gpu_sub_path.c_str(),
                opt.execution.gpu_special_mult_path.empty()
                    ? nullptr
                    : opt.execution.gpu_special_mult_path.c_str());
            if (prep != 0) {
                std::cerr << "GPU: backend prepare failed" << std::endl;
                out->fail(FailureSource::Preparation);
                return ecm_stage1::take_computation_result(std::move(raw), ECM_ERROR,
                                                           ecm_stage1_stop_requested() != 0);
            }
        }

        auto* factors = out->factors;
        auto* array_found = out->array_found;

        // Resolve saved sigma before invoking the generator in the runtime.
        uint64_t firstsigma64 = opt.identity.sigma_fixed ? opt.identity.fixed_sigma64 : 0;
        const SigmaGenerator generate_sigma =
            opt.identity.sigma_fixed ? nullptr : +[](uint32_t count, int param) -> uint64_t {
            return param == 0 ? random_sigma_u64() : uint64_t(gpu_pick_random_sigma(count));
        };

        std::cout << "Using B1=" << B1 << ", B2=" << B2 << " (" << curves << " curves, "
                  << ecm_backend_name() << ")" << std::endl;

        // Execute: runtime accepts a compatible layout or constructs a fresh batch.
        float gputime = 0.0f;
        ecm_cuda_set_verbose(opt.execution.verbose);
        phase = FailureSource::Execution;
        ConstructionCapture capture{N, &firstsigma64, out};
        const auto run = execute_mont_batch(
            factors, array_found, N, batch_s, curves, &firstsigma64, opt.persistence.ckpt_ms,
            &gputime, opt.execution.verbose, opt.execution.selection.gpu_param,
            uint64_t(std::floor(B1 + 0.0001)),
            opt.execution.selection.scalar_construction == ScalarConstruction::PrimePowerLcmTimes12
                ? 12u
                : 1u,
            capture_construction_factor, &capture, generate_sigma, &requested);
        int ret = run.code;
        // Collect: transfer the exact restored/generated identity and normalized values.
        out->firstsigma = uint32_t(firstsigma64);
        out->firstsigma64 = firstsigma64;
        for (uint32_t i = 0; i < curves; ++i)
            out->sigmas[i] = firstsigma64 + i;
        if (ret == ECM_ERROR) {
            if (run.termination == Termination::Stopped)
                out->stop();
            else if (run.termination == Termination::SampleLimit)
                out->completion = LegacyExecutionResult::Completion::SampleLimit;
            else
                out->fail(run.failure == Failure::Preparation   ? FailureSource::Preparation
                          : run.failure == Failure::Device      ? FailureSource::Device
                          : run.failure == Failure::Persistence ? FailureSource::Persistence
                          : run.failure == Failure::Publication ? FailureSource::Publication
                                                                : FailureSource::Execution);
        }
        if (run.degenerate_curve < curves)
            out->set_curve(run.degenerate_curve, LegacyExecutionResult::CurveKind::Degenerate,
                           nullptr, firstsigma64 + run.degenerate_curve);
        if (ret != ECM_ERROR)
            for (uint32_t i = 0; i < curves; ++i)
                out->set_curve(i,
                               array_found[i] == ECM_NO_FACTOR_FOUND
                                   ? LegacyExecutionResult::CurveKind::Stage2Point
                                   : LegacyExecutionResult::CurveKind::Factor,
                               factors[i], firstsigma64 + i);
        if (ret != ECM_ERROR)
            out->validate_curves(N);
        phase = FailureSource::Execution;

        std::cout << "GPU stage1 returned: " << ret << " gputime=" << gputime << " ms\n";
        if (ret != ECM_ERROR)
            out->completion = LegacyExecutionResult::Completion::Completed;
        out->ret = ret;
        return ecm_stage1::take_computation_result(std::move(raw), ret,
                                                   ecm_stage1_stop_requested() != 0);
    } catch (const std::exception& error) {
        out->fail(out->publish_failed ? FailureSource::Publication : phase);
        std::cerr << "ERROR: GPU Montgomery Stage1: " << error.what() << std::endl;
        return ecm_stage1::take_computation_result(std::move(raw), ECM_ERROR,
                                                   ecm_stage1_stop_requested() != 0);
    }
}

} // namespace ecm_cuda_stage1
