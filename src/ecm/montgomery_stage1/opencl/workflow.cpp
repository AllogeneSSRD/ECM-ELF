#include "ecm/io/results.h"
#include "workflow.h"
#include "../../stage1/preparation.h"
#include "ecm/stage1/gpu_backend.h"
#include "api.h"
#include "ecm/io/save.h"
#include "../../../io/stage1_checkpoint.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace ecm_stage1;
namespace ecm_opencl_stage1 {
ecm_stage1::ComputationResult run_mont_stage1(const ecm_stage1::Stage1Request &request,ecm_stage1::ExecutionOptions &opt, ecm_stage1_checkpoint::Context& checkpoint) {
    const auto N=request.modulus;const auto n=N;
    const auto B1=request.b1,B2=request.b2,b1=B1,b2=B2;
    const auto curves=request.curves;
    const auto &n_expr=request.expression;
    LegacyExecutionResult raw;auto *out=&raw;

    out->ret = ECM_ERROR;
    out->prepare_failed = false;
    out->curves = curves;
    out->firstsigma = 0;
    out->factors = nullptr;
    out->array_found = nullptr;

    if (curves == 0) {
        std::cerr << "gpucurves must be > 0" << std::endl;
        return ecm_stage1::take_computation_result(std::move(raw),ECM_ERROR,ecm_stage1_stop_requested()!=0);
    }

    // Setup params.
    ecm_params params;
    ecm_init(params);
    params->gpu = 1;
    params->gpu_number_of_curves = curves;
    params->gpu_checkpoint_interval_ms = opt.persistence.ckpt_ms;
    params->gpu_param = opt.execution.selection.gpu_param;
    if (!opt.execution.gpu_mul_path.empty()) {
        strncpy(params->gpu_mul_path, opt.execution.gpu_mul_path.c_str(), sizeof(params->gpu_mul_path) - 1u);
        params->gpu_mul_path[sizeof(params->gpu_mul_path) - 1u] = '\0';
    }
    if (!opt.execution.gpu_sqr_path.empty()) {
        strncpy(params->gpu_sqr_path, opt.execution.gpu_sqr_path.c_str(), sizeof(params->gpu_sqr_path) - 1u);
        params->gpu_sqr_path[sizeof(params->gpu_sqr_path) - 1u] = '\0';
    }
    if (!opt.execution.gpu_add_path.empty()) {
        strncpy(params->gpu_add_path, opt.execution.gpu_add_path.c_str(), sizeof(params->gpu_add_path) - 1u);
        params->gpu_add_path[sizeof(params->gpu_add_path) - 1u] = '\0';
    }
    if (!opt.execution.gpu_sub_path.empty()) {
        strncpy(params->gpu_sub_path, opt.execution.gpu_sub_path.c_str(), sizeof(params->gpu_sub_path) - 1u);
        params->gpu_sub_path[sizeof(params->gpu_sub_path) - 1u] = '\0';
    }
    if (!opt.execution.gpu_special_mult_path.empty()) {
        strncpy(params->gpu_special_mult_path, opt.execution.gpu_special_mult_path.c_str(),
                sizeof(params->gpu_special_mult_path) - 1u);
        params->gpu_special_mult_path[sizeof(params->gpu_special_mult_path) - 1u] = '\0';
    }
    params->verbose = opt.execution.verbose ? 1 : 0;
    params->param = ECM_PARAM_BATCH_32BITS_D;

    mpz_t batch_s;
    mpz_init(batch_s);
    if (!compute_batch_s(batch_s, B1,
                         opt.execution.selection.scalar_construction == ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12 ? 12u : 1u, opt.persistence.exponent_cache_dir)) {
        std::cerr << "Failed to compute batch_s" << std::endl;
        mpz_clear(batch_s);
        ecm_clear(params);
        return ecm_stage1::take_computation_result(std::move(raw),ECM_ERROR,ecm_stage1_stop_requested()!=0);
    }
    mpz_set(params->batch_s, batch_s);
    params->batch_last_B1_used = B1;

    {
        const int prep = ecm_backend_prepare((size_t)mpz_sizeinbase(N, 2), params->verbose,
                                             opt.execution.device_index,
                                             params->gpu_mul_path[0] ? params->gpu_mul_path : nullptr,
                                             params->gpu_sqr_path[0] ? params->gpu_sqr_path : nullptr,
                                             params->gpu_add_path[0] ? params->gpu_add_path : nullptr,
                                             params->gpu_sub_path[0] ? params->gpu_sub_path : nullptr,
                                             params->gpu_special_mult_path[0] ? params->gpu_special_mult_path
                                                                             : nullptr);
        if (prep != 0) {
            std::cerr << "GPU: backend prepare failed" << std::endl;
            out->prepare_failed = true;
            mpz_clear(batch_s);
            ecm_clear(params);
            return ecm_stage1::take_computation_result(std::move(raw),ECM_ERROR,ecm_stage1_stop_requested()!=0);
        }
    }

    mpz_t *factors = (mpz_t *)malloc(sizeof(mpz_t) * curves);
    int *array_found = (int *)malloc(sizeof(int) * curves);
    for (uint32_t i = 0; i < curves; i++) {
        mpz_init(factors[i]);
        array_found[i] = ECM_NO_FACTOR_FOUND;
    }

    /* Curve index (sigma) of the first curve.
       gpu_param = 3 : the batch parametrization carries d = sigma/2^32 as a 32-bit
                       kernel parameter, so sigma must stay inside 2^32;
       gpu_param = 0 : Suyama sigma is a 64-bit value on the CPU path too, so use the
                       same 53-bit random generator there (Prime95's ECMSTAGE2 reader
                       only accepts sigma < 2^63, and random_sigma_u64() is < 2^53). */
    const bool gpu_param0 = (params->gpu_param == 0);
    uint64_t firstsigma64 = opt.identity.sigma_fixed
        ? opt.identity.fixed_sigma64
        : (gpu_param0 ? random_sigma_u64() : (uint64_t)gpu_pick_random_sigma(curves));
    firstsigma64=checkpoint.sigma(0,firstsigma64);
    const uint32_t firstsigma = (uint32_t)(firstsigma64 & 0xFFFFFFFFull);
    if (!gpu_param0 &&
        (firstsigma64 > 0xFFFFFFFFull || curves > 0x100000000ull - firstsigma64)) {
        std::cerr << "sigma range overflows uint32 (sigma + curves > 2^32)" << std::endl;
        for (uint32_t i = 0; i < curves; i++) mpz_clear(factors[i]);
        free(factors);
        free(array_found);
        mpz_clear(batch_s);
        ecm_clear(params);
        return ecm_stage1::take_computation_result(std::move(raw),ECM_ERROR,ecm_stage1_stop_requested()!=0);
    }
    for(uint32_t i=1;i<curves;++i)checkpoint.sigma(i,firstsigma64+i);
    mpz_t batch_d;
    mpz_init(batch_d);
    /* batch_d is the batch parametrization's curve constant; only param3 uses it */
    gpu_compute_batch_d(batch_d, firstsigma, N);

    // B1/B2 are fixed for this run. The *actual* sigma is printed by the backend
    // after it applies any checkpoint resume (the checkpoint may override the
    // freshly-computed sigma), so it is not printed here.
    std::cout << "Using B1=" << B1 << ", B2=" << B2
              << " (" << curves << " curves, " << ecm_backend_name() << ")" << std::endl;

    float gputime = 0.0f;
    int ret = ecm_backend_stage1(&checkpoint, factors, array_found, N, params->batch_s, curves,
                                       &firstsigma64, params->gpu_checkpoint_interval_ms,
                                       &gputime, params->verbose, params->gpu_param,
                                       (uint64_t)std::floor(B1 + 0.0001),
                                       opt.execution.selection.scalar_construction == ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12 ? 12u : 1u,
                                       params->gpu_mul_path[0] ? params->gpu_mul_path : nullptr,
                                       params->gpu_sqr_path[0] ? params->gpu_sqr_path : nullptr,
                                       params->gpu_add_path[0] ? params->gpu_add_path : nullptr,
                                       params->gpu_sub_path[0] ? params->gpu_sub_path : nullptr,
                                       params->gpu_special_mult_path[0] ? params->gpu_special_mult_path
                                                                       : nullptr);

    std::cout << "GPU stage1 returned: " << ret << " gputime=" << gputime << " ms\n";
    if(ret!=ECM_ERROR)for(uint32_t i=0;i<curves;++i)if(array_found[i]!=ECM_NO_FACTOR_FOUND &&
        !validate_stage1_factor(N,factors[i])){out->publish_failed=true;ret=ECM_ERROR;break;}

    mpz_clear(batch_d);
    mpz_clear(batch_s);
    ecm_clear(params);

    out->ret = ret;
    out->firstsigma = (uint32_t)(firstsigma64 & 0xFFFFFFFFull);
    out->firstsigma64 = firstsigma64;
    out->factors = factors;
    out->array_found = array_found;
    /* GPU batches use the contiguous convention: curve i has sigma = firstsigma + i
       (the same rule as -sigma / -gpucurves). */
    out->sigmas = (uint64_t *)malloc(sizeof(uint64_t) * (curves ? curves : 1));
    if (out->sigmas != nullptr) {
        for (uint32_t i = 0; i < curves; i++) out->sigmas[i] = firstsigma64 + i;
    }
    return ecm_stage1::take_computation_result(std::move(raw),ret,ecm_stage1_stop_requested()!=0);
}
}
