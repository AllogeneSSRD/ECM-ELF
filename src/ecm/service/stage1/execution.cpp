#include "execution.h"
#include "preparation.h"
#include "../../backend/cpu/stage1/legacy_executor.h"
#if ECM_HAS_CUDA_EDWARDS
#include "../../backend/cuda/stage1/edwards/host_workflow.h"
#endif
#include "ecm/service/execution/legacy_gpu.h"
#include "../execution/legacy/cgbn_stage1.h"
#include "../../storage/save/ecm_save.h"
#include "../../storage/save/ecm_edwards_io.h"
#include "../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include "../../storage/checkpoint/ecm_stage1_legacy_gpu.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace ecm_stage1 {
static int run_mont_gpu_stage1(const mpz_t N, double B1, double B2, uint32_t curves,
                           const std::string &savefilename, bool saveappend,
                           const std::string &n_expr,
                           Stage1ExecutionOptions &opt, LegacyStage1Result *out) {
    out->ret = ECM_ERROR;
    out->prepare_failed = false;
    out->curves = curves;
    out->firstsigma = 0;
    out->factors = nullptr;
    out->array_found = nullptr;

    if (curves == 0) {
        std::cerr << "gpucurves must be > 0" << std::endl;
        return ECM_ERROR;
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
                         opt.execution.selection.scalar_construction == ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12 ? 12u : 1u)) {
        std::cerr << "Failed to compute batch_s" << std::endl;
        mpz_clear(batch_s);
        ecm_clear(params);
        return ECM_ERROR;
    }
    mpz_set(params->batch_s, batch_s);
    params->batch_last_B1_used = B1;
    {
        const char *algorithm=std::getenv("ECM_GPU_STAGE1_ALGO");
        if(std::string(ecm_backend_name())=="CUDA/CGBN" && (!algorithm || !*algorithm || std::string(algorithm)=="ladder"))
            ecm_stage1_import_legacy_gpu(N,batch_s,curves,uint32_t(params->gpu_param),
                opt.identity.sigma_fixed,opt.identity.fixed_sigma64,[&](uint64_t sigma,uint32_t count){
                    std::vector<uint64_t> sigmas(count);for(uint32_t i=0;i<count;++i)sigmas[i]=sigma+i;
                    return stage1_results_complete(opt,N,uint64_t(B1),false,unsigned(params->gpu_param),sigmas);
                });
    }

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
            return ECM_ERROR;
        }
    }

    std::string resolved_save = savefilename;
    if (!resolved_save.empty()) {
        resolved_save = opencl_ecm_resolve_data_path(resolved_save.c_str());
        if (!opencl_ecm_check_save_file_writable(resolved_save, saveappend)) {
            out->publish_failed = true;
            mpz_clear(batch_s);
            ecm_clear(params);
            return ECM_ERROR;
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
    firstsigma64=ecm_stage1_batch_sigma(0,firstsigma64);
    const uint32_t firstsigma = (uint32_t)(firstsigma64 & 0xFFFFFFFFull);
    if (!gpu_param0 &&
        (firstsigma64 > 0xFFFFFFFFull || curves > 0x100000000ull - firstsigma64)) {
        std::cerr << "sigma range overflows uint32 (sigma + curves > 2^32)" << std::endl;
        for (uint32_t i = 0; i < curves; i++) mpz_clear(factors[i]);
        free(factors);
        free(array_found);
        mpz_clear(batch_s);
        ecm_clear(params);
        return ECM_ERROR;
    }
    for(uint32_t i=1;i<curves;++i)ecm_stage1_batch_sigma(i,firstsigma64+i);
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
    int ret = ecm_backend_stage1(factors, array_found, N, params->batch_s, curves,
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
        !stage1_publish_factor(opt,N,uint64_t(B1),firstsigma64+i,factors[i])){out->publish_failed=true;ret=ECM_ERROR;break;}

    if (ret != ECM_ERROR && !resolved_save.empty()) {
        /* The save must describe the SAME curve family the GPU just ran:
             gpu_param = 3 -> batch parametrization, written with PARAM=3, and with
                              the batch path's historical N rewriting (N divided by
                              the factors found so far);
             gpu_param = 0 -> Suyama param0, written in the param0 form (no PARAM=),
                              which is what gmp-ecm -param 0 and Prime95 sigma_type=1
                              read back for stage 2.  It must carry the ORIGINAL N:
                              gmp-ecm checks the line checksum against N, so the
                              factor-stripped expression makes it reject every line
                              with "bad checksum". */
        const std::string n_expr_stripped =
            opencl_ecm_build_saved_n_expr(n_expr, N, curves, factors, array_found);
        const std::string n_expr_param0 =
            n_expr.empty() ? std::to_string(mpz_sizeinbase(N, 2)) : n_expr;
        bool wrote = false;
        if (params->gpu_param == 0) {
            std::vector<uint64_t> sigmas(curves);
            for (uint32_t i = 0; i < curves; i++) sigmas[i] = firstsigma64 + i;
            std::vector<int> hit(curves, 0);
            /* factors[i] holds the normalized x for a miss and the factor for a hit,
               exactly like the CPU path's xs[] -- and findfactor()/array_found tells
               us which is which. */
            for (uint32_t i = 0; i < curves; i++)
                hit[i] = (array_found[i] != ECM_NO_FACTOR_FOUND) ? 1 : 0;
            wrote = ecm_append_save_lines_mont(resolved_save, N, B1, sigmas.data(), curves,
                                               factors, hit.data(), n_expr_param0);
            if (!wrote)
                std::cerr << "Failed to append param0 save lines into " << resolved_save << std::endl;
        } else if (params->gpu_param == 2) {
            /* param2 ("batch 2", 6-torsion): PARAM=2 with the ORIGINAL N for the same
               checksum reason as param0, and SIGMA = the scalar multiplier sigma0+i the
               GPU used.  gmp-ecm reads this back with -param 2; Prime95 cannot read it
               (sigma_type only accepts 0/1/3).  See
               docs/performance/STAGE1.md */
            wrote = opencl_ecm_append_save_lines(resolved_save, N, B1, firstsigma, curves, factors,
                                                 n_expr_param0, 2);
            if (!wrote)
                std::cerr << "Failed to append param2 save lines into " << resolved_save << std::endl;
        } else {
            wrote = opencl_ecm_append_save_lines(resolved_save, N, B1, firstsigma, curves, factors,
                                                 n_expr_stripped);
            if (!wrote)
                std::cerr << "Failed to append GPU save lines into " << resolved_save << std::endl;
        }
        if (!wrote) {
            out->publish_failed = true;
            ret = ECM_ERROR;
        }
    }

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
    return ret;
}

int execute_stage1(const mpz_t N, double B1, double B2, uint32_t curves,
                           const std::string &savefilename, bool saveappend,
                           const std::string &n_expr,
                           Stage1ExecutionOptions &opt, LegacyStage1Result *out) {
    ecm_stage1_set_backup_limit(opt.persistence.ckpt_backups);
#if ECM_HAS_CUDA_EDWARDS
    if (opt.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu &&
        opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards)
        return ecm_cuda_edwards::run_stage1(N,B1,B2,curves,savefilename,saveappend,opt,out);
#endif
    if (opt.execution.selection.target == ecm_stage1::ExecutionTarget::Cpu &&
        opt.execution.selection.curve == ecm_stage1::CurveModel::Montgomery) {
        return ecm_cpu_stage1::run_mont_stage1(N, B1, B2, curves, savefilename, saveappend,
                               n_expr, opt, out);
    }
    if (opt.execution.selection.target == ecm_stage1::ExecutionTarget::Cpu &&
        opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards) {
        return ecm_cpu_stage1::run_edwards_stage1(N, B1, B2, curves, savefilename, saveappend,
                                  n_expr, opt, out);
    }

    return run_mont_gpu_stage1(N,B1,B2,curves,savefilename,saveappend,n_expr,opt,out);
}
} // namespace ecm_stage1
