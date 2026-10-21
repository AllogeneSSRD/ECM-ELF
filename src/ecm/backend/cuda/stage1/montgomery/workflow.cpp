#include "../execution.h"
#include "../../../../service/stage1/preparation.h"
#include "ecm/service/execution/legacy_gpu.h"
#include "../../../../service/stage1/gpu_common.h"
#include "ecm/storage/results/publication.h"
#include "../../../../storage/save/ecm_save.h"
#include "../../../../storage/save/ecm_edwards_io.h"
#include "../../../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include "../../../../storage/checkpoint/ecm_stage1_legacy_gpu.h"
#include "../../runtime/cuda_ecm_shim.h"
#include "runtime.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace ecm_stage1;
namespace ecm_cuda_stage1 {
namespace {
struct Parameters {
    ecm_params value;
    Parameters(){ecm_init(value);}
    Parameters(const Parameters &)=delete;
    Parameters &operator=(const Parameters &)=delete;
    ~Parameters(){ecm_clear(value);}
};
struct FactorPublication {
    const Stage1Options &options;
    mpz_srcptr modulus;
    uint64_t b1;
    const uint64_t *sigma;
};
bool publish_construction_factor(uint32_t lane,mpz_srcptr factor,void *context) {
    const auto &p=*static_cast<FactorPublication *>(context);
    return stage1_publish_factor(p.options,p.modulus,p.b1,*p.sigma+lane,factor);
}
}

int run_mont_stage1(const mpz_t N, double B1, double B2, uint32_t curves,
                           const std::string &savefilename, bool saveappend,
                           const std::string &n_expr,
                           const Stage1Options &opt, ExecutionResult *out) {
    *out=ExecutionResult{};
    auto phase=FailureSource::Preparation;
    try {
        if(!curves)throw std::invalid_argument("gpucurves must be > 0");
        out->allocate_curves(curves);

        // Setup params.
        Parameters owned_params;
        auto *params=owned_params.value;
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

        ecm_edwards::Integer owned_scalar;
        auto *batch_s=owned_scalar.v;
        if (!compute_batch_s(batch_s, B1,
                             opt.execution.selection.scalar_construction == ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12 ? 12u : 1u)) {
            std::cerr << "Failed to compute batch_s" << std::endl;
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
                        return ecm_results::contains_stage1_curves(opt.persistence.results_file,N,uint64_t(B1),false,unsigned(params->gpu_param),sigmas);
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
                out->fail(FailureSource::Preparation);
                return ECM_ERROR;
            }
        }

        std::string resolved_save = savefilename;
        if (!resolved_save.empty()) {
            resolved_save = opencl_ecm_resolve_data_path(resolved_save.c_str());
            if (!opencl_ecm_check_save_file_writable(resolved_save, saveappend)) {
                out->fail(FailureSource::Publication);
                return ECM_ERROR;
            }
        }

        auto *factors=out->factors;
        auto *array_found=out->array_found;

        // Resolve saved sigma before invoking the generator in the runtime.
        uint64_t firstsigma64=opt.identity.sigma_fixed?opt.identity.fixed_sigma64:0;
        const SigmaGenerator generate_sigma=opt.identity.sigma_fixed?nullptr:
            +[](uint32_t count,int param)->uint64_t {
                return param==0?random_sigma_u64():uint64_t(gpu_pick_random_sigma(count));
            };

        std::cout << "Using B1=" << B1 << ", B2=" << B2
                  << " (" << curves << " curves, " << ecm_backend_name() << ")" << std::endl;

        float gputime = 0.0f;
        ecm_cuda_set_verbose(params->verbose);
        phase=FailureSource::Execution;
        FactorPublication publication{opt,N,uint64_t(B1),&firstsigma64};
        const auto run=execute_mont_batch(factors,array_found,N,params->batch_s,curves,
            &firstsigma64,params->gpu_checkpoint_interval_ms,&gputime,params->verbose,
            params->gpu_param,uint64_t(std::floor(B1+0.0001)),
            opt.execution.selection.scalar_construction==ScalarConstruction::PrimePowerLcmTimes12?12u:1u,
            publish_construction_factor,&publication,generate_sigma);
        int ret=run.code;
        const auto firstsigma=uint32_t(firstsigma64);
        out->firstsigma=uint32_t(firstsigma64);out->firstsigma64=firstsigma64;
        for(uint32_t i=0;i<curves;++i)out->sigmas[i]=firstsigma64+i;
        if(ret==ECM_ERROR) {
            if(run.termination==Termination::Stopped)out->stop();
            else if(run.termination==Termination::SampleLimit)out->completion=ExecutionResult::Completion::SampleLimit;
            else out->fail(run.failure==Failure::Preparation?FailureSource::Preparation:
                run.failure==Failure::Device?FailureSource::Device:
                run.failure==Failure::Persistence?FailureSource::Persistence:
                run.failure==Failure::Publication?FailureSource::Publication:FailureSource::Execution);
        }
        phase=FailureSource::Publication;

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
                out->fail(FailureSource::Publication);
                ret = ECM_ERROR;
            }
        }


        out->ret=ret;
        return ret;
    } catch(const std::exception &error) {
        out->fail(out->publish_failed?FailureSource::Publication:phase);
        std::cerr<<"ERROR: GPU Montgomery Stage1: "<<error.what()<<std::endl;
        return ECM_ERROR;
    }
}

} // namespace ecm_cuda_stage1
