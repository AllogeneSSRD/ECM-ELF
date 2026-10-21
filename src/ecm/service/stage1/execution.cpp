#include "execution.h"
#include "../../backend/cpu/stage1/legacy_executor.h"
#include "../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#if ECM_HAS_CUDA_STAGE1
#include "../../backend/cuda/stage1/execution.h"
#else
#include "../../backend/opencl/stage1/montgomery/workflow.h"
#endif

namespace ecm_stage1 {

int execute_stage1(const mpz_t N, double B1, double B2, uint32_t curves,
                           const std::string &savefilename, bool saveappend,
                           const std::string &n_expr,
                           ExecutionOptions &opt, ExecutionResult *out) {
    ecm_stage1_set_backup_limit(opt.persistence.ckpt_backups);
#if ECM_HAS_CUDA_STAGE1
    if (opt.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu &&
        opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards)
        return ecm_cuda_stage1::run_edwards_stage1(N,B1,B2,curves,savefilename,saveappend,n_expr,opt,out);
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

#if ECM_HAS_CUDA_STAGE1
    return ecm_cuda_stage1::run_mont_stage1(N,B1,B2,curves,savefilename,saveappend,n_expr,opt,out);
#else
    return ecm_opencl_stage1::run_mont_stage1(N,B1,B2,curves,savefilename,saveappend,n_expr,opt,out);
#endif
}
} // namespace ecm_stage1
