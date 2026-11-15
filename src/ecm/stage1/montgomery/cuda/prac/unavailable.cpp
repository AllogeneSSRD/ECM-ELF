#include "runtime.h"
#include "ecm/cuda/runtime/cuda_ecm_shim.h"

int ecm_cuda_stage1::run_mont_resident(ecm_stage1_checkpoint::Context&, mpz_t*, int*,
    mpz_srcptr, mpz_srcptr, uint32_t, uint64_t*, unsigned long, float*, uint64_t, uint32_t, bool,
    RunResult&, const ecm_cuda_mont_host::ConstructionResults&, SigmaGenerator,
    const ecm_cuda_mont_host::MontRequestedSpec&) {
    outputf(OUTPUT_ERROR, "GPU: resident/PRAC kernels are not built; rebuild with "
                         "-EnablePrac or -DECM_CUDA_ENABLE_PRAC=ON\n");
    return ECM_ERROR;
}
