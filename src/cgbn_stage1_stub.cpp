#include <gmp.h>
#include <cstdint>
#include "ecm/service/execution/ecm.h"
#include <iostream>

extern "C" int cgbn_ecm_stage1(mpz_t *factors, int *array_found,
                               const mpz_t N, const mpz_t s,
                               uint32_t curves, uint32_t *sigma,
                               unsigned long checkpoint_interval_ms,
                               float *gputime, int verbose,
                               const char *gpu_mul_path, const char *gpu_sqr_path,
                               const char *gpu_add_path, const char *gpu_sub_path,
                               const char *gpu_special_mult_path)
{
    (void)gpu_mul_path;
    (void)gpu_sqr_path;
    (void)gpu_add_path;
    (void)gpu_sub_path;
    (void)gpu_special_mult_path;
    // Minimal stub: mark no factors found, set gputime to 0
    std::cout << "Warning: cgbn_ecm_stage1 is a stub that does not perform any computation.\n";
    if(gputime) *gputime = 0.0f;
    for(uint32_t i=0;i<curves;i++){
        array_found[i] = ECM_NO_FACTOR_FOUND;
        if(factors) mpz_set_ui(factors[i], 0);
    }
    return ECM_NO_FACTOR_FOUND;
}