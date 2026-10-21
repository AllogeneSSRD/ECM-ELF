/* ecm_gpu_common.cpp — backend-neutral GPU helpers shared by all ECM backends.

   These two routines are pure GMP arithmetic (no OpenCL/CUDA), used by the
   driver to pick a starting sigma and to compute the batch-mode d value. They
   were previously defined inside the OpenCL stage-1 translation unit; they live
   here so both the `ecm` (OpenCL) and `ecm_cuda` (CUDA) executables can link
   them without pulling in a specific GPU backend.
*/

#include "gpu_common.h"

#include <gmp.h>

#include <chrono>
#include <cstdint>
#include <ctime>

#ifdef _WIN32
#include <io.h>
#include <process.h>
#ifndef getpid
#define getpid _getpid
#endif
#else
#include <unistd.h>
#endif

// Uniform sigma in [1, UINT32_MAX - curves] for ECM_PARAM_BATCH_32BITS_D.
extern "C" uint32_t gpu_pick_random_sigma(uint32_t curves) {
    if (curves == 0 || (uint64_t)curves >= (uint64_t)UINT32_MAX) {
        return 2u;
    }

    gmp_randstate_t rng;
    gmp_randinit_default(rng);
    unsigned long seed = (unsigned long)time(nullptr);
    seed ^= (unsigned long)getpid() * 0x9e3779b9ul;
    seed ^= (unsigned long)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    gmp_randseed_ui(rng, seed);

    mpz_t range, r;
    mpz_init(range);
    mpz_init(r);
    mpz_set_ui(range, 0);
    mpz_setbit(range, 32);
    mpz_sub_ui(range, range, (unsigned long)curves);
    mpz_urandomm(r, rng, range);
    uint32_t sigma = (uint32_t)mpz_get_ui(r) + 1u;

    mpz_clear(range);
    mpz_clear(r);
    gmp_randclear(rng);
    return sigma;
}

extern "C" void gpu_compute_batch_d(mpz_t d_out, uint32_t sigma, const mpz_t N) {
    mpz_t pow2_32, inv;
    mpz_init(pow2_32);
    mpz_init(inv);
    mpz_ui_pow_ui(pow2_32, 2, 32);
    mpz_invert(inv, pow2_32, N);
    mpz_set_ui(d_out, sigma);
    mpz_mul(d_out, d_out, inv);
    mpz_mod(d_out, d_out, N);
    mpz_clear(pow2_32);
    mpz_clear(inv);
}
