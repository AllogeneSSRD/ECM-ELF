#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include "cgbn_stage1_kernel.h"
#include "cgbn_stage1_prac_kernel.cuh"
#include "cgbn_stage1_prac_single_add.cuh"
#include "cgbn_stage1_prac_constants.h"

// Match the single-compact/cap168 body, with a runtime control in the same TU.
// INIT/EXPORT remain the ordinary kernels: R, layout and checkpoint ABI agree.
template<class P, int MODE>
__global__ void __maxnreg__(168) kernel_suyama_constants(
    cgbn_error_report_t *report, uint64_t total, uint64_t start, uint64_t length,
    uint32_t *control, uint32_t *data, uint32_t count, uint32_t unused, uint32_t np0) {
    (void)total; (void)unused;
    static_assert(P::BITS == 4608 && P::TPI == 16, "constant policies require 4608/TPI16");
    int instance = (blockIdx.x * blockDim.x + threadIdx.x) / P::TPI;
    if (instance >= count) return;
    curve_t<P> c(CHECK_ERROR ? cgbn_report_monitor : cgbn_no_checks, report, instance);
    using bn = typename curve_t<P>::bn_t;
    auto *mem = reinterpret_cast<typename curve_t<P>::mem_t *>(data) + 7 * instance;
    bn n;
    if constexpr (MODE == ECM_DOMAIN_PRAC_CONSTANT_M4423) {
        // CGBN load distributes consecutive words: index=lane*9+limb.
        // Only lane15 has the tail: FFFF,FFFF,FFFF,7F,0,0,0,0,0.
        const bool tail = (threadIdx.x & 15) == 15;
        #pragma unroll
        for (int limb = 0; limb < 9; ++limb) {
            if (limb < 3) n._limbs[limb] = 0xffffffffu;
            else n._limbs[limb] = tail ? (limb == 3 ? 0x7fu : 0u) : 0xffffffffu;
        }
    } else cgbn_load(c._env, n, mem);
    if constexpr (MODE != ECM_DOMAIN_PRAC_CONSTANT_RUNTIME) np0 = 1;
    bn ax, az, bx, bz, cx, cz, a24;
    cgbn_load(c._env, a24, mem + 1);
    cgbn_load(c._env, ax, mem + 3); cgbn_load(c._env, az, mem + 4);
    auto *primes = reinterpret_cast<const EcmPracPrime *>(control);
    for (uint64_t i = start; i < start + length; ++i) {
        EcmPracPrime entry = primes[i];
        for (uint32_t repeat = 0; repeat < entry.repetitions; ++repeat) {
            if (entry.p == 2) { prac_dbl<P, true>(c, ax, az, ax, az, a24, n, np0); continue; }
            prac_odd_shared_dbl(c, ax, az, bx, bz, cx, cz, a24, n, np0, entry.p, entry.d);
        }
    }
    cgbn_store(c._env, mem + 3, ax); cgbn_store(c._env, mem + 4, az);
}

cgbn_stage1_kernel_fn cgbn_stage1_domain_constants(uint32_t bits, uint32_t *tpi, int mode) {
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4608)
    if (bits == 4608) {
        *tpi = 16;
        switch (mode) {
        case ECM_DOMAIN_PRAC_CONSTANT_RUNTIME:
            return kernel_suyama_constants<cgbn_params_t<16,4608>, ECM_DOMAIN_PRAC_CONSTANT_RUNTIME>;
        case ECM_DOMAIN_PRAC_CONSTANT_NP0:
            return kernel_suyama_constants<cgbn_params_t<16,4608>, ECM_DOMAIN_PRAC_CONSTANT_NP0>;
        case ECM_DOMAIN_PRAC_CONSTANT_M4423:
            return kernel_suyama_constants<cgbn_params_t<16,4608>, ECM_DOMAIN_PRAC_CONSTANT_M4423>;
        }
    }
#endif
    *tpi = 0; return nullptr;
}
