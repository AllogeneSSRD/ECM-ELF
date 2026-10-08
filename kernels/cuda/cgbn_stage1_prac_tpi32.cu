// Lightweight dispatch; heavy TPI32 instantiations live in four separate TUs.
#include <gmp.h>
#include <cgbn.h>
#include <cuda_runtime.h>
#include "cgbn_stage1_kernel.h"

cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32_9216_10240(uint32_t bits, uint32_t *tpi, int mode);
cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32_11264_12288(uint32_t bits, uint32_t *tpi, int mode);
cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32_13312_14336(uint32_t bits, uint32_t *tpi, int mode);
cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32_15360_16384(uint32_t bits, uint32_t *tpi, int mode);

cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32(uint32_t bits, uint32_t *tpi, int mode) {
    if (bits == 9216 || bits == 10240) return cgbn_stage1_domain_tpi32_9216_10240(bits, tpi, mode);
    if (bits == 11264 || bits == 12288) return cgbn_stage1_domain_tpi32_11264_12288(bits, tpi, mode);
    if (bits == 13312 || bits == 14336) return cgbn_stage1_domain_tpi32_13312_14336(bits, tpi, mode);
    if (bits == 15360 || bits == 16384) return cgbn_stage1_domain_tpi32_15360_16384(bits, tpi, mode);
    *tpi = 0; return nullptr;
}
