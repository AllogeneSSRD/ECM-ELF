#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include "cgbn_stage1_kernel.h"
#include "cgbn_stage1_prac_kernel.cuh"
#include "cgbn_stage1_prac_constants.h"
cgbn_stage1_kernel_fn cgbn_stage1_domain_small(uint32_t, uint32_t *, int);
cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi16(uint32_t, uint32_t *, int);
cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32(uint32_t, uint32_t *, int);
cgbn_stage1_kernel_fn cgbn_stage1_domain_alternate32(uint32_t bits, uint32_t *tpi, int mode);
cgbn_stage1_kernel_fn cgbn_stage1_domain_single_add(uint32_t bits, uint32_t *tpi, int mode);
cgbn_stage1_kernel_fn cgbn_stage1_domain_single_compact32(uint32_t bits, uint32_t *tpi, int mode);
cgbn_stage1_kernel_fn cgbn_stage1_domain_dispatch(uint32_t bits, uint32_t *tpi, int mode, uint32_t requested_tpi) {
    if (mode == ECM_DOMAIN_PRAC_CONSTANT_RUNTIME || mode == ECM_DOMAIN_PRAC_CONSTANT_NP0 ||
        mode == ECM_DOMAIN_PRAC_CONSTANT_M4423) {
        if (!requested_tpi || requested_tpi == 16)
            return cgbn_stage1_domain_constants(bits, tpi, mode);
        *tpi = 0; return nullptr;
    }
    if (mode == ECM_DOMAIN_PRAC_SINGLE_ADD || mode == ECM_DOMAIN_PRAC_SINGLE_ADD_168 ||
        mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT || mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT_168 ||
        mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT_128) {
        if (!requested_tpi || requested_tpi == 16)
            return cgbn_stage1_domain_single_add(bits, tpi, mode);
        if (requested_tpi == 32 && (mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT ||
                                    mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT_168))
            return cgbn_stage1_domain_single_compact32(bits, tpi, mode);
        *tpi = 0; return nullptr;
    }
    cgbn_stage1_kernel_fn fn;
    fn = cgbn_stage1_domain_small(bits, tpi, mode);
    if (!fn) fn = cgbn_stage1_domain_tpi16(bits, tpi, mode);
    if (!fn) fn = cgbn_stage1_domain_tpi32(bits, tpi, mode);
    if (fn && (!requested_tpi || requested_tpi == *tpi)) return fn;
    if (requested_tpi == 32) return cgbn_stage1_domain_alternate32(bits, tpi, mode);
    *tpi = 0; return nullptr;
}
