#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include "cgbn_stage1_kernel.h"
#include "cgbn_stage1_prac_kernel.cuh"
cgbn_stage1_kernel_fn cgbn_stage1_domain_small(uint32_t, uint32_t *, int);
cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi16(uint32_t, uint32_t *, int);
cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32(uint32_t, uint32_t *, int);
cgbn_stage1_kernel_fn cgbn_stage1_domain_alternate32(uint32_t bits, uint32_t *tpi, int mode);
cgbn_stage1_kernel_fn cgbn_stage1_domain_single_add(uint32_t bits, uint32_t *tpi, int mode);
cgbn_stage1_kernel_fn cgbn_stage1_domain_dispatch(uint32_t bits, uint32_t *tpi, int mode, uint32_t requested_tpi) {
    if (mode == ECM_DOMAIN_PRAC_SINGLE_ADD || mode == ECM_DOMAIN_PRAC_SINGLE_ADD_168) {
        if (!requested_tpi || requested_tpi == 16)
            return cgbn_stage1_domain_single_add(bits, tpi, mode);
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
