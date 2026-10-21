#pragma once
// Include CGBN's public types before this interface. No project device templates.
#include <cstdint>
#ifndef ECM_TPB
#define ECM_TPB 128
#endif
#define CARRY_BITS 6
#ifndef ECM_CUDA_FULL_BUILD
#define IS_DEV_BUILD
#endif
#ifndef ECM_MERS_FOLD
#define ECM_MERS_FOLD 0
#endif
#ifndef ECM_NO_PARAM2
#define ECM_NO_PARAM2 0
#endif
const uint32_t TPB_DEFAULT=ECM_TPB;
typedef void (*cgbn_stage1_kernel_fn)(cgbn_error_report_t *, uint64_t, uint64_t,
                                      uint64_t, uint32_t*, uint32_t*, uint32_t,
                                      uint32_t, uint32_t);

cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi4(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi8(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi16(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi32(uint32_t BITS, uint32_t *TPI_out);

/* Suyama param0 variants (method = gpu + gpu_param = 0).  Kept in their own TU so
   the extra template instantiations do not slow down the param3 kernel builds; the
   instantiated grid mirrors the param3 one exactly (see the answer to "can the two
   share instantiations?" in docs/architecture/STAGE1.md). */
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi4(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi8(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi16(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi32(uint32_t BITS, uint32_t *TPI_out);

/* param2 ("batch 2", 6-torsion, method = gpu + gpu_param = 2) variants: the same kernel
   body instantiated with CONST_DIFF = true, so the differential addition drops the xdiff
   multiply (5M+4S vs param0's 6M+4S).  Its own TU for the same reason as the Suyama set,
   and its own family rather than a runtime flag (that cost 34%, docs §5.6). */
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi4(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi8(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi16(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi32(uint32_t BITS, uint32_t *TPI_out);




void cgbn_stage1_launch(cgbn_stage1_kernel_fn kernel,dim3 grid,dim3 block,
    cgbn_error_report_t *report,uint64_t scalar_bits,uint64_t start,uint64_t count,
    uint32_t *scalar,uint32_t *data,uint32_t curves,uint32_t sigma,uint32_t np0);
