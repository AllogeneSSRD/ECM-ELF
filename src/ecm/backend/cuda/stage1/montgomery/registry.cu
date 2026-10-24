#include "../../arithmetic/cgbn/types.cuh"
#include "registry.cuh"
#include "api.h"
#include "../../runtime/cuda_ecm_shim.h"
#include <cuda_runtime.h>
void cgbn_stage1_launch(cgbn_stage1_kernel_fn kernel,dim3 grid,dim3 block,
    cgbn_error_report_t *report,uint64_t scalar_bits,uint64_t start,uint64_t count,
    uint32_t *scalar,uint32_t *data,uint32_t curves,uint32_t sigma,uint32_t np0) {
    kernel<<<grid,block>>>(report,scalar_bits,start,count,scalar,data,curves,sigma,np0);
}
#ifdef ECM_CUDA_ISOLATE_TIERS
#define ECM_CUDA_MONT_ONLY
#ifdef IS_DEV_BUILD
#define ECM_CUDA_MONT_DEV_ONLY
#endif
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) \
cgbn_stage1_kernel_fn ecm_mont_param0_w##W(uint32_t,uint32_t*); \
cgbn_stage1_kernel_fn ecm_mont_param2_w##W(uint32_t,uint32_t*); \
cgbn_stage1_kernel_fn ecm_mont_param3_w##W(uint32_t,uint32_t*);
#include "../../dispatch/tiers.def"
#undef ECM_CUDA_TIER
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) \
if (BITS == W) { \
    if (param == 0) return ecm_mont_param0_w##W(BITS,TPI_out); \
    if (param == 2) return ecm_mont_param2_w##W(BITS,TPI_out); \
    return ecm_mont_param3_w##W(BITS,TPI_out); }
static cgbn_stage1_kernel_fn isolated_dispatch(int param,uint32_t BITS,uint32_t *TPI_out) {
#include "../../dispatch/tiers.def"
    return nullptr;
}
#undef ECM_CUDA_TIER
#undef ECM_CUDA_MONT_DEV_ONLY
#undef ECM_CUDA_MONT_ONLY
cgbn_stage1_kernel_fn cgbn_stage1_kernel_dispatch(uint32_t b,uint32_t *t) {return isolated_dispatch(3,b,t);}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_dispatch(uint32_t b,uint32_t *t) {return isolated_dispatch(0,b,t);}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_dispatch(uint32_t b,uint32_t *t) {return isolated_dispatch(2,b,t);}
#else
cgbn_stage1_kernel_fn ecm_mont_param3_g0(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param3_g1(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param3_g2(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param3_g3(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param3_g4(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param3_g5(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param3_g6(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi4(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param3_g0(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi8(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param3_g1(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi16(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param3_g2(bits,tpi))return fn;
    if(auto fn=ecm_mont_param3_g3(bits,tpi))return fn;
    if(auto fn=ecm_mont_param3_g4(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi32(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param3_g5(bits,tpi))return fn;
    if(auto fn=ecm_mont_param3_g6(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn ecm_mont_param0_g0(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param0_g1(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param0_g2(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param0_g3(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param0_g4(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param0_g5(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param0_g6(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi4(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param0_g0(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi8(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param0_g1(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi16(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param0_g2(bits,tpi))return fn;
    if(auto fn=ecm_mont_param0_g3(bits,tpi))return fn;
    if(auto fn=ecm_mont_param0_g4(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi32(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param0_g5(bits,tpi))return fn;
    if(auto fn=ecm_mont_param0_g6(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn ecm_mont_param2_g0(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param2_g1(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param2_g2(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param2_g3(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param2_g4(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param2_g5(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn ecm_mont_param2_g6(uint32_t,uint32_t*);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi4(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param2_g0(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi8(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param2_g1(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi16(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param2_g2(bits,tpi))return fn;
    if(auto fn=ecm_mont_param2_g3(bits,tpi))return fn;
    if(auto fn=ecm_mont_param2_g4(bits,tpi))return fn;
    return nullptr;
}
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi32(uint32_t bits,uint32_t *tpi) {
    if(auto fn=ecm_mont_param2_g5(bits,tpi))return fn;
    if(auto fn=ecm_mont_param2_g6(bits,tpi))return fn;
    return nullptr;
}

#endif

void ecm_build_tier_list_default(std::vector<uint32_t> &out) {
  out.clear();
#define ECM_CUDA_ALL_TIERS
#ifdef IS_DEV_BUILD
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) if constexpr (M && D) out.push_back(W);
#else
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) if constexpr (M) out.push_back(W);
#endif
#include "../../dispatch/tiers.def"
#undef ECM_CUDA_TIER
#undef ECM_CUDA_ALL_TIERS
#ifdef IS_DEV_BUILD
    outputf(OUTPUT_ALWAYS, "Warning: Dev buils, only support N<1024.\n");
    outputf(OUTPUT_ALWAYS, "Warning: Using dev build with only 2 kernels. Consider adding more kernels for better performance on large inputs.\n");
#endif
}

// param0 uses the same container grid as param3, but the list is built without the
// dev-build guards: the selection loop skips the tiers this build has no kernel for.
void ecm_build_tier_list_param0(std::vector<uint32_t> &out) {
  out.clear();
#define ECM_CUDA_ALL_TIERS
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) if constexpr (M) out.push_back(W);
#include "../../dispatch/tiers.def"
#undef ECM_CUDA_TIER
#undef ECM_CUDA_ALL_TIERS
}



#ifndef ECM_CUDA_ISOLATE_TIERS
cgbn_stage1_kernel_fn cgbn_stage1_kernel_dispatch(uint32_t BITS, uint32_t *TPI_out) {
    cgbn_stage1_kernel_fn k = cgbn_stage1_kernel_tpi4(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_tpi8(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_tpi16(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_tpi32(BITS, TPI_out);
    return k;
}

/* Suyama param0 kernel lookup (see cgbn_stage1_kernels_suyama.cu). */
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_dispatch(uint32_t BITS, uint32_t *TPI_out) {
    cgbn_stage1_kernel_fn k = cgbn_stage1_kernel_suyama_tpi4(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_suyama_tpi8(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_suyama_tpi16(BITS, TPI_out);
    if (k != nullptr) return k;
    return cgbn_stage1_kernel_suyama_tpi32(BITS, TPI_out);
}

/* param2 kernel lookup (see cgbn_stage1_kernels_param2.cu). */
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_dispatch(uint32_t BITS, uint32_t *TPI_out) {
    cgbn_stage1_kernel_fn k = cgbn_stage1_kernel_param2_tpi4(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_param2_tpi8(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_param2_tpi16(BITS, TPI_out);
    if (k != nullptr) return k;
    return cgbn_stage1_kernel_param2_tpi32(BITS, TPI_out);
}

#endif

/* ── D4: kernel-tier query (ecm_cuda.exe --gpu-info) ─────────────────────────────
   Reports the tiers this build carries plus the TPB/TPI each one launches with, using the
   SAME tier lists and the SAME per-TPI dispatch as the run path. That is what makes the
   `gpucurves` recommendation (`curves = blocks_per_sm * sm_count * (tpb / tpi)`)
   trustworthy: the numbers cannot drift from what the kernel would actually pick.

   Pure query: no kernel launch, no device call, no file access. `want_bits` is the bit
   length of the number the task will factor (CARRY_BITS is added here, so callers pass the
   plain bit size); 0 means "every tier". Returns the number of entries written, or -1 when
   no tier is large enough. */
int ecm_cuda_tier_list(int param0, int param2, uint32_t want_bits,
                       ecm_cuda_tier *out, int max_out, uint32_t *carry_bits) {
    if (carry_bits != nullptr) *carry_bits = (uint32_t)CARRY_BITS;
    if (out == nullptr || max_out <= 0) return -1;

    (void)param0;
#if ECM_NO_PARAM2
    if (param2) return -1;
#else
    (void)param2;
#endif
    struct Tier {uint32_t bits,tpi;};
#define ECM_CUDA_MONT_ONLY
#ifdef IS_DEV_BUILD
#define ECM_CUDA_MONT_DEV_ONLY
#endif
    static const Tier tiers[]={
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) {W,T},
#include "../../dispatch/tiers.def"
#undef ECM_CUDA_TIER
        {0,0}};
#undef ECM_CUDA_MONT_DEV_ONLY
#undef ECM_CUDA_MONT_ONLY

    int written = 0;
    for (const Tier &tier : tiers) {
        const uint32_t bits = tier.bits;
        if (!bits) continue;
        if (want_bits != 0u && bits < want_bits + (uint32_t)CARRY_BITS) continue;
        if (written < max_out) {
            out[written].bits = bits;
            out[written].tpb = (uint32_t)TPB_DEFAULT;
            out[written].tpi = tier.tpi;
        }
        ++written;
        if (want_bits != 0u) break;          /* the kernel would stop at the first fit */
    }
    if (want_bits != 0u && written == 0) return -1;
    return written > max_out ? max_out : written;
}

int ecm_cuda_fold_build(void) {
#if ECM_MERS_FOLD
    return 1;
#else
    return 0;
#endif
}

int ecm_cuda_tier_occupancy(int param0, int param2, uint32_t bits,
                            uint32_t *blocks_per_sm) {
    if (blocks_per_sm == nullptr) return 0;
    *blocks_per_sm = 0;

    uint32_t tpi_u32 = 0;
    cgbn_stage1_kernel_fn k = param0 ? cgbn_stage1_kernel_suyama_dispatch(bits, &tpi_u32)
                             : param2 ? cgbn_stage1_kernel_param2_dispatch(bits, &tpi_u32)
                                      : cgbn_stage1_kernel_dispatch(bits, &tpi_u32);
    if (k == nullptr) return 0;   /* tier not instantiated in this build */

    int occ = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, (int)TPB_DEFAULT, 0)
            != cudaSuccess || occ <= 0) {
        return 0;
    }
    *blocks_per_sm = (uint32_t)occ;
    return 1;
}
