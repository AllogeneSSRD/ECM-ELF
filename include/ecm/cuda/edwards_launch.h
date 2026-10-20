#pragma once
#include <cstdint>
namespace ecm_edwards_cuda_detail {
// data has ten contiguous container slots/curve; scalar/codes are uint32 words.
// Host owns all buffers until the existing synchronization/readback completes.
// NAF data and dictionary retain Montgomery residues across scalar slices.
struct Kernels {
    uint32_t bits,tpi,tpb;
    void (*binary)(void*,void*,const uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t);
    void (*prepare)(void*,void*,void*,uint32_t,uint32_t);
    void (*naf)(void*,void*,void*,void*,const uint32_t*,const uint32_t*,
                uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,int);
};
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) const Kernels &kernels_##W();
#include "tiers.def"
#undef ECM_CUDA_TIER
}
