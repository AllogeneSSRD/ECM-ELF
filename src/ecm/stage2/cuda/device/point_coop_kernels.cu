#include <gmp.h>
#include <ecm/cuda/cgbn.h>
#include "../launch.h"
#include "../point_coop_plan.h"
#include "point_coop_chain.cuh"
#define ECM_STAGE2_BABY_WHEEL_COOP_ONLY
#include "../../../../../kernels/cuda/stage2_baby_wheel.cuh"
#undef ECM_STAGE2_BABY_WHEEL_COOP_ONLY

// One finite dispatch shared by both the production chain and its mandatory
// arithmetic checks. The selected template preserves the caller's actual radix.
#define S2G_COOP_RUN(W,T,K,...) K<W,T><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(__VA_ARGS__)
#define S2G_COOP_DISPATCH(K,...) do { \
    if(tpi!=32) { S2G_COOP_RUN(256,16,K,__VA_ARGS__); } \
    else switch(s2g_coop_capacity_words(nw,tpi)) { \
        case 160: S2G_COOP_RUN(160,32,K,__VA_ARGS__); break; \
        case 192: S2G_COOP_RUN(192,32,K,__VA_ARGS__); break; \
        case 224: S2G_COOP_RUN(224,32,K,__VA_ARGS__); break; \
        default: S2G_COOP_RUN(256,32,K,__VA_ARGS__); break; \
    } \
} while(0)

void ecm_launch_s2g_coop_chain_kernel(EcmCudaLaunch cfg,int tpi,const unsigned long long *n,
    unsigned long long ninv,int nw,const unsigned long long *dx,const unsigned long long *dz,
    const unsigned long long *sx,const unsigned long long *sz,const unsigned long long *ex,
    const unsigned long long *ez,unsigned long long chains,unsigned long long points,
    unsigned long long length,size_t seed_stride,int mersenne_bits,
    unsigned long long *out_x,unsigned long long *out_z)
{
    S2G_COOP_DISPATCH(stage2_coop_chain,n,ninv,nw,dx,dz,sx,sz,
        ex,ez,chains,points,length,seed_stride,mersenne_bits,out_x,out_z);
}

void ecm_launch_s2g_coop_mont_test_kernel(EcmCudaLaunch cfg,int tpi,const unsigned long long *a,
    const unsigned long long *b,const unsigned long long *n,unsigned long long ninv,int nw,
    int cases,int mersenne_bits,bool square,unsigned long long *out)
{
    S2G_COOP_DISPATCH(stage2_coop_mont_test,a,b,n,ninv,nw,cases,mersenne_bits,square,out);
}

void ecm_launch_s2g_baby_wheel_coop(EcmCudaLaunch cfg, int nw,
    const unsigned long long* n, unsigned long long ninv, const unsigned long long* one,
    const ecm_stage2::baby_wheel::Segment* segments, unsigned count, const unsigned* slots,
    const unsigned long long* sx, const unsigned long long* sz, int mersenne_bits,
    unsigned long long* out_x, unsigned long long* out_z, unsigned long long* guard) {
    using ecm_stage2::baby_wheel::coop_chain;
#define S2G_WHEEL_RUN(W) coop_chain<W, 32><<<cfg.grid, cfg.block, cfg.shared, cfg.stream>>>( \
    n, ninv, nw, one, segments, count, slots, sx, sz, mersenne_bits, out_x, out_z, guard)
    switch (s2g_coop_capacity_words(nw, 32)) {
        case 160: S2G_WHEEL_RUN(160); break;
        case 192: S2G_WHEEL_RUN(192); break;
        case 224: S2G_WHEEL_RUN(224); break;
        default: S2G_WHEEL_RUN(256); break;
    }
#undef S2G_WHEEL_RUN
}

#undef S2G_COOP_DISPATCH
#undef S2G_COOP_RUN
