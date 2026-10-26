// Device arithmetic and thin launchers; no GMP, progress or checkpoint policy.
#pragma once
#include "../../../../cuda/arithmetic/cgbn/types.cuh"
#include "../launch.h"
#include "../point.cuh"
#include "../naf.cuh"
namespace {
template<uint32_t W,uint32_t T> struct Params {
    static const uint32_t BITS=W,TPI=T,TPB=128,MAX_ROTATION=1,SHM_LIMIT=0;
    static const bool CONSTANT_TIME=false;
};
template<class P> __global__ void slice(cgbn_error_report_t *report,
        cgbn_mem_t<P::BITS> *data,const uint32_t *scalar,uint32_t total_bits,
        uint32_t start,uint32_t stop,uint32_t count) {
    using Context=cgbn_context_t<P::TPI,P>;
    using Env=cgbn_env_t<Context,P::BITS>;
    const uint32_t i=(blockIdx.x*blockDim.x+threadIdx.x)/P::TPI;
    if(i>=count)return;
    Context ctx(cgbn_report_monitor,report,i);Env env(ctx);
    auto *slot=data+10*i;
    typename Env::cgbn_t n,d;
    cgbn_edwards_point<Env> p,r;
    cgbn_load(env,n,slot);cgbn_load(env,d,slot+1);
    const uint32_t np0=cgbn_bn2mont(env,d,d,n);
    cgbn_load(env,p.x,slot+2);cgbn_bn2mont(env,p.x,p.x,n);
    cgbn_load(env,p.y,slot+3);cgbn_bn2mont(env,p.y,p.y,n);
    cgbn_load(env,p.z,slot+4);cgbn_bn2mont(env,p.z,p.z,n);
    cgbn_load(env,p.t,slot+5);cgbn_bn2mont(env,p.t,p.t,n);
    cgbn_load(env,r.x,slot+6);cgbn_bn2mont(env,r.x,r.x,n);
    cgbn_load(env,r.y,slot+7);cgbn_bn2mont(env,r.y,r.y,n);
    cgbn_load(env,r.z,slot+8);cgbn_bn2mont(env,r.z,r.z,n);
    cgbn_load(env,r.t,slot+9);cgbn_bn2mont(env,r.t,r.t,n);
    cgbn_edwards_field<Env> f{env,n,np0};
    for(uint32_t pos=start;pos<stop;++pos) {
        f.dbl(r,r);const uint32_t bit=total_bits-1-pos;
        if((scalar[bit/32]>>(bit%32))&1u)f.add_point(r,r,p,d);
    }
    cgbn_mont2bn(env,r.x,r.x,n,np0);cgbn_store(env,slot+6,r.x);
    cgbn_mont2bn(env,r.y,r.y,n,np0);cgbn_store(env,slot+7,r.y);
    cgbn_mont2bn(env,r.z,r.z,n,np0);cgbn_store(env,slot+8,r.z);
    cgbn_mont2bn(env,r.t,r.t,n,np0);cgbn_store(env,slot+9,r.t);
}
}
namespace ecm_edwards_cuda_detail {
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) \
const Kernels &kernels_##W() { \
    using P=Params<W,T>;using Mem=cgbn_mem_t<W>; \
    static_assert(sizeof(Mem)==W/8,"CGBN buffer stride"); \
    static_assert(P::TPB==ET,"Edwards block geometry"); \
    static const Kernels k{W,T,ET, \
        [](void *r,void *d,const uint32_t *s,uint32_t b,uint32_t a,uint32_t z,uint32_t n){ \
            slice<P><<<(n+128/T-1)/(128/T),128>>>(static_cast<cgbn_error_report_t*>(r),static_cast<Mem*>(d),s,b,a,z,n);}, \
        [](void *r,void *d,void *q,uint32_t e,uint32_t n){ \
            edwards_naf_prepare<P><<<(n+128/T-1)/(128/T),128>>>(static_cast<cgbn_error_report_t*>(r),static_cast<Mem*>(d),static_cast<Mem*>(q),e,n);}, \
        [](void *r,void *d,void *q,void *o,const uint32_t *nz,const uint32_t *c,uint32_t b,uint32_t a,uint32_t z,uint32_t e,uint32_t w,uint32_t n,int dbl){ \
            edwards_naf_slice<P><<<(n+128/T-1)/(128/T),128>>>(static_cast<cgbn_error_report_t*>(r),static_cast<Mem*>(d),static_cast<Mem*>(q),static_cast<Mem*>(o),nz,c,b,a,z,e,w,n,dbl);} }; \
    return k; \
}
#include "../../../../cuda/dispatch/tiers.def"
#undef ECM_CUDA_TIER
}
