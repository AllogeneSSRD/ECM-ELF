#pragma once
#include "point_coop_field.cuh"

// Same binary ladder and projective scale as the scalar point backend. Return
// both neighbours so the arithmetic can also serve paired seeds independently.
template<class Env>
__device__ __forceinline__ void stage2_coop_ladder_pair(const Stage2PaddedCoopField<Env> &f,
    unsigned long long k,const typename Env::cgbn_t &px,const typename Env::cgbn_t &pz,
    const typename Env::cgbn_t &a24,const typename Env::cgbn_t &mone,
    typename Env::cgbn_t &ax,typename Env::cgbn_t &az,
    typename Env::cgbn_t &bx,typename Env::cgbn_t &bz)
{
    if(!k) {
        cgbn_set(f.e,ax,mone);cgbn_set_ui32(f.e,az,0);
        cgbn_set(f.e,bx,px);cgbn_set(f.e,bz,pz);return;
    }
    cgbn_set(f.e,ax,px);cgbn_set(f.e,az,pz);f.xdbl(bx,bz,px,pz,a24);
    const int top=63-__clzll(k);
    for(int bit=top-1;bit>=0;--bit) {
        if((k>>bit)&1ull) {f.xadd(ax,az,ax,az,bx,bz,px,pz);f.xdbl(bx,bz,bx,bz,a24);}
        else {f.xadd(bx,bz,ax,az,bx,bz,px,pz);f.xdbl(ax,az,ax,az,a24);}
    }
}

template<int WORDS,int TPI>
__global__ void stage2_coop_ladder(const unsigned long long *n,unsigned long long ninv,int nw,
    const unsigned long long *qx,const unsigned long long *qz,const unsigned long long *a24,
    const unsigned long long *mone,const unsigned long long *scalars,int points,int mersenne_bits,
    bool normal_output,unsigned long long *out_x,unsigned long long *out_z)
{
    const int i=(blockIdx.x*blockDim.x+threadIdx.x)/TPI;if(i>=points)return;
    using Ctx=cgbn_context_t<TPI>;using Env=cgbn_env_t<Ctx,64*WORDS>;using Bn=typename Env::cgbn_t;
    Ctx ctx(cgbn_no_checks);Env e(ctx);Bn bn,px,pz,da,one,ax,az,bx,bz;
    stage2_coop_load<Env>(bn,n,nw);stage2_coop_load<Env>(px,qx,nw);stage2_coop_load<Env>(pz,qz,nw);
    stage2_coop_load<Env>(da,a24,nw);stage2_coop_load<Env>(one,mone,nw);
    const Stage2PaddedCoopField<Env> f{e,bn,(uint32_t)ninv,mersenne_bits,64*nw};
    stage2_coop_ladder_pair(f,scalars[i],px,pz,da,one,ax,az,bx,bz);
    if(normal_output) {cgbn_set_ui32(e,one,1);f.mul(ax,ax,one);f.mul(az,az,one);}
    stage2_coop_store<Env>(out_x+(size_t)i*nw,ax,nw);
    stage2_coop_store<Env>(out_z+(size_t)i*nw,az,nw);
}
