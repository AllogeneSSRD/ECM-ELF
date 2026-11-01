#pragma once
#include "point_coop_field.cuh"

template<int WORDS,int TPI>
__global__ void stage2_coop_chain(const unsigned long long *n,unsigned long long ninv,int nw,
    const unsigned long long *dx,const unsigned long long *dz,const unsigned long long *sx,
    const unsigned long long *sz,const unsigned long long *ex,const unsigned long long *ez,
    unsigned long long chains,unsigned long long points,unsigned long long length,size_t seed_stride,
    int mersenne_bits,unsigned long long *out_x,unsigned long long *out_z)
{
    const unsigned long long chain=(blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x)/TPI;
    const unsigned long long start=chain*length;if(chain>=chains || start>=points)return;
    const unsigned long long end=start+length<points?start+length:points;
    using Ctx=cgbn_context_t<TPI>;using Env=cgbn_env_t<Ctx,64*WORDS>;using Bn=typename Env::cgbn_t;
    Ctx ctx(cgbn_no_checks);Env e(ctx);Bn bn,xd,zd,xa,za,xb,zb;
    stage2_coop_load<Env>(bn,n,nw);stage2_coop_load<Env>(xd,dx,nw);stage2_coop_load<Env>(zd,dz,nw);
    stage2_coop_load<Env>(xa,sx+(size_t)chain*seed_stride*nw,nw);stage2_coop_load<Env>(za,sz+(size_t)chain*seed_stride*nw,nw);
    stage2_coop_load<Env>(xb,ex+(size_t)chain*seed_stride*nw,nw);stage2_coop_load<Env>(zb,ez+(size_t)chain*seed_stride*nw,nw);
    const Stage2PaddedCoopField<Env> f{e,bn,(uint32_t)ninv,mersenne_bits,64*nw};
    stage2_coop_store<Env>(out_x+start*nw,xa,nw);stage2_coop_store<Env>(out_z+start*nw,za,nw);
    if(end>start+1) {
        stage2_coop_store<Env>(out_x+(start+1)*nw,xb,nw);stage2_coop_store<Env>(out_z+(start+1)*nw,zb,nw);
    }
    for(unsigned long long k=start+2;k<end;++k) {
        f.xadd(xa,za,xb,zb,xd,zd,xa,za);
        cgbn_swap(e,xa,xb);cgbn_swap(e,za,zb);
        stage2_coop_store<Env>(out_x+k*nw,xb,nw);
        stage2_coop_store<Env>(out_z+k*nw,zb,nw);
    }
}

template<int WORDS,int TPI>
__global__ void stage2_coop_mont_test(const unsigned long long *a,const unsigned long long *b,
    const unsigned long long *n,unsigned long long ninv,int nw,int cases,int mersenne_bits,
    bool square,unsigned long long *out)
{
    const int i=(blockIdx.x*blockDim.x+threadIdx.x)/TPI;if(i>=cases)return;
    using Ctx=cgbn_context_t<TPI>;using Env=cgbn_env_t<Ctx,64*WORDS>;using Bn=typename Env::cgbn_t;
    Ctx ctx(cgbn_no_checks);Env e(ctx);Bn x,y,bn;
    stage2_coop_load<Env>(x,a+(size_t)i*nw,nw);stage2_coop_load<Env>(bn,n,nw);
    if(square)cgbn_set(e,y,x);else stage2_coop_load<Env>(y,b+(size_t)i*nw,nw);
    const Stage2PaddedCoopField<Env> f{e,bn,(uint32_t)ninv,mersenne_bits,64*nw};
    f.mul(x,x,y);stage2_coop_store<Env>(out+(size_t)i*nw,x,nw);
}
