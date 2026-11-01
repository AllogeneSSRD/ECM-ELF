#pragma once
#include "point_coop_field.cuh"
// Opt-in adapter for a fixed CGBN capacity and arbitrary actual 64-bit
// limb count. Keep the scalar engine's R=2^(64*nw), rather than silently changing
// the coordinate representation when the cooperative container is padded.
template<class Env>
struct Stage2PaddedCoopField {
    using Bn=typename Env::cgbn_t;
    const Env &e;
    const Bn &n;
    uint32_t np0;
    int mersenne_bits,radix_bits;

    // Share a single out-of-line product body: compiling one copy per xADD
    // call site is prohibitively expensive for a first wide-point experiment.
    __device__ __noinline__ void mul(Bn &r,const Bn &a,const Bn &b) const {
        const Stage2CoopField<Env> full{e,n,np0,mersenne_bits};
        const int delta=Env::BITS-radix_bits;
        if(!delta){full.mul(r,a,b);return;}
        typename Env::cgbn_wide_t t;
        cgbn_mul_wide(e,t,a,b);
        Bn low,high,v;
        if(mersenne_bits) {
            full.reduce_mersenne(v,t); // ab / R_container, already canonical
            const int left=delta%mersenne_bits;
            if(left) {
                cgbn_shift_left(e,low,v,left);cgbn_bitwise_and(e,low,low,n);
                cgbn_shift_right(e,high,v,mersenne_bits-left);
                cgbn_bitwise_ior(e,v,low,high);
            }
            cgbn_set(e,r,v);return;
        }
        // REDC_container(ab*2^delta)=ab/R_actual. The shifted numerator is
        // < n*R_container, so the wide REDC input bound still holds.
        cgbn_shift_right(e,v,t._low,Env::BITS-delta);
        cgbn_shift_left(e,high,t._high,delta);cgbn_bitwise_ior(e,t._high,high,v);
        cgbn_shift_left(e,t._low,t._low,delta);
        cgbn_mont_reduce_wide(e,r,t,n,np0);full.canonical(r);
    }
    __device__ __forceinline__ void add(Bn &r,const Bn &a,const Bn &b) const {
        const int carry=cgbn_add(e,r,a,b);
        if(carry || cgbn_compare(e,r,n)>=0)cgbn_sub(e,r,r,n);
    }
    __device__ __forceinline__ void sub(Bn &r,const Bn &a,const Bn &b) const {
        const int borrow=cgbn_sub(e,r,a,b);if(borrow)cgbn_add(e,r,r,n);
    }
    __device__ __forceinline__ void half(Bn &r,const Bn &a) const {
        int carry=0;
        if(cgbn_get_ui32(e,a)&1)carry=cgbn_add(e,r,a,n);else cgbn_set(e,r,a);
        cgbn_shift_right(e,r,r,1);
        cgbn_insert_bits_ui32(e,r,r,Env::BITS-1,1,carry);
    }
    __device__ __forceinline__ void xadd(Bn &rx,Bn &rz,const Bn &px,const Bn &pz,
        const Bn &qx,const Bn &qz,const Bn &dx,const Bn &dz) const {
        Bn a,b,tx,tz;
        add(a,px,pz);sub(b,px,pz);add(tx,qx,qz);sub(tz,qx,qz);
        mul(a,a,tz);mul(b,b,tx);add(tx,a,b);sub(tz,a,b);
        half(tx,tx);half(tz,tz);mul(tx,tx,tx);mul(tz,tz,tz);
        mul(a,dz,tx);mul(b,dx,tz);
        cgbn_set(e,rx,a);cgbn_set(e,rz,b); // outputs may alias the difference
    }
};

template<class Env>
__device__ __forceinline__ void stage2_coop_load(typename Env::cgbn_t &r,const unsigned long long *p,int nw) {
    const auto *words=reinterpret_cast<const uint32_t*>(p);
    const int first=(threadIdx.x%Env::TPI)*Env::LIMBS;
    #pragma unroll
    for(int i=0;i<Env::LIMBS;++i)r._limbs[i]=first+i<2*nw?words[first+i]:0;
}
template<class Env>
__device__ __forceinline__ void stage2_coop_store(unsigned long long *p,const typename Env::cgbn_t &a,int nw) {
    auto *words=reinterpret_cast<uint32_t*>(p);
    const int first=(threadIdx.x%Env::TPI)*Env::LIMBS;
    #pragma unroll
    for(int i=0;i<Env::LIMBS;++i)if(first+i<2*nw)words[first+i]=a._limbs[i];
}

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
