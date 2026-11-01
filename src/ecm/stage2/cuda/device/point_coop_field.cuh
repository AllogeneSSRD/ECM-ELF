#pragma once
// Cooperative field adapter. CGBN owns distributed 32-bit limbs; BITS is the
// same 64*nw Montgomery radix as the scalar production reference. This header
// uses the same canonical-residue contract as the scalar point backend.
template<class Env>
struct Stage2CoopField {
    using Bn=typename Env::cgbn_t;
    const Env &e;
    const Bn &n;
    uint32_t np0;
    int mersenne_bits;

    __device__ __forceinline__ void canonical(Bn &r) const {
        // This CGBN Montgomery core only removes n on radix overflow and can
        // return a representative in [n,2n). Our point contract requires [0,n).
        if(cgbn_compare(e,r,n)>=0)cgbn_sub(e,r,r,n);
    }

    __device__ __forceinline__ void reduce_mersenne(Bn &r,const typename Env::cgbn_wide_t &t) const {
        Bn low,high,v,tail;
        const int d=Env::BITS-mersenne_bits;
        // ab < n^2: low_p(ab)+high_p(ab) < 2n. Do not discard the
        // carry at the container boundary when p==BITS.
        cgbn_bitwise_and(e,low,t._low,n);
        cgbn_shift_right(e,high,t._low,mersenne_bits);
        cgbn_shift_left(e,v,t._high,d);
        cgbn_bitwise_ior(e,high,high,v);
        const int carry=cgbn_add(e,v,low,high);
        if(carry || cgbn_compare(e,v,n)>=0)cgbn_sub(e,v,v,n);
        // Montgomery R^-1 is rotation by (BITS-p) mod p in the p-bit ring.
        const int rotate=d%mersenne_bits;
        if(rotate) {
            cgbn_shift_right(e,tail,v,rotate);
            cgbn_shift_left(e,v,v,mersenne_bits-rotate);
            cgbn_bitwise_and(e,v,v,n);
            cgbn_bitwise_ior(e,v,v,tail);
        }
        cgbn_set(e,r,v);
    }
    __device__ __forceinline__ void mul(Bn &r,const Bn &a,const Bn &b) const {
        if(!mersenne_bits){cgbn_mont_mul(e,r,a,b,n,np0);canonical(r);return;}
        typename Env::cgbn_wide_t t;
        cgbn_mul_wide(e,t,a,b);reduce_mersenne(r,t);
    }
    __device__ __forceinline__ void square(Bn &r,const Bn &a) const {
        if(!mersenne_bits){cgbn_mont_sqr(e,r,a,n,np0);canonical(r);return;}
        typename Env::cgbn_wide_t t;
        cgbn_sqr_wide(e,t,a);reduce_mersenne(r,t);
    }
};

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
    __device__ __forceinline__ void xdbl(Bn &rx,Bn &rz,const Bn &px,const Bn &pz,const Bn &a24) const {
        Bn a,b,aa,bb;
        add(a,px,pz);sub(b,px,pz);mul(aa,a,a);mul(bb,b,b);
        sub(a,aa,bb);mul(b,a24,a);add(b,b,bb);
        mul(aa,aa,bb);mul(bb,a,b);
        // Both inputs have been consumed before either aliased output is stored.
        cgbn_set(e,rx,aa);cgbn_set(e,rz,bb);
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
