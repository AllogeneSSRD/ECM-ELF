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
