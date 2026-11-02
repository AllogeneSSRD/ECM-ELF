#pragma once
#include "square.cuh"

// The two-digit WMAD reduction schedule below is derived from CGBN's
// core_mont_wmad.cu. Its license applies to that schedule:
// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

namespace ecm_cgbn {

// REDC consumes D + C, where D=a0^2+B^2*a1^2, C=2*B*a0*a1 and
// B=2^(BITS/2). Feed low(C)/2^32 into the unaligned accumulator and stream
// high(C) separately. No full-width addition of D+C or its carry propagation
// is needed. The original Montgomery radix and final canonicalisation remain.
template<unsigned Depth,class Env,bool StreamHigh=false,bool Canonical=true,unsigned Layout=0>
__device__ __forceinline__ void mont_square_carrysave(const Env &env,
    typename Env::cgbn_t &r,const typename Env::cgbn_t &a,
    const typename Env::cgbn_t &n,uint32_t np0) {
#if !defined(__CUDA_ARCH__) || !defined(XMP_WMAD)
    // CGBN's host parser and pre-Volta backends do not expose WMAD internals.
    cgbn_mont_sqr(env,r,a,n,np0);
#else
    if constexpr(!Depth || !square_split_eligible<Env>()) {
        cgbn_mont_sqr(env,r,a,n,np0);
    } else {
        constexpr unsigned T=Env::TPI,L=Env::LIMBS,H=T/2,J=L/2,NW=T*L;
        using S=SquareSplitTypes<Env>;
        using XE=typename S::CrossEnv;using XC=typename S::CrossContext;
        using Core=cgbn::core_t<Env>;
        const unsigned lane=threadIdx.x&(T-1),base=(threadIdx.x&31)&~(T-1);
        constexpr unsigned bits=0xffffffffu>>(32-T);
        const unsigned mask=bits<<base;
        typename Env::cgbn_wide_t diagonal;typename S::CrossWide cross;
        square_split_parts<Depth,Env,Layout>(env,diagonal,cross,a);
        XC xc(cgbn_no_checks);XE xe(xc);
        const unsigned low_bit=__shfl_sync(mask,cross._low._limbs[J-1],T-1,T)>>31;
        const unsigned high_bit=__shfl_sync(mask,cross._high._limbs[J-1],T-1,T)>>31;
        cgbn_shift_left(xe,cross._low,cross._low,1);
        cgbn_shift_left(xe,cross._high,cross._high,1);
        if(!lane)cross._high._limbs[0]|=low_bit;

        uint32_t ra[L+2],ru[L+1];
        #pragma unroll
        for(unsigned j=0;j<L;++j) {
            ra[j]=diagonal._low._limbs[j];
            const unsigned value=__shfl_sync(mask,cross._low._limbs[(j+1)%J],
                2*(lane%H)+(j+1)/J,T);
            ru[j]=(lane>=H && !(lane==T-1 && j==L-1))?value:0;
        }
        const unsigned first=__shfl_sync(mask,cross._low._limbs[0],0,T);
        if(lane==H-1)ru[L-1]=first;
        // WMAD assumes the last unaligned digit is empty at each two-digit
        // boundary. Route its initial contribution to the next lane's c.
        uint32_t c=__shfl_up_sync(mask,ru[L-1],1,T);
        if(!lane)c=0;
        ru[L-1]=0;ru[L]=0;ra[L]=0;ra[L+1]=0;
        if constexpr(!StreamHigh) {
            // high(D)+high(C) cannot overflow R: their sum is <= floor(a^2/R).
            // Resolve this half once rather than retaining a second stream
            // across every dependent REDC digit. The low half stays carry-save.
            typename Env::cgbn_t high_c;
            #pragma unroll
            for(unsigned j=0;j<L;++j) {
                const auto v=__shfl_sync(mask,cross._high._limbs[j%J],2*(lane%H)+j/J,T);
                high_c._limbs[j]=lane<H?v:(lane==H && j==0?high_bit:0);
            }
            cgbn_add(env,diagonal._high,diagonal._high,high_c);
        }
        #pragma nounroll
        for(unsigned thread=0;thread<T;thread+=2) {
            #pragma unroll
            for(unsigned word=0;word<2*L;word+=2) {
                cgbn::chain_t<> unaligned0;
                ra[0]=unaligned0.add(ra[0],c);
                const uint32_t q0=__shfl_sync(mask,ra[0],0,T)*np0;
                #pragma unroll
                for(unsigned j=0;j<L-1;j+=2) {
                    ru[j]=unaligned0.madlo(n._limbs[j+1],q0,ru[j]);
                    ru[j+1]=unaligned0.madhi(n._limbs[j+1],q0,ru[j+1]);
                }
                cgbn::chain_t<> aligned0;
                #pragma unroll
                for(unsigned j=0;j<L;j+=2) {
                    ra[j]=aligned0.madlo(n._limbs[j],q0,ra[j]);
                    ra[j+1]=aligned0.madhi(n._limbs[j],q0,ra[j+1]);
                }
                ra[L]=aligned0.add(ra[L],0);
                uint32_t t0=__shfl_sync(mask,ra[0],threadIdx.x+1,T);

                cgbn::chain_t<> aligned1;
                ru[0]=aligned1.add(ru[0],ra[1]);
                const uint32_t q1=__shfl_sync(mask,ru[0],0,T)*np0;
                #pragma unroll
                for(unsigned j=0;j<L;j+=2) {
                    ra[j]=aligned1.madlo(n._limbs[j+1],q1,ra[j+2]);
                    ra[j+1]=aligned1.madhi(n._limbs[j+1],q1,ra[j+3]);
                }
                ra[L]=aligned1.add(0,0);
                cgbn::chain_t<> unaligned1;
                uint32_t t1=unaligned1.madlo(n._limbs[0],q1,ru[0]);
                c=unaligned1.madhi(n._limbs[0],q1,ru[1]);
                #pragma unroll
                for(unsigned j=0;j<L-2;j+=2) {
                    ru[j]=unaligned1.madlo(n._limbs[j+2],q1,ru[j+2]);
                    ru[j+1]=unaligned1.madhi(n._limbs[j+2],q1,ru[j+3]);
                }
                ru[L-2]=unaligned1.add(ru[L],0);ru[L-1]=0;ru[L]=0;
                t1=__shfl_sync(mask,t1,threadIdx.x+1,T);
                const uint32_t d0=__shfl_sync(mask,diagonal._high._limbs[word%L],thread+word/L,T);
                const uint32_t d1=__shfl_sync(mask,diagonal._high._limbs[(word+1)%L],thread+(word+1)/L,T);
                if(lane==T-1){t0=d0;t1=d1;}
                cgbn::chain_t<3> input_d;
                ra[L-2]=input_d.add(ra[L-2],t0);
                ra[L-1]=input_d.add(ra[L-1],t1);
                ra[L]=input_d.add(0,0);
                if constexpr(StreamHigh) {
                    const unsigned k0=thread*L+word,k1=k0+1;
                    uint32_t s0=__shfl_sync(mask,cross._high._limbs[word%J],k0/J,T);
                    uint32_t s1=__shfl_sync(mask,cross._high._limbs[(word+1)%J],k1/J,T);
                    s0=lane==T-1?(k0<NW/2?s0:k0==NW/2?high_bit:0):0;
                    s1=lane==T-1?(k1<NW/2?s1:k1==NW/2?high_bit:0):0;
                    cgbn::chain_t<3> input_c;
                    ra[L-2]=input_c.add(ra[L-2],s0);
                    ra[L-1]=input_c.add(ra[L-1],s1);
                    ra[L]=input_c.add(ra[L],0);
                }
                ra[L+1]=0;
            }
        }
        cgbn::chain_t<> finish;
        ra[0]=finish.add(ra[0],c);
        #pragma unroll
        for(unsigned j=1;j<L;++j)ra[j]=finish.add(ra[j],ru[j-1]);
        c=finish.add(ra[L],0);
        c=Core::fast_propagate_add(c,ra);
        if(c) {
            const uint32_t first_n=n._limbs[0]-(lane==0);
            cgbn::chain_t<L+1> subtract;
            ra[0]=subtract.add(ra[0],~first_n);
            #pragma unroll
            for(unsigned j=1;j<L;++j)ra[j]=subtract.add(ra[j],~n._limbs[j]);
            c=subtract.add(0,0);Core::fast_propagate_add(c,ra);
        }
        #pragma unroll
        for(unsigned j=0;j<L;++j)r._limbs[j]=ra[j];
    }
#endif
    if constexpr(Canonical)if(cgbn_compare(env,r,n)>=0)cgbn_sub(env,r,r,n);
}

}
