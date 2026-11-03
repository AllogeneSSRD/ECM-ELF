#pragma once
#include "../../../../src/ecm/cuda/arithmetic/cgbn/square.cuh"

// Two-word WMAD schedule derived from CGBN core_mont_wmad.cu.
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
// Row i contributes a_i * (a_i*beta^i + 2*sum_{j>i}a_j*beta^j).
// The first doubled digit after the diagonal must exclude the diagonal's
// outgoing bit. Masking an already-doubled whole integer alone is incorrect.
__device__ __forceinline__ uint32_t triangular_digit(uint32_t a,uint32_t twice,
    unsigned position,unsigned row) {
    return position<row ? 0 : position==row ? a : position==row+1 ? twice&~1u : twice;
}

template<class Env>
__device__ __forceinline__ void mont_square_triangular(const Env &env,
    typename Env::cgbn_t &r,const typename Env::cgbn_t &a,
    const typename Env::cgbn_t &n,uint32_t np0) {
#if !defined(__CUDA_ARCH__) || !defined(XMP_WMAD)
    cgbn_mont_sqr(env,r,a,n,np0);
#else
    if constexpr(!square_split_eligible<Env>())cgbn_mont_sqr(env,r,a,n,np0);
    else {
        constexpr unsigned L=Env::LIMBS,T=Env::TPI,NW=T*L;
        using Core=cgbn::core_t<Env>;
        const unsigned lane=threadIdx.x&(T-1),base=(threadIdx.x&31)&~(T-1);
        const unsigned mask=(0xffffffffu>>(32-T))<<base;
        typename Env::cgbn_t twice;cgbn_shift_left(env,twice,a,1);
        const uint32_t top=__shfl_sync(mask,a._limbs[L-1],T-1,T)>>31;
        uint32_t ra[L+2]={},ru[L+1]={},c=0;
        #pragma nounroll
        for(unsigned thread=0;thread<T;thread+=2) {
            #pragma unroll
            for(unsigned word=0;word<2*L;word+=2) {
                const unsigned row=thread*L+word;
                const uint32_t a0=__shfl_sync(mask,a._limbs[word%L],thread+word/L,T);
                const uint32_t a1=__shfl_sync(mask,a._limbs[(word+1)%L],thread+(word+1)/L,T);
                cgbn::chain_t<> p0odd;
                ra[0]=p0odd.add(ra[0],c);
                #pragma unroll
                for(unsigned j=0;j<L-1;j+=2) {
                    const auto v=triangular_digit(a._limbs[j+1],twice._limbs[j+1],lane*L+j+1,row);
                    ru[j]=p0odd.madlo(v,a0,ru[j]);ru[j+1]=p0odd.madhi(v,a0,ru[j+1]);
                }
                cgbn::chain_t<> p0even;
                #pragma unroll
                for(unsigned j=0;j<L;j+=2) {
                    const auto v=triangular_digit(a._limbs[j],twice._limbs[j],lane*L+j,row);
                    ra[j]=p0even.madlo(v,a0,ra[j]);ra[j+1]=p0even.madhi(v,a0,ra[j+1]);
                }
                ra[L]=p0even.add(ra[L],0);
                const uint32_t q0=__shfl_sync(mask,ra[0],0,T)*np0;
                cgbn::chain_t<> n0even;
                #pragma unroll
                for(unsigned j=0;j<L;j+=2) {
                    ra[j]=n0even.madlo(n._limbs[j],q0,ra[j]);ra[j+1]=n0even.madhi(n._limbs[j],q0,ra[j+1]);
                }
                ra[L]=n0even.add(ra[L],0);
                cgbn::chain_t<> n0odd;
                #pragma unroll
                for(unsigned j=0;j<L-1;j+=2) {
                    ru[j]=n0odd.madlo(n._limbs[j+1],q0,ru[j]);ru[j+1]=n0odd.madhi(n._limbs[j+1],q0,ru[j+1]);
                }
                ru[L]=n0odd.add(ru[L],0);
                uint32_t t0=ra[0];
                cgbn::chain_t<> p1even;
                #pragma unroll
                for(unsigned j=0;j<L;j+=2) {
                    const auto v=triangular_digit(a._limbs[j],twice._limbs[j],lane*L+j,row+1);
                    ru[j]=p1even.madlo(v,a1,ru[j]);ru[j+1]=p1even.madhi(v,a1,ru[j+1]);
                }
                ru[L]=p1even.add(ru[L],0);
                cgbn::chain_t<> p1odd;
                #pragma unroll
                for(unsigned j=0;j<L-1;j+=2) {
                    const auto v=triangular_digit(a._limbs[j+1],twice._limbs[j+1],lane*L+j+1,row+1);
                    ra[j+2]=p1odd.madlo(v,a1,ra[j+2]);ra[j+3]=p1odd.madhi(v,a1,ra[j+3]);
                }
                cgbn::chain_t<> n1odd;
                ru[0]=n1odd.add(ru[0],ra[1]);
                const uint32_t q1=__shfl_sync(mask,ru[0],0,T)*np0;
                #pragma unroll
                for(unsigned j=0;j<L;j+=2) {
                    ra[j]=n1odd.madlo(n._limbs[j+1],q1,ra[j+2]);ra[j+1]=n1odd.madhi(n._limbs[j+1],q1,ra[j+3]);
                }
                ra[L]=n1odd.add(0,0);
                cgbn::chain_t<> n1even;
                uint32_t t1=n1even.madlo(n._limbs[0],q1,ru[0]);
                c=n1even.madhi(n._limbs[0],q1,ru[1]);
                #pragma unroll
                for(unsigned j=0;j<L-2;j+=2) {
                    ru[j]=n1even.madlo(n._limbs[j+2],q1,ru[j+2]);ru[j+1]=n1even.madhi(n._limbs[j+2],q1,ru[j+3]);
                }
                ru[L-2]=n1even.add(ru[L],0);ru[L-1]=0;ru[L]=0;
                t0=__shfl_sync(mask,t0,threadIdx.x+1,T);
                t1=__shfl_sync(mask,t1,threadIdx.x+1,T);
                // Doubling may have a digit at beta^NW. It cannot influence
                // either low quotient word. After two REDC steps, inject at
                // NW-2/NW-1; do not truncate it or expand every live array.
                if(lane==T-1) {t0=top*a0;t1=row+1==NW-1 ? 0 : top*a1;}
                cgbn::chain_t<3> feed;
                ra[L-2]=feed.add(ra[L-2],t0);ra[L-1]=feed.add(ra[L-1],t1);ra[L]=feed.add(ra[L],0);ra[L+1]=0;
            }
        }
        cgbn::chain_t<> finish;
        ra[0]=finish.add(ra[0],c);
        #pragma unroll
        for(unsigned j=1;j<L;++j)ra[j]=finish.add(ra[j],ru[j-1]);
        uint32_t extra=finish.add(ra[L],ru[L-1]);
        uint32_t incoming=__shfl_up_sync(mask,extra,1,T);
        if(lane!=T-1)extra=0;if(!lane)incoming=0;
        cgbn::chain_t<> collect;
        ra[0]=collect.add(ra[0],incoming);
        #pragma unroll
        for(unsigned j=1;j<L;++j)ra[j]=collect.add(ra[j],0);
        c=collect.add(extra,0);c=Core::fast_propagate_add(c,ra);
        if(c) {
            const uint32_t first=n._limbs[0]-(lane==0);cgbn::chain_t<> subtract;
            ra[0]=subtract.add(ra[0],~first);
            #pragma unroll
            for(unsigned j=1;j<L;++j)ra[j]=subtract.add(ra[j],~n._limbs[j]);
            c=subtract.add(0,0);Core::fast_propagate_add(c,ra);
        }
        #pragma unroll
        for(unsigned j=0;j<L;++j)r._limbs[j]=ra[j];
    }
#endif
    if(cgbn_compare(env,r,n)>=0)cgbn_sub(env,r,r,n);
}
}
