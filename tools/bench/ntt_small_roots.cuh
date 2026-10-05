#pragma once

// Experimental stage2/3 roots. The includer supplies the production Goldilocks
// operations and tile helpers. Method 1 accepts any table; method 2 requires
// the standard primitive root generated from 7, checked by the host probe.
template<int S>
__device__ __forceinline__ unsigned long long small_root_shift(unsigned long long v)
{
    return gl_reduce(v<<S,v>>(64-S));
}
__device__ __forceinline__ unsigned long long small_root_value(unsigned long long v,
    unsigned long long w,int st)
{
    if(st!=2 && st!=3)return tile_warp_twiddle(v,w,st);
    if(w==1)return v;
    if(w==(1ull<<24))return small_root_shift<24>(v);
    if(w==GL_P-(1ull<<24))return gl_sub_dev(0,small_root_shift<24>(v));
    if(w==(1ull<<48))return small_root_shift<48>(v);
    if(w==GL_P-(1ull<<48))return gl_sub_dev(0,small_root_shift<48>(v));
    constexpr unsigned long long r72=(1ull<<40)-(1ull<<8);
    if(w==r72)return gl_sub_dev(small_root_shift<40>(v),small_root_shift<8>(v));
    if(w==GL_P-r72)return gl_sub_dev(small_root_shift<8>(v),small_root_shift<40>(v));
    if(st==3) {
        if(w==(1ull<<12))return small_root_shift<12>(v);
        if(w==GL_P-(1ull<<12))return gl_sub_dev(0,small_root_shift<12>(v));
        if(w==(1ull<<36))return small_root_shift<36>(v);
        if(w==GL_P-(1ull<<36))return gl_sub_dev(0,small_root_shift<36>(v));
        if(w==(1ull<<60))return small_root_shift<60>(v);
        if(w==GL_P-(1ull<<60))return gl_sub_dev(0,small_root_shift<60>(v));
        constexpr unsigned long long r84=(1ull<<52)-(1ull<<20);
        if(w==r84)return gl_sub_dev(small_root_shift<52>(v),small_root_shift<20>(v));
        if(w==GL_P-r84)return gl_sub_dev(small_root_shift<20>(v),small_root_shift<52>(v));
    }
    return gl_mul(v,w);
}

// Exact reduction of any128-bit input, using 2^64=epsilon and 2^96=-1.
// Carry/borrow corrections subtract/add epsilon rather than running four folds.
__device__ __forceinline__ unsigned long long small_root_reduce(unsigned long long lo,
    unsigned long long hi)
{
    return gl_reduce128_short(lo,hi);
}
template<bool INV,bool FAST=false>
__device__ __forceinline__ unsigned long long small_root_index(unsigned long long v,
    unsigned int j,int st)
{
    // gamma=2^12 has order16; standard r16=gamma^13, inverse=gamma^3.
    const unsigned int e=((j*(INV ? 3u : 13u))<<(3-st))&15u;
    const unsigned int bits=12*(e&7u),s=bits&63u;
    const auto a=v<<s;
    const auto b=(v>>((64-s)&63u))&(0ull-(unsigned long long)(s!=0));
    const bool wide=bits>=64;
    // v*2^bits may need148 bits. Keep the high20, using 2^128=-2^32.
    const auto reduced=FAST ? small_root_reduce(wide ? 0ull : a,wide ? a : b) : gl_reduce(wide ? 0ull : a,wide ? a : b);
    const auto value=gl_sub_dev(reduced,wide ? b<<32 : 0ull);
    return (e&8u) ? gl_sub_dev(0,value) : value;
}
template<int METHOD,bool INV>
__device__ __forceinline__ unsigned long long small_root_twiddle(unsigned long long v,
    const unsigned long long *table,int st,unsigned int j)
{
    if constexpr(METHOD>=2)if(st==2 || st==3)return small_root_index<INV,METHOD==3>(v,j,st);
    const auto w=__ldg(table+(1u<<st)-1+j);
    if constexpr(METHOD==1)return small_root_value(v,w,st);
    return tile_warp_twiddle(v,w,st);
}
template<int METHOD,bool INV>
__device__ __forceinline__ void small_root_pair(unsigned long long &lo,unsigned long long &hi,
    const unsigned long long *table)
{
    const unsigned int lane=threadIdx.x&31;
    #pragma unroll
    for(int q=0;q<6;++q) {
        const int st=INV ? q : 5-q;const unsigned int half=1u<<st,j=lane&(half-1);
        const auto u=lo,v=INV ? small_root_twiddle<METHOD,INV>(hi,table,st,j) : hi;
        lo=gl_add_dev(u,v);hi=INV ? gl_sub_dev(u,v) : small_root_twiddle<METHOD,INV>(gl_sub_dev(u,v),table,st,j);
        if(INV ? st<5 : st>0) {
            const unsigned int mask=1u<<(INV ? st : st-1);const bool lower=(lane&mask)==0;
            const auto peer=__shfl_xor_sync(0xffffffffu,lower ? hi : lo,mask);
            lo=lower ? lo : peer;hi=lower ? peer : hi;
        }
    }
}
template<int METHOD,bool INV>
__device__ __forceinline__ unsigned long long small_root_warp5(unsigned long long v,
    const unsigned long long *table)
{
    const unsigned int lane=threadIdx.x&31;
    #pragma unroll
    for(int q=0;q<5;++q) {
        const int st=INV ? q : 4-q;const unsigned int half=1u<<st,j=lane&(half-1);const bool upper=(lane&half)!=0;
        if(INV && upper)v=small_root_twiddle<METHOD,INV>(v,table,st,j);
        const auto peer=__shfl_xor_sync(0xffffffffu,v,half);
        v=INV ? (upper ? gl_sub_dev(peer,v) : gl_add_dev(v,peer)) :
            (upper ? small_root_twiddle<METHOD,INV>(gl_sub_dev(peer,v),table,st,j) : gl_add_dev(v,peer));
    }
    return v;
}
template<int METHOD,bool INV>
__global__ __launch_bounds__(FUSE_TILE_THREADS,3)
void small_root_tile(unsigned long long *a,const unsigned long long *b,
    unsigned long long n,int t,const unsigned long long *table,unsigned long long scale,unsigned long long stride)
{
    const auto tile=1ull<<t,base=(unsigned long long)blockIdx.x*tile;
    a+=blockIdx.y*stride;if(b)b+=blockIdx.y*stride;
    extern __shared__ unsigned long long sm[];
    for(unsigned int i=threadIdx.x;i<tile;i+=blockDim.x)sm[i]=a[base+i];
    __syncthreads();
    if(INV) {
        for(unsigned int i=threadIdx.x;i<tile;i+=blockDim.x)sm[i]=gl_mul(gl_mul(sm[i],b[base+i]),scale);
        __syncthreads();
    }
    const int wt=t>=6 ? 6 : 5;
    if(!INV) {
        int st=t-1;
        for(;st>=wt+1;st-=2){tile_radix4_block(sm,tile,st,table);__syncthreads();}
        if(st==wt)tile_radix2_stage<false>(sm,tile,st,table);
        if(wt==6)for(unsigned int i=(threadIdx.x>>5)*64+(threadIdx.x&31);i<tile;i+=2*blockDim.x) {
            auto lo=sm[i],hi=sm[i+32];small_root_pair<METHOD,false>(lo,hi,table);
            const unsigned int dst=(i&~63u)+2*(threadIdx.x&31);a[base+dst]=lo;a[base+dst+1]=hi;
        }
        else for(unsigned int i=threadIdx.x;i<tile;i+=blockDim.x)a[base+i]=small_root_warp5<METHOD,false>(sm[i],table);
    } else {
        if(wt==6)for(unsigned int i=(threadIdx.x>>5)*64+(threadIdx.x&31);i<tile;i+=2*blockDim.x) {
            const unsigned int src=(i&~63u)+2*(threadIdx.x&31);auto lo=sm[src],hi=sm[src+1];
            small_root_pair<METHOD,true>(lo,hi,table);__syncwarp();sm[i]=lo;sm[i+32]=hi;
        }
        else for(unsigned int i=threadIdx.x;i<tile;i+=blockDim.x)sm[i]=small_root_warp5<METHOD,true>(sm[i],table);
        __syncthreads();int st=wt;
        for(;st+1<t;st+=2){tile_radix4_block_inv(sm,tile,st,table);__syncthreads();}
        if(st<t)tile_radix2_stage<true>(sm,tile,st,table);
        for(unsigned int i=threadIdx.x;i<tile;i+=blockDim.x)a[base+i]=sm[i];
    }
}
