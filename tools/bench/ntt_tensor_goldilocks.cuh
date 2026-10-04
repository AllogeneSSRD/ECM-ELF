#pragma once

// Standalone exact Goldilocks experiment. The includer provides gl_reduce/gl_sub.
// Fragment layout follows the NVIDIA PTX m16n8k16 integer instruction contract.
// No reference-library source is imported; no production planner calls this code.
__device__ __forceinline__ unsigned int tc_pack_bytes(const unsigned long long x[4],int byte)
{
    unsigned int p=0;
    #pragma unroll
    for(int i=0;i<4;++i)p|=(unsigned int)((x[i]>>(8*byte))&255ull)<<(8*i);
    return p;
}

__device__ __forceinline__ void tc_mma_u8(unsigned int s[4],unsigned int a0,
                                        unsigned int a1,unsigned int b)
{
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+r"(s[0]),"+r"(s[1]),"+r"(s[2]),"+r"(s[3])
        : "r"(a0),"r"(a1),"r"(b));
#else
    asm volatile("trap;");
#endif
}

template<int D>
__device__ __forceinline__ void tc_sum_diagonals(const unsigned int a0[8],
    const unsigned int a1[8],const unsigned int b[8],unsigned int carry[4],
    unsigned long long lo[4],unsigned long long hi[4])
{
    unsigned int sum[4]={};
    #pragma unroll
    for(int i=0;i<8;++i)if(D-i>=0 && D-i<8)tc_mma_u8(sum,a0[i],a1[i],b[D-i]);
    // A diagonal has at most 16*8*255^2=8323200; carry cannot overflow s32.
    #pragma unroll
    for(int j=0;j<4;++j) {
        const unsigned int value=sum[j]+carry[j];carry[j]=value>>8;
        if constexpr(D<8)lo[j]|=(unsigned long long)(value&255u)<<(8*D);
        else hi[j]|=(unsigned long long)(value&255u)<<(8*(D-8));
    }
    if constexpr(D<14)tc_sum_diagonals<D+1>(a0,a1,b,carry,lo,hi);
}

__device__ __forceinline__ void tc_dot16_exact(const unsigned long long ar0[4],
    const unsigned long long ar1[4],const unsigned long long br[4],
    unsigned long long output[4],unsigned int top[4])
{
    unsigned int a0[8],a1[8],b[8],carry[4]={};
    unsigned long long lo[4]={},hi[4]={};
    #pragma unroll
    for(int i=0;i<8;++i) {
        a0[i]=tc_pack_bytes(ar0,i);a1[i]=tc_pack_bytes(ar1,i);b[i]=tc_pack_bytes(br,i);
    }
    tc_sum_diagonals<0>(a0,a1,b,carry,lo,hi);
    #pragma unroll
    for(int j=0;j<4;++j) {
        hi[j]|=(unsigned long long)(carry[j]&255u)<<56;
        top[j]=carry[j]>>8; // the complete sum has at most 132 bits, not 128
        // q=2^64-2^32+1: 2^128 == -2^32 (mod q).
        output[j]=gl_sub_dev(gl_reduce(lo[j],hi[j]),(unsigned long long)top[j]<<32);
    }
}

// Prepacked common roots: [byte][A-half][lane]. All lanes load consecutive
// u32 words; only B byte extraction remains in the runtime tile path.
__device__ __forceinline__ void tc_dot16_packed(const unsigned int *roots,
    const unsigned long long br[4],unsigned long long output[4])
{
    const unsigned int lane=threadIdx.x&31;
    unsigned int a0[8],a1[8],b[8],carry[4]={};
    unsigned long long lo[4]={},hi[4]={};
    #pragma unroll
    for(int i=0;i<8;++i) {
        a0[i]=__ldg(roots+i*64+lane);a1[i]=__ldg(roots+i*64+32+lane);b[i]=tc_pack_bytes(br,i);
    }
    tc_sum_diagonals<0>(a0,a1,b,carry,lo,hi);
    #pragma unroll
    for(int j=0;j<4;++j) {
        hi[j]|=(unsigned long long)(carry[j]&255u)<<56;
        output[j]=gl_sub_dev(gl_reduce(lo[j],hi[j]),(unsigned long long)(carry[j]>>8)<<32);
    }
}

__device__ __forceinline__ unsigned int tc_rev4(unsigned int x)
{
    return ((x&1)<<3)|((x&2)<<1)|((x&4)>>1)|((x&8)>>3);
}

// Two independent 64-word transforms/warp: CUDA upper two stages, exact MMA
// lower four stages. Input/output permutations match the existing warp6 tail.
template<bool INV>
__device__ __forceinline__ void tc_warp6_tile(unsigned long long *sm,unsigned long long *global,
    unsigned long long tile,const unsigned long long *table,const unsigned int *roots,
    unsigned int offset)
{
    const unsigned int lane=threadIdx.x&31,group=lane>>2,tid=lane&3;
    unsigned long long br[4],out[4];
    #pragma unroll
    for(int i=0;i<4;++i) {
        const unsigned int row=tid*4+i,index=offset+group*16+(INV ? tc_rev4(row) : row);
        br[i]=index<tile ? sm[index] : 0;
    }
    if(!INV) {
        #pragma unroll
        for(int q=0;q<2;++q) {
            const int st=5-q;const unsigned int upper_bit=1u<<(st-4),shuffle=1u<<(st-2);
            const bool upper=(group&upper_bit)!=0;
            #pragma unroll
            for(int i=0;i<4;++i) {
                const unsigned int j=(group*16+tid*4+i)&((1u<<st)-1);
                const auto peer=__shfl_xor_sync(0xffffffffu,br[i],shuffle);
                br[i]=upper ? gl_mul(gl_sub_dev(peer,br[i]),__ldg(table+(1u<<st)-1+j)) : gl_add_dev(br[i],peer);
            }
        }
    }
    tc_dot16_packed(roots,br,out);
    if(INV) {
        // stage 4 pairs adjacent columns, held in adjacent output registers.
        #pragma unroll
        for(int r=0;r<2;++r) {
            const unsigned int row=group+8*r;const auto u=out[2*r];
            const auto v=gl_mul(out[2*r+1],__ldg(table+15+row));
            out[2*r]=gl_add_dev(u,v);out[2*r+1]=gl_sub_dev(u,v);
        }
        // stage 5 pairs columns separated by two (lane bit zero).
        #pragma unroll
        for(int r=0;r<4;++r) {
            const unsigned int row=group+(r>=2 ? 8 : 0),j=(r&1)*16+row;
            const bool upper=(tid&1)!=0;
            const auto value=upper ? gl_mul(out[r],__ldg(table+31+j)) : out[r];
            const auto peer=__shfl_xor_sync(0xffffffffu,value,1);
            out[r]=upper ? gl_sub_dev(peer,value) : gl_add_dev(value,peer);
        }
    }
    #pragma unroll
    for(int r=0;r<4;++r) {
        const unsigned int row=group+(r>=2 ? 8 : 0),col=tid*2+(r&1);
        const unsigned int index=offset+col*16+(INV ? row : tc_rev4(row));
        if(index<tile)(INV ? sm : global)[index]=out[r];
    }
}

// Actual tile boundary contract: DIF natural -> bit-reversed; inverse DIT
// bit-reversed -> natural, including the same B/1N fusion as the production tile.
// This standalone experiment reuses the production shared radix4 high stages.
template<bool INV>
__global__ void tc_goldilocks_tile(unsigned long long *a,const unsigned long long *b,
    unsigned long long n,int t,const unsigned long long *table,unsigned long long scale,
    unsigned long long stride,const unsigned int *roots)
{
    const unsigned long long tile=1ull<<t,base=(unsigned long long)blockIdx.x*tile;
    a+=blockIdx.y*stride;if(b)b+=blockIdx.y*stride;
    extern __shared__ unsigned long long sm[];
    for(unsigned int i=threadIdx.x;i<tile;i+=blockDim.x)sm[i]=a[base+i];
    __syncthreads();
    if(INV) {
        for(unsigned int i=threadIdx.x;i<tile;i+=blockDim.x)sm[i]=gl_mul(gl_mul(sm[i],b[base+i]),scale);
        __syncthreads();
    } else {
        int st=t-1;
        for(;st>=7;st-=2) {
            tile_radix4_block(sm,tile,st,table);
            __syncthreads(); // the shared radix4 helper leaves the CTA fence to its caller
        }
        if(st==6)tile_radix2_stage<false>(sm,tile,st,table);
    }
    for(unsigned int offset=(threadIdx.x>>5)*128;offset<tile;offset+=(blockDim.x/32)*128)
        tc_warp6_tile<INV>(sm,a+base,tile,table,roots,offset);
    if(INV) {
        __syncthreads();int st=6;
        for(;st+1<t;st+=2) {
            tile_radix4_block_inv(sm,tile,st,table);
            __syncthreads();
        }
        if(st<t)tile_radix2_stage<true>(sm,tile,st,table);
        for(unsigned int i=threadIdx.x;i<tile;i+=blockDim.x)a[base+i]=sm[i];
    }
}

// A is a common row-major 16x16 matrix. Each warp multiplies one row-major
// 16x8 B and writes 16x8 C, in natural order. B==C is allowed; A must be disjoint.
// Threads/warp alignment and uniform whole-warp exit are required by mma.sync.
__global__ void tc_goldilocks_mat16(const unsigned long long *a,const unsigned long long *b,
    unsigned long long *c,unsigned long long batches,unsigned int *tops=nullptr)
{
    const auto batch=((unsigned long long)blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(batch>=batches)return;
    const unsigned int lane=threadIdx.x&31,group=lane>>2,tid=lane&3;
    unsigned long long ar0[4],ar1[4],br[4],out[4];unsigned int top[4];
    #pragma unroll
    for(int i=0;i<4;++i) {
        ar0[i]=a[group*16+tid*4+i];ar1[i]=a[(group+8)*16+tid*4+i];
        br[i]=b[batch*128+(tid*4+i)*8+group];
    }
    tc_dot16_exact(ar0,ar1,br,out,top);
    #pragma unroll
    for(int i=0;i<4;++i) {
        const auto index=batch*128+(group+(i>=2 ? 8 : 0))*8+tid*2+(i&1);
        c[index]=out[i];if(tops)tops[index]=top[i];
    }
}
