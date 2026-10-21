#pragma once
#include "../launch.h"
__global__ void s4_scatter_result_kernel(const unsigned long long *src,
    size_t stride,size_t first,size_t count,size_t m,int W,
    const unsigned long long *offsets,unsigned long long *dst)
{
    const size_t gid=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    const size_t words=count*(size_t)W;
    if(gid>=m*words)return;
    const size_t slice=gid/words,j=gid%words;
    dst[offsets[slice]+j]=src[(slice*stride+first)*W+j];
}

__global__ void groot_input_digest_kernel(const unsigned long long *input,size_t count,
    unsigned long long base,unsigned long long *digest) {
    unsigned long long sum=0,xorv=0;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<count;i+=gridDim.x*(size_t)blockDim.x) {
        const auto index=base+i,value=input[i];
        sum+=groot_mix64(value^groot_mix64(index+0x9e3779b97f4a7c15ull));
        xorv^=groot_mix64(value+groot_mix64(index+0xd1b54a32d192ed03ull));
    }
    __shared__ unsigned long long sums[256],xors[256];
    sums[threadIdx.x]=sum;xors[threadIdx.x]=xorv;__syncthreads();
    for(unsigned d=128;d;d/=2) {
        if(threadIdx.x<d){sums[threadIdx.x]+=sums[threadIdx.x+d];xors[threadIdx.x]^=xors[threadIdx.x+d];}
        __syncthreads();
    }
    if(!threadIdx.x){atomicAdd(digest,sums[0]);atomicXor(digest+1,xors[0]);}
}

__global__ void scaled_root_reverse_kernel(const unsigned long long *src,size_t count,
    size_t degree,int W,unsigned long long *dst) {
    const size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(i>=degree*(size_t)W)return;
    const size_t c=degree-1-i/W;
    dst[i]=c<count?src[c*W+i%W]:0ull;
}

__global__ void fold_reverse_kernel(const unsigned long long *src,size_t top,
    size_t count,int W,unsigned long long *out) {
    const size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(i<count*(size_t)W)out[i]=src[(top-i/W)*W+i%W];
}

__global__ void fold_subtract_kernel(const unsigned long long *a,const unsigned long long *b,
    const unsigned long long *n,int W,size_t count,unsigned long long *out,
    unsigned long long *length) {
    const size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;if(i>=count)return;
    a+=i*W;b+=i*W;out+=i*W;
    unsigned long long borrow=0;
    for(int j=0;j<W;++j) {
        const unsigned long long t=a[j]-b[j],b1=a[j]<b[j],v=t-borrow,b2=t<borrow;
        out[j]=v;borrow=b1|b2;
    }
    if(borrow) {
        unsigned long long carry=0;
        for(int j=0;j<W;++j) {
            const unsigned long long t=out[j]+n[j],c1=t<out[j],v=t+carry,c2=v<t;
            out[j]=v;carry=c1|c2;
        }
    }
    unsigned long long any=0;for(int j=0;j<W;++j)any|=out[j];
    if(any)atomicMax(length,(unsigned long long)i+1);
}

void ecm_launch_s4_scatter_result_kernel(EcmCudaLaunch cfg, const unsigned long long *src, size_t stride, size_t first, size_t count, size_t m, int W, const unsigned long long *offsets, unsigned long long *dst) {
    s4_scatter_result_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(src,stride,first,count,m,W,offsets,dst);
}

void ecm_launch_groot_input_digest_kernel(EcmCudaLaunch cfg, const unsigned long long *input, size_t count, unsigned long long base, unsigned long long *digest) {
    groot_input_digest_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(input,count,base,digest);
}

void ecm_launch_scaled_root_reverse_kernel(EcmCudaLaunch cfg, const unsigned long long *src, size_t count, size_t degree, int W, unsigned long long *dst) {
    scaled_root_reverse_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(src,count,degree,W,dst);
}

void ecm_launch_fold_reverse_kernel(EcmCudaLaunch cfg, const unsigned long long *src, size_t top, size_t count, int W, unsigned long long *out) {
    fold_reverse_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(src,top,count,W,out);
}

void ecm_launch_fold_subtract_kernel(EcmCudaLaunch cfg, const unsigned long long *a, const unsigned long long *b, const unsigned long long *n, int W, size_t count, unsigned long long *out, unsigned long long *length) {
    fold_subtract_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(a,b,n,W,count,out,length);
}
